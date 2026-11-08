from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gnnsurrogate.data import _run, apply, edges, features, flat_table, sample
from gnnsurrogate.models import CorridorGNN, Scaler, gnn_predict, hcm_node_delay, to_graphs, train_gnn
from gnnsurrogate.registry import drift_report, production, promote, register
from gnnsurrogate.sim.ntcip_ctrl.timing.corridor_config import build_corridor

ROOT = Path(__file__).resolve().parents[1]
BASE = build_corridor(ROOT / "configs" / "corridor.yaml")


def test_scenario_application_preserves_cycle_and_barriers() -> None:
    for s in sample(30, 4, BASE.cycle_s, seed=1):
        c = apply(BASE, s)
        for it in c.intersections:
            t = it.timing
            assert sum(t.phases[p].split_s for p in t.ring1[0] + t.ring1[1]) == pytest.approx(t.cycle_s)
            assert 0 <= t.offset_s < t.cycle_s


def test_features_and_edges_shapes() -> None:
    x = features(BASE, True)
    ei, ea = edges(BASE)
    assert x.shape == (4, 10) and ei.shape == (2, 6) and ea.shape == (6, 4)
    assert np.allclose(np.hypot(ea[:, 1], ea[:, 2]), 1.0)


def test_one_simulation_row() -> None:
    s = sample(1, 4, BASE.cycle_s, seed=3)[0]
    r = _run({"corridor": BASE.model_dump(mode="json"), "scenario": s.__dict__, "seed": 1, "duration_s": 700.0, "warmup_s": 100.0})
    assert len(r["y_node"]) == 4 and all(d >= 0 for d in r["y_node"])
    assert r["fuel_gal_h"] > 0 and r["co2_kg_h"] == pytest.approx(r["fuel_gal_h"] * 8.887, rel=1e-6)


def _fake_rows(n: int) -> list[dict]:  # type: ignore[type-arg]
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        x = rng.uniform(0, 1, (4, 10))
        ei, ea = edges(BASE)
        y = (5 + 20 * x[:, 0] + 10 * x[:, 2]).tolist()
        rows.append({"x": x.tolist(), "edge_index": ei.tolist(), "edge_attr": ea.tolist(), "y_node": y,
                     "fuel_gal_h": 50 + sum(y), "co2_kg_h": (50 + sum(y)) * 8.887, "runtime_s": 0.1})
    return rows


def test_gnn_learns_a_simple_function() -> None:
    rows = _fake_rows(300)
    nx = np.vstack([r["x"] for r in rows])
    ne = np.vstack([r["edge_attr"] for r in rows])
    g = to_graphs(rows, Scaler.fit(nx), Scaler.fit(ne))
    model, hist = train_gnn(g[:240], g[240:], epochs=80, patience=80)
    pn, _ = gnn_predict(model, g[240:])
    y = np.asarray([r["y_node"] for r in rows[240:]])
    assert np.mean(np.abs(pn - y)) < 2.0
    assert hist[-1] < hist[0]


def test_flat_table_and_hcm_baseline() -> None:
    X, yn, yg = flat_table(_fake_rows(5))
    assert X.shape == (5, 40 + 24) and yn.shape == (5, 4) and yg.shape == (5, 2)
    d = hcm_node_delay(features(BASE, False), BASE.cycle_s)
    assert d.shape == (4,) and np.all(d > 0)


def test_registry_and_drift(tmp_path: Path) -> None:
    d1 = register(tmp_path, "m", lambda d: (d / "model.txt").write_text("a"), {"metrics": {"r2": 0.9}})
    d2 = register(tmp_path, "m", lambda d: (d / "model.txt").write_text("b"), {"metrics": {"r2": 0.95}})
    assert d1.name == "v1" and d2.name == "v2"
    promote(tmp_path, "m", "v2")
    assert production(tmp_path, "m") == d2
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)})
    same = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
    shift = pd.DataFrame({"a": rng.normal(1, 1, 200), "b": rng.normal(0, 1, 200)})
    assert not drift_report(ref, same)["drift"].any()
    rep = drift_report(ref, shift)
    assert rep.loc[rep["feature"] == "a", "drift"].item() and not rep.loc[rep["feature"] == "b", "drift"].item()


def test_model_forward_shapes() -> None:
    m = CorridorGNN(10, 4)
    x = torch.randn(8, 10)
    ei = torch.tensor([[0, 1, 1, 2, 4, 5, 5, 6], [1, 0, 2, 1, 5, 4, 6, 5]])
    pn, pg = m(x, ei, torch.randn(8, 4), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
    assert pn.shape == (8,) and pg.shape == (2, 2)


@pytest.mark.skipif(not (ROOT / "registry" / "lightgbm" / "PRODUCTION").exists(), reason="run `gnnsurrogate train` first")
def test_api_predict_cache_and_drift() -> None:
    from fastapi.testclient import TestClient

    from gnnsurrogate.api.app import create_app

    cl = TestClient(create_app(ROOT / "registry"))
    node = {"vol_eb": 900, "vol_wb": 800, "vol_side": 500, "vol_left": 250, "green_art": 50, "green_side": 30, "offset_s": 0}
    body = {"nodes": [node] * 4}
    r1 = cl.post("/v1/predict", json=body)
    assert r1.status_code == 200 and r1.headers["x-cache"] == "MISS"
    out = r1.json()
    assert len(out["delay_s_per_veh"]) == 4 and all(d > 0 for d in out["delay_s_per_veh"]) and out["fuel_gal_h"] > 0
    assert cl.post("/v1/predict", json=body).headers["x-cache"] == "HIT"
    g = cl.post("/v1/predict", json={**body, "model": "lightgbm"}).json()
    assert out["model"] == "gnn" and g["model"] == "lightgbm" and len(g["delay_s_per_veh"]) == 4
    assert cl.post("/v1/predict", json={"nodes": [node] * 3}).status_code == 422
    assert cl.post("/v1/drift", json=[body] * 5).status_code == 422
    assert cl.post("/v1/drift", json=[body] * 25).status_code == 200
