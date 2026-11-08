"""Scenario sampling and simulation-based dataset generation.

Each scenario perturbs the base 4-signal corridor (``configs/``):

* demand: global factor U(0.6, 1.25) times per-intersection factors for side-street through
  U(0.5, 1.6) and left turns U(0.5, 1.8), and per-direction arterial factors U(0.6, 1.3);
* timing: per-intersection split shift U(-8, +8) s moved between the coordinated phases (2 + 6) and the
  side-street through phases (4 + 8), which keeps the barrier sums equal; offsets U(0, C);
* control: fixed-time or actuated-coordinated with equal probability.

The closed-loop mesoscopic model (vendored from ``synchro-vissim-ntcip-controller``) then gives:

* per-intersection average stop-line delay (s/veh), the node targets;
* corridor fuel (gal/h) with the Synchro/TRANSYT-style model
  ``F = VMT k1 + delay_vh k2 + stops k3``, with ``k1 = 0.075283 - 0.0015892 S + 0.000015066 S^2`` gal/veh-mi,
  ``k2 = 0.7329`` gal/veh-h and ``k3 = 0.0000061411 S^2`` gal/stop at cruise speed S = 40 mph,
  and CO2 = 8,887 g per gallon of gasoline. These are the graph targets.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .sim.ntcip_ctrl.corridor.meso import simulate
from .sim.ntcip_ctrl.timing.corridor_config import build_corridor
from .sim.ntcip_ctrl.timing.models import Corridor, IntersectionTiming

SPEED_MPH = 40.0
K1 = 0.075283 - 0.0015892 * SPEED_MPH + 0.000015066 * SPEED_MPH ** 2
K2 = 0.7329
K3 = 0.0000061411 * SPEED_MPH ** 2
CO2_G_PER_GAL = 8887.0
APPROACH_MI = 0.25  # length of each approach counted for VMT
NODE_FEATURES = ("vol_eb", "vol_wb", "vol_side", "vol_left", "green_art", "green_side", "off_sin", "off_cos", "actuated", "has_side_lefts")


@dataclass(frozen=True)
class Scenario:
    sid: int
    demand: float
    art: tuple[float, float]
    side: tuple[float, ...]
    left: tuple[float, ...]
    shift: tuple[float, ...]
    offsets: tuple[float, ...]
    actuated: bool


def sample(n: int, n_int: int, cycle: float, seed: int = 0) -> list[Scenario]:
    r = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(Scenario(
            i, float(r.uniform(0.6, 1.25)), (float(r.uniform(0.6, 1.3)), float(r.uniform(0.6, 1.3))),
            tuple(float(x) for x in r.uniform(0.5, 1.6, n_int)), tuple(float(x) for x in r.uniform(0.5, 1.8, n_int)),
            tuple(float(x) for x in r.uniform(-8, 8, n_int)), tuple(float(x) for x in r.uniform(0, cycle, n_int)),
            bool(r.random() < 0.5)))
    return out


def apply(base: Corridor, s: Scenario) -> Corridor:
    """Return a corridor with the scenario's demand, split shifts and offsets."""
    ints = []
    for k, it in enumerate(base.intersections):
        t: IntersectionTiming = it.timing
        sh = s.shift[k]
        try:
            # model_validate (not model_copy) so min-green and barrier checks run on the shifted splits
            phases = {ph: pp.model_dump() for ph, pp in t.phases.items()}
            for ph, sign in ((2, 1), (6, 1), (4, -1), (8, -1)):
                if ph in phases:
                    phases[ph]["split_s"] += sign * sh
            t2 = IntersectionTiming.model_validate({**t.model_dump(), "phases": phases, "offset_s": s.offsets[k]})
        except ValueError:
            t2 = t.model_copy(update={"offset_s": s.offsets[k]})  # infeasible shift: keep original splits
        aps = {}
        for ph, ap in it.approaches.items():
            f = s.demand
            if ph == 2:
                f *= s.art[0]
            elif ph == 6:
                f *= s.art[1]
            elif ph in (4, 8):
                f *= s.side[k]
            else:
                f *= s.left[k]
            aps[ph] = ap.model_copy(update={"volume_vph": ap.volume_vph * f})
        ints.append(it.model_copy(update={"timing": t2, "approaches": aps}))
    return base.model_copy(update={"intersections": ints})


def features(c: Corridor, actuated: bool) -> np.ndarray:
    """Node feature matrix (n_int, len(NODE_FEATURES))."""
    rows = []
    for it in c.intersections:
        t, a = it.timing, it.approaches
        vol = lambda ph: a[ph].volume_vph if ph in a else 0.0  # noqa: E731
        g = lambda ph: t.phases[ph].green_s if ph in t.phases else 0.0  # noqa: E731
        ang = 2 * np.pi * t.offset_s / t.cycle_s
        rows.append([vol(2), vol(6), vol(4) + vol(8), vol(1) + vol(5) + vol(3) + vol(7), g(2), g(4) or g(8),
                     np.sin(ang), np.cos(ang), float(actuated), float(3 in t.phases)])
    return np.asarray(rows, float)


def edges(c: Corridor) -> tuple[np.ndarray, np.ndarray]:
    """Directed edges between adjacent intersections with features
    [travel time / C, sin and cos of (offset difference - travel time) phase, direction]."""
    src, dst, feat = [], [], []
    x = [it.position_m for it in c.intersections]
    for k in range(len(x) - 1):
        for a, b, direction in ((k, k + 1, 1.0), (k + 1, k, -1.0)):
            tt = abs(x[b] - x[a]) / c.speed_mps
            dphi = 2 * np.pi * (c.intersections[b].timing.offset_s - c.intersections[a].timing.offset_s - tt) / c.cycle_s
            src.append(a)
            dst.append(b)
            feat.append([tt / c.cycle_s, np.sin(dphi), np.cos(dphi), direction])
    return np.array([src, dst], int), np.asarray(feat, float)


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    base = Corridor.model_validate(payload["corridor"])
    s = Scenario(**payload["scenario"])
    c = apply(base, s)
    res = simulate(c, mode="actuated" if s.actuated else "fixed", seed=payload["seed"], duration_s=payload["duration_s"],
                   warmup_s=payload["warmup_s"])
    mv = res.movements
    hours = (payload["duration_s"] - payload["warmup_s"]) / 3600.0
    node_delay = []
    for it in c.intersections:
        m = mv[mv["int_id"] == it.timing.int_id]
        w = m["served"].to_numpy(float)
        node_delay.append(float((w * m["mean_delay_s"].fillna(0)).sum() / max(w.sum(), 1.0)))
    served = float(mv["served"].sum())
    delay_vh = float((mv["served"] * mv["mean_delay_s"].fillna(0)).sum()) / 3600.0
    stops = float((mv["served"] * mv["stop_rate"].fillna(0)).sum())
    fuel_gal = served * APPROACH_MI * K1 + delay_vh * K2 + stops * K3
    edge_index, edge_attr = edges(c)
    return {"sid": s.sid, "x": features(c, s.actuated).tolist(), "edge_index": edge_index.tolist(), "edge_attr": edge_attr.tolist(),
            "y_node": node_delay, "fuel_gal_h": fuel_gal / hours, "co2_kg_h": fuel_gal * CO2_G_PER_GAL / 1000.0 / hours,
            "runtime_s": res.runtime_s}


def generate(config: str | Path, out: str | Path, n: int, *, seed: int = 0, workers: int = 2, duration_s: float = 1800.0,
             warmup_s: float = 300.0) -> Path:
    base = build_corridor(config)
    scen = sample(n, len(base.intersections), base.cycle_s, seed)
    payloads = [{"corridor": base.model_dump(mode="json"), "scenario": s.__dict__, "seed": 1000 + s.sid,
                 "duration_s": duration_s, "warmup_s": warmup_s} for s in scen]
    with ProcessPoolExecutor(max(1, min(workers, os.cpu_count() or 1))) as pool:
        rows = list(pool.map(_run, payloads, chunksize=8))
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def load(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def flat_table(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Flattened features for tree / linear models (node features + edge phase features), targets."""
    feats, ynode, ygraph = [], [], []
    for r in rows:
        x = np.asarray(r["x"]).ravel()
        e = np.asarray(r["edge_attr"]).ravel()
        feats.append(np.r_[x, e])
        ynode.append(r["y_node"])
        ygraph.append([r["fuel_gal_h"], r["co2_kg_h"]])
    n_int = len(rows[0]["x"])
    cols = [f"n{k}_{f}" for k in range(n_int) for f in NODE_FEATURES]
    cols += [f"e{k}_{f}" for k in range(len(rows[0]["edge_attr"])) for f in ("tt", "dphi_sin", "dphi_cos", "dir")]
    return pd.DataFrame(feats, columns=cols), np.asarray(ynode), np.asarray(ygraph)
