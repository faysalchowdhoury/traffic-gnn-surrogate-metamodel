"""Surrogate serving.

``POST /v1/predict``  scenario (per-intersection volumes, greens, offsets, mode) -> per-intersection delay,
                     corridor fuel and CO2, from the PRODUCTION GNN (default) or LightGBM
``POST /v1/drift``    batch of scenarios -> KS drift report against the training sample

Models load once from the registry at start-up. Responses are memoised in an LRU keyed by the request
hash (a Redis cache would slot in the same way, as in the safety-ssam-trajectory-engine repository).
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .. import __version__
from ..data import NODE_FEATURES
from ..models import CorridorGNN, Scaler, gnn_predict, to_graphs
from ..registry import drift_report, production


class Node(BaseModel):
    vol_eb: float = Field(ge=0, le=4000)
    vol_wb: float = Field(ge=0, le=4000)
    vol_side: float = Field(ge=0, le=3000)
    vol_left: float = Field(ge=0, le=2000)
    green_art: float = Field(gt=0, le=150)
    green_side: float = Field(gt=0, le=150)
    offset_s: float = Field(ge=0, le=300)
    has_side_lefts: bool = True


class Scenario(BaseModel):
    nodes: list[Node] = Field(min_length=4, max_length=4)
    positions_m: list[float] = Field(default_factory=lambda: [0.0, 420.0, 820.0, 1350.0], min_length=4, max_length=4)
    cycle_s: float = 100.0
    speed_mps: float = 17.88
    actuated: bool = True
    model: Literal["lightgbm", "gnn"] = "gnn"


def scenario_row(s: Scenario) -> dict:  # type: ignore[type-arg]
    x = [[n.vol_eb, n.vol_wb, n.vol_side, n.vol_left, n.green_art, n.green_side, np.sin(2 * np.pi * n.offset_s / s.cycle_s),
          np.cos(2 * np.pi * n.offset_s / s.cycle_s), float(s.actuated), float(n.has_side_lefts)] for n in s.nodes]
    src, dst, ea = [], [], []
    for k in range(3):
        for a, b, d in ((k, k + 1, 1.0), (k + 1, k, -1.0)):
            tt = abs(s.positions_m[b] - s.positions_m[a]) / s.speed_mps
            dphi = 2 * np.pi * (s.nodes[b].offset_s - s.nodes[a].offset_s - tt) / s.cycle_s
            src.append(a)
            dst.append(b)
            ea.append([tt / s.cycle_s, np.sin(dphi), np.cos(dphi), d])
    return {"x": x, "edge_index": [src, dst], "edge_attr": ea}


def create_app(registry_root: str | Path = "registry", cache_size: int = 4096) -> FastAPI:
    root = Path(registry_root)
    app = FastAPI(title="Corridor surrogate", version=__version__)
    lgb_dir = production(root, "lightgbm")
    lgb_meta = json.loads((lgb_dir / "meta.json").read_text())
    boosters = [lgb.Booster(model_file=str(lgb_dir / f"model_{t}.txt")) for t in lgb_meta["targets"]]
    gnn_dir = production(root, "gnn")
    gm = json.loads((gnn_dir / "meta.json").read_text())
    gnn = CorridorGNN(gm["in_dim"], gm["edge_dim"])
    gnn.load_state_dict(torch.load(gnn_dir / "model.pt", weights_only=True))
    gnn.eval()
    xs, es = Scaler.from_dict(gm["x_scaler"]), Scaler.from_dict(gm["e_scaler"])
    train_feats = pd.read_csv(root / "training_features.csv")
    cache: OrderedDict[str, bytes] = OrderedDict()
    torch.set_num_threads(1)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__, "lightgbm": lgb_meta["version"], "gnn": gm["version"]}

    @app.post("/v1/predict")
    def predict(s: Scenario) -> Response:
        key = hashlib.sha1(s.model_dump_json().encode()).hexdigest()
        if key in cache:
            cache.move_to_end(key)
            return Response(cache[key], media_type="application/json", headers={"x-cache": "HIT"})
        row = scenario_row(s)
        if s.model == "lightgbm":
            flat = np.r_[np.asarray(row["x"]).ravel(), np.asarray(row["edge_attr"]).ravel()][None, :]
            y = [float(np.expm1(b.predict(flat)[0])) for b in boosters]
        else:
            pn, pg = gnn_predict(gnn, to_graphs([row], xs, es))
            y = [*map(float, pn[0]), *map(float, pg[0])]
        body = json.dumps({"model": s.model, "delay_s_per_veh": [round(v, 2) for v in y[:4]],
                           "fuel_gal_h": round(y[4], 2), "co2_kg_h": round(y[5], 1)}).encode()
        cache[key] = body
        while len(cache) > cache_size:
            cache.popitem(last=False)
        return Response(body, media_type="application/json", headers={"x-cache": "MISS"})

    @app.post("/v1/drift")
    def drift(batch: list[Scenario]) -> list[dict[str, object]]:
        if len(batch) < 20:
            raise HTTPException(422, "send at least 20 scenarios for a meaningful KS test")
        flat = [np.r_[np.asarray(scenario_row(s)["x"]).ravel(), np.asarray(scenario_row(s)["edge_attr"]).ravel()] for s in batch]
        df = pd.DataFrame(flat, columns=train_feats.columns)
        return drift_report(train_feats, df).head(10).to_dict("records")

    return app


assert len(NODE_FEATURES) == 10
