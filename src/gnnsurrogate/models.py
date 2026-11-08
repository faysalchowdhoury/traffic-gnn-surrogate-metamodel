"""Surrogate models.

* :class:`CorridorGNN`: node encoder, then 3 residual **GINE** message-passing layers (edge-conditioned,
  so the offset-phase relation between neighbouring signals enters the messages), a node head for
  per-intersection delay and a graph head (mean + sum pooling) for corridor fuel and CO2.
* LightGBM: one gradient-boosted regressor per target on flattened node and edge features.
* Ridge: linear baseline on the same features.
* HCM: analytic per-lane-group delay (random arrivals, PF = 1) from the scenario's volumes and greens,
  i.e. what a deterministic tool would say without simulation.

All learned models are fitted on ``log1p`` targets (delays are right-skewed) and standardised inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

from .data import NODE_FEATURES


class CorridorGNN(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.enc = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)), edge_dim=edge_dim)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.node_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.graph_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.enc(x))
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + torch.relu(conv(h, edge_index, edge_attr)))
        g = torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=1)
        return self.node_head(h).squeeze(-1), self.graph_head(g)


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, a: np.ndarray) -> "Scaler":
        return cls(a.mean(axis=0), np.where(a.std(axis=0) > 1e-9, a.std(axis=0), 1.0))

    def __call__(self, a: np.ndarray) -> np.ndarray:
        return (a - self.mean) / self.std

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> "Scaler":
        return cls(np.asarray(d["mean"]), np.asarray(d["std"]))


def to_graphs(rows: list[dict], xs: Scaler, es: Scaler) -> list[Data]:  # type: ignore[type-arg]
    out = []
    for r in rows:
        x = torch.tensor(xs(np.asarray(r["x"], float)), dtype=torch.float32)
        ea = torch.tensor(es(np.asarray(r["edge_attr"], float)), dtype=torch.float32)
        d = Data(x=x, edge_index=torch.tensor(r["edge_index"], dtype=torch.long), edge_attr=ea)
        if "y_node" in r:
            d.y_node = torch.tensor(np.log1p(r["y_node"]), dtype=torch.float32)
            d.y_graph = torch.tensor(np.log1p([[r["fuel_gal_h"], r["co2_kg_h"]]]), dtype=torch.float32)
        out.append(d)
    return out


def train_gnn(train: list[Data], val: list[Data], *, epochs: int = 300, lr: float = 3e-3, batch: int = 64, seed: int = 0,
              patience: int = 40) -> tuple[CorridorGNN, list[float]]:
    torch.manual_seed(seed)
    model = CorridorGNN(train[0].x.shape[1], train[0].edge_attr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    vb = Batch.from_data_list(val)
    best, best_state, bad, hist = np.inf, None, 0, []
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        idx = rng.permutation(len(train))
        for k in range(0, len(idx), batch):
            b = Batch.from_data_list([train[i] for i in idx[k:k + batch]])
            pn, pg = model(b.x, b.edge_index, b.edge_attr, b.batch)
            loss = nn.functional.mse_loss(pn, b.y_node) + nn.functional.mse_loss(pg, b.y_graph)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pn, pg = model(vb.x, vb.edge_index, vb.edge_attr, vb.batch)
            vl = float(nn.functional.mse_loss(pn, vb.y_node) + nn.functional.mse_loss(pg, vb.y_graph))
        hist.append(vl)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, hist


def gnn_predict(model: CorridorGNN, graphs: list[Data]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    b = Batch.from_data_list(graphs)
    with torch.no_grad():
        pn, pg = model(b.x, b.edge_index, b.edge_attr, b.batch)
    n_int = graphs[0].x.shape[0]
    return np.expm1(pn.numpy()).reshape(-1, n_int), np.expm1(pg.numpy())


def hcm_node_delay(x: np.ndarray, cycle: float) -> np.ndarray:
    """HCM-style node delay from raw node features (no progression: P = g/C), volume weighted."""
    from .sim.ntcip_ctrl.timing.hcm import lane_group_delay

    f = {k: i for i, k in enumerate(NODE_FEATURES)}
    out = []
    for row in x:
        groups = [
            (row[f["vol_eb"]], 3600.0, row[f["green_art"]]),
            (row[f["vol_wb"]], 3600.0, row[f["green_art"]]),
            (row[f["vol_side"]], 2 * 1800.0, row[f["green_side"]]),
            (row[f["vol_left"]], 4 * 1700.0, max(cycle - row[f["green_art"]] - row[f["green_side"]] - 24.0, 5.0)),
        ]
        num = den = 0.0
        for v, s, g in groups:
            if v <= 0:
                continue
            d = lane_group_delay(v, s, g, cycle).delay_s
            num += v * min(d, 600.0)
            den += v
        out.append(num / den if den else 0.0)
    return np.asarray(out)
