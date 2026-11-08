"""Train, evaluate, register and benchmark all surrogates."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from .data import flat_table, load
from .models import Scaler, gnn_predict, hcm_node_delay, to_graphs, train_gnn
from .registry import data_hash, drift_report, promote, register

TARGETS = ["delay_int1", "delay_int2", "delay_int3", "delay_int4", "fuel_gal_h", "co2_kg_h"]


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p)),
            "mape": float(np.mean(np.abs(p - y) / np.maximum(np.abs(y), 1e-6)) * 100)}


def run(data_path: str | Path, out_dir: str | Path, registry_root: str | Path, *, seed: int = 0, cycle: float = 100.0) -> dict[str, Any]:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rows = load(data_path)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    n_tr, n_va = int(0.7 * len(rows)), int(0.15 * len(rows))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    X, Yn, Yg = flat_table(rows)
    Y = np.c_[Yn, Yg]
    res: dict[str, Any] = {"n_scenarios": len(rows), "split": [len(tr), len(va), len(te)], "models": {}, "latency_ms": {}}

    # ---- ridge
    xs = Scaler.fit(X.to_numpy()[tr])
    t0 = time.perf_counter()
    ridge = Ridge(alpha=1.0).fit(xs(X.to_numpy()[tr]), np.log1p(Y[tr]))
    ridge_s = time.perf_counter() - t0
    p_ridge = np.expm1(ridge.predict(xs(X.to_numpy()[te])))
    # ---- LightGBM
    t0 = time.perf_counter()
    boosters = []
    for k in range(Y.shape[1]):
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
        m.fit(X.iloc[tr], np.log1p(Y[tr, k]), eval_set=[(X.iloc[va], np.log1p(Y[va, k]))],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        boosters.append(m)
    lgb_s = time.perf_counter() - t0
    p_lgb = np.column_stack([np.expm1(m.predict(X.iloc[te])) for m in boosters])
    # ---- GNN
    nx = np.vstack([np.asarray(rows[i]["x"]) for i in tr])
    ne = np.vstack([np.asarray(rows[i]["edge_attr"]) for i in tr])
    gxs, ges = Scaler.fit(nx), Scaler.fit(ne)
    g_all = to_graphs(rows, gxs, ges)
    t0 = time.perf_counter()
    gnn, hist = train_gnn([g_all[i] for i in tr], [g_all[i] for i in va], seed=seed)
    gnn_s = time.perf_counter() - t0
    pn, pg = gnn_predict(gnn, [g_all[i] for i in te])
    p_gnn = np.c_[pn, pg]
    # ---- HCM analytic (node delays only)
    p_hcm = np.vstack([hcm_node_delay(np.asarray(rows[i]["x"]), cycle) for i in te])

    for name, p, fit_s in (("ridge", p_ridge, ridge_s), ("lightgbm", p_lgb, lgb_s), ("gnn", p_gnn, gnn_s)):
        res["models"][name] = {"fit_s": fit_s, **{t: _metrics(Y[te, k], p[:, k]) for k, t in enumerate(TARGETS)}}
        res["models"][name]["mean_node_mae"] = float(np.mean([res["models"][name][t]["mae"] for t in TARGETS[:4]]))
        res["models"][name]["mean_node_r2"] = float(np.mean([res["models"][name][t]["r2"] for t in TARGETS[:4]]))
    res["models"]["hcm"] = {t: _metrics(Y[te, k], p_hcm[:, k]) for k, t in enumerate(TARGETS[:4])}
    res["models"]["hcm"]["mean_node_mae"] = float(np.mean([res["models"]["hcm"][t]["mae"] for t in TARGETS[:4]]))
    res["models"]["hcm"]["mean_node_r2"] = float(np.mean([res["models"]["hcm"][t]["r2"] for t in TARGETS[:4]]))

    # ---- single-scenario latency (in-process, 2,000 calls)
    one = X.iloc[[te[0]]]
    g_one = [g_all[te[0]]]

    def timeit(fn, n: int = 2000) -> dict[str, float]:  # type: ignore[no-untyped-def]
        fn()
        t = []
        for _ in range(n):
            s = time.perf_counter()
            fn()
            t.append((time.perf_counter() - s) * 1000)
        a = np.asarray(t)
        return {"p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99))}

    torch.set_num_threads(1)
    res["latency_ms"]["lightgbm_6_targets"] = timeit(lambda: [m.predict(one) for m in boosters], 1000)
    res["latency_ms"]["gnn_all_targets"] = timeit(lambda: gnn_predict(gnn, g_one), 1000)
    res["latency_ms"]["ridge"] = timeit(lambda: ridge.predict(xs(one.to_numpy())), 2000)
    sim_rt = float(np.mean([r["runtime_s"] for r in rows]))
    res["simulation_s_per_scenario"] = sim_rt

    # ---- registry
    dh = data_hash(data_path)
    reg = Path(registry_root)

    def save_lgb(d: Path) -> None:
        for k, m in enumerate(boosters):
            m.booster_.save_model(str(d / f"model_{TARGETS[k]}.txt"))

    d_lgb = register(reg, "lightgbm", save_lgb, {"features": list(X.columns), "targets": TARGETS, "data_hash": dh,
                                                  "metrics": res["models"]["lightgbm"]})
    d_gnn = register(reg, "gnn", lambda d: torch.save(gnn.state_dict(), d / "model.pt"),
                     {"targets": TARGETS, "data_hash": dh, "metrics": res["models"]["gnn"], "x_scaler": gxs.to_dict(),
                      "e_scaler": ges.to_dict(), "in_dim": int(nx.shape[1]), "edge_dim": int(ne.shape[1])})
    promote(reg, "lightgbm", d_lgb.name)
    promote(reg, "gnn", d_gnn.name)
    X.iloc[tr].to_csv(reg / "training_features.csv", index=False)

    # ---- drift demonstration: +30% demand on held-out scenarios
    shifted = X.iloc[te].copy()
    vol_cols = [c for c in X.columns if "vol_" in c]
    shifted[vol_cols] *= 1.3
    dr = drift_report(X.iloc[tr], shifted)
    same = drift_report(X.iloc[tr], X.iloc[te])
    res["drift"] = {"shifted_flagged": int(dr["drift"].sum()), "unshifted_flagged": int(same["drift"].sum()),
                    "features": len(dr), "top": dr.head(5).to_dict("records")}
    res["gnn_val_curve"] = hist
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    _plot(Y[te], {"Ridge": p_ridge, "LightGBM": p_lgb, "GNN": p_gnn}, p_hcm, out / "figures")
    return res


def _plot(y: np.ndarray, preds: dict[str, np.ndarray], hcm: np.ndarray, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, (name, p) in zip(axes, [("HCM analytic", hcm), *preds.items()]):
        yy, pp = y[:, :4].ravel(), p[:, :4].ravel()
        ax.scatter(yy, pp, s=5, alpha=0.4)
        lim = [0, float(np.percentile(yy, 99.5)) * 1.1]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"{name}: R² {r2_score(yy, pp):.2f}", fontsize=10)
        ax.set_xlabel("simulated intersection delay (s/veh)")
    axes[0].set_ylabel("predicted")
    fig.tight_layout()
    fig.savefig(fig_dir / "parity.png", dpi=130)
    plt.close(fig)


def write_report(s: dict[str, Any], out: Path) -> Path:
    L = ["# Surrogate results", "", f"{s['n_scenarios']} simulated scenarios (train/val/test {s['split']}); simulation cost "
         f"{s['simulation_s_per_scenario']:.2f} s per scenario.", "",
         "| Model | Mean node-delay MAE (s/veh) | Mean node R² | Fuel MAE (gal/h) | Fuel R² | CO2 R² | Fit time (s) |", "|---|---|---|---|---|---|---|"]
    for name in ("hcm", "ridge", "lightgbm", "gnn"):
        m = s["models"][name]
        fuel = m.get("fuel_gal_h", {"mae": float("nan"), "r2": float("nan")})
        co2 = m.get("co2_kg_h", {"r2": float("nan")})
        L.append(f"| {name} | {m['mean_node_mae']:.2f} | {m['mean_node_r2']:.3f} | {fuel['mae']:.2f} | {fuel['r2']:.3f} | {co2['r2']:.3f} | "
                 f"{m.get('fit_s', float('nan')):.1f} |")
    L += ["", "## Single-scenario inference latency (in-process)", "", "| Model | p50 (ms) | p99 (ms) |", "|---|---|---|"]
    for k, v in s["latency_ms"].items():
        L.append(f"| {k} | {v['p50']:.3f} | {v['p99']:.3f} |")
    d = s["drift"]
    L += ["", f"## Drift check", "", f"Held-out scenarios vs training: {d['unshifted_flagged']}/{d['features']} features flagged. "
          f"Same scenarios with +30% demand: {d['shifted_flagged']}/{d['features']} flagged.", "", "![Parity](figures/parity.png)", ""]
    md = out / "RESULTS.md"
    md.write_text("\n".join(L), encoding="utf-8")
    return md
