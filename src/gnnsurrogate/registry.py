"""File-based model registry and input-drift monitoring.

Layout::

    registry/<model>/v<N>/model.{pt,txt,json}   artefact
    registry/<model>/v<N>/meta.json             metrics, feature schema, scalers, data hash, created
    registry/<model>/PRODUCTION                 version string served by the API

Promotion is explicit (``promote``), so a retrained model never replaces production silently.
Drift: two-sample KS tests of a batch of incoming feature vectors against the training sample,
Bonferroni-corrected across features.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def data_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def register(root: str | Path, name: str, write_artifact, meta: dict[str, Any]) -> Path:  # type: ignore[no-untyped-def]
    base = Path(root) / name
    base.mkdir(parents=True, exist_ok=True)
    versions = sorted(int(p.name[1:]) for p in base.glob("v*") if p.name[1:].isdigit())
    v = (versions[-1] + 1) if versions else 1
    d = base / f"v{v}"
    d.mkdir()
    write_artifact(d)
    meta = {**meta, "version": f"v{v}", "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    return d


def promote(root: str | Path, name: str, version: str) -> None:
    d = Path(root) / name / version
    if not (d / "meta.json").is_file():
        raise FileNotFoundError(f"{name}/{version} not registered")
    (Path(root) / name / "PRODUCTION").write_text(version, encoding="utf-8")


def production(root: str | Path, name: str) -> Path:
    p = Path(root) / name / "PRODUCTION"
    if not p.is_file():
        raise FileNotFoundError(f"no production version for {name}")
    return Path(root) / name / p.read_text(encoding="utf-8").strip()


def drift_report(reference: pd.DataFrame, batch: pd.DataFrame, alpha: float = 0.01) -> pd.DataFrame:
    rows = []
    cols = [c for c in reference.columns if c in batch.columns and reference[c].std() > 0]
    for c in cols:
        r = stats.ks_2samp(reference[c].to_numpy(float), batch[c].to_numpy(float))
        rows.append({"feature": c, "ks_D": float(r.statistic), "p_bonferroni": min(1.0, float(r.pvalue) * len(cols))})
    df = pd.DataFrame(rows).sort_values("ks_D", ascending=False)
    df["drift"] = df["p_bonferroni"] < alpha
    return df.reset_index(drop=True)
