"""Build a :class:`Corridor` from a UTDF file plus a small geometry YAML."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from ..core.config import StrictModel, load_model
from ..core.exceptions import ConfigurationError
from .models import Approach, Corridor, CorridorIntersection
from .utdf import parse_utdf

MPH_TO_MPS = 0.44704


class CorridorConfig(StrictModel):
    name: str
    utdf: Path
    speed_mph: float = Field(gt=5.0, le=80.0)
    positions_m: dict[int, float]
    lanes: dict[int, int] = Field(default_factory=dict)
    sat_flow_vphpl: dict[str, float] = Field(default_factory=lambda: {"through": 1800.0, "left": 1700.0})


def build_corridor(config_path: str | Path, utdf_override: str | Path | None = None) -> Corridor:
    cfg_path = Path(config_path).resolve()
    cfg = load_model(cfg_path, CorridorConfig)
    utdf = Path(utdf_override) if utdf_override else (cfg.utdf if cfg.utdf.is_absolute() else cfg_path.parent / cfg.utdf)
    timings, volumes = parse_utdf(utdf)
    missing = set(cfg.positions_m) - set(timings)
    if missing:
        raise ConfigurationError("positions given for intersections absent from UTDF", context={"missing": sorted(missing)})
    items = []
    for iid, pos in sorted(cfg.positions_m.items(), key=lambda kv: kv[1]):
        t = timings[iid]
        approaches = {}
        for ph in t.phases:
            left = ph % 2 == 1
            approaches[ph] = Approach(
                volume_vph=volumes.get(iid, {}).get(ph, 0.0),
                lanes=cfg.lanes.get(ph, 1),
                sat_flow_vphpl=cfg.sat_flow_vphpl["left" if left else "through"],
            )
        items.append(CorridorIntersection(timing=t, position_m=pos, approaches=approaches))
    return Corridor(name=cfg.name, speed_mps=cfg.speed_mph * MPH_TO_MPS, intersections=items)
