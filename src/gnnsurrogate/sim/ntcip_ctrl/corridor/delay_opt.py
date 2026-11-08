"""Simulation-based offset optimisation (minimise total vehicle delay).

Bandwidth maximisation (MAXBAND/PASSER style) rewards wide green bands but ignores
volumes and queues, and on unevenly spaced arterials it often settles on one-way
progression. Synchro instead optimises offsets on delay. This module does the latter
with the closed-loop mesoscopic model as the objective:

* coordinate descent over intersections 2..N on a ``step_s`` grid;
* every candidate is evaluated on the **same** random seeds (common random numbers), so
  candidate differences are not swamped by demand noise;
* objective = total stop-line delay of all served vehicles (veh-h), i.e. volume weighted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.logging import get_logger
from ..nema.controller import ControllerMode
from ..timing.models import Corridor
from .meso import simulate

log = get_logger(__name__)


@dataclass
class DelayOptResult:
    offsets_s: tuple[float, ...]
    objective_veh_h: float
    evaluations: int
    history: list[tuple[tuple[float, ...], float]] = field(default_factory=list)


def total_delay_veh_h(corridor: Corridor, offsets: list[float], *, mode: ControllerMode, seeds: list[int],
                      duration_s: float, warmup_s: float) -> float:
    tot = 0.0
    for s in seeds:
        mv = simulate(corridor, mode=mode, offsets=offsets, seed=s, duration_s=duration_s, warmup_s=warmup_s).movements
        tot += float((mv["served"] * mv["mean_delay_s"].fillna(0.0)).sum()) / 3600.0
    return tot / len(seeds)


def optimize_offsets_delay(
    corridor: Corridor,
    *,
    mode: ControllerMode = "actuated",
    start: list[float] | None = None,
    seeds: list[int] | None = None,
    step_s: float = 5.0,
    sweeps: int = 3,
    duration_s: float = 1500.0,
    warmup_s: float = 300.0,
) -> DelayOptResult:
    """Coordinate-descent minimisation of total delay; returns the best offsets found."""
    seeds = seeds or [1001, 1002]
    c = corridor.cycle_s
    off = list(start) if start is not None else [i.timing.offset_s for i in corridor.intersections]
    off[0] = 0.0
    cand = np.arange(0.0, c, step_s)
    best = total_delay_veh_h(corridor, off, mode=mode, seeds=seeds, duration_s=duration_s, warmup_s=warmup_s)
    evals = 1
    history = [(tuple(off), best)]
    for sweep in range(sweeps):
        improved = False
        for k in range(1, len(off)):
            for o in cand:
                if abs(o - off[k]) < 1e-9:
                    continue
                trial = off.copy()
                trial[k] = float(o)
                val = total_delay_veh_h(corridor, trial, mode=mode, seeds=seeds, duration_s=duration_s,
                                        warmup_s=warmup_s)
                evals += 1
                if val < best - 1e-9:
                    best, off, improved = val, trial, True
                    history.append((tuple(off), best))
        log.info("delay_opt.sweep", extra={"sweep": sweep, "objective_veh_h": round(best, 3), "offsets": off})
        if not improved:
            break
    return DelayOptResult(tuple(off), best, evals, history)
