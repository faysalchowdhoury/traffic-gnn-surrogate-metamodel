"""HCM 7th Edition signalized lane-group control delay (Chapter 19, Eq. 19-18 to 19-26).

``d = d1 * PF + d2 + d3`` with

* uniform delay ``d1 = 0.5 C (1 - g/C)^2 / (1 - min(1, X) g/C)``
* progression adjustment ``PF = (1 - P) / (1 - g/C)`` (P = share of arrivals on green)
* incremental delay ``d2 = 900 T [(X - 1) + sqrt((X - 1)^2 + 8 k I X / (c T))]``
* initial-queue delay ``d3 = 0`` (no residual queue assumed)

This is the deterministic "Synchro-side" estimate used to sanity-check the simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

LOS_THRESHOLDS: tuple[tuple[float, str], ...] = ((10, "A"), (20, "B"), (35, "C"), (55, "D"), (80, "E"))


def los(delay_s: float, v_c: float | None = None) -> str:
    if delay_s is None or math.isnan(delay_s):
        return "n/a"
    if v_c is not None and v_c > 1.0:
        return "F"
    for lim, letter in LOS_THRESHOLDS:
        if delay_s <= lim:
            return letter
    return "F"


@dataclass(frozen=True)
class LaneGroupDelay:
    capacity_vph: float
    v_c: float
    d1: float
    pf: float
    d2: float

    @property
    def delay_s(self) -> float:
        return self.d1 * self.pf + self.d2

    @property
    def los(self) -> str:
        return los(self.delay_s, self.v_c)


def lane_group_delay(
    volume_vph: float,
    sat_flow_vph: float,
    effective_green_s: float,
    cycle_s: float,
    *,
    arrivals_on_green: float | None = None,
    analysis_h: float = 0.25,
    k: float = 0.5,
    upstream_i: float = 1.0,
) -> LaneGroupDelay:
    """HCM control delay for one lane group (s/veh)."""
    g_c = max(min(effective_green_s / cycle_s, 1.0), 1e-6)
    cap = sat_flow_vph * g_c
    x = volume_vph / cap if cap > 0 else math.inf
    d1 = 0.5 * cycle_s * (1.0 - g_c) ** 2 / (1.0 - min(1.0, x) * g_c)
    if arrivals_on_green is None:
        pf = 1.0
    else:
        pf = (1.0 - arrivals_on_green) / (1.0 - g_c) if g_c < 1.0 else 1.0
    t = analysis_h
    d2 = 900.0 * t * ((x - 1.0) + math.sqrt((x - 1.0) ** 2 + 8.0 * k * upstream_i * x / (cap * t))) if cap > 0 else math.inf
    return LaneGroupDelay(cap, x, d1, pf, d2)
