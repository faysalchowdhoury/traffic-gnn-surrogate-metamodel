"""Nominal (fixed-time) phase schedule and coordination force-off points.

Local cycle time ``tc = (t - offset) mod C`` is referenced to the **start of green of the
coordinated phases** (Synchro "Referenced to: Begin of Green", reference phase 2+6).
With the ring sequence ``[[1, 2], [3, 4]]`` the cycle seen from tc = 0 is::

    | phi2 (coord) | phi3 phi4 | phi1 | -> back to phi2
      barrier A tail  barrier B  barrier A head

so leading lefts in barrier A are timed at the *end* of the local cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import IntersectionTiming


@dataclass(frozen=True)
class PhaseWindow:
    phase: int
    start: float  # green start, local cycle time
    green_end: float  # = force-off point
    yellow_end: float
    end: float


def nominal_schedule(timing: IntersectionTiming) -> dict[int, PhaseWindow]:
    """Phase windows in local cycle time for both rings (values may exceed C; use mod C)."""
    c = timing.cycle_s
    out: dict[int, PhaseWindow] = {}
    for ring in (timing.ring1, timing.ring2):
        group_a, group_b = ring
        coord = [p for p in group_a if p in timing.coord_phases]
        if not coord:
            raise ValueError("each ring's barrier A group must contain a coordinated phase")
        k = group_a.index(coord[0])
        # from tc=0: coord phase and any lagging phases of barrier A, then barrier B, then leading A phases
        order = group_a[k:] + group_b + group_a[:k]
        t = 0.0
        for ph in order:
            pp = timing.phases[ph]
            out[ph] = PhaseWindow(ph, t, t + pp.green_s, t + pp.green_s + pp.yellow_s, t + pp.split_s)
            t += pp.split_s
        if abs(t - c) > 0.05:
            raise ValueError("ring does not sum to the cycle length")
    return out


def coord_green_windows(timing: IntersectionTiming, phase: int) -> tuple[float, float]:
    """``(start, end)`` of a phase's nominal green in local cycle time."""
    w = nominal_schedule(timing)[phase]
    return w.start, w.green_end
