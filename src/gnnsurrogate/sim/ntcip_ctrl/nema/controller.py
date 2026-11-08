"""Tick-based NEMA TS2 dual-ring, ring-barrier controller.

Modes
-----
``fixed``     pretimed: every phase is served every cycle and ends at its force-off.
``actuated``  actuated-coordinated with **fixed force-offs**:

* coordinated phases (default 2 + 6) rest in green and yield at their force-off point only
  if a conflicting phase has a call;
* non-coordinated phases need a call, time at least minimum green, extend by the passage
  (vehicle extension) timer and terminate on **gap-out**, **max-out** or **force-off**;
* a phase is skipped when it has no call or when less than its minimum green remains
  before its force-off (its permissive window has closed);
* unused time returns to the coordinated phases (**early return to green**).

Barrier rule: phases on the same side of a barrier time concurrently in the two rings,
and both rings must be ready to cross before either may. A ring that finishes early
holds its last phase in green. If one ring has no demand in a barrier group, its through
phase is brought up with the other ring's phase (**dual entry**).

Local cycle time ``tc`` is referenced to the begin of green of the coordinated phases.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from ..core.exceptions import ConfigurationError
from ..timing.models import BARRIER_A, BARRIER_B, IntersectionTiming, PhaseParams
from ..timing.schedule import nominal_schedule

ControllerMode = Literal["fixed", "actuated"]
_EPS = 1e-6


class TerminationReason(str, Enum):
    GAP_OUT = "gap_out"
    MAX_OUT = "max_out"
    FORCE_OFF = "force_off"
    YIELD = "coord_yield"


@dataclass
class _Ring:
    index: int
    groups: list[list[int]]  # [barrier A group, barrier B group]
    phase: int
    interval: str = "G"  # G, Y, R, W (waiting at barrier after clearance)
    timer: float = 0.0
    green_start_tc: float = 0.0
    ready: bool = False
    next_phase: int | None = None


@dataclass
class PhaseStats:
    green_time_s: float = 0.0
    services: int = 0
    terminations: Counter = field(default_factory=Counter)


class NemaController:
    """Emulated dual-ring controller.

    Args:
        timing: validated timing plan (cycle, offset, splits, sequence, intervals).
        mode: ``"fixed"`` or ``"actuated"``.
        dt: tick length in seconds.
        t0: absolute time of the first :meth:`step` call.
    """

    def __init__(self, timing: IntersectionTiming, mode: ControllerMode = "actuated", *, dt: float = 0.1,
                 t0: float = 0.0) -> None:
        if mode not in ("fixed", "actuated"):
            raise ConfigurationError("mode must be 'fixed' or 'actuated'", context={"mode": mode})
        self.timing = timing
        self.mode: ControllerMode = mode
        self.dt = dt
        self.schedule = nominal_schedule(timing)
        self.stats: dict[int, PhaseStats] = {p: PhaseStats() for p in timing.phases}
        self.gap: dict[int, float] = {p: 1e9 for p in timing.phases}
        self._calls: set[int] = set()
        self._locked: set[int] = set()  # detector memory: calls placed on non-green phases persist until served
        self.barrier = 0
        c1 = next(p for p in timing.ring1[0] if p in timing.coord_phases)
        c2 = next(p for p in timing.ring2[0] if p in timing.coord_phases)
        self.rings = [_Ring(0, timing.ring1, c1), _Ring(1, timing.ring2, c2)]
        for r in self.rings:
            self.stats[r.phase].services += 1
        # start at coordinated begin-of-green, then pre-roll with no demand so that the
        # controller is in its nominal state at t0
        tc0 = (t0 - timing.offset_s) % timing.cycle_s
        self.t = t0 - tc0
        while self.t < t0 - _EPS:
            self.step(set(), set())

    # ------------------------------------------------------------ properties
    @property
    def tc(self) -> float:
        return (self.t - self.timing.offset_s) % self.timing.cycle_s

    def params(self, phase: int) -> PhaseParams:
        return self.timing.phases[phase]

    def phase_state(self, phase: int) -> str:
        """``"G"``, ``"Y"`` or ``"R"`` for one phase."""
        for r in self.rings:
            if r.phase == phase and r.interval in ("G", "Y"):
                return r.interval
        return "R"

    def states(self) -> dict[int, str]:
        return {p: self.phase_state(p) for p in self.timing.phases}

    def greens(self) -> set[int]:
        return {r.phase for r in self.rings if r.interval == "G"}

    def update_timing(self, timing: IntersectionTiming) -> None:
        """Hot-swap timing parameters (e.g. from an NTCIP SET). Sequence/cycle changes take effect next cycle."""
        if set(timing.phases) != set(self.timing.phases):
            raise ConfigurationError("phase set cannot change at runtime")
        self.timing = timing
        self.schedule = nominal_schedule(timing)

    # ----------------------------------------------------------------- logic
    def _has_call(self, phase: int) -> bool:
        if self.mode == "fixed" or phase in self.timing.coord_phases:
            return True
        rc = self.params(phase).recall
        return rc in ("min", "max") or phase in self._calls or phase in self._locked

    def _force_off_elapsed(self, phase: int, start_tc: float) -> float:
        """Green time available from ``start_tc`` until the phase force-off (fixed force-off)."""
        fo = self.schedule[phase].green_end % self.timing.cycle_s
        return (fo - start_tc) % self.timing.cycle_s

    def _eligible(self, phase: int) -> bool:
        """Permissive-window check for a non-coordinated phase at the current tc.

        A phase may start only between the coordinated yield point and its own force-off,
        and only if at least its minimum green fits before the force-off.
        """
        if self.mode == "fixed" or phase in self.timing.coord_phases:
            return True
        c = self.timing.cycle_s
        remaining = self._force_off_elapsed(phase, self.tc)
        span = (self.schedule[phase].green_end - self._yield_tc()) % c
        return self.params(phase).min_green_s - _EPS <= remaining <= span + _EPS

    def _yield_tc(self) -> float:
        return min(self.schedule[p].green_end for p in self.timing.coord_phases) % self.timing.cycle_s

    def _wants_to_end(self, ring: _Ring) -> TerminationReason | None:
        p = ring.phase
        pp = self.params(p)
        g = ring.timer
        fo = self._force_off_elapsed(p, ring.green_start_tc)
        if self.mode == "fixed":
            return TerminationReason.FORCE_OFF if g >= fo - _EPS else None
        if p in self.timing.coord_phases:
            conflicting = any(q in self._calls or q in self._locked or self.params(q).recall != "none"
                              for q in self.timing.phases if q not in self.timing.coord_phases)
            return TerminationReason.YIELD if (g >= fo - _EPS and conflicting) else None
        if g < pp.min_green_s - _EPS:
            return None
        if g >= fo - _EPS:
            return TerminationReason.FORCE_OFF
        if g >= pp.max_s - _EPS:
            return TerminationReason.MAX_OUT
        if pp.recall == "max":
            return None
        if self.gap[p] >= pp.passage_s - _EPS:
            return TerminationReason.GAP_OUT
        return None

    def _next_in_group(self, ring: _Ring) -> int | None:
        group = ring.groups[self.barrier]
        if ring.phase not in group:
            return None
        for q in group[group.index(ring.phase) + 1 :]:
            if self._has_call(q) and self._eligible(q):
                return q
        return None

    def _start_green(self, ring: _Ring, phase: int) -> None:
        ring.phase, ring.interval, ring.timer = phase, "G", 0.0
        ring.green_start_tc = self.tc
        ring.ready = False
        ring.next_phase = None
        self._locked.discard(phase)
        self.stats[phase].services += 1

    def _cross_barrier(self) -> None:
        for _ in range(2):
            self.barrier = 1 - self.barrier
            picks: list[int | None] = []
            for ring in self.rings:
                pick = next((q for q in ring.groups[self.barrier] if self._has_call(q) and self._eligible(q)), None)
                picks.append(pick)
            if picks[0] is None and picks[1] is None:
                continue  # no demand on this side of the barrier: skip it
            for i, ring in enumerate(self.rings):
                if picks[i] is None:
                    other = picks[1 - i]
                    partner = {2: 6, 6: 2, 4: 8, 8: 4}.get(other or -1)
                    group = ring.groups[self.barrier]
                    picks[i] = partner if partner in group else group[-1]
            for ring, pick in zip(self.rings, picks):
                self._start_green(ring, pick)  # type: ignore[arg-type]
            return
        # nothing anywhere (should not happen: coordinated phases are always called)
        self.barrier = 0
        for ring in self.rings:
            self._start_green(ring, next(p for p in ring.groups[0] if p in self.timing.coord_phases))

    def step(self, calls: set[int], actuations: set[int]) -> None:
        """Advance one tick.

        Args:
            calls: phases with a vehicle call (presence on a stop-bar detector / waiting queue).
            actuations: phases whose detector was actuated during this tick (resets passage timers).
        """
        dt = self.dt
        self._calls = set(calls)
        live_greens = self.greens()
        self._locked |= {p for p in calls if p in self.timing.phases and p not in live_greens}
        for p in self.gap:
            self.gap[p] = 0.0 if (p in actuations) else self.gap[p] + dt

        # 1) green termination decisions (evaluated at the start of the tick)
        reasons: dict[int, TerminationReason] = {}
        for ring in self.rings:
            if ring.interval != "G":
                continue
            why = self._wants_to_end(ring)
            if why is None:
                ring.ready = False
                continue
            nxt = self._next_in_group(ring)
            if nxt is not None:  # move to the next phase in this ring, same barrier group
                ring.next_phase = nxt
                ring.interval, ring.timer, ring.ready = "Y", 0.0, False
                self.stats[ring.phase].terminations[why.value] += 1
            else:  # last phase before the barrier: hold green until the other ring is ready
                ring.ready = True
                reasons[ring.index] = why

        def barrier_bound(r: _Ring) -> bool:
            return (r.interval == "G" and r.ready) or (r.interval in ("Y", "R") and r.next_phase is None) or r.interval == "W"

        if any(r.interval == "G" and r.ready for r in self.rings) and all(barrier_bound(r) for r in self.rings):
            for ring in self.rings:
                if ring.interval == "G":
                    self.stats[ring.phase].terminations[reasons[ring.index].value] += 1
                    ring.interval, ring.timer, ring.next_phase, ring.ready = "Y", 0.0, None, False

        # 2) advance the clock and interval timers; greens that start now start at the new time
        self.t += dt
        for ring in self.rings:
            ring.timer += dt
            if ring.interval == "G":
                self.stats[ring.phase].green_time_s += dt
            elif ring.interval == "Y" and ring.timer >= self.params(ring.phase).yellow_s - _EPS:
                ring.interval, ring.timer = "R", 0.0
            elif ring.interval == "R" and ring.timer >= self.params(ring.phase).red_clear_s - _EPS:
                if ring.next_phase is not None:
                    self._start_green(ring, ring.next_phase)
                else:
                    ring.interval = "W"
        if all(r.interval == "W" for r in self.rings):
            self._cross_barrier()
