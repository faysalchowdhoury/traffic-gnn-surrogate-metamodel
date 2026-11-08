"""Typed signal timing for one intersection and a corridor.

NEMA eight-phase convention used throughout (right-hand traffic):

=====  ======  =====  ======
Ring1  move    Ring2  move
=====  ======  =====  ======
phi1   WBL     phi5   EBL
phi2   EBT     phi6   WBT
phi3   NBL     phi7   SBL
phi4   SBT     phi8   NBT
=====  ======  =====  ======

Barrier A = {1, 2, 5, 6}, barrier B = {3, 4, 7, 8}. Coordinated phases default to 2 and 6.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..core.config import StrictModel

BARRIER_A: frozenset[int] = frozenset({1, 2, 5, 6})
BARRIER_B: frozenset[int] = frozenset({3, 4, 7, 8})
RING1: frozenset[int] = frozenset({1, 2, 3, 4})
PHASE_MOVEMENT: dict[int, str] = {1: "WBL", 2: "EBT", 3: "NBL", 4: "SBT", 5: "EBL", 6: "WBT", 7: "SBL", 8: "NBT"}
MOVEMENT_PHASE: dict[str, int] = {v: k for k, v in PHASE_MOVEMENT.items()}
DUAL_ENTRY_PARTNER: dict[int, int] = {2: 6, 6: 2, 4: 8, 8: 4}

Recall = Literal["none", "min", "max"]


class PhaseParams(StrictModel):
    """Per-phase controller parameters (seconds)."""

    split_s: float = Field(gt=0.0, description="Split = green + yellow + red clearance")
    min_green_s: float = Field(5.0, ge=1.0)
    max_green_s: float | None = Field(None, gt=0.0, description="Max 1; defaults to split - Y - AR")
    passage_s: float = Field(3.0, ge=0.0, le=10.0, description="vehicle extension / passage")
    yellow_s: float = Field(4.0, ge=3.0, le=6.0)
    red_clear_s: float = Field(1.0, ge=0.0, le=6.0)
    recall: Recall = "none"
    dual_entry: bool = False

    @property
    def green_s(self) -> float:
        return self.split_s - self.yellow_s - self.red_clear_s

    @property
    def max_s(self) -> float:
        return self.max_green_s if self.max_green_s is not None else self.green_s

    @model_validator(mode="after")
    def _check(self) -> "PhaseParams":
        if self.green_s < self.min_green_s - 1e-9:
            raise ValueError(f"split {self.split_s}s leaves {self.green_s:.1f}s green < min green {self.min_green_s}s")
        return self


class IntersectionTiming(StrictModel):
    """One controller's coordinated timing plan."""

    int_id: int
    name: str = ""
    cycle_s: float = Field(gt=10.0, le=300.0)
    offset_s: float = Field(0.0, ge=0.0)
    coord_phases: tuple[int, int] = (2, 6)
    ring1: list[list[int]] = Field(default_factory=lambda: [[1, 2], [3, 4]])
    ring2: list[list[int]] = Field(default_factory=lambda: [[5, 6], [7, 8]])
    phases: dict[int, PhaseParams]

    @model_validator(mode="after")
    def _validate(self) -> "IntersectionTiming":
        if len(self.ring1) != 2 or len(self.ring2) != 2:
            raise ValueError("each ring needs exactly two barrier groups")
        groups = (BARRIER_A, BARRIER_B)
        for ring, allowed in ((self.ring1, {1, 2, 3, 4}), (self.ring2, {5, 6, 7, 8})):
            for b, group in enumerate(ring):
                for ph in group:
                    if ph not in allowed or ph not in groups[b]:
                        raise ValueError(f"phase {ph} illegal in ring/barrier position {b + 1}")
                    if ph not in self.phases:
                        raise ValueError(f"phase {ph} sequenced but not timed")
        for b in range(2):
            s1 = sum(self.phases[p].split_s for p in self.ring1[b])
            s2 = sum(self.phases[p].split_s for p in self.ring2[b])
            if abs(s1 - s2) > 0.05:
                raise ValueError(f"int {self.int_id}: barrier {b + 1} ring sums differ ({s1:.1f} vs {s2:.1f} s)")
        total = sum(sum(self.phases[p].split_s for p in self.ring1[b]) for b in range(2))
        if abs(total - self.cycle_s) > 0.05:
            raise ValueError(f"int {self.int_id}: splits sum to {total:.1f}s, cycle is {self.cycle_s:.1f}s")
        for cp in self.coord_phases:
            if cp not in BARRIER_A:
                raise ValueError("coordinated phases must be in barrier A (1, 2, 5, 6)")
        return self

    def ring_of(self, phase: int) -> list[list[int]]:
        return self.ring1 if phase in RING1 else self.ring2


class Approach(StrictModel):
    """Lane-group data used by the simulation and HCM checks."""

    volume_vph: float = Field(ge=0.0)
    lanes: int = Field(1, ge=1, le=6)
    sat_flow_vphpl: float = Field(1800.0, gt=500.0, le=2400.0)


class CorridorIntersection(StrictModel):
    timing: IntersectionTiming
    position_m: float = Field(ge=0.0, description="distance along the arterial from the first intersection")
    approaches: dict[int, Approach] = Field(description="keyed by NEMA phase")


class Corridor(StrictModel):
    """An arterial: intersections ordered west -> east (EB = phase 2 = increasing position)."""

    name: str
    speed_mps: float = Field(gt=1.0, le=40.0)
    intersections: list[CorridorIntersection] = Field(min_length=2)

    @model_validator(mode="after")
    def _check(self) -> "Corridor":
        pos = [i.position_m for i in self.intersections]
        if pos != sorted(pos) or len(set(pos)) != len(pos):
            raise ValueError("intersections must be ordered by strictly increasing position")
        cycles = {round(i.timing.cycle_s, 3) for i in self.intersections}
        if len(cycles) != 1:
            raise ValueError(f"coordinated corridor needs a common cycle, got {sorted(cycles)}")
        return self

    @property
    def cycle_s(self) -> float:
        return self.intersections[0].timing.cycle_s
