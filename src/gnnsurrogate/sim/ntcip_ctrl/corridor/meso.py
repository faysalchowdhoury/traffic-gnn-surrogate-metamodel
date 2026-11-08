"""Closed-loop mesoscopic corridor simulation.

Every intersection runs a :class:`~ntcip_ctrl.nema.NemaController`. Each lane group
(NEMA phase) is a FIFO vertical queue discharging at its saturation headway during the
effective green: from 2 s after green onset (start-up lost time, waived for vehicles that
arrive moving after onset) until 2 s into yellow (extension of effective green). Stop-bar
presence places a call; each arrival is a detector actuation that resets the passage timer.

Arterial through vehicles (EB = phi2, WB = phi6) leave one stop line and arrive at the next
after ``distance / speed``, so platoon structure, and therefore offsets, matter. Side-street
and left-turn demand is Poisson.

Delay is stop-line queueing delay (departure - arrival). It excludes acceleration and
deceleration delay, so it is a lower bound on HCM control delay.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ..nema.controller import ControllerMode, NemaController
from ..timing.models import Corridor

STARTUP_LOST_S = 2.0
YELLOW_EXTENSION_S = 2.0
STOP_THRESHOLD_S = 1.0


@dataclass
class _LaneGroup:
    phase: int
    headway: float
    queue: deque[tuple[float, int, bool]]  # (arrival time, trip id, arrived on green)
    next_free: float = -1e9
    records: list[tuple[float, float, int, bool]] | None = None  # (arr, dep, trip, arrived_on_green)


@dataclass(frozen=True)
class SimResult:
    trips: pd.DataFrame  # arterial end-to-end trips
    movements: pd.DataFrame  # per intersection x phase
    controllers: pd.DataFrame  # per intersection x phase green/terminations
    runtime_s: float


def _poisson(rng: np.random.Generator, vph: float, horizon: float) -> np.ndarray:
    if vph <= 0:
        return np.empty(0)
    mean = 3600.0 / vph
    n = int(vph * horizon / 3600.0 * 1.5) + 20
    t = np.cumsum(rng.exponential(mean, n))
    while t[-1] < horizon:
        t = np.concatenate([t, t[-1] + np.cumsum(rng.exponential(mean, n))])
    return t[t < horizon]


def simulate(
    corridor: Corridor,
    *,
    mode: ControllerMode,
    offsets: list[float] | None = None,
    seed: int = 1,
    duration_s: float = 3900.0,
    warmup_s: float = 300.0,
    dt: float = 0.1,
) -> SimResult:
    """Run the corridor and return trip, movement and controller tables (post warm-up)."""
    tick = perf_counter()
    rng = np.random.default_rng(seed)
    ints = corridor.intersections
    n = len(ints)
    timings = [
        it.timing if offsets is None else it.timing.model_copy(update={"offset_s": float(offsets[k]) % it.timing.cycle_s})
        for k, it in enumerate(ints)
    ]
    ctrls = [NemaController(t, mode, dt=dt, t0=0.0) for t in timings]
    groups: list[dict[int, _LaneGroup]] = []
    pending: list[list[tuple[float, int, int]]] = []  # heap per intersection: (time, phase, trip)
    for k, it in enumerate(ints):
        g: dict[int, _LaneGroup] = {}
        heap: list[tuple[float, int, int]] = []
        for ph, ap in it.approaches.items():
            if ph not in timings[k].phases:
                continue
            g[ph] = _LaneGroup(ph, 3600.0 / (ap.sat_flow_vphpl * ap.lanes), deque(), records=[])
            external = ph not in (2, 6) or (ph == 2 and k == 0) or (ph == 6 and k == n - 1)
            if external:
                for a in _poisson(rng, ap.volume_vph, duration_s):
                    trip = -1
                    if ph in (2, 6):
                        trip = len(heap) + 1_000_000 * (k + 1) + (0 if ph == 2 else 500_000)
                    heap.append((float(a), ph, trip))
        heapq.heapify(heap)
        groups.append(g)
        pending.append(heap)

    trip_start: dict[int, float] = {}
    trip_delay: dict[int, float] = {}
    trip_stops: dict[int, int] = {}
    trips_done: list[dict[str, Any]] = []
    x = [it.position_m for it in ints]
    v = corridor.speed_mps

    steps = int(round(duration_s / dt))
    for i in range(steps):
        t = i * dt
        t_end = t + dt
        for k in range(n):
            ctrl, g, heap = ctrls[k], groups[k], pending[k]
            states = {r.phase: (r.interval, r.timer) for r in ctrl.rings if r.interval in ("G", "Y")}
            actuated: set[int] = set()
            while heap and heap[0][0] < t_end:
                arr, ph, trip = heapq.heappop(heap)
                if ph not in g:
                    continue
                if trip >= 0 and trip not in trip_start:
                    trip_start[trip] = arr
                    trip_delay[trip] = 0.0
                    trip_stops[trip] = 0
                on_green = ph in states and states[ph][0] == "G"
                g[ph].queue.append((arr, trip, on_green))
                actuated.add(ph)
            # discharge during this tick
            for ph, lg in g.items():
                st = states.get(ph)
                if st is None:
                    continue
                interval, timer = st
                onset = t - timer
                while lg.queue:
                    arr, trip, on_green = lg.queue[0]
                    if interval == "G":
                        ok = timer >= STARTUP_LOST_S or arr >= onset
                    else:
                        ok = timer < YELLOW_EXTENSION_S
                    dep = max(lg.next_free, arr, t)
                    if not ok or dep >= t_end:
                        break
                    lg.queue.popleft()
                    lg.next_free = dep + lg.headway
                    delay = dep - arr
                    if arr >= warmup_s:
                        lg.records.append((arr, dep, trip, on_green))  # type: ignore[union-attr]
                    if trip >= 0:
                        trip_delay[trip] += delay
                        trip_stops[trip] += int(delay > STOP_THRESHOLD_S)
                        nxt = k + 1 if ph == 2 else k - 1
                        if 0 <= nxt < n:
                            heapq.heappush(pending[nxt], (dep + abs(x[nxt] - x[k]) / v, ph, trip))
                        else:
                            start = trip_start.pop(trip)
                            if start >= warmup_s:
                                trips_done.append({
                                    "trip": trip, "direction": "EB" if ph == 2 else "WB", "enter_s": start,
                                    "exit_s": dep, "delay_s": trip_delay.pop(trip), "stops": trip_stops.pop(trip),
                                    "free_flow_s": abs(x[-1] - x[0]) / v,
                                })
                            else:
                                trip_delay.pop(trip)
                                trip_stops.pop(trip)
            presence = {ph for ph, lg in g.items() if lg.queue}
            ctrl.step(presence | actuated, actuated | presence)

    # ---- tables
    rows = []
    for k, (it, g) in enumerate(zip(ints, groups)):
        for ph, lg in g.items():
            rec = np.array(lg.records, dtype=float).reshape(-1, 4) if lg.records else np.empty((0, 4))
            delay = rec[:, 1] - rec[:, 0] if len(rec) else np.empty(0)
            rows.append({
                "int_id": it.timing.int_id, "phase": ph, "served": len(rec),
                "served_vph": len(rec) / ((duration_s - warmup_s) / 3600.0),
                "mean_delay_s": float(delay.mean()) if delay.size else math.nan,
                "p95_delay_s": float(np.percentile(delay, 95)) if delay.size else math.nan,
                "stop_rate": float((delay > STOP_THRESHOLD_S).mean()) if delay.size else math.nan,
                "arrivals_on_green": float(rec[:, 3].mean()) if delay.size else math.nan,
                "residual_queue": len(lg.queue),
            })
    movements = pd.DataFrame(rows)
    crow = []
    analysis_cycles = (duration_s) / corridor.cycle_s
    for it, ctrl in zip(ints, ctrls):
        for ph, s in ctrl.stats.items():
            crow.append({"int_id": it.timing.int_id, "phase": ph, "avg_green_s": s.green_time_s / analysis_cycles,
                         "services_per_cycle": s.services / analysis_cycles, **{f"n_{k}": v for k, v in s.terminations.items()}})
    controllers = pd.DataFrame(crow).fillna(0)
    return SimResult(pd.DataFrame(trips_done), movements, controllers, perf_counter() - tick)
