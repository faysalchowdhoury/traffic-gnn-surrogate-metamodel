"""Two-way progression bandwidth (time-space green band) and offset optimisation.

For a vehicle entering intersection 0 eastbound at time ``tau`` and travelling at the
progression speed ``v``, its arrival at intersection ``k`` is ``tau + x_k / v``. It is inside the
**green band** if every arrival falls in the coordinated EB phase (phi2) green window:

``(tau + x_k / v - offset_k) mod C  in  [s2_k, e2_k)`` for all ``k``.

The EB bandwidth is the measure of such ``tau`` over one cycle; WB uses phi6 and positions
measured from the last intersection. Offsets are optimised by vectorised coordinate descent
with random restarts over a 1-s grid (the first intersection is the reference, offset 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..timing.models import Corridor
from ..timing.schedule import coord_green_windows

GRID_S = 0.1


@dataclass(frozen=True)
class BandResult:
    eb_s: float
    wb_s: float
    offsets_s: tuple[float, ...]

    @property
    def total_s(self) -> float:
        return self.eb_s + self.wb_s

    def efficiency(self, cycle_s: float) -> float:
        """Bandwidth efficiency (EB + WB) / (2 C), the usual MAXBAND/PASSER measure."""
        return self.total_s / (2.0 * cycle_s)


def _windows(corridor: Corridor, phase: int) -> np.ndarray:
    return np.array([coord_green_windows(i.timing, phase) for i in corridor.intersections], dtype=float)


def _in_green(local: np.ndarray, win: np.ndarray, cycle: float) -> np.ndarray:
    s, e = win[..., 0], win[..., 1]
    return ((local - s) % cycle) < (e - s)


def _direction_masks(corridor: Corridor, offsets: np.ndarray, eb: bool, tau: np.ndarray) -> np.ndarray:
    """Boolean (n_int, n_tau) green indicator for one direction."""
    c = corridor.cycle_s
    x = np.array([i.position_m for i in corridor.intersections])
    travel = (x - x[0]) / corridor.speed_mps if eb else (x[-1] - x) / corridor.speed_mps
    win = _windows(corridor, 2 if eb else 6)
    local = (tau[None, :] + travel[:, None] - offsets[:, None]) % c
    return _in_green(local, win[:, None, :], c)


def bandwidth(corridor: Corridor, offsets: list[float] | np.ndarray | None = None) -> BandResult:
    """EB and WB bandwidth (s) for the given offsets (defaults to the timings' offsets)."""
    off = np.asarray(offsets if offsets is not None else [i.timing.offset_s for i in corridor.intersections], float)
    tau = np.arange(0.0, corridor.cycle_s, GRID_S)
    eb = _direction_masks(corridor, off, True, tau).all(axis=0).sum() * GRID_S
    wb = _direction_masks(corridor, off, False, tau).all(axis=0).sum() * GRID_S
    return BandResult(float(eb), float(wb), tuple(float(o) for o in off))


def optimize_offsets(
    corridor: Corridor,
    *,
    step_s: float = 1.0,
    restarts: int = 24,
    seed: int = 0,
    balance_weight: float = 0.05,
) -> BandResult:
    """Maximise ``EB + WB - balance_weight * |EB - WB|`` by coordinate descent.

    Each coordinate update evaluates every candidate offset of one intersection at once
    (``n_candidates x n_tau`` boolean arrays), so a 4-intersection, C = 100 s corridor
    optimises in well under a second.
    """
    n = len(corridor.intersections)
    c = corridor.cycle_s
    rng = np.random.default_rng(seed)
    tau = np.arange(0.0, c, GRID_S)
    cand = np.arange(0.0, c, step_s)
    x = np.array([i.position_m for i in corridor.intersections])
    trav_e = (x - x[0]) / corridor.speed_mps
    trav_w = (x[-1] - x) / corridor.speed_mps
    win_e, win_w = _windows(corridor, 2), _windows(corridor, 6)

    def masks(k: int, offs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        le = (tau[None, :] + trav_e[k] - offs[:, None]) % c
        lw = (tau[None, :] + trav_w[k] - offs[:, None]) % c
        return _in_green(le, win_e[k], c), _in_green(lw, win_w[k], c)

    def score(e: float, w: float) -> float:
        return e + w - balance_weight * abs(e - w)

    starts = [np.array([i.timing.offset_s for i in corridor.intersections], float)]
    starts += [np.concatenate([[0.0], rng.choice(cand, n - 1)]) for _ in range(restarts)]
    best: tuple[float, np.ndarray] = (-np.inf, starts[0])
    for off in starts:
        off = off.copy()
        off[0] = 0.0
        me = np.empty((n, tau.size), bool)
        mw = np.empty((n, tau.size), bool)
        for k in range(n):
            a, b = masks(k, off[k : k + 1])
            me[k], mw[k] = a[0], b[0]
        improved = True
        while improved:
            improved = False
            for k in range(1, n):
                rest_e = np.delete(me, k, axis=0).all(axis=0)
                rest_w = np.delete(mw, k, axis=0).all(axis=0)
                ce, cw = masks(k, cand)
                e = (ce & rest_e).sum(axis=1) * GRID_S
                w = (cw & rest_w).sum(axis=1) * GRID_S
                s = e + w - balance_weight * np.abs(e - w)
                j = int(np.argmax(s))
                if s[j] > score(float(((me.all(axis=0)).sum()) * GRID_S), float(mw.all(axis=0).sum() * GRID_S)) + 1e-9:
                    off[k] = cand[j]
                    me[k], mw[k] = ce[j], cw[j]
                    improved = True
        sc = score(float(me.all(axis=0).sum() * GRID_S), float(mw.all(axis=0).sum() * GRID_S))
        if sc > best[0] + 1e-9:
            best = (sc, off.copy())
    return bandwidth(corridor, best[1])
