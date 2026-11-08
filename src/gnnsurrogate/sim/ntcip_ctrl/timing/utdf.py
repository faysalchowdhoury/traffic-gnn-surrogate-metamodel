"""Synchro UTDF (Universal Traffic Data Format) combined-file reader and writer.

A combined UTDF file is a CSV divided into bracketed sections. Each section has a title
line, a header row starting with ``RECORDNAME`` (or ``DATE``/``INTID`` for volumes) and data
rows. The records used here are::

    [Timeplans]
    Timing Plan Settings
    RECORDNAME,INTID,DATA
    Cycle Length,101,100
    Offset,101,0
    Reference Phase,101,206          # 206 = phases 2 + 6

    [Phases]
    Phases Data
    RECORDNAME,INTID,D1,D2,D3,D4,D5,D6,D7,D8
    BRP,101,111,112,211,212,121,122,221,222   # barrier-ring-position
    MinGreen,101,...
    MaxGreen,101,...
    VehExt,101,...
    Yellow,101,...
    AllRed,101,...
    Recall,101,0,3,0,0,0,3,0,0      # 0 none, 1 min, 2 ped, 3 max, 4 coord(C-Max)
    DualEntry,101,...

    [Volume]
    Volume Data
    DATE,TIME,INTID,NBL,NBT,NBR,SBL,SBT,SBR,EBL,EBT,EBR,WBL,WBT,WBR

The record names follow Synchro's documented UTDF layout. They have been exercised
against the example files in ``examples/``, not against every Synchro release; parse
errors name the offending section and record so a new variant is quick to map.
Split = MaxGreen + Yellow + AllRed (coordinated plans store the split in MaxGreen).
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from pathlib import Path

from ..core.exceptions import UTDFParseError
from .models import MOVEMENT_PHASE, IntersectionTiming, PhaseParams

_RECALL_CODES = {0: "none", 1: "min", 2: "none", 3: "max", 4: "max", 5: "max"}
_PHASE_COLS = [f"D{i}" for i in range(1, 9)]


def _sections(text: str) -> dict[str, list[list[str]]]:
    out: dict[str, list[list[str]]] = {}
    current: str | None = None
    for row in csv.reader(io.StringIO(text)):
        if not row or all(not c.strip() for c in row):
            continue
        first = row[0].strip()
        if first.startswith("[") and first.endswith("]"):
            current = first[1:-1].strip().lower()
            out[current] = []
            continue
        if current is not None:
            out[current].append([c.strip() for c in row])
    return out


def _table(rows: list[list[str]], section: str) -> tuple[list[str], list[list[str]]]:
    for i, row in enumerate(rows):
        if row and row[0].upper() in ("RECORDNAME", "DATE", "INTID"):
            return [c.upper() for c in row], rows[i + 1 :]
    raise UTDFParseError("section has no header row", context={"section": section})


def _num(value: str, *, section: str, record: str) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise UTDFParseError("non-numeric value", context={"section": section, "record": record, "value": value}) from exc


def parse_utdf(source: str | Path) -> tuple[dict[int, IntersectionTiming], dict[int, dict[int, float]]]:
    """Parse a combined UTDF file.

    Returns:
        ``(timings, volumes)`` where ``volumes[int_id][phase] = vph`` (through + right
        turns are not split; right turns are dropped, which is conservative for through).
    """
    if isinstance(source, Path) or "\n" not in source:
        p = Path(source)
        if not p.is_file():
            raise UTDFParseError("UTDF file not found", context={"path": str(p)})
        text = p.read_text(encoding="utf-8-sig")
    else:
        text = source
    sec = _sections(text)
    for required in ("timeplans", "phases"):
        if required not in sec:
            raise UTDFParseError("missing section", context={"section": f"[{required.title()}]"})

    plans: dict[int, dict[str, float]] = defaultdict(dict)
    header, rows = _table(sec["timeplans"], "Timeplans")
    for row in rows:
        rec = dict(zip(header, row))
        name, int_id = rec.get("RECORDNAME", ""), rec.get("INTID", "")
        if not int_id:
            continue
        val = _num(rec.get("DATA", ""), section="Timeplans", record=name)
        if val is not None:
            plans[int(float(int_id))][name.lower()] = val

    phase_rec: dict[int, dict[str, dict[int, float]]] = defaultdict(dict)
    header, rows = _table(sec["phases"], "Phases")
    for row in rows:
        rec = dict(zip(header, row))
        name, int_id = rec.get("RECORDNAME", ""), rec.get("INTID", "")
        if not int_id:
            continue
        vals: dict[int, float] = {}
        for i, col in enumerate(_PHASE_COLS, start=1):
            v = _num(rec.get(col, ""), section="Phases", record=name)
            if v is not None:
                vals[i] = v
        phase_rec[int(float(int_id))][name.lower()] = vals

    timings: dict[int, IntersectionTiming] = {}
    for int_id, recs in phase_rec.items():
        if int_id not in plans or "cycle length" not in plans[int_id]:
            raise UTDFParseError("no Cycle Length in [Timeplans]", context={"intid": int_id})
        for need in ("brp", "maxgreen", "yellow", "allred"):
            if need not in recs:
                raise UTDFParseError("missing phase record", context={"intid": int_id, "record": need})
        brp = recs["brp"]
        ring1: list[list[int]] = [[], []]
        ring2: list[list[int]] = [[], []]
        for ph, code in sorted(brp.items(), key=lambda kv: kv[1]):
            c = int(code)
            barrier, ring = c // 100, (c // 10) % 10
            if barrier not in (1, 2) or ring not in (1, 2):
                raise UTDFParseError("bad BRP code", context={"intid": int_id, "phase": ph, "brp": c})
            (ring1 if ring == 1 else ring2)[barrier - 1].append(ph)
        ref = int(plans[int_id].get("reference phase", 206))
        coord = (ref // 100, ref % 100) if ref >= 100 else (ref, ref)
        phases: dict[int, PhaseParams] = {}
        for ph in brp:
            g = recs["maxgreen"].get(ph)
            if g is None:
                raise UTDFParseError("phase in BRP has no MaxGreen", context={"intid": int_id, "phase": ph})
            y = recs["yellow"].get(ph, 4.0)
            ar = recs["allred"].get(ph, 1.0)
            phases[ph] = PhaseParams(
                split_s=g + y + ar,
                min_green_s=recs.get("mingreen", {}).get(ph, min(5.0, g)),
                passage_s=recs.get("vehext", {}).get(ph, 3.0),
                yellow_s=y,
                red_clear_s=ar,
                recall=_RECALL_CODES.get(int(recs.get("recall", {}).get(ph, 0)), "none"),  # type: ignore[arg-type]
                dual_entry=bool(recs.get("dualentry", {}).get(ph, 0)),
            )
        timings[int_id] = IntersectionTiming(
            int_id=int_id,
            cycle_s=plans[int_id]["cycle length"],
            offset_s=plans[int_id].get("offset", 0.0) % plans[int_id]["cycle length"],
            coord_phases=coord,
            ring1=ring1,
            ring2=ring2,
            phases=phases,
        )

    volumes: dict[int, dict[int, float]] = defaultdict(dict)
    if "volume" in sec:
        header, rows = _table(sec["volume"], "Volume")
        for row in rows:
            rec = dict(zip(header, row))
            if not rec.get("INTID"):
                continue
            iid = int(float(rec["INTID"]))
            for mv, ph in MOVEMENT_PHASE.items():
                v = _num(rec.get(mv, ""), section="Volume", record=mv)
                if v is not None:
                    volumes[iid][ph] = volumes[iid].get(ph, 0.0) + v
    return timings, dict(volumes)


def write_utdf(timings: dict[int, IntersectionTiming], path: str | Path,
               volumes: dict[int, dict[int, float]] | None = None) -> Path:
    """Write timings (e.g. optimised offsets) back to a combined UTDF file."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["[Timeplans]"])
    w.writerow(["Timing Plan Settings"])
    w.writerow(["RECORDNAME", "INTID", "DATA"])
    for iid, t in sorted(timings.items()):
        ref = t.coord_phases[0] * 100 + t.coord_phases[1]
        for name, val in (("Control Type", 3), ("Cycle Length", t.cycle_s), ("Lock Timings", 0),
                          ("Referenced To", 0), ("Reference Phase", ref), ("Offset", round(t.offset_s, 1))):
            w.writerow([name, iid, val])
    w.writerow([])
    w.writerow(["[Phases]"])
    w.writerow(["Phases Data"])
    w.writerow(["RECORDNAME", "INTID", *_PHASE_COLS])
    rev_recall = {"none": 0, "min": 1, "max": 3}
    for iid, t in sorted(timings.items()):
        brp: dict[int, int] = {}
        for r, ring in enumerate((t.ring1, t.ring2), start=1):
            for b, group in enumerate(ring, start=1):
                for pos, ph in enumerate(group, start=1):
                    brp[ph] = b * 100 + r * 10 + pos

        def row(name: str, fn) -> None:  # type: ignore[no-untyped-def]
            w.writerow([name, iid, *[(fn(t.phases[p]) if p in t.phases else "") for p in range(1, 9)]])

        w.writerow(["BRP", iid, *[brp.get(p, "") for p in range(1, 9)]])
        row("MinGreen", lambda pp: pp.min_green_s)
        row("MaxGreen", lambda pp: round(pp.green_s, 1))
        row("VehExt", lambda pp: pp.passage_s)
        row("Yellow", lambda pp: pp.yellow_s)
        row("AllRed", lambda pp: pp.red_clear_s)
        row("Recall", lambda pp: rev_recall[pp.recall])
        row("DualEntry", lambda pp: int(pp.dual_entry))
    if volumes:
        w.writerow([])
        w.writerow(["[Volume]"])
        w.writerow(["Volume Data"])
        cols = ["NBL", "NBT", "NBR", "SBL", "SBT", "SBR", "EBL", "EBT", "EBR", "WBL", "WBT", "WBR"]
        w.writerow(["DATE", "TIME", "INTID", *cols])
        for iid, vols in sorted(volumes.items()):
            w.writerow(["01/01/2026", "1700", iid, *[vols.get(MOVEMENT_PHASE[c], "") if c in MOVEMENT_PHASE else ""
                                                      for c in cols]])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(buf.getvalue(), encoding="utf-8")
    return p
