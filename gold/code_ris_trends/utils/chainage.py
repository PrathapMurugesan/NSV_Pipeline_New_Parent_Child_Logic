"""master_chainage in the new silver layout -> one row per 10 m per lane (the layout the gold uses).

New layout: one row per 10 m (chainage_start / chainage_end) for the whole road, with
  lanes_present ('L1,L2,L3,R1,R2,R3'), lhs_pavement_type / rhs_pavement_type and per lane
  <lane>_start_chainage / <lane>_end_chainage (l1_..., r3_...); no lane, source_system or survey date.

Output (old layout): upc ('_' form), source_system, defect_detected_on, zone, phase, cycle, lane,
  start_chainage, end_chainage, pavement_type.
  - a lane is kept on a row only when it is listed in lanes_present
  - lane chainage = <lane>_start/_end_chainage, else the row's chainage_start / chainage_end
  - pavement_type = lhs_pavement_type for L lanes, rhs_pavement_type for R lanes
  - source_system / defect_detected_on: the chainage has no survey, so every upc / phase / cycle gets
    the surveys (source_system + detected_on) found in the defect tables of that upc / phase / cycle
A table that already has lane / start_chainage / end_chainage (old layout) is returned unchanged.
"""
import logging
import re

import pandas as pd

log = logging.getLogger(__name__)

OUT_COLUMNS = ["upc", "source_system", "defect_detected_on", "zone", "phase", "cycle", "lane",
               "start_chainage", "end_chainage", "pavement_type"]
SURVEY_KEYS = ["upc", "phase", "cycle"]
_LANE_COL = re.compile(r"^([lr]\d+)_start_chainage$", re.IGNORECASE)


def _underscore(series):
    return series.astype("string").str.strip().str.replace("/", "_", regex=False)


def surveys_from(frames):
    """Distinct upc ('_' form) / phase / cycle / source_system / detected_on of the defect tables."""
    cols = SURVEY_KEYS + ["source_system", "detected_on"]
    parts = [f[cols] for f in frames if f is not None and set(cols).issubset(f.columns) and len(f)]
    if not parts:
        return pd.DataFrame(columns=cols)
    s = pd.concat(parts, ignore_index=True)
    s = s.assign(upc=_underscore(s["upc"]), phase=s["phase"].astype("string"), cycle=s["cycle"].astype("string"))
    return s.drop_duplicates().rename(columns={"detected_on": "defect_detected_on"})


def to_lane_chainage(chainage, survey_frames):
    if {"lane", "start_chainage", "end_chainage"}.issubset(chainage.columns):
        return chainage  # old layout

    ch = chainage.reset_index(drop=True)
    upc = ch["upc_code"] if "upc_code" in ch.columns else ch["upc"]
    present = ch["lanes_present"].astype("string").fillna("").str.upper().str.replace(" ", "", regex=False)
    present = present.str.split(",")
    lanes = sorted({m.group(1).upper() for c in ch.columns if (m := _LANE_COL.match(str(c)))})

    parts = []
    for lane in lanes:
        has = present.map(lambda xs, l=lane: l in xs).astype(bool)
        if not has.any():
            continue
        lo = lane.lower()
        side = "lhs" if lo.startswith("l") else "rhs"
        rows = ch[has]
        parts.append(pd.DataFrame({
            "upc": _underscore(upc[has]),
            "zone": rows["zone"],
            "phase": rows["phase"].astype("string"),
            "cycle": rows["cycle"].astype("string"),
            "lane": lane,
            "start_chainage": pd.to_numeric(rows[f"{lo}_start_chainage"], errors="coerce")
                                .fillna(pd.to_numeric(rows["chainage_start"], errors="coerce")),
            "end_chainage": pd.to_numeric(rows[f"{lo}_end_chainage"], errors="coerce")
                              .fillna(pd.to_numeric(rows["chainage_end"], errors="coerce")),
            "pavement_type": rows.get(f"{side}_pavement_type"),
        }))
    if not parts:
        raise ValueError("master_chainage: no lane columns (<lane>_start_chainage) matched lanes_present")
    long = pd.concat(parts, ignore_index=True)

    surveys = surveys_from(survey_frames)
    out = long.merge(surveys, on=SURVEY_KEYS, how="inner")
    missing = long[SURVEY_KEYS].drop_duplicates().merge(surveys[SURVEY_KEYS].drop_duplicates(),
                                                        on=SURVEY_KEYS, how="left", indicator=True)
    n_missing = int((missing["_merge"] == "left_only").sum())
    if n_missing:
        log.warning("master_chainage: %s upc/phase/cycle without any survey in the defect tables — skipped",
                    n_missing)
    log.info("master_chainage: %s rows (one per 10 m) -> %s lane rows | lanes: %s",
             f"{len(ch):,}", f"{len(out):,}", ", ".join(lanes))
    return out[OUT_COLUMNS].reset_index(drop=True)
