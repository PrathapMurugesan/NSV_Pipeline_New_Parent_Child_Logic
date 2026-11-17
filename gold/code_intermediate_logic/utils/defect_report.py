"""Defect report: metric per reporting-interval window, flagged where metric_value > threshold_value.

New silver structure (file names in utils/tables.py):
  roughness / rutting : child_<d>_as_per_ca (CA, at the CA reporting interval) | child_100m_<d> (IRC, 100 m)
                        child_10m_<d> (10 m, CA & IRC rows)
  cracking            : master_cracking_ca (CA, 100 m) | master_cracking_irc (IRC, 100 m)
                        child_10m_cracking (10 m, CA & IRC rows)

Source per defect / standard / upc / phase / cycle (first table with rows for that standard wins):
  CA  : as_per_ca table as-is  ->  not found: 10m CA rows averaged into the CA reporting interval
  IRC : interval 100 m -> child 100m as-is  ->  not found: 10m IRC rows averaged into 100 m
        other interval -> 10m IRC rows averaged into the interval  ->  not found: child 100m as-is
Every source is filtered on its standard first (rows_for): CA windows only use 'As per CA' rows and IRC
windows only 'As per IRC' rows. The new 10m tables hold both in one row ('As per CA & IRC'); there
CA reads ca_threshold_value / ca_breach_flag and IRC reads irc_threshold_value / irc_breach_flag.
"""
import logging

import numpy as np
import pandas as pd

from utils.helpers import standard_mask, to_int
from utils.reporting_intervals import RI_KEYS

log = logging.getLogger(__name__)


# defect -> 10m base table key (other keys: <defect>_as_per_ca, <defect>_child_100m)
DEFECTS = {"roughness": "roughness", "rutting": "rutting", "cracking": "cracking"}

CHILD_100M_INTERVAL = 100

REPORT_TYPES = {"ca": "As per CA", "irc": "As per IRC"}

SRC_COLS = ["source_system", "upc", "phase", "cycle", "lane",
            "start_chainage", "end_chainage", "detected_on",
            "metric_value", "threshold_name", "threshold_value"]

STD_COLS = ["ca_threshold_value", "irc_threshold_value", "ca_breach_flag", "irc_breach_flag"]


def rows_for(src, rt):
    """Rows of one standard with its threshold_value / breach_flag. A combined 'As per CA & IRC' row (new
    10 m tables) gives CA its ca_threshold_value / ca_breach_flag and IRC its irc_threshold_value /
    irc_breach_flag, so CA always uses CA values and IRC always IRC values."""
    rows = src[standard_mask(src["threshold_name"], REPORT_TYPES[rt])]
    if f"{rt}_threshold_value" not in rows.columns:
        return rows
    return rows.assign(
        threshold_value=rows[f"{rt}_threshold_value"].fillna(rows["threshold_value"]),
        breach_flag=rows[f"{rt}_breach_flag"].fillna(rows["breach_flag"]),
    )


WINDOW_KEYS = ["source_system", "upc", "phase", "cycle", "lane", "detected_on"]


def prepare_source(df):
    if df is None:  # table not delivered by silver
        return pd.DataFrame({c: pd.Series(dtype="object") for c in SRC_COLS + ["breach_flag"] + STD_COLS})
    out = df[SRC_COLS].copy()
    # silver's own breach decision (e.g. master_cracking_ca: breach if ANY lane breaches, while
    # metric_value is the average of the lanes); missing column -> decided from metric vs threshold
    out["breach_flag"] = pd.to_numeric(df["breach_flag"], errors="coerce") if "breach_flag" in df.columns else np.nan
    # per-standard columns of the new 10 m tables (one row for CA & IRC)
    for col in ("ca_threshold_value", "irc_threshold_value", "ca_breach_flag", "irc_breach_flag"):
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    for col in ("start_chainage", "end_chainage", "metric_value", "threshold_value"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _candidates(defect, base, interval, rt, sources):
    """Source preference for one defect / standard / interval -> list of (interval, table_key, needs_agg).
    interval None means the source segments are used as-is.

    CA  : <defect>_as_per_ca as-is (already at the CA reporting interval)
          -> not found: CA rows of the 10m table averaged into the CA reporting interval
    IRC : 100 m interval -> child 100m as-is, else 10m averaged
          other interval -> 10m averaged into the interval, else child 100m as-is
    """
    averaged = (interval, base, True)
    if rt == "ca":
        return [(None, f"{defect}_as_per_ca", False), averaged]
    child_100m = (None, f"{defect}_child_100m", False)
    if interval == CHILD_100M_INTERVAL:
        return [child_100m, averaged]
    return [averaged, child_100m]


def _has_rows(ri_rows, table_key, rt, sources):
    """Boolean mask: ri_rows whose upc/phase/cycle have rows for this standard in the table."""
    src = sources[table_key]
    avail = rows_for(src, rt)[RI_KEYS].drop_duplicates()
    return (ri_rows.merge(avail, on=RI_KEYS, how="left", indicator=True)["_merge"] == "both").to_numpy()


def _source_groups(defect, base, rt, ri_rt, interval_col, sources):
    """Routing for one defect / standard -> list of (interval, table_key, needs_agg, ri_rows)."""
    groups = []
    for iv in sorted(ri_rt[interval_col].unique()):
        remaining = ri_rt[ri_rt[interval_col] == iv]
        for interval, table_key, needs_agg in _candidates(defect, base, int(iv), rt, sources):
            found = _has_rows(remaining, table_key, rt, sources)
            if found.any():
                groups.append((interval, table_key, needs_agg, remaining[found]))
            remaining = remaining[~found]
    return groups


def _report_part(defect, rt, interval, table_key, needs_agg, ri_rows, sources):
    src = rows_for(sources[table_key], rt)
    matched = src.merge(ri_rows[RI_KEYS], on=RI_KEYS, how="inner")

    if needs_agg:
        window_start = np.floor(np.fmin(matched["start_chainage"], matched["end_chainage"]) / interval) * interval
        matched = (
            matched.assign(start_chainage=window_start, end_chainage=window_start + interval)
            .groupby(WINDOW_KEYS + ["start_chainage", "end_chainage"], dropna=False)
            .agg(metric_value=("metric_value", "mean"), threshold_value=("threshold_value", "max"))
            .reset_index()
        )

    # Flag: as-is tables use silver's breach_flag where it is filled; averaged windows (and rows
    # without a breach_flag) use metric_value > threshold_value
    computed = (matched["metric_value"] > matched["threshold_value"]).astype(int)
    if needs_agg:
        defect_flag = computed
    else:
        defect_flag = (matched["breach_flag"].fillna(computed) == 1).astype(int)

    # as_per_ca / child 100m used as-is: actual segment length; otherwise the RI window
    if interval is None:
        reporting_interval = to_int((matched["end_chainage"] - matched["start_chainage"]).abs())
    else:
        reporting_interval = pd.Series(int(interval), index=matched.index, dtype="Int64")

    return pd.DataFrame({
        "source_system": matched["source_system"],
        "upc": matched["upc"],
        "upc_code_as_per_om_master": matched["upc"],
        "phase": matched["phase"],
        "cycle": matched["cycle"],
        "defect_detected_on": matched["detected_on"],
        "lane": matched["lane"],
        "start_chainage": matched["start_chainage"],
        "end_chainage": matched["end_chainage"],
        "defect_type": f"{defect}_{rt}",
        "reporting_interval": reporting_interval,
        "metric_value": matched["metric_value"],
        "threshold_value": matched["threshold_value"],
        "defect_flag": defect_flag,
        "source_table": table_key,
    })


def build_defect_report(sources, ri):
    """sources: notebook table key -> prepared source DataFrame; ri: wide reporting intervals."""
    parts = []
    for defect, base in DEFECTS.items():
        for rt in REPORT_TYPES:
            interval_col = f"{defect}_interval_{rt}"
            ri_rt = ri[RI_KEYS + [interval_col]]
            for interval, table_key, needs_agg, ri_rows in _source_groups(defect, base, rt, ri_rt, interval_col, sources):
                parts.append(_report_part(defect, rt, interval, table_key, needs_agg, ri_rows, sources))
    log.info("Defect report parts: %s", len(parts))

    report = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["upc", "defect_type", "phase", "cycle", "lane", "start_chainage"], kind="stable")
        .reset_index(drop=True)
    )

    log.info("Full report: %s rows | defect types: %s",
             f"{len(report):,}", sorted(report["defect_type"].unique()))
    counts = report.groupby(["upc", "defect_type", "source_table"]).size()
    for (upc, dt, table), n in counts.items():
        log.info("  %s | %-14s | %-22s | %s rows", upc, dt, table, f"{n:,}")
    return report
