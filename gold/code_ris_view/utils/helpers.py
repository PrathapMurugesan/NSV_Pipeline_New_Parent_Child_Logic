"""Small type-normalisation helpers shared by the loaders."""
import json
from datetime import date

import pandas as pd


def to_str_id(series):
    """Cast phase/cycle style keys to string, '2.0' -> '2', so joins match across sources."""
    return series.astype("string").str.replace(r"\.0$", "", regex=True)


def to_date(series):
    """'0202-07-14' -> datetime.date, matching the date objects in the parquet tables.
    (pd.to_datetime cannot hold years before 1677, which occur in the source data.)"""
    return series.map(lambda v: date.fromisoformat(str(v)[:10]) if pd.notna(v) else None)


SIDES = ("LHS", "RHS")


def _lane_detail(raw):
    """measures JSON -> (lane_values, lane_breach) dicts, or None when it holds no lane values."""
    if not isinstance(raw, str) or "lane_values" not in raw:
        return None
    try:
        m = json.loads(raw)
    except ValueError:
        return None
    values = m.get("lane_values")
    if not isinstance(values, dict) or not values:
        return None
    return values, m.get("lane_breach") or {}


def explode_sides(df):
    """Side-level rows (lane = 'LHS' / 'RHS', e.g. master_roughness, master_cracking_ca: value = average of
    the side's lanes, breach if ANY lane breaches) -> one row per lane from the measures JSON:
    lane = each key of lane_values, metric_value = that lane's value, breach_flag = lane_breach[lane].
    Other columns (defect_id, id, pci, affected_length_*, ...) are copied, so a defect is still counted
    once (same defect_id) and PCI applies to every lane of the side. Rows without lane detail stay as they are."""
    if "lane" not in df.columns or "measures" not in df.columns:
        return df
    side = df["lane"].astype("string").str.strip().str.upper().isin(SIDES).fillna(False).astype(bool)
    if not side.any():
        return df
    detail = df.loc[side, "measures"].map(_lane_detail)
    has = detail.notna()
    if not has.any():
        return df
    src = df.loc[side][has.to_numpy()]
    detail = detail[has]
    reps = detail.map(lambda d: len(d[0])).to_numpy()
    lanes = [lane for d in detail for lane in d[0]]
    values = [d[0][lane] for d in detail for lane in d[0]]
    breach = [d[1].get(lane) for d in detail for lane in d[0]]
    exploded = src.loc[src.index.repeat(reps)].reset_index(drop=True)
    exploded["lane"] = lanes
    exploded["metric_value"] = pd.to_numeric(pd.Series(values), errors="coerce")
    if "breach_flag" in exploded.columns:
        exploded["breach_flag"] = pd.to_numeric(pd.Series(breach), errors="coerce").fillna(exploded["breach_flag"])
    keep = df.drop(index=src.index)
    return pd.concat([keep, exploded], ignore_index=True)
