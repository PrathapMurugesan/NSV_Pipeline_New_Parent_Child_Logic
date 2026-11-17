"""Small helpers that reproduce Spark semantics the notebook relied on."""
import json
import operator
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

LT, LE, GT, GE = operator.lt, operator.le, operator.gt, operator.ge


def upc_slash(series):
    """N_02015_03002_GJ -> N/02015/03002/GJ (defect tables use '/', master_chainage uses '_')."""
    return series.astype("string").str.replace("_", "/", regex=False)


def upc_underscore(series):
    return series.astype("string").str.replace("/", "_", regex=False)


def to_str_id(series):
    """Cast phase/cycle style keys to string, '2.0' -> '2', so joins match across sources."""
    return series.astype("string").str.replace(r"\.0$", "", regex=True)


def spark_round(series, decimals):
    """Round half-up like Spark's F.round (pandas .round() rounds half-to-even)."""
    exp = Decimal(1).scaleb(-decimals)

    def _round(x):
        if pd.isna(x):
            return np.nan
        return float(Decimal(repr(float(x))).quantize(exp, rounding=ROUND_HALF_UP))

    return series.map(_round).astype("float64")


def null_safe_compare(left, right, op):
    """Element-wise comparison returning nullable booleans: <NA> when either side is null,
    so &, | and ~ follow Spark's three-valued logic instead of treating NaN as False."""
    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    result = pd.Series(op(left, right), index=left.index, dtype="boolean")
    result[left.isna() | right.isna()] = pd.NA
    return result


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


def standard_mask(threshold_name, standard):
    """Rows usable for one standard ('As per CA' / 'As per IRC'): rows of that standard, plus rows that carry
    both standards in one row (new child_10m tables: threshold_name 'As per CA & IRC', with
    ca_threshold_value / ca_breach_flag and irc_threshold_value / irc_breach_flag columns)."""
    name = threshold_name.astype("string").str.upper()
    combined = name.str.contains("CA", regex=False) & name.str.contains("IRC", regex=False)
    return ((name == standard.upper()) | combined).fillna(False).astype(bool)
