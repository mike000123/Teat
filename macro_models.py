from __future__ import annotations

import numpy as np
import pandas as pd


def compute_thresholds_from_window(df, start, end, keys, dirs, min_points: int = 24) -> dict:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    w = df.loc[start_dt:end_dt].copy()
    if w.empty:
        return {}

    def q(series, p):
        s = series.dropna()
        return float(s.quantile(p))

    thr = {}
    for col in keys:
        if col not in w.columns:
            continue
        s = w[col].dropna()
        if len(s) < min_points:
            continue

        higher_is_bullish = bool(dirs.get(col, False))
        if higher_is_bullish:
            thr[col] = {"bull": q(s, 0.67), "bear": q(s, 0.33)}
        else:
            thr[col] = {"bull": q(s, 0.33), "bear": q(s, 0.67)}
    return thr


def structural_thresholds() -> dict:
    return {
        "REAL_YIELD_CPI": {"bull": 1.0, "bear": 2.0},
        "CPI_YOY": {"bull": 3.5, "bear": 2.0},
        "USD_12M_CHG": {"bull": -3.0, "bear": 3.0},
        "CURVE_10Y_3M": {"bull": 0.0, "bear": 1.5},
        "DEFICIT_GDP": {"bull": -6.0, "bear": -3.0},
        "REAL_YIELD_TIPS10": {"bull": 0.75, "bear": 1.75},
        "HY_OAS": {"bull": 4.5, "bear": 3.2},
    }


def market_structural_thresholds() -> dict:
    return {
        "REAL_YIELD_CPI": {"bull": 1.5, "bear": 2.5},
        "USD_12M_CHG": {"bull": -2.0, "bear": 5.0},
        "CURVE_10Y_3M": {"bull": 1.0, "bear": -0.25},
        "HY_OAS": {"bull": 3.5, "bear": 5.0},
        "QQQ_ABOVE_MA200": {"bull": 1.0, "bear": 0.0},
        "QQQ_MA50_SLOPE_20D": {"bull": 0.0, "bear": -2.0},
        "MARKET_BREADTH_ABOVE_MA200": {"bull": 60.0, "bear": 40.0},
    }


def get_market_dirs() -> dict:
    return {
        "REAL_YIELD_CPI": False,
        "USD_12M_CHG": False,
        "CURVE_10Y_3M": True,
        "HY_OAS": False,
        "QQQ_ABOVE_MA200": True,
        "QQQ_MA50_SLOPE_20D": True,
        "MARKET_BREADTH_ABOVE_MA200": True,
    }


def get_market_weights() -> dict:
    return {
        "REAL_YIELD_CPI": 0.18,
        "USD_12M_CHG": 0.12,
        "CURVE_10Y_3M": 0.12,
        "HY_OAS": 0.20,
        "QQQ_ABOVE_MA200": 0.14,
        "QQQ_MA50_SLOPE_20D": 0.09,
        "MARKET_BREADTH_ABOVE_MA200": 0.15,
    }


def get_accel_dirs() -> dict:
    return {
        "GOLD_6M_RET": True,
        "REALYIELD_3M_CHG": False,
        "STRESS_3M_CHG": True,
        "USD_3M_CHG": False,
    }


def compute_accel_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["GOLD_6M_RET"] = 100.0 * out["GOLD_USD"].pct_change(6) if "GOLD_USD" in out.columns else np.nan

    if "REAL_YIELD_TIPS10" in out.columns and out["REAL_YIELD_TIPS10"].dropna().size > 0:
        ry = out["REAL_YIELD_TIPS10"]
    else:
        ry = out["REAL_YIELD_CPI"] if "REAL_YIELD_CPI" in out.columns else pd.Series(index=out.index, dtype=float)

    out["REALYIELD_3M_CHG"] = ry.diff(3)
    out["STRESS_3M_CHG"] = out["HY_OAS"].diff(3) if "HY_OAS" in out.columns else np.nan
    out["USD_3M_CHG"] = 100.0 * out["USD_TWEX_SPLICE"].pct_change(3) if "USD_TWEX_SPLICE" in out.columns else np.nan
    return out


def accel_thresholds_fixed() -> dict:
    return {
        "GOLD_6M_RET": {"bull": 8.0, "bear": -8.0},
        "REALYIELD_3M_CHG": {"bull": -0.30, "bear": 0.30},
        "STRESS_3M_CHG": {"bull": 0.60, "bear": -0.60},
        "USD_3M_CHG": {"bull": -3.0, "bear": 3.0},
    }


def compute_accel_thresholds_quantiles(df: pd.DataFrame, keys: list[str], dirs: dict, start: str = "2000-01-01") -> dict:
    w = df.loc[pd.to_datetime(start):].copy()
    thr = {}
    for k in keys:
        if k not in w.columns:
            continue
        s = w[k].dropna()
        if len(s) < 60:
            continue

        higher_is_bullish = bool(dirs.get(k, False))
        if higher_is_bullish:
            thr[k] = {"bull": float(s.quantile(0.67)), "bear": float(s.quantile(0.33))}
        else:
            thr[k] = {"bull": float(s.quantile(0.33)), "bear": float(s.quantile(0.67))}
    return thr


def state_score(value: float, bull: float, bear: float, higher_is_bullish: bool) -> int:
    if higher_is_bullish:
        if value >= bull:
            return 1
        if value <= bear:
            return -1
        return 0
    else:
        if value <= bull:
            return 1
        if value >= bear:
            return -1
        return 0


def label_signal(sig: float) -> str:
    if sig >= 0.60:
        return "Structural Bull"
    if sig >= 0.20:
        return "Positive"
    if sig > -0.20:
        return "Neutral"
    if sig > -0.60:
        return "Vulnerable"
    return "Structural Headwind"


def compute_signal(df, thresholds, weights, dirs) -> pd.DataFrame:
    out = df.copy()
    for col, higher_bull in dirs.items():
        bull = thresholds[col]["bull"]
        bear = thresholds[col]["bear"]
        out[col + "_STATE"] = out[col].apply(lambda v: state_score(v, bull, bear, higher_bull))

    contrib_cols = []
    for col in dirs.keys():
        ccol = col + "_CONTRIB"
        out[ccol] = weights[col] * out[col + "_STATE"]
        contrib_cols.append(ccol)

    out["SIGNAL"] = out[contrib_cols].sum(axis=1)
    out["REGIME"] = out["SIGNAL"].apply(label_signal)
    return out


def bucket_accel(x: float) -> str:
    if pd.isna(x):
        return "No data"
    if x >= 0.60:
        return "Acceleration Bull"
    if x >= 0.20:
        return "Positive"
    if x > -0.20:
        return "Neutral"
    if x > -0.60:
        return "Vulnerable"
    return "Acceleration Headwind"


def run_structural(res_base: pd.DataFrame, weights_struct: dict, dirs_struct: dict):
    thresholds = structural_thresholds()
    candidate_keys = [k for k in dirs_struct.keys() if k in weights_struct]
    core_keys = [k for k in candidate_keys if (k in thresholds) and (k in res_base.columns) and (res_base[k].dropna().size > 0)]

    dirs = {k: dirs_struct[k] for k in core_keys}
    weights = {k: weights_struct[k] for k in core_keys}
    tot = sum(weights.values()) or 1.0
    weights = {k: v / tot for k, v in weights.items()}
    thresholds = {k: thresholds[k] for k in core_keys}

    res = compute_signal(res_base, thresholds, weights, dirs)
    return res, core_keys, thresholds, dirs, weights


def run_market_structural(res_base: pd.DataFrame):
    thresholds = market_structural_thresholds()
    dirs_market = get_market_dirs()
    weights_market = get_market_weights()

    candidate_keys = [k for k in dirs_market.keys() if k in thresholds]
    core_keys = [k for k in candidate_keys if (k in res_base.columns) and (res_base[k].dropna().size > 0)]

    dirs = {k: dirs_market[k] for k in core_keys}
    weights = {k: weights_market[k] for k in core_keys}
    tot = sum(weights.values()) or 1.0
    weights = {k: v / tot for k, v in weights.items()}
    thresholds = {k: thresholds[k] for k in core_keys}

    res = compute_signal(res_base, thresholds, weights, dirs)
    return res, core_keys, thresholds, dirs, weights


def run_crisis(res_base: pd.DataFrame, weights_crisis: dict, dirs_crisis: dict, win_start: str, win_end: str, conditioned_stats_fn=None):
    candidate_keys = [k for k in dirs_crisis.keys() if k in weights_crisis]
    thresholds = compute_thresholds_from_window(res_base, win_start, win_end, keys=candidate_keys, dirs=dirs_crisis)

    ui_meta = {"error": None, "warning": None}
    if not thresholds:
        avail_start = res_base.index.min()
        avail_end = res_base.index.max()
        ui_meta["error"] = (
            f"No data available in the selected crisis window ({win_start} → {win_end}).\n\n"
            f"Available dataset range is approximately: {avail_start.date()} → {avail_end.date()}.\n\n"
            "Pick a later crisis window (e.g., 1974/1980/2011/2020) or switch to Custom and use an overlapping range."
        )
        res = res_base.copy()
        res["SIGNAL"] = np.nan
        res["REGIME"] = "No data"
        core_keys = []
        dirs = {}
        weights = {}
        gold_stats = {
            "3m": {"ok": False, "reason": "no-overlap"},
            "6m": {"ok": False, "reason": "no-overlap"},
        }
        return res, core_keys, {}, dirs, weights, gold_stats, ui_meta

    core_keys = list(thresholds.keys())
    if len(core_keys) < 2:
        ui_meta["warning"] = (
            "Crisis window has very limited usable indicator history. "
            "Similarity score may be unstable (too few indicators available)."
        )

    dirs = {k: dirs_crisis[k] for k in core_keys}
    weights = {k: weights_crisis.get(k, 0.0) for k in core_keys}
    tot = sum(weights.values()) or 1.0
    weights = {k: v / tot for k, v in weights.items()}

    res = compute_signal(res_base, thresholds, weights, dirs)

    if conditioned_stats_fn is not None:
        gold_stats_6m = conditioned_stats_fn(
            res_base, win_start, win_end, thresholds, weights, dirs,
            horizon_months=6, band=0.15
        )
        gold_stats_3m = conditioned_stats_fn(
            res_base, win_start, win_end, thresholds, weights, dirs,
            horizon_months=3, band=0.15
        )
        gold_stats = {"3m": gold_stats_3m, "6m": gold_stats_6m}
    else:
        gold_stats = {
            "3m": {"ok": False, "reason": "conditioned-stats-fn-missing"},
            "6m": {"ok": False, "reason": "conditioned-stats-fn-missing"},
        }

    return res, core_keys, thresholds, dirs, weights, gold_stats, ui_meta


def run_accel(res_accel_base: pd.DataFrame, accel_method: str, w_g: float, w_ry: float, w_st: float, w_u: float):
    dirs_accel = get_accel_dirs()
    accel_keys = list(dirs_accel.keys())

    raw_w = {"GOLD_6M_RET": w_g, "REALYIELD_3M_CHG": w_ry, "STRESS_3M_CHG": w_st, "USD_3M_CHG": w_u}
    active = {k: v for k, v in raw_w.items() if v > 0}
    tot = sum(active.values()) or 1.0
    weights = {k: v / tot for k, v in active.items()}

    if accel_method.startswith("Fixed"):
        thresholds = accel_thresholds_fixed()
    else:
        thresholds = compute_accel_thresholds_quantiles(res_accel_base, accel_keys, dirs_accel)

    core_keys = [k for k in accel_keys if (k in thresholds and k in weights and k in res_accel_base.columns)]
    dirs = {k: dirs_accel[k] for k in core_keys}
    thresholds = {k: thresholds[k] for k in core_keys}
    weights = {k: weights[k] for k in core_keys}
    tot = sum(weights.values()) or 1.0
    weights = {k: v / tot for k, v in weights.items()}

    res = compute_signal(res_accel_base, thresholds, weights, dirs)
    res["REGIME"] = res["SIGNAL"].apply(bucket_accel)
    return res, core_keys, thresholds, dirs, weights
