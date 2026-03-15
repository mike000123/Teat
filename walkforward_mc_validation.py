from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import mc_simulator as mc


@dataclass
class MCValidationConfig:
    horizon_steps: int = 60
    n_sims: int = 1000
    block_size: int = 5
    seed: int = 7
    benchmark_symbol: str = "QQQ"
    rsi_period: int = 14
    rsi_slope_bars: int = 5
    vol_window: int = 20
    rs_window: int = 63
    ma_fast: int = 50
    ma_slow: int = 200


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    d = s.diff()
    up = d.clip(lower=0.0)
    down = (-d).clip(lower=0.0)
    avg_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def build_tactical_state_series(
    price: pd.Series,
    benchmark_price: Optional[pd.Series] = None,
    cfg: Optional[MCValidationConfig] = None,
) -> tuple[pd.Series, dict]:
    cfg = cfg or MCValidationConfig()
    px = pd.to_numeric(price, errors="coerce").dropna().astype(float)
    if px.empty:
        return pd.Series(dtype=object), {}

    r = rsi(px, period=cfg.rsi_period).reindex(px.index)
    slope = r.diff(cfg.rsi_slope_bars)

    ma200 = px.rolling(cfg.ma_slow).mean()
    above_ma200 = px > ma200

    ma50 = px.rolling(cfg.ma_fast).mean()
    ma50_slope = ma50.pct_change(20)

    vol = px.pct_change().rolling(cfg.vol_window).std() * np.sqrt(252)
    vol_q75 = vol.dropna().quantile(0.75) if not vol.dropna().empty else np.nan
    vol_regime = pd.Series(np.where(vol > vol_q75, "HIGH", "LOW"), index=px.index)

    if benchmark_price is not None and not benchmark_price.empty:
        bench = pd.to_numeric(benchmark_price, errors="coerce").dropna().astype(float)
        bench = bench.reindex(px.index).ffill()
        rs_vs_bench = px.pct_change(cfg.rs_window) - bench.pct_change(cfg.rs_window)
    else:
        rs_vs_bench = pd.Series(index=px.index, dtype=float)

    state = pd.Series(
        [
            mc.classify_accel_state(
                r.iloc[i],
                slope.iloc[i],
                bool(above_ma200.iloc[i]) if pd.notna(above_ma200.iloc[i]) else None,
                rs_vs_bench.iloc[i] if i < len(rs_vs_bench) else None,
                ma50_slope.iloc[i] if i < len(ma50_slope) else None,
                vol_regime.iloc[i] if i < len(vol_regime) else None,
            )
            for i in range(len(px))
        ],
        index=px.index,
        name="STATE",
    )

    latest = {
        "rsi": float(r.dropna().iloc[-1]) if r.dropna().shape[0] else np.nan,
        "rsi_slope": float(slope.dropna().iloc[-1]) if slope.dropna().shape[0] else np.nan,
        "above_ma200": bool(above_ma200.dropna().iloc[-1]) if above_ma200.dropna().shape[0] else None,
        "ma50_slope_20d": float(ma50_slope.dropna().iloc[-1]) if ma50_slope.dropna().shape[0] else np.nan,
        "vol_regime": str(vol_regime.dropna().iloc[-1]) if vol_regime.dropna().shape[0] else "UNKNOWN",
        "rs_vs_bench": float(rs_vs_bench.dropna().iloc[-1]) if rs_vs_bench.dropna().shape[0] else np.nan,
        "state_now": str(state.iloc[-1]) if len(state) else "UNKNOWN",
    }
    return state, latest


def validate_one_replay_date(
    price_hist: pd.Series,
    future_price: Optional[pd.Series] = None,
    benchmark_hist: Optional[pd.Series] = None,
    cfg: Optional[MCValidationConfig] = None,
) -> dict:
    cfg = cfg or MCValidationConfig()
    px = pd.to_numeric(price_hist, errors="coerce").dropna().astype(float)
    if px.shape[0] < max(120, cfg.ma_slow + 5):
        return {"mc_error": "Not enough history for MC validation"}

    state, meta = build_tactical_state_series(px, benchmark_hist, cfg=cfg)
    paths, state_now = mc.monte_carlo_paths_by_tactical_state_block(
        price=px,
        state=state,
        state_now=meta.get("state_now"),
        horizon_steps=int(cfg.horizon_steps),
        n_sims=int(cfg.n_sims),
        block_size=int(cfg.block_size),
        seed=int(cfg.seed),
    )
    if paths is None:
        return {
            "mc_state": state_now,
            "mc_error": f"Not enough history in tactical state ({state_now})",
        }

    bands = mc.mc_percentiles(paths)
    start_price = float(px.iloc[-1])
    p10_end = float(bands["p10"].iloc[-1])
    p50_end = float(bands["p50"].iloc[-1])
    p90_end = float(bands["p90"].iloc[-1])

    out = {
        "mc_state": state_now,
        "mc_start_price": start_price,
        "mc_p10_price": p10_end,
        "mc_p50_price": p50_end,
        "mc_p90_price": p90_end,
        "mc_p10_ret_pct": (p10_end / start_price - 1.0) * 100.0,
        "mc_p50_ret_pct": (p50_end / start_price - 1.0) * 100.0,
        "mc_p90_ret_pct": (p90_end / start_price - 1.0) * 100.0,
        "mc_prob_up_pct": float((paths[:, -1] > start_price).mean()) * 100.0,
        "mc_rsi": meta.get("rsi", np.nan),
        "mc_rsi_slope": meta.get("rsi_slope", np.nan),
        "mc_vol_regime": meta.get("vol_regime", "UNKNOWN"),
    }

    if future_price is not None:
        fut = pd.to_numeric(future_price, errors="coerce").dropna().astype(float)
        if fut.shape[0]:
            actual_end = float(fut.iloc[-1])
            actual_ret = (actual_end / start_price - 1.0) * 100.0 if start_price > 0 else np.nan
            out.update({
                "mc_actual_end_price": actual_end,
                "mc_actual_ret_pct": actual_ret,
                "mc_inside_p10_p90": bool(out["mc_p10_ret_pct"] <= actual_ret <= out["mc_p90_ret_pct"]),
                "mc_below_p10": bool(actual_ret < out["mc_p10_ret_pct"]),
                "mc_above_p90": bool(actual_ret > out["mc_p90_ret_pct"]),
                "mc_direction_hit": bool(np.sign(actual_ret) == np.sign(out["mc_p50_ret_pct"])) if pd.notna(actual_ret) else False,
                "mc_abs_err_p50": abs(actual_ret - out["mc_p50_ret_pct"]) if pd.notna(actual_ret) else np.nan,
            })
    return out


def run_walkforward_mc_validation(
    replay_df: pd.DataFrame,
    full_price: pd.Series,
    benchmark_price: Optional[pd.Series] = None,
    horizon_months: int = 1,
    cfg: Optional[MCValidationConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or MCValidationConfig()
    if replay_df is None or replay_df.empty:
        return pd.DataFrame()

    px = pd.to_numeric(full_price, errors="coerce").dropna().astype(float).sort_index()
    bench = None
    if benchmark_price is not None:
        bench = pd.to_numeric(benchmark_price, errors="coerce").dropna().astype(float).sort_index()

    rows = []
    for _, row in replay_df.iterrows():
        dt = pd.Timestamp(row["as_of_date"])
        hist_px = px.loc[px.index <= dt]
        future_cut = dt + pd.DateOffset(months=int(horizon_months))
        future_px = px.loc[(px.index >= dt) & (px.index <= future_cut)]
        hist_bench = bench.loc[bench.index <= dt] if bench is not None else None

        mc_row = validate_one_replay_date(
            price_hist=hist_px,
            future_price=future_px,
            benchmark_hist=hist_bench,
            cfg=cfg,
        )
        rows.append({"as_of_date": dt, **mc_row})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
        out = out.sort_values("as_of_date").reset_index(drop=True)
        replay = replay_df.copy()
        replay["as_of_date"] = pd.to_datetime(replay["as_of_date"], errors="coerce")
        out = replay.merge(out, on="as_of_date", how="left")
    return out


def summarize_mc_validation(validation_df: pd.DataFrame) -> dict:
    if validation_df is None or validation_df.empty:
        return {}

    df = validation_df.copy()

    def _num_col(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype="float64")

    valid = pd.DataFrame({
        "actual": _num_col("mc_actual_ret_pct"),
        "p50": _num_col("mc_p50_ret_pct"),
        "inside": _num_col("mc_inside_p10_p90"),
        "below": _num_col("mc_below_p10"),
        "above": _num_col("mc_above_p90"),
        "dir": _num_col("mc_direction_hit"),
        "err": _num_col("mc_abs_err_p50"),
    }, index=df.index).dropna(subset=["actual", "p50"])

    if valid.empty:
        return {"rows": 0}

    err_nonnull = valid["err"].dropna()

    return {
        "rows": int(len(valid)),
        "inside_p10_p90_pct": float(valid["inside"].fillna(0).astype(bool).mean()) * 100.0,
        "below_p10_pct": float(valid["below"].fillna(0).astype(bool).mean()) * 100.0,
        "above_p90_pct": float(valid["above"].fillna(0).astype(bool).mean()) * 100.0,
        "direction_hit_pct": float(valid["dir"].fillna(0).astype(bool).mean()) * 100.0,
        "median_abs_err_pct": float(err_nonnull.median()) if len(err_nonnull) else np.nan,
        "mean_abs_err_pct": float(err_nonnull.mean()) if len(err_nonnull) else np.nan,
    }
