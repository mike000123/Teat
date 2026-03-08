from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any

import pandas as pd

import macro_models as mm
from macro_fusion import build_macro_state
from strategy_engine import decide_trade


DEFAULT_GOLD_DIRS = {
    "REAL_YIELD_CPI": False,
    "CPI_YOY": True,
    "USD_12M_CHG": False,
    "CURVE_10Y_3M": False,
    "DEFICIT_GDP": False,
    "REAL_YIELD_TIPS10": False,
    "HY_OAS": True,
}

DEFAULT_GOLD_WEIGHTS = {
    "REAL_YIELD_CPI": 0.35,
    "CPI_YOY": 0.20,
    "USD_12M_CHG": 0.25,
    "CURVE_10Y_3M": 0.10,
    "DEFICIT_GDP": 0.10,
    "REAL_YIELD_TIPS10": 0.0,
    "HY_OAS": 0.0,
}

DEFAULT_ACCEL_WEIGHTS = {
    "GOLD_6M_RET": 0.40,
    "REALYIELD_3M_CHG": 0.25,
    "STRESS_3M_CHG": 0.20,
    "USD_3M_CHG": 0.15,
}


@dataclass
class MacroBacktestConfig:
    ticker: str = "GLD"
    start_date: str = "2006-01-01"
    step: int = 1
    structural_dirs: Optional[dict[str, bool]] = None
    structural_weights: Optional[dict[str, float]] = None
    accel_method: str = "Rolling quantiles (from 2000)"
    accel_weights: Optional[dict[str, float]] = None
    price_col: str = "GOLD_USD"
    forward_return_months: Optional[int] = 1


def walk_forward_dates(index: pd.DatetimeIndex, start_date: str, step: int = 5):
    idx = pd.DatetimeIndex(index).sort_values().unique()
    idx = idx[idx >= pd.Timestamp(start_date)]
    return idx[::step]


def run_walk_forward_backtest(
    price_df: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    simulate_one_date: Callable[[pd.Timestamp], dict],
    start_date: str,
    step: int = 5,
) -> pd.DataFrame:
    rows = []

    for dt in walk_forward_dates(date_index, start_date=start_date, step=step):
        try:
            row = simulate_one_date(dt)
            if row is None:
                row = {}
            row["as_of_date"] = dt
            rows.append(row)
        except Exception as e:
            rows.append({
                "as_of_date": dt,
                "error": str(e),
            })

    return pd.DataFrame(rows)


def _latest_value(df: pd.DataFrame, col: str, dt: pd.Timestamp):
    try:
        val = df.loc[:dt, col].iloc[-1]
        if pd.isna(val):
            return None
        return val.item() if hasattr(val, "item") else val
    except Exception:
        return None


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    active = {k: float(v) for k, v in weights.items() if float(v) > 0}
    tot = sum(active.values()) or 1.0
    return {k: v / tot for k, v in active.items()}


def simulate_macro_strategy_date(
    panel_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    *,
    config: Optional[MacroBacktestConfig] = None,
    tactical_snapshot_fn: Optional[Callable[[pd.DataFrame, pd.Timestamp, str], Optional[dict[str, Any]]]] = None,
    mc_snapshot_fn: Optional[Callable[[pd.DataFrame, pd.Timestamp, str, Optional[dict[str, Any]]], Optional[dict[str, Any]]]] = None,
) -> dict[str, Any]:
    """
    Compute the full model stack for one replay date:
    gold structural, market structural, acceleration, fused macro state,
    and a strategy decision.
    """
    cfg = config or MacroBacktestConfig()
    dt = pd.Timestamp(as_of_date)
    hist = panel_df.loc[:dt].copy()
    if hist.empty:
        raise ValueError("No history available up to the requested replay date.")

    structural_dirs = cfg.structural_dirs or DEFAULT_GOLD_DIRS
    structural_weights = _normalize_weights(cfg.structural_weights or DEFAULT_GOLD_WEIGHTS)
    accel_w = _normalize_weights(cfg.accel_weights or DEFAULT_ACCEL_WEIGHTS)

    gold_res, gold_keys, _, _, _ = mm.run_structural(hist, structural_weights, structural_dirs)
    market_res, market_keys, _, _, _ = mm.run_market_structural(hist)
    accel_base = mm.compute_accel_features(hist)
    accel_res, accel_keys, _, _, _ = mm.run_accel(
        accel_base,
        cfg.accel_method,
        accel_w.get("GOLD_6M_RET", 0.0),
        accel_w.get("REALYIELD_3M_CHG", 0.0),
        accel_w.get("STRESS_3M_CHG", 0.0),
        accel_w.get("USD_3M_CHG", 0.0),
    )

    gold_signal = _latest_value(gold_res, "SIGNAL", dt)
    gold_regime = _latest_value(gold_res, "REGIME", dt)
    market_signal = _latest_value(market_res, "SIGNAL", dt)
    market_regime = _latest_value(market_res, "REGIME", dt)
    accel_signal = _latest_value(accel_res, "SIGNAL", dt)
    accel_regime = _latest_value(accel_res, "REGIME", dt)

    tactical = tactical_snapshot_fn(hist, dt, cfg.ticker) if tactical_snapshot_fn else {}
    tactical = tactical or {}

    mc = mc_snapshot_fn(hist, dt, cfg.ticker, tactical) if mc_snapshot_fn else {}
    mc = mc or {}

    macro_state = build_macro_state(
        asset=cfg.ticker,
        as_of_date=str(dt.date()),
        gold_signal=gold_signal,
        gold_regime=gold_regime,
        market_signal=market_signal,
        market_regime=market_regime,
        accel_signal=accel_signal,
        accel_regime=accel_regime,
        tactical_state=tactical.get("tactical_state"),
        screener_verdict=tactical.get("screener_verdict"),
        above_ma200=tactical.get("above_ma200"),
        rsi_daily=tactical.get("rsi_daily"),
        rsi_slope=tactical.get("rsi_slope"),
        vol_regime=tactical.get("vol_regime"),
        p10_ret=mc.get("p10_ret"),
        p50_ret=mc.get("p50_ret"),
        p90_ret=mc.get("p90_ret"),
        prob_higher_pct=mc.get("prob_higher_pct"),
    )

    decision = decide_trade(
        ticker=cfg.ticker,
        structural_regime=gold_regime,
        tactical_state=tactical.get("tactical_state"),
        mc_typical_pct=mc.get("p50_ret"),
        mc_prob_higher_pct=mc.get("prob_higher_pct"),
        above_ma200=tactical.get("above_ma200"),
        macro_state=macro_state,
        mc_downside_pct=mc.get("p10_ret"),
    )

    row: dict[str, Any] = {
        "gold_signal": gold_signal,
        "gold_regime": gold_regime,
        "gold_active_indicators": ",".join(gold_keys),
        "market_signal": market_signal,
        "market_regime": market_regime,
        "market_active_indicators": ",".join(market_keys),
        "accel_signal": accel_signal,
        "accel_regime": accel_regime,
        "accel_active_indicators": ",".join(accel_keys),
        "macro_score": macro_state.fused_score,
        "macro_state": macro_state.fused_state,
        "macro_narrative": macro_state.narrative,
        "tactical_state": tactical.get("tactical_state"),
        "screener_verdict": tactical.get("screener_verdict"),
        "above_ma200": tactical.get("above_ma200"),
        "rsi_daily": tactical.get("rsi_daily"),
        "rsi_slope": tactical.get("rsi_slope"),
        "vol_regime": tactical.get("vol_regime"),
        "mc_p10_ret": mc.get("p10_ret"),
        "mc_p50_ret": mc.get("p50_ret"),
        "mc_p90_ret": mc.get("p90_ret"),
        "mc_prob_higher_pct": mc.get("prob_higher_pct"),
        "decision": decision.action,
        "decision_confidence": decision.confidence,
        "position_size_pct": decision.position_size_pct,
        "decision_reason": decision.reason,
        "stop_loss_pct": decision.stop_loss_pct,
        "take_profit_pct": decision.take_profit_pct,
    }

    if cfg.price_col in panel_df.columns:
        row["price"] = _latest_value(panel_df, cfg.price_col, dt)
        if cfg.forward_return_months:
            fut_idx = panel_df.index[panel_df.index > dt]
            if len(fut_idx) >= int(cfg.forward_return_months):
                fut_dt = fut_idx[int(cfg.forward_return_months) - 1]
                p0 = row["price"]
                p1 = _latest_value(panel_df, cfg.price_col, fut_dt)
                if p0 not in (None, 0) and p1 is not None:
                    row[f"fwd_{int(cfg.forward_return_months)}m_ret_pct"] = 100.0 * (float(p1) / float(p0) - 1.0)

    return row


def run_macro_strategy_backtest(
    panel_df: pd.DataFrame,
    *,
    config: Optional[MacroBacktestConfig] = None,
    tactical_snapshot_fn: Optional[Callable[[pd.DataFrame, pd.Timestamp, str], Optional[dict[str, Any]]]] = None,
    mc_snapshot_fn: Optional[Callable[[pd.DataFrame, pd.Timestamp, str, Optional[dict[str, Any]]], Optional[dict[str, Any]]]] = None,
) -> pd.DataFrame:
    """
    Walk forward through a monthly macro panel and return one row per replay date,
    including model packs, fused macro state, and strategy decision.
    """
    cfg = config or MacroBacktestConfig()

    def _simulate(dt: pd.Timestamp) -> dict[str, Any]:
        return simulate_macro_strategy_date(
            panel_df,
            dt,
            config=cfg,
            tactical_snapshot_fn=tactical_snapshot_fn,
            mc_snapshot_fn=mc_snapshot_fn,
        )

    return run_walk_forward_backtest(
        price_df=panel_df,
        date_index=pd.DatetimeIndex(panel_df.index),
        simulate_one_date=_simulate,
        start_date=cfg.start_date,
        step=cfg.step,
    )
