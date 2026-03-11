from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import pandas as pd


@dataclass
class MacroBacktestConfig:
    ticker: str = "GOLD"
    start_date: str = "2005-01-01"
    step: int = 1
    price_col: str = "GOLD_USD"
    forward_return_months: int = 1
    trading_mode: str = "long_flat"
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0

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
    """
    Walk-forward backtest skeleton.

    Parameters
    ----------
    price_df : pd.DataFrame
        Kept for future use / compatibility.
    date_index : pd.DatetimeIndex
        Dates over which to run the simulation.
    simulate_one_date : callable
        Function that accepts one pd.Timestamp and returns a dict.
    start_date : str
        First date to begin replay.
    step : int
        Use every Nth date.

    Returns
    -------
    pd.DataFrame
        One row per simulated date.
    """
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

def _nearest_price_at_or_before(price_series: pd.Series, dt: pd.Timestamp) -> float | None:
    s = pd.to_numeric(price_series, errors="coerce").dropna().sort_index()
    s = s[s.index <= pd.Timestamp(dt)]
    if s.empty:
        return None
    val = pd.to_numeric(s.iloc[-1], errors="coerce")
    return None if pd.isna(val) else float(val)


def run_macro_strategy_backtest(
    res_base: pd.DataFrame | None = None,
    compute_one_date: Optional[Callable[[pd.DataFrame, pd.Timestamp], dict]] = None,
    config: Optional[MacroBacktestConfig] = None,
    *,
    price_series: pd.Series | None = None,
    date_index: pd.DatetimeIndex | None = None,
    simulate_one_date: Optional[Callable[[pd.Timestamp], dict]] = None,
    start_date: str | pd.Timestamp | None = None,
    step_months: int = 1,
    forward_months: int = 1,
) -> pd.DataFrame:

    # -----------------------------
    # Style A: config-based API
    # -----------------------------
    if config is not None and res_base is not None and compute_one_date is not None:
        if res_base.empty or config.price_col not in res_base.columns:
            return pd.DataFrame()

        idx = pd.DatetimeIndex(res_base.index).sort_values().unique()
        replay_dates = walk_forward_dates(idx, start_date=str(config.start_date), step=int(config.step))
        px = pd.to_numeric(res_base[config.price_col], errors="coerce").dropna().sort_index()

        rows = []
        for dt in replay_dates:
            hist_df = res_base.loc[res_base.index <= pd.Timestamp(dt)].copy()
            if hist_df.empty:
                continue

            try:
                row = compute_one_date(hist_df, pd.Timestamp(dt)) or {}
            except Exception as e:
                row = {"error": str(e)}

            row["as_of_date"] = pd.Timestamp(dt)

            price_now = _nearest_price_at_or_before(px, pd.Timestamp(dt))
            row["price"] = price_now

            future_cut = pd.Timestamp(dt) + pd.DateOffset(months=int(config.forward_return_months))
            future_series = px[(px.index >= pd.Timestamp(dt)) & (px.index <= future_cut)]

            price_future = None
            if not future_series.empty:
                v = pd.to_numeric(future_series.iloc[-1], errors="coerce")
                price_future = None if pd.isna(v) else float(v)

            fwd_col = f"fwd_{int(config.forward_return_months)}m_ret_pct"
            if price_now is not None and price_future is not None and price_now > 0:
                row[fwd_col] = (price_future / price_now - 1.0) * 100.0
            else:
                row[fwd_col] = np.nan

            rows.append(row)

        out = pd.DataFrame(rows)
        if not out.empty:
            out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
            out = out.sort_values("as_of_date").reset_index(drop=True)
        return out

    # -----------------------------
    # Style B: legacy API
    # -----------------------------
    if price_series is None or date_index is None or simulate_one_date is None or start_date is None:
        raise TypeError(
            "run_macro_strategy_backtest() requires either "
            "(res_base, compute_one_date=..., config=...) "
            "or "
            "(price_series=..., date_index=..., simulate_one_date=..., start_date=..., ...)"
        )

    ps = pd.to_numeric(price_series, errors="coerce").dropna().sort_index()
    if ps.empty:
        return pd.DataFrame()

    rows = []
    replay_dates = walk_forward_dates(
        pd.DatetimeIndex(date_index),
        start_date=str(pd.Timestamp(start_date).date()),
        step=max(1, int(step_months)),
    )

    for dt in replay_dates:
        try:
            row = simulate_one_date(pd.Timestamp(dt)) or {}
        except Exception as e:
            row = {"error": str(e)}

        row["as_of_date"] = pd.Timestamp(dt)

        price_now = _nearest_price_at_or_before(ps, pd.Timestamp(dt))
        row["price"] = price_now

        future_cut = pd.Timestamp(dt) + pd.DateOffset(months=int(forward_months))
        future_series = ps[(ps.index >= pd.Timestamp(dt)) & (ps.index <= future_cut)]

        price_future = None
        if not future_series.empty:
            v = pd.to_numeric(future_series.iloc[-1], errors="coerce")
            price_future = None if pd.isna(v) else float(v)

        fwd_col = f"fwd_{int(forward_months)}m_ret_pct"
        if price_now is not None and price_future is not None and price_now > 0:
            row[fwd_col] = (price_future / price_now - 1.0) * 100.0
        else:
            row[fwd_col] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
        out = out.sort_values("as_of_date").reset_index(drop=True)
    return out


def _position_label(x: float) -> str:
    if pd.isna(x):
        return "Unknown"
    if x > 0:
        return "Long"
    if x < 0:
        return "Short"
    return "Flat"


def _decision_to_target_position(decision: str, size_pct: float | None, trading_mode: str = "long_flat") -> float:
    d = str(decision or "").upper()
    sz = pd.to_numeric(size_pct, errors="coerce")
    if pd.isna(sz):
        sz = 100.0
    target = max(0.0, min(1.0, float(sz) / 100.0))

    if trading_mode == "long_short":
        if d == "BUY":
            return target
        if d == "SELL":
            return -target
        return np.nan

    # long_flat
    if d == "BUY":
        return target
    if d == "SELL":
        return 0.0
    return np.nan


def build_portfolio_backtest(
    bt_df: pd.DataFrame,
    res_base: pd.DataFrame | None = None,
    price_col: str | None = None,
    initial_capital: float = 100.0,
    trading_mode: str = "long_flat",
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    execution_lag_steps: int = 0,
    min_hold_steps: int = 0,
    confirmation_steps: int = 1,
    rebalance_mode: str = "on_change",
    rebalance_threshold_pct: float = 0.0,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> pd.DataFrame:
    if bt_df is None or bt_df.empty:
        return pd.DataFrame()

    df = bt_df.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
    df = df.sort_values("as_of_date").reset_index(drop=True)

    # infer forward-return column if present
    fwd_cols = [c for c in df.columns if c.startswith("fwd_") and c.endswith("m_ret_pct")]
    fwd_col = fwd_cols[0] if fwd_cols else None

    # price / next price / step return
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["next_price"] = df["price"].shift(-1)
    df["asset_period_return_pct"] = np.where(
        (df["price"] > 0) & df["next_price"].notna(),
        (df["next_price"] / df["price"] - 1.0) * 100.0,
        np.nan,
    )

    raw_actions = df.get("decision", pd.Series(index=df.index, dtype=object)).astype(str).str.upper()
    raw_sizes = df["position_size_pct"] if "position_size_pct" in df.columns else pd.Series(100.0, index=df.index)
    raw_sizes = pd.to_numeric(raw_sizes, errors="coerce").fillna(100.0)

    # confirmation logic
    conf_n = max(1, int(confirmation_steps))
    confirmed_actions = []
    streak_action = None
    streak_count = 0
    for act in raw_actions:
        if act == streak_action:
            streak_count += 1
        else:
            streak_action = act
            streak_count = 1
        confirmed_actions.append(act if streak_count >= conf_n else "WATCH")

    df["raw_action"] = raw_actions
    df["confirmed_action"] = confirmed_actions
    df["signal_confirmed"] = df["raw_action"] == df["confirmed_action"]

    # lagged execution signal
    lag = max(0, int(execution_lag_steps))
    exec_action = pd.Series(df["confirmed_action"]).shift(lag).fillna("WATCH")
    exec_size = pd.Series(raw_sizes).shift(lag).fillna(raw_sizes.iloc[0] if len(raw_sizes) else 100.0)

    equity = float(initial_capital)
    benchmark = float(initial_capital)

    applied_positions = []
    next_positions = []
    position_before_labels = []
    position_after_labels = []
    executed_trades = []
    turnovers = []
    strat_rets = []
    equity_curve = []
    benchmark_curve = []
    drawdowns = []
    hold_steps = []
    stop_flags = []
    take_flags = []

    current_pos = 0.0
    current_peak = equity
    start_price = df["price"].dropna().iloc[0] if df["price"].dropna().shape[0] else np.nan
    hold_counter = 0

    for i in range(len(df)):
        before_pos = current_pos
        act = str(exec_action.iloc[i]).upper()
        size = float(exec_size.iloc[i]) if not pd.isna(exec_size.iloc[i]) else 100.0

        target = _decision_to_target_position(act, size, trading_mode=trading_mode)
        if pd.isna(target):
            target = current_pos

        # minimum hold
        can_change = hold_counter >= max(0, int(min_hold_steps))
        if not can_change and target != current_pos:
            target = current_pos

        # rebalance modes
        if rebalance_mode == "on_change":
            if np.sign(target) == np.sign(current_pos) and abs(target - current_pos) < 1e-12:
                target = current_pos
        elif rebalance_mode == "threshold":
            thr = max(0.0, float(rebalance_threshold_pct)) / 100.0
            if abs(target - current_pos) < thr:
                target = current_pos

        turnover = abs(target - current_pos)
        cost_pct = turnover * (float(transaction_cost_bps) + float(slippage_bps)) / 10000.0

        step_ret_pct = df.loc[i, "asset_period_return_pct"]
        step_ret = 0.0 if pd.isna(step_ret_pct) else float(step_ret_pct) / 100.0

        gross = target * step_ret

        stop_hit = False
        take_hit = False
        if stop_loss_pct is not None and not pd.isna(stop_loss_pct):
            if gross <= -(float(stop_loss_pct) / 100.0):
                gross = -(float(stop_loss_pct) / 100.0)
                stop_hit = True
        if take_profit_pct is not None and not pd.isna(take_profit_pct):
            if gross >= (float(take_profit_pct) / 100.0):
                gross = (float(take_profit_pct) / 100.0)
                take_hit = True

        net = gross - cost_pct
        equity *= (1.0 + net)

        # benchmark = normalized underlying asset
        px = df.loc[i, "price"]
        if pd.notna(px) and pd.notna(start_price) and start_price > 0:
            benchmark = 100.0 * (float(px) / float(start_price))

        current_peak = max(current_peak, equity)
        dd = (equity / current_peak - 1.0) * 100.0 if current_peak > 0 else 0.0

        trade_label = "No change"
        if target != before_pos:
            if target > before_pos:
                trade_label = "Increase long" if before_pos >= 0 else "Flip to long"
            elif target < before_pos:
                trade_label = "Reduce / exit" if target >= 0 else "Flip to short"

        applied_positions.append(before_pos * 100.0)
        next_positions.append(target * 100.0)
        position_before_labels.append(_position_label(before_pos))
        position_after_labels.append(_position_label(target))
        executed_trades.append(trade_label)
        turnovers.append(turnover * 100.0)
        strat_rets.append(net * 100.0)
        equity_curve.append(equity)
        benchmark_curve.append(benchmark)
        drawdowns.append(dd)
        hold_steps.append(hold_counter)
        stop_flags.append(stop_hit)
        take_flags.append(take_hit)

        if target == current_pos:
            hold_counter += 1
        else:
            hold_counter = 0

        current_pos = 0.0 if (stop_hit or take_hit) else target

    df["signal_action"] = df["raw_action"]
    df["executed_trade"] = executed_trades
    df["position_before_label"] = position_before_labels
    df["position_before_pct"] = applied_positions
    df["position_after_label"] = position_after_labels
    df["position_after_pct"] = next_positions
    df["applied_position_pct"] = applied_positions
    df["next_position_pct"] = next_positions
    df["hold_steps_in_position"] = hold_steps
    df["turnover_pct"] = turnovers
    df["strategy_period_return_pct"] = strat_rets
    df["equity_curve"] = equity_curve
    df["benchmark_curve"] = benchmark_curve
    df["drawdown_pct"] = drawdowns
    df["stop_triggered"] = stop_flags
    df["take_profit_triggered"] = take_flags

    if fwd_col and fwd_col in df.columns and "fwd_1m_ret_pct" not in df.columns:
        df["fwd_1m_ret_pct"] = df[fwd_col]

    return df


def compute_backtest_analytics(portfolio_df: pd.DataFrame, periods_per_year: float = 12.0) -> dict:
    if portfolio_df is None or portfolio_df.empty:
        return {}

    df = portfolio_df.copy()
    rows = int(len(df))

    strat = pd.to_numeric(df.get("strategy_period_return_pct"), errors="coerce") / 100.0
    eq = pd.to_numeric(df.get("equity_curve"), errors="coerce")
    dd = pd.to_numeric(df.get("drawdown_pct"), errors="coerce")
    turnover = pd.to_numeric(df.get("turnover_pct"), errors="coerce")

    eq_valid = eq.dropna()
    if eq_valid.shape[0]:
        start_equity = float(eq_valid.iloc[0])
        end_equity = float(eq_valid.iloc[-1])
        total_return_pct = ((end_equity / start_equity) - 1.0) * 100.0 if start_equity != 0 else np.nan
    else:
        total_return_pct = np.nan
    max_drawdown_pct = float(dd.min()) if dd.dropna().shape[0] else np.nan
    avg_turnover_pct = float(turnover.mean()) if turnover.dropna().shape[0] else np.nan

    # buy hit rate with aligned index
    decision = df.get("decision", pd.Series(index=df.index, dtype=object)).astype(str).str.upper()
    fwd = pd.to_numeric(df.get("fwd_1m_ret_pct"), errors="coerce")
    valid = pd.DataFrame({"decision": decision, "fwd": fwd}).dropna(subset=["fwd"])
    buy_hit_rate_pct = float((valid.loc[valid["decision"] == "BUY", "fwd"] > 0).mean() * 100.0) if (valid["decision"] == "BUY").any() else np.nan

    if strat.dropna().shape[0] >= 2:
        mean_r = float(strat.mean())
        vol_r = float(strat.std(ddof=0))
        years = max(rows / float(periods_per_year), 1e-9)

        if eq_valid.shape[0]:
            start_equity = float(eq_valid.iloc[0])
            end_equity = float(eq_valid.iloc[-1])
            cagr_pct = ((end_equity / start_equity) ** (1.0 / years) - 1.0) * 100.0 if start_equity > 0 else np.nan
        else:
            cagr_pct = np.nan

        volatility_pct = vol_r * np.sqrt(periods_per_year) * 100.0
        sharpe_like = (mean_r / vol_r) * np.sqrt(periods_per_year) if vol_r > 0 else np.nan
    else:
        cagr_pct = np.nan
        volatility_pct = np.nan
        sharpe_like = np.nan

    confirmed_rows = int(pd.to_numeric(df.get("signal_confirmed"), errors="coerce").fillna(0).astype(bool).sum()) if "signal_confirmed" in df.columns else 0
    stop_events = int(pd.to_numeric(df.get("stop_triggered"), errors="coerce").fillna(0).astype(bool).sum()) if "stop_triggered" in df.columns else 0
    take_profit_events = int(pd.to_numeric(df.get("take_profit_triggered"), errors="coerce").fillna(0).astype(bool).sum()) if "take_profit_triggered" in df.columns else 0

    return {
        "rows": rows,
        "buy_hit_rate_pct": buy_hit_rate_pct,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "cagr_pct": cagr_pct,
        "volatility_pct": volatility_pct,
        "sharpe_like": sharpe_like,
        "avg_turnover_pct": avg_turnover_pct,
        "confirmed_rows": confirmed_rows,
        "stop_events": stop_events,
        "take_profit_events": take_profit_events,
    }