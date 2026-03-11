# mc_simulator.py
import numpy as np
import pandas as pd



def classify_accel_state(rsi_val, rsi_slope, above_ma200, rs_vs_qqq_3m, ma50_slope_20d, vol_regime
):
    if pd.isna(rsi_val):
        return "UNKNOWN"

    bull_trend = (above_ma200 is True) and pd.notna(ma50_slope_20d) and (ma50_slope_20d > 0)
    bear_trend = (above_ma200 is False) and pd.notna(ma50_slope_20d) and (ma50_slope_20d < 0)
    leader = pd.notna(rs_vs_qqq_3m) and (rs_vs_qqq_3m > 0)
    high_vol = str(vol_regime).upper() == "HIGH"

    if bull_trend and leader:
        if rsi_val <= 45:
            core = "BULL_PULLBACK"
        elif rsi_slope >= 2:
            core = "BULL_CONTINUATION"
        else:
            core = "BULL_MATURE"
    elif bear_trend and (not leader):
        if rsi_val >= 55:
            core = "BEAR_BOUNCE"
        elif rsi_slope <= -2:
            core = "BEAR_CONTINUATION"
        else:
            core = "BEAR_WEAK"
    else:
        core = "MIXED"

    vol = "HIGHVOL" if high_vol else "LOWVOL"
    return f"{core}_{vol}"


def monte_carlo_paths_by_tactical_state(
    price: pd.Series,
    state: pd.Series,
    state_now: str | None = None,
    horizon_steps: int = 60,
    n_sims: int = 2000,
    seed: int = 7
):
    """
    Bootstraps 1-step returns from historical bars where 'state' == state_now.
    - price: Series of prices (daily or intraday, consistent spacing preferred)
    - state: Series of same index labeling each bar's tactical state
    """
    px = price.dropna().astype(float).copy()
    stt = state.reindex(px.index).astype(str)

    tmp = pd.DataFrame({"PX": px, "STATE": stt})
    tmp["RET_1"] = tmp["PX"].pct_change()

    if state_now is None:
        state_now = str(tmp["STATE"].iloc[-1])

    pool = tmp.loc[tmp["STATE"] == str(state_now), "RET_1"].dropna()
    if pool.empty:
        return None, state_now

    start_price = float(tmp["PX"].iloc[-1])
    rng = np.random.default_rng(seed)
    rets = rng.choice(pool.values, size=(n_sims, horizon_steps), replace=True)
    paths = start_price * np.cumprod(1.0 + rets, axis=1)
    return paths, state_now

def monte_carlo_paths_by_tactical_state_block(
    price: pd.Series,
    state: pd.Series,
    state_now: str | None = None,
    horizon_steps: int = 60,
    n_sims: int = 2000,
    block_size: int = 5,
    seed: int = 7,
):
    """
    Block-bootstrap Monte Carlo.
    Samples contiguous blocks of returns (default 5 bars) from historical periods
    where the tactical state matches `state_now`.

    This preserves some short-term momentum/mean-reversion structure better than
    single-step bootstrap.

    Parameters
    ----------
    price : pd.Series
        Price series (daily or intraday).
    state : pd.Series
        Tactical state label aligned to the same index as price.
    state_now : str | None
        Current state to condition on. If None, uses latest state.
    horizon_steps : int
        Number of future bars to simulate.
    n_sims : int
        Number of Monte Carlo paths.
    block_size : int
        Number of consecutive returns per sampled block.
    seed : int
        RNG seed.

    Returns
    -------
    paths : np.ndarray | None
        Shape (n_sims, horizon_steps), simulated price paths.
    state_now : str
        State used for conditioning.
    """
    px = price.dropna().astype(float).copy()
    stt = state.reindex(px.index).astype(str)

    tmp = pd.DataFrame({"PX": px, "STATE": stt})
    tmp["RET_1"] = tmp["PX"].pct_change()

    if state_now is None:
        state_now = str(tmp["STATE"].iloc[-1])

    # Keep only rows in the matching state
    tmp_state = tmp.loc[tmp["STATE"] == str(state_now)].copy()
    ret_series = tmp_state["RET_1"].dropna()

    if ret_series.empty or len(ret_series) < max(block_size * 3, 20):
        return None, state_now

    rets = ret_series.to_numpy()
    n = len(rets)

    # Build overlapping contiguous blocks
    blocks = []
    for i in range(0, n - block_size + 1):
        blk = rets[i:i + block_size]
        if len(blk) == block_size and np.isfinite(blk).all():
            blocks.append(blk)

    if not blocks:
        return None, state_now

    blocks = np.array(blocks)
    start_price = float(tmp["PX"].iloc[-1])

    rng = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(horizon_steps / block_size))

    paths = np.empty((n_sims, horizon_steps), dtype=float)

    for s in range(n_sims):
        chosen_idx = rng.integers(0, len(blocks), size=n_blocks_needed)
        sim_rets = blocks[chosen_idx].reshape(-1)[:horizon_steps]
        sim_path = start_price * np.cumprod(1.0 + sim_rets)
        paths[s, :] = sim_path

    return paths, state_now

def monte_carlo_paths_by_regime(
    df: pd.DataFrame,
    price_col: str = "GOLD_USD",
    regime_col: str = "REGIME",
    regime_now: str | None = None,
    horizon_months: int = 12,
    n_sims: int = 2000,
    seed: int = 7
):
    """
    Bootstraps 1M returns from historical months matching regime_now.
    Returns array (n_sims, horizon_months) of simulated price paths.
    """
    tmp = df[[price_col, regime_col]].dropna().copy()
    tmp["RET_1M"] = tmp[price_col].pct_change()

    if regime_now is None:
        regime_now = str(tmp[regime_col].iloc[-1])

    pool = tmp.loc[tmp[regime_col].astype(str) == str(regime_now), "RET_1M"].dropna()
    if pool.empty:
        return None, regime_now

    start_price = float(tmp[price_col].iloc[-1])
    rng = np.random.default_rng(seed)
    rets = rng.choice(pool.values, size=(n_sims, horizon_months), replace=True)
    paths = start_price * np.cumprod(1.0 + rets, axis=1)
    return paths, regime_now

def mc_percentiles(paths: np.ndarray, ps=(10, 50, 90)) -> pd.DataFrame:
    p = {f"p{q}": np.percentile(paths, q, axis=0) for q in ps}
    return pd.DataFrame(p)

def monte_carlo_cone_by_tactical_state_block(
    price: pd.Series,
    state: pd.Series,
    state_now: str | None = None,
    horizon_steps: int = 12,
    n_sims: int = 1000,
    block_size: int = 5,
    seed: int = 7,
):
    """
    Returns step-by-step p10 / p50 / p90 price cone from a chosen anchor history.

    Parameters
    ----------
    price : pd.Series
        Historical price series up to the anchor date only.
    state : pd.Series
        Tactical state series aligned to price.
    state_now : str | None
        Current tactical state at the anchor date. If None, latest state is used.
    horizon_steps : int
        Number of future bars to simulate.
    n_sims : int
        Number of Monte Carlo paths.
    block_size : int
        Block size for block bootstrap.
    seed : int
        RNG seed.

    Returns
    -------
    cone_df : pd.DataFrame
        Columns: step, p10, p50, p90
    state_used : str
        Tactical state used for conditioning.
    """
    paths, state_used = monte_carlo_paths_by_tactical_state_block(
        price=price,
        state=state,
        state_now=state_now,
        horizon_steps=horizon_steps,
        n_sims=n_sims,
        block_size=block_size,
        seed=seed,
    )

    if paths is None:
        return pd.DataFrame(columns=["step", "p10", "p50", "p90"]), state_used

    bands = mc_percentiles(paths).copy()
    bands["step"] = np.arange(1, len(bands) + 1)

    cols = ["step", "p10", "p50", "p90"]
    return bands[cols], state_used