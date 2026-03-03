# mc_simulator.py
import numpy as np
import pandas as pd

def classify_accel_state(rsi_val: float, rsi_slope: float) -> str:
    """
    Very simple tactical state classifier.
    You can refine thresholds later.
    """
    if pd.isna(rsi_val):
        return "UNKNOWN"

    # Momentum direction from slope
    if pd.isna(rsi_slope):
        slope_bucket = "FLAT"
    elif rsi_slope >= 2:
        slope_bucket = "UP"
    elif rsi_slope <= -2:
        slope_bucket = "DOWN"
    else:
        slope_bucket = "FLAT"

    # Level bucket
    if rsi_val >= 60:
        level = "STRONG"
    elif rsi_val <= 40:
        level = "WEAK"
    else:
        level = "NEUTRAL"

    return f"{level}_{slope_bucket}"


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