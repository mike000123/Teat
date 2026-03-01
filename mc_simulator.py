# mc_simulator.py
import numpy as np
import pandas as pd

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