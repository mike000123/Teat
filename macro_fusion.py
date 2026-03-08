from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


FUSED_STATE_LABELS = [
    (0.60, "Risk-On Bullish"),
    (0.20, "Constructive"),
    (-0.20, "Balanced"),
    (-0.60, "Cautious"),
]


@dataclass
class RegimeSnapshot:
    signal: Optional[float] = None
    regime: Optional[str] = None
    indicator_states: dict[str, int] = field(default_factory=dict)
    indicator_contribs: dict[str, float] = field(default_factory=dict)
    active_indicators: list[str] = field(default_factory=list)


@dataclass
class TacticalSnapshot:
    ticker: str
    tactical_state: Optional[str] = None
    screener_score: Optional[float] = None
    screener_verdict: Optional[str] = None
    above_ma200: Optional[bool] = None
    rsi_daily: Optional[float] = None
    rsi_slope: Optional[float] = None
    vol_regime: Optional[str] = None


@dataclass
class MonteCarloSnapshot:
    p10_ret: Optional[float] = None
    p50_ret: Optional[float] = None
    p90_ret: Optional[float] = None
    prob_higher_pct: Optional[float] = None
    state_conditioned_on: Optional[str] = None


@dataclass
class MacroState:
    as_of_date: Optional[str]
    asset: str
    gold_structural: RegimeSnapshot
    market_structural: RegimeSnapshot
    acceleration: RegimeSnapshot
    tactical: Optional[TacticalSnapshot]
    monte_carlo: Optional[MonteCarloSnapshot]
    fused_state: str
    fused_score: float
    narrative: str


def _to_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _bucket_fused(score: float) -> str:
    for threshold, label in FUSED_STATE_LABELS:
        if score >= threshold:
            return label
    return "Defensive"


def tactical_bias_from_state(
    tactical_state: Optional[str],
    screener_verdict: Optional[str] = None,
) -> float:
    state = str(tactical_state or "").upper()
    verdict = str(screener_verdict or "").upper()

    if verdict == "BUY":
        return 1.0
    if verdict == "SELL":
        return -1.0
    if verdict == "WATCH":
        return 0.25

    if state.startswith("BULL_PULLBACK") or state.startswith("BULL_CONTINUATION"):
        return 1.0
    if state.startswith("BULL_MATURE"):
        return 0.50
    if state.startswith("BEAR_CONTINUATION") or state.startswith("BEAR_WEAK"):
        return -1.0
    if state.startswith("BEAR_BOUNCE"):
        return -0.50
    if state.startswith("MIXED"):
        return 0.0
    return 0.0


def build_macro_state(
    *,
    asset: str,
    as_of_date: Optional[str],
    gold_signal: Optional[float],
    gold_regime: Optional[str],
    market_signal: Optional[float],
    market_regime: Optional[str],
    accel_signal: Optional[float],
    accel_regime: Optional[str],
    tactical_state: Optional[str] = None,
    screener_verdict: Optional[str] = None,
    above_ma200: Optional[bool] = None,
    rsi_daily: Optional[float] = None,
    rsi_slope: Optional[float] = None,
    vol_regime: Optional[str] = None,
    p10_ret: Optional[float] = None,
    p50_ret: Optional[float] = None,
    p90_ret: Optional[float] = None,
    prob_higher_pct: Optional[float] = None,
) -> MacroState:
    asset_u = str(asset).upper()
    tactical = TacticalSnapshot(
        ticker=asset,
        tactical_state=tactical_state,
        screener_verdict=screener_verdict,
        above_ma200=above_ma200,
        rsi_daily=rsi_daily,
        rsi_slope=rsi_slope,
        vol_regime=vol_regime,
    )
    mc = MonteCarloSnapshot(
        p10_ret=p10_ret,
        p50_ret=p50_ret,
        p90_ret=p90_ret,
        prob_higher_pct=prob_higher_pct,
        state_conditioned_on=tactical_state,
    )

    tactical_bias = tactical_bias_from_state(tactical_state=tactical_state, screener_verdict=screener_verdict)

    gold_w, market_w, accel_w, tactical_w = (0.45, 0.25, 0.20, 0.10) if asset_u in {"GLD", "GC=F", "XAUUSD", "GOLD"} else (0.15, 0.45, 0.20, 0.20)

    fused_score = (
        gold_w * _to_float(gold_signal)
        + market_w * _to_float(market_signal)
        + accel_w * _to_float(accel_signal)
        + tactical_w * tactical_bias
    )
    fused_state = _bucket_fused(fused_score)

    pieces = [
        f"gold={gold_regime or 'N/A'} ({_to_float(gold_signal):+.2f})",
        f"market={market_regime or 'N/A'} ({_to_float(market_signal):+.2f})",
        f"accel={accel_regime or 'N/A'} ({_to_float(accel_signal):+.2f})",
    ]
    if tactical_state:
        pieces.append(f"tactical={tactical_state}")
    if prob_higher_pct is not None:
        pieces.append(f"MC↑={_to_float(prob_higher_pct):.1f}%")

    return MacroState(
        as_of_date=as_of_date,
        asset=asset,
        gold_structural=RegimeSnapshot(signal=gold_signal, regime=gold_regime),
        market_structural=RegimeSnapshot(signal=market_signal, regime=market_regime),
        acceleration=RegimeSnapshot(signal=accel_signal, regime=accel_regime),
        tactical=tactical,
        monte_carlo=mc,
        fused_state=fused_state,
        fused_score=float(fused_score),
        narrative=" | ".join(pieces),
    )
