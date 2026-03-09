from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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


def tactical_bias_from_state(tactical_state: Optional[str], screener_verdict: Optional[str] = None) -> float:
    state = str(tactical_state or '').upper()
    verdict = str(screener_verdict or '').upper()
    if verdict == 'BUY':
        return 1.0
    if verdict == 'WATCH':
        return 0.35
    if verdict == 'SELL':
        return -1.0
    if state.startswith('BULL_PULLBACK') or state.startswith('BULL_CONTINUATION'):
        return 0.75
    if state.startswith('BULL_MATURE'):
        return 0.35
    if state.startswith('BEAR_CONTINUATION') or state.startswith('BEAR_WEAK'):
        return -0.75
    if state.startswith('BEAR_BOUNCE'):
        return -0.35
    return 0.0


def _bucket_fused(score: float) -> str:
    if score >= 0.60:
        return 'Risk-On Bullish'
    if score >= 0.20:
        return 'Constructive'
    if score > -0.20:
        return 'Balanced'
    if score > -0.60:
        return 'Cautious'
    return 'Defensive'


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
    tactical_snapshot: Optional[TacticalSnapshot] = None,
    monte_carlo_snapshot: Optional[MonteCarloSnapshot] = None,
) -> MacroState:
    tactical_bias = tactical_bias_from_state(
        getattr(tactical_snapshot, 'tactical_state', None),
        getattr(tactical_snapshot, 'screener_verdict', None),
    )

    if str(asset).lower().startswith('gold') or str(asset).upper() in {'GLD', 'GC=F'}:
        fused_score = (
            0.45 * _to_float(gold_signal)
            + 0.25 * _to_float(market_signal)
            + 0.20 * _to_float(accel_signal)
            + 0.10 * tactical_bias
        )
    else:
        fused_score = (
            0.15 * _to_float(gold_signal)
            + 0.45 * _to_float(market_signal)
            + 0.20 * _to_float(accel_signal)
            + 0.20 * tactical_bias
        )

    fused_state = _bucket_fused(fused_score)
    narrative = (
        f"Gold {gold_regime or '—'} | Market {market_regime or '—'} | "
        f"Accel {accel_regime or '—'} | Tactical {getattr(tactical_snapshot, 'tactical_state', '—') or '—'}"
    )

    return MacroState(
        as_of_date=as_of_date,
        asset=asset,
        gold_structural=RegimeSnapshot(signal=_to_float(gold_signal, None), regime=gold_regime),
        market_structural=RegimeSnapshot(signal=_to_float(market_signal, None), regime=market_regime),
        acceleration=RegimeSnapshot(signal=_to_float(accel_signal, None), regime=accel_regime),
        tactical=tactical_snapshot,
        monte_carlo=monte_carlo_snapshot,
        fused_state=fused_state,
        fused_score=fused_score,
        narrative=narrative,
    )
