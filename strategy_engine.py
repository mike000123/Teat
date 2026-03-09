from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class StrategyDecision:
    ticker: str
    action: str
    confidence: float
    position_size_pct: float
    reason: str
    stop_loss_pct: float
    take_profit_pct: float


def _safe_float(x: Any) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def _macro_attr(obj: Any, name: str, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def decide_trade(
    *,
    ticker: str,
    structural_regime: Optional[str] = None,
    tactical_state: Optional[str] = None,
    mc_typical_pct: Optional[float] = None,
    mc_prob_higher_pct: Optional[float] = None,
    above_ma200: Optional[bool] = None,
    macro_state: Optional[Any] = None,
    mc_downside_pct: Optional[float] = None,
) -> StrategyDecision:
    fused_score = _safe_float(_macro_attr(macro_state, 'fused_score', None))

    if structural_regime is None and macro_state is not None:
        structural_regime = _macro_attr(_macro_attr(macro_state, 'gold_structural', None), 'regime', None)
    if tactical_state is None and macro_state is not None:
        tactical_state = _macro_attr(_macro_attr(macro_state, 'tactical', None), 'tactical_state', None)
    if above_ma200 is None and macro_state is not None:
        above_ma200 = _macro_attr(_macro_attr(macro_state, 'tactical', None), 'above_ma200', None)
    if mc_typical_pct is None and macro_state is not None:
        mc_typical_pct = _safe_float(_macro_attr(_macro_attr(macro_state, 'monte_carlo', None), 'p50_ret', None))
    if mc_prob_higher_pct is None and macro_state is not None:
        mc_prob_higher_pct = _safe_float(_macro_attr(_macro_attr(macro_state, 'monte_carlo', None), 'prob_higher_pct', None))
    if mc_downside_pct is None and macro_state is not None:
        mc_downside_pct = _safe_float(_macro_attr(_macro_attr(macro_state, 'monte_carlo', None), 'p10_ret', None))

    structural_ok = structural_regime in {'Structural Bull', 'Positive', None}
    tactical_ok = tactical_state in {
        'BULL_PULLBACK_LOWVOL', 'BULL_CONTINUATION_LOWVOL', 'BULL_MATURE_LOWVOL', None
    }
    tactical_bad = tactical_state in {
        'BEAR_CONTINUATION_HIGHVOL', 'BEAR_WEAK_HIGHVOL', 'BEAR_BOUNCE_HIGHVOL'
    }
    mc_ok = (
        mc_typical_pct is not None and mc_prob_higher_pct is not None and mc_typical_pct > 0 and mc_prob_higher_pct >= 55
    )
    downside_bad = (mc_downside_pct is not None and mc_downside_pct < -12)

    if fused_score is not None:
        if fused_score >= 0.60 and tactical_ok and (above_ma200 is True or above_ma200 is None) and not downside_bad:
            conf = 0.68
            if mc_prob_higher_pct is not None:
                conf = min(0.95, conf + max(0.0, mc_prob_higher_pct - 55.0) / 100.0)
            size = 0.075
            if fused_score >= 0.80:
                size = 0.10
            if mc_prob_higher_pct is not None and mc_prob_higher_pct >= 60:
                size += 0.025
            return StrategyDecision(ticker, 'BUY', conf, min(size, 0.125), f'Fused macro supportive ({fused_score:.2f})', 0.06, 0.12)
        if fused_score >= 0.20 and not tactical_bad:
            return StrategyDecision(ticker, 'WATCH', 0.58, 0.0, f'Macro constructive but entry not fully confirmed ({fused_score:.2f})', 0.0, 0.0)
        if fused_score <= -0.60 or tactical_bad:
            return StrategyDecision(ticker, 'SELL', 0.65, 0.0, f'Macro/tactical headwind ({fused_score:.2f})', 0.0, 0.0)
        return StrategyDecision(ticker, 'HOLD', 0.45, 0.0, 'Macro state mixed / balanced', 0.0, 0.0)

    if structural_ok and tactical_ok and mc_ok and (above_ma200 is True or above_ma200 is None):
        return StrategyDecision(
            ticker=ticker,
            action='BUY',
            confidence=min(0.95, 0.50 + ((mc_prob_higher_pct or 50) - 50) / 100),
            position_size_pct=0.10 if (mc_prob_higher_pct or 0) >= 60 else 0.05,
            reason=f"{structural_regime or 'No structural filter'} + {tactical_state} + MC supportive",
            stop_loss_pct=0.06,
            take_profit_pct=0.12,
        )

    if tactical_bad or (mc_typical_pct is not None and mc_typical_pct < 0):
        return StrategyDecision(ticker, 'SELL', 0.60, 0.0, f"{tactical_state or 'Weak tactical state'} / MC unsupportive", 0.0, 0.0)

    return StrategyDecision(ticker, 'HOLD', 0.40, 0.0, 'Conditions not strong enough', 0.0, 0.0)
