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
    """
    Backward-compatible rule-based strategy engine.

    Supports both:
    - legacy inputs (structural_regime, tactical_state, mc_typical_pct, ...)
    - fused macro_state input from macro_fusion.build_macro_state(...)
    """

    fused_score = _safe_float(_macro_attr(macro_state, "fused_score", None))
    fused_state = _macro_attr(macro_state, "fused_state", None)

    if structural_regime is None and macro_state is not None:
        structural_regime = _macro_attr(_macro_attr(macro_state, "gold_structural", None), "regime", None)

    if tactical_state is None and macro_state is not None:
        tactical_state = _macro_attr(_macro_attr(macro_state, "tactical", None), "tactical_state", None)

    if above_ma200 is None and macro_state is not None:
        above_ma200 = _macro_attr(_macro_attr(macro_state, "tactical", None), "above_ma200", None)

    if mc_typical_pct is None and macro_state is not None:
        mc_typical_pct = _safe_float(_macro_attr(_macro_attr(macro_state, "monte_carlo", None), "p50_ret", None))

    if mc_prob_higher_pct is None and macro_state is not None:
        mc_prob_higher_pct = _safe_float(_macro_attr(_macro_attr(macro_state, "monte_carlo", None), "prob_higher_pct", None))

    if mc_downside_pct is None and macro_state is not None:
        mc_downside_pct = _safe_float(_macro_attr(_macro_attr(macro_state, "monte_carlo", None), "p10_ret", None))

    structural_ok = structural_regime in {"Structural Bull", "Positive", None}
    fused_good = fused_score is not None and fused_score >= 0.20
    fused_strong = fused_score is not None and fused_score >= 0.60
    fused_bad = fused_score is not None and fused_score <= -0.20
    fused_very_bad = fused_score is not None and fused_score <= -0.60

    tactical_ok = tactical_state in {
        "BULL_PULLBACK_LOWVOL",
        "BULL_CONTINUATION_LOWVOL",
        "BULL_MATURE_LOWVOL",
    }
    tactical_soft_ok = tactical_state in {
        "BULL_PULLBACK_LOWVOL",
        "BULL_CONTINUATION_LOWVOL",
        "BULL_MATURE_LOWVOL",
        "MIXED_LOWVOL",
    }
    tactical_bad = tactical_state in {
        "BEAR_CONTINUATION_HIGHVOL",
        "BEAR_WEAK_HIGHVOL",
        "BEAR_BOUNCE_HIGHVOL",
        "MIXED_HIGHVOL",
    }

    mc_ok = (
        mc_typical_pct is not None
        and mc_prob_higher_pct is not None
        and mc_typical_pct > 0
        and mc_prob_higher_pct >= 55
    )
    mc_soft_ok = (
        mc_typical_pct is not None
        and mc_prob_higher_pct is not None
        and mc_typical_pct >= 0
        and mc_prob_higher_pct >= 52
    )
    downside_heavy = (mc_downside_pct is not None and mc_downside_pct <= -8.0)

    if (fused_strong or structural_ok) and tactical_ok and mc_ok and (above_ma200 is True or above_ma200 is None):
        conf = 0.58
        if fused_score is not None:
            conf += max(0.0, min(0.20, (fused_score - 0.20) * 0.25))
        if mc_prob_higher_pct is not None:
            conf += max(0.0, min(0.15, (mc_prob_higher_pct - 55.0) / 100.0))
        conf = min(0.95, conf)

        pos = 0.05
        if fused_strong:
            pos += 0.025
        if mc_prob_higher_pct is not None and mc_prob_higher_pct >= 60:
            pos += 0.025
        if downside_heavy:
            pos -= 0.025
        pos = max(0.025, min(0.125, pos))

        return StrategyDecision(
            ticker=ticker,
            action="BUY",
            confidence=conf,
            position_size_pct=pos,
            reason=(
                f"{fused_state or structural_regime or 'Constructive macro'}"
                f" + {tactical_state} + MC supportive"
            ),
            stop_loss_pct=0.06 if not downside_heavy else 0.05,
            take_profit_pct=0.12,
        )

    if (fused_good and tactical_soft_ok and mc_soft_ok and (above_ma200 is True or above_ma200 is None)):
        return StrategyDecision(
            ticker=ticker,
            action="WATCH",
            confidence=0.55 if fused_score is None else min(0.80, 0.52 + max(0.0, fused_score) * 0.20),
            position_size_pct=0.00,
            reason=f"Macro constructive but setup not fully confirmed ({tactical_state or 'no tactical confirmation'})",
            stop_loss_pct=0.00,
            take_profit_pct=0.00,
        )

    if tactical_bad or fused_very_bad or (fused_bad and (mc_typical_pct is not None and mc_typical_pct < 0)):
        return StrategyDecision(
            ticker=ticker,
            action="SELL",
            confidence=0.65 if fused_very_bad or tactical_bad else 0.58,
            position_size_pct=0.00,
            reason=f"{tactical_state or 'Weak tactical state'} / macro or MC unsupportive",
            stop_loss_pct=0.00,
            take_profit_pct=0.00,
        )

    return StrategyDecision(
        ticker=ticker,
        action="HOLD",
        confidence=0.40 if fused_score is None else 0.45,
        position_size_pct=0.00,
        reason="Conditions not strong enough",
        stop_loss_pct=0.00,
        take_profit_pct=0.00,
    )
