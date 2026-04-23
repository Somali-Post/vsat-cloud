from __future__ import annotations

import random
import time
from typing import Any, Dict, Iterable, Mapping


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_postflop_state(metrics: Mapping[str, Any], glyphs: Iterable[Mapping[str, Any]]) -> bool:
    stage_value = str(metrics.get("Stage", "")).strip().lower()
    if stage_value in {"flop", "turn", "river", "postflop", "post_flop"}:
        return True
    if _to_float(metrics.get("Street", -1), default=-1) >= 1:
        return True

    for glyph in glyphs:
        name = str(glyph.get("name", "")).lower()
        if any(token in name for token in ("flop", "turn", "river", "postflop", "post_flop")):
            return True
    return False


def _first_metric(metrics: Mapping[str, Any], keys: Iterable[str], default: float = 0.0) -> float:
    for key in keys:
        if key in metrics:
            return _to_float(metrics.get(key), default=default)
    return default


def _extract_glyph_names(glyphs: Iterable[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for glyph in glyphs:
        names.append(str(glyph.get("name", "")).lower().replace(" ", "").replace("-", "").replace("_", ""))
    return names


def _has_premium_hand(glyphs: Iterable[Mapping[str, Any]]) -> bool:
    names = _extract_glyph_names(glyphs)
    if not names:
        return False

    premium_tokens = {
        "aa",
        "kk",
        "qq",
        "jj",
        "ak",
        "aceace",
        "kingking",
        "queenqueen",
        "jackjack",
        "aceking",
        "pocketaces",
        "pocketkings",
    }
    return any(any(token in name for token in premium_tokens) for name in names)


def _compute_tournament_bubble_distance(metrics: Mapping[str, Any]) -> float:
    rank = _first_metric(metrics, ("TournamentRank", "tournament_rank"), default=0.0)
    paid_places = _first_metric(metrics, ("Payout_Proximity", "PayoutProximity", "payout_proximity"), default=0.0)
    players_left = _first_metric(metrics, ("Players_Left", "PlayersLeft", "players_left"), default=0.0)

    if rank > 0.0 and paid_places > 0.0:
        return rank - paid_places
    if players_left > 0.0 and paid_places > 0.0:
        return players_left - paid_places
    return float("inf")


def calculate_raw_decision(state: Dict[str, Any], game_type: str = "CASH") -> Dict[str, Any]:
    """
    Compute a raw decision before UI snapping and humanized latency.

    Returns:
        {
            "action": str,
            "raw_value": float,       # continuous 0..3 scale before UI snapping
            "complexity": int,        # 1, 2, or 3
            "is_postflop": bool
        }
    """

    metrics = state.get("metrics", {}) or {}
    glyphs = state.get("glyphs", []) or []
    active_turn = bool(state.get("active_turn", False))
    normalized_game_type = str(game_type or "CASH").strip().upper()

    total_value = _first_metric(metrics, ("TotalValue", "TotalPot", "PotSize", "Pot"), default=0.0)
    pressure = _first_metric(metrics, ("Pressure",), default=0.0)
    edge = _first_metric(metrics, ("Edge",), default=0.0)
    volatility = _first_metric(metrics, ("Volatility",), default=0.0)
    confidence = _first_metric(metrics, ("Confidence",), default=50.0)
    incoming_bet = _first_metric(
        metrics,
        ("CurrentStake", "IncomingBet", "ToCall", "CallAmount", "BetToCall"),
        default=0.0,
    )
    hero_stack = _first_metric(
        metrics,
        ("HeroStack", "Stack", "EffectiveStack", "EffStack"),
        default=0.0,
    )

    pot_odds = 0.0
    if incoming_bet > 0.0 and (total_value + incoming_bet) > 0.0:
        pot_odds = incoming_bet / (total_value + incoming_bet)

    # Pressure-based complexity with hard override for high total value.
    pressure_signal = (
        pressure
        + max(0.0, volatility) * 0.6
        + (20.0 if active_turn else 0.0)
        + (8.0 * min(len(glyphs), 3))
    )
    if pressure_signal < 45.0:
        complexity = 1
    elif pressure_signal < 90.0:
        complexity = 2
    else:
        complexity = 3
    if total_value > 1000.0:
        complexity = 3

    # Aggression score drives action and raw size target.
    aggression = (
        pressure * 0.015
        + edge * 0.03
        + confidence * 0.01
        + (0.25 if active_turn else 0.0)
        + (0.2 * complexity)
        - (pot_odds * 1.4)
    )

    # Tournament ICM preservation near the payout bubble.
    bubble_distance = _compute_tournament_bubble_distance(metrics)
    icm_multiplier = 1.0
    if normalized_game_type == "TOURNAMENT":
        if bubble_distance <= 8:
            icm_multiplier = 0.55
        elif bubble_distance <= 15:
            icm_multiplier = 0.7
        elif bubble_distance <= 30:
            icm_multiplier = 0.85
        aggression *= icm_multiplier

    # Defensive override for near-all-in decisions unless premium cards are detected.
    pot_ratio = incoming_bet / max(total_value, 1e-6) if incoming_bet > 0.0 else 0.0
    stack_ratio = incoming_bet / hero_stack if (incoming_bet > 0.0 and hero_stack > 0.0) else 0.0
    all_in_pressure = incoming_bet > 0.0 and (pot_ratio >= 1.2 or stack_ratio >= 0.65)
    premium_hand = _has_premium_hand(glyphs)
    if all_in_pressure and not premium_hand:
        return {
            "action": "Fold",
            "raw_value": 0.0,
            "complexity": int(complexity),
            "is_postflop": bool(_is_postflop_state(metrics, glyphs)),
            "pot_odds": float(pot_odds),
            "incoming_bet": float(incoming_bet),
            "all_in_pressure": True,
            "game_type": normalized_game_type,
            "icm_multiplier": float(icm_multiplier),
        }
    if all_in_pressure and premium_hand:
        aggression += 0.65

    if total_value <= 0.0:
        action = "Check"
        raw_value = 0.0
    elif aggression < 1.25:
        action = "Fold"
        raw_value = 0.0
    elif aggression < 2.1:
        action = "Call"
        raw_value = 1.0
    elif aggression < 2.9:
        action = "Bet"
        raw_value = 2.0
    else:
        action = "Raise"
        raw_value = 3.0

    is_postflop = _is_postflop_state(metrics, glyphs)
    return {
        "action": action,
        "raw_value": float(raw_value),
        "complexity": int(complexity),
        "is_postflop": bool(is_postflop),
        "pot_odds": float(pot_odds),
        "incoming_bet": float(incoming_bet),
        "all_in_pressure": bool(all_in_pressure),
        "game_type": normalized_game_type,
        "icm_multiplier": float(icm_multiplier),
    }


class DecisionHumanizer:
    """Snaps raw math outputs to UI language and applies biomimetic delay."""

    _PREFLOP_STRINGS = ["Min", "3 BB", "Pot", "Max"]
    _POSTFLOP_STRINGS = ["Min", "50% of Pot", "Pot", "Max"]

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()

    def normalize_value(self, value: float, is_postflop: bool) -> str:
        labels = self._POSTFLOP_STRINGS if is_postflop else self._PREFLOP_STRINGS
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return labels[0]

        snapped_index = int(round(max(0.0, min(3.0, numeric))))
        return labels[snapped_index]

    def simulate_latency(self, complexity: int) -> float:
        level = max(1, min(3, int(complexity)))
        if level == 1:
            duration = self._rng.uniform(1.0, 2.5)
        elif level == 2:
            duration = self._rng.uniform(3.0, 5.0)
        else:
            duration = self._rng.uniform(6.0, 9.0)

        time.sleep(duration)
        return duration
