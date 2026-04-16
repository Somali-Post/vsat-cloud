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


def calculate_raw_decision(state: Dict[str, Any]) -> Dict[str, Any]:
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

    total_value = _to_float(metrics.get("TotalValue", 0.0))
    pressure = _to_float(metrics.get("Pressure", 0.0))
    edge = _to_float(metrics.get("Edge", 0.0))
    volatility = _to_float(metrics.get("Volatility", 0.0))
    confidence = _to_float(metrics.get("Confidence", 50.0), default=50.0)

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
    )

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
