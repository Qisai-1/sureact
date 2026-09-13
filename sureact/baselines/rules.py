from __future__ import annotations

import random
from typing import Any


class RandomLegal:
    """Random legal action."""

    def __init__(self, seed: int = 0) -> None:
        self.name = "B1_random_legal"
        self._rng = random.Random(seed)

    def choose(self, checkpoint: dict[str, Any]) -> str:
        legal = list(checkpoint.get("legal_actions") or checkpoint.get("per_action", {}))
        return self._rng.choice(legal) if legal else "ASK"


class AlwaysAction:

    def __init__(self, action: str) -> None:
        self.action = action
        self.name = f"B2_always_{action.lower()}"

    def choose(self, checkpoint: dict[str, Any]) -> str:
        return self.action


class AskOrAct:

    HEDGES = ("not sure", "maybe", "i think", "unclear", "don't know", "some", "either")

    def __init__(self, act_action: str = "ACT_REVERSIBLE") -> None:
        self.act_action = act_action
        self.name = "B7_ask_or_act"

    def choose(self, checkpoint: dict[str, Any]) -> str:
        legal = list(checkpoint.get("legal_actions") or checkpoint.get("per_action", {}))
        hist = checkpoint.get("history") or []
        text = ""
        for m in reversed(hist):
            role = str(m.get("sender") or m.get("role") or "").lower()
            if "user" in role:
                text = str(m.get("content") or "").lower()
                break
        ambiguous = text.endswith("?") or any(h in text for h in self.HEDGES)
        want = "ASK" if ambiguous else self.act_action
        if want in legal:
            return want
        other = self.act_action if want == "ASK" else "ASK"
        return other if other in legal else (legal[0] if legal else "ASK")
