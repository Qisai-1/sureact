from __future__ import annotations

from typing import Any


class ValueController:
    """Choose the action with the highest predicted Q."""

    def __init__(
        self,
        model: Any,
        tok: Any,
        lam: float,
        eta: float,
        profile: Any,
        name: str = "SUREACT_value",
        max_len: int = 1024,
        device: str = "cuda",
        default_action: str | None = None,
        margin: float = 0.0,
        backbone_hint: bool = False,
    ) -> None:
        self.name = name
        self._model, self._tok = model, tok
        self._lam, self._eta, self._profile = lam, eta, profile
        self._max_len, self._device = max_len, device
        self._cache: dict[str, dict[str, float]] = {}
        self._default = default_action
        self._margin = margin
        self._hint = backbone_hint

    def q_of(self, p: dict[str, float]) -> float:
        return p["success"] - self._lam * p["cost"] - self._eta * p["critical_error"]

    def prime(self, checkpoints: list[dict[str, Any]]) -> None:
        from sureact.baselines.scalar import checkpoint_key
        from sureact.value.dataset import render_state
        from sureact.value.model import predict

        keys, texts = [], []
        for cp in checkpoints:
            base = checkpoint_key(cp)
            for a in cp.get("legal_actions") or list(cp.get("per_action", {})):
                k = f"{base}|{a}"
                if k in self._cache:
                    continue
                keys.append(k)
                texts.append(render_state(cp, a, backbone_hint=self._hint))
        if not texts:
            return
        for k, p in zip(keys, predict(
            self._model, self._tok, texts, max_len=self._max_len, device=self._device
        )):
            self._cache[k] = p

    def choose(self, checkpoint: dict[str, Any]) -> str:
        from sureact.baselines.scalar import checkpoint_key

        base = checkpoint_key(checkpoint)
        legal = list(checkpoint.get("legal_actions") or checkpoint.get("per_action", {}))
        if not legal:
            return "ASK"
        qs = {}
        for a in legal:
            p = self._cache.get(f"{base}|{a}")
            if p is None:
                raise KeyError(
                    f"no prediction for {base}|{a}; call prime() on the eval set first"
                )
            qs[a] = self.q_of(p)
        best = max(qs, key=qs.__getitem__)
        if self._default is None or self._default not in qs:
            return best
        return best if qs[best] - qs[self._default] > self._margin else self._default
