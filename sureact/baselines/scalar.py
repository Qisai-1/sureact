from __future__ import annotations

import itertools
from typing import Any


def checkpoint_key(cp: dict[str, Any]) -> str:
    env = cp.get("env") or "?"
    domain = cp.get("domain") or "?"
    fork = cp.get("fork_at")
    if fork is None:
        fork = cp.get("step_idx")
    src = str(cp.get("_src") or cp.get("_model") or "?").split("/")[-1]
    return f"{env}:{domain}:{src}:{cp.get('task_id')}:{fork}"


class ScalarPolicy:
    """Threshold rule over a confidence score."""

    def __init__(
        self,
        name: str,
        scores: dict[str, float],
        thresholds: list[float],
        mapping: list[str],
        default: str = "ASK",
    ) -> None:
        self.name = name
        self._scores = scores
        self._thresholds = thresholds
        self._mapping = mapping
        self._default = default

    def choose(self, checkpoint: dict[str, Any]) -> str:
        key = checkpoint_key(checkpoint)
        if key not in self._scores:
            return self._default
        band = sum(1 for t in self._thresholds if self._scores[key] >= t)
        want = self._mapping[band]
        legal = list(checkpoint.get("legal_actions") or checkpoint.get("per_action", {}))
        return want if (not legal or want in legal) else self._default


def fit_threshold_policy(
    name: str,
    checkpoints: list[dict[str, Any]],
    scores: dict[str, float],
    lam: float,
    eta: float,
    profile: Any,
    max_bands: int = 3,
    grid: int = 20,
) -> ScalarPolicy:
    from sureact.scoring import q

    actions = sorted({a for cp in checkpoints for a in (cp.get("per_action") or {})})
    if not actions:
        return ScalarPolicy(name, scores, [], ["ASK"])

    scored = []
    for cp in checkpoints:
        per = cp.get("per_action") or {}
        if not per:
            continue
        key = checkpoint_key(cp)
        if key not in scores:
            continue
        qs = {a: q(o, lam, eta, profile) for a, o in per.items()}
        scored.append((scores[key], qs))
    if not scored:
        return ScalarPolicy(name, scores, [], [actions[0]])

    worst = min(min(qs.values()) for _, qs in scored)

    def utility(thresholds: list[float], mapping: list[str]) -> float:
        tot = 0.0
        for s, qs in scored:
            band = sum(1 for t in thresholds if s >= t)
            tot += qs.get(mapping[band], worst)
        return tot / len(scored)

    cuts = [i / grid for i in range(1, grid)]
    best = (utility([], [actions[0]]), [], [actions[0]])
    for n_bands in range(1, max_bands + 1):
        thr_sets = [[]] if n_bands == 1 else list(itertools.combinations(cuts, n_bands - 1))
        for thresholds in thr_sets:
            for mapping in itertools.product(actions, repeat=n_bands):
                u = utility(list(thresholds), list(mapping))
                if u > best[0]:
                    best = (u, list(thresholds), list(mapping))
    return ScalarPolicy(name, scores, best[1], best[2], default=best[2][0])
