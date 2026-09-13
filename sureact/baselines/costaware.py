from __future__ import annotations

from typing import Any

from .scalar import checkpoint_key

INFO_ACTIONS = ("ASK", "SEARCH")


class CostAwarePolicy:
    """Cost-aware threshold policy."""

    def __init__(
        self,
        name: str,
        scores: dict[str, float],
        h: float,
        lam: float,
        costs: dict[str, float],
        default: str,
        info_actions: tuple[str, ...],
    ) -> None:
        self.name = name
        self._scores = scores
        self._h = h
        self._lam = lam
        self._costs = costs
        self._default = default
        self._info = info_actions

    def choose(self, checkpoint: dict[str, Any]) -> str:
        key = checkpoint_key(checkpoint)
        if key not in self._scores:
            return self._default
        legal = [a for a in self._info if a in (checkpoint.get("per_action") or {})]
        if not legal:
            return self._default
        benefit = (1.0 - self._scores[key]) * self._h
        best, best_net = self._default, 0.0
        for a in legal:
            net = benefit - self._lam * self._costs.get(a, 0.0)
            if net > best_net:
                best, best_net = a, net
        return best


def fit_cost_aware_policy(
    name: str,
    checkpoints: list[dict[str, Any]],
    scores: dict[str, float],
    lam: float,
    eta: float,
    profile: Any,
    grid: int = 81,
) -> CostAwarePolicy:
    from sureact.scoring import q

    actions = sorted({a for cp in checkpoints for a in (cp.get("per_action") or {})})
    if not actions:
        return CostAwarePolicy(name, scores, 0.0, lam, {}, "ASK", ())

    from sureact.schema import MetaAction

    costs: dict[str, float] = {}
    for a in actions:
        try:
            costs[a] = float(profile.cost_of(MetaAction(a)))
        except (ValueError, KeyError, AttributeError):
            continue

    scored: list[tuple[float, dict[str, float], list[str]]] = []
    for cp in checkpoints:
        per = cp.get("per_action") or {}
        key = checkpoint_key(cp)
        if not per or key not in scores:
            continue
        qs = {a: q(o, lam, eta, profile) for a, o in per.items()}
        scored.append((scores[key], qs, [a for a in INFO_ACTIONS if a in per]))
    if not scored:
        return CostAwarePolicy(name, scores, 0.0, lam, costs, actions[0], INFO_ACTIONS)

    worst = min(min(qs.values()) for _, qs, _ in scored)
    default = max(
        actions,
        key=lambda a: sum(qs.get(a, worst) for _, qs, _ in scored) / len(scored),
    )

    def utility(h: float) -> float:
        tot = 0.0
        for s, qs, legal in scored:
            benefit = (1.0 - s) * h
            pick, best_net = default, 0.0
            for a in legal:
                net = benefit - lam * costs.get(a, 0.0)
                if net > best_net:
                    pick, best_net = a, net
            tot += qs.get(pick, worst)
        return tot / len(scored)

    span = max(max(qs.values()) - min(qs.values()) for _, qs, _ in scored)
    hi = max(2.0 * span, 2.0 * lam * (max(costs.values()) if costs else 1.0))
    best_h, best_u = 0.0, float("-inf")
    for i in range(grid):
        h = hi * i / (grid - 1)
        u = utility(h)
        if u > best_u:
            best_h, best_u = h, u
    return CostAwarePolicy(name, scores, best_h, lam, costs, default, INFO_ACTIONS)
