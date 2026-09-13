from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class Controller(Protocol):
    """A policy that picks a meta-action."""

    name: str

    def choose(self, checkpoint: dict[str, Any]) -> str:
        """Return a meta-action without reading `per_action`."""
        ...


@dataclass
class Evaluation:

    name: str
    n: int
    utility: float
    success: float
    cost: float
    critical_error: float | None
    unlabelled: int = 0
    n_risk_judgeable: int = 0
    chosen: dict[str, int] = field(default_factory=dict)

    def row(self) -> str:
        crit = "  n/a" if self.critical_error is None else f"{self.critical_error:5.3f}"
        flag = f"  [{self.unlabelled} off-menu]" if self.unlabelled else ""
        return (
            f"  {self.name:26} U={self.utility:+.3f}  succ={self.success:.3f}"
            f"  cost={self.cost:.3f}  crit={crit}{flag}"
        )


def evaluate(
    controller: Controller,
    checkpoints: list[dict[str, Any]],
    lam: float,
    eta: float,
    profile: Any,
) -> Evaluation:
    """Score a controller against labelled checkpoints."""
    from sureact.scoring import q, risk_judgeable

    n = us = ss = cs = 0.0
    crit_sum = 0.0
    n_crit = 0
    unlabelled = 0
    chosen: dict[str, int] = {}

    for cp in checkpoints:
        per = cp.get("per_action") or {}
        if not per:
            continue
        action = controller.choose(cp)
        chosen[action] = chosen.get(action, 0) + 1
        outcome = per.get(action)
        if outcome is None:
            unlabelled += 1
            outcome = min(per.values(), key=lambda o: q(o, lam, eta, profile))
        n += 1
        us += q(outcome, lam, eta, profile)
        ss += outcome.get("success") or 0.0
        from sureact.schema import MetaAction

        counts = outcome.get("action_counts") or {}
        cs += (
            sum(profile.cost_of(MetaAction(a)) * c for a, c in counts.items())
            if counts
            else (outcome.get("cost") or 0.0)
        )
        if risk_judgeable(outcome):
            n_crit += 1
            crit_sum += outcome["critical_error"]

    k = n or 1
    return Evaluation(
        name=controller.name,
        n=int(n),
        utility=us / k,
        success=ss / k,
        cost=cs / k,
        critical_error=(crit_sum / n_crit) if n_crit else None,
        unlabelled=unlabelled,
        n_risk_judgeable=n_crit,
        chosen=chosen,
    )
