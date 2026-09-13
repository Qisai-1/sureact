from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


class MetaAction(str, Enum):
    """Meta-actions available to the agent."""

    ASK = "ASK"
    SEARCH = "SEARCH"
    VERIFY = "VERIFY"
    ACT_REVERSIBLE = "ACT_REVERSIBLE"
    COMMIT_IRREVERSIBLE = "COMMIT_IRREVERSIBLE"
    ABSTAIN_HANDOFF = "ABSTAIN_HANDOFF"
    ABSTAIN_REFUSE = "ABSTAIN_REFUSE"

    @property
    def is_abstain(self) -> bool:
        return self in (MetaAction.ABSTAIN_HANDOFF, MetaAction.ABSTAIN_REFUSE)

    @property
    def is_read_only(self) -> bool:
        return self in (MetaAction.SEARCH, MetaAction.VERIFY)

    @property
    def is_mutating(self) -> bool:
        return self in (MetaAction.ACT_REVERSIBLE, MetaAction.COMMIT_IRREVERSIBLE)


ALL_ACTIONS: tuple[MetaAction, ...] = tuple(MetaAction)


class UncertaintySource(str, Enum):
    """Sources of uncertainty."""

    GOAL = "goal"
    OBSERVATION = "observation"
    ACTION = "action"
    OUTCOME = "outcome"


SOURCE_ORDER: tuple[UncertaintySource, ...] = (
    UncertaintySource.GOAL,
    UncertaintySource.OBSERVATION,
    UncertaintySource.ACTION,
    UncertaintySource.OUTCOME,
)


@dataclass(frozen=True)
class SourceVector:
    """Per-source uncertainty in [0, 1]."""

    goal: float = 0.0
    observation: float = 0.0
    action: float = 0.0
    outcome: float = 0.0

    def __post_init__(self) -> None:
        for name in ("goal", "observation", "action", "outcome"):
            v = getattr(self, name)
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"SourceVector.{name}={v} outside [0,1]")

    def as_list(self) -> list[float]:
        return [self.goal, self.observation, self.action, self.outcome]

    @classmethod
    def from_list(cls, xs: list[float]) -> SourceVector:
        if len(xs) != 4:
            raise ValueError(f"expected 4 components in SOURCE_ORDER, got {len(xs)}")
        return cls(*[float(x) for x in xs])

    def dominant(self) -> UncertaintySource:
        return SOURCE_ORDER[max(range(4), key=lambda i: self.as_list()[i])]


@dataclass
class ActionOutcome:

    action: MetaAction
    success: float
    critical_error: float | None
    cost: float
    steps: float
    delta_u: SourceVector
    n_seeds: int = 1
    terminal_state_ok: float = 0.0
    action_counts: dict[str, float] = field(default_factory=dict)

    def cost_under(self, profile: CostProfile) -> float:
        if not self.action_counts:
            return self.cost
        return sum(profile.cost_of(MetaAction(a)) * c for a, c in self.action_counts.items())

    def risk_judgeable(self) -> bool:
        return self.critical_error is not None

    def q(self, lam: float, eta: float, profile: CostProfile | None = None) -> float:
        if profile is None and self.action_counts and not self.cost:
            raise ValueError(
                "q() needs a cost profile when cost comes from action_counts"
            )
        cost = self.cost if profile is None else self.cost_under(profile)
        return self.success - lam * cost - eta * (self.critical_error or 0.0)


@dataclass
class FreeArmOutcome:

    outcome: ActionOutcome
    first_action_counts: dict[str, float] = field(default_factory=dict)
    n_failed: int = 0
    n_unclassified: int = 0

    @property
    def action(self) -> MetaAction:
        """The backbone's own choice at this checkpoint."""
        return self.outcome.action


@dataclass
class Checkpoint:
    """One decision point."""

    task_id: str
    variant_id: str
    base_task_id: str
    step_idx: int
    domain: str
    env: str
    history: list[dict[str, Any]]
    legal_actions: list[MetaAction]
    u_true: SourceVector
    per_action: dict[str, ActionOutcome] = field(default_factory=dict)
    env_state_hash: str = ""
    schema_version: str = SCHEMA_VERSION
    scenario_template: str = ""
    tool_graph_signature: str = ""
    perturbation_seed: int = 0
    is_native: bool = False
    dropped_actions: dict[str, str] = field(default_factory=dict)
    free_arm: FreeArmOutcome | None = None

    def valid_set(self, lam: float, eta: float, eps: float = 0.05) -> list[MetaAction]:
        """Actions within epsilon of the best."""
        if not self.per_action:
            return []
        qs = {a: o.q(lam, eta) for a, o in self.per_action.items()}
        best = max(qs.values())
        return [MetaAction(a) for a, q in qs.items() if q >= best - eps]

    def a_star(self, lam: float, eta: float) -> MetaAction:
        qs = {a: o.q(lam, eta) for a, o in self.per_action.items()}
        return MetaAction(max(qs, key=qs.__getitem__))

    def regret(self, chosen: MetaAction, lam: float, eta: float) -> float:
        """Regret of the chosen action."""
        qs = {a: o.q(lam, eta) for a, o in self.per_action.items()}
        if chosen.value not in qs:
            raise KeyError(
                f"no counterfactual rollout for {chosen.value} at {self.task_id}#{self.step_idx}"
            )
        return max(qs.values()) - qs[chosen.value]

    def free_regret(
        self, lam: float, eta: float, profile: CostProfile | None = None
    ) -> float | None:
        if self.free_arm is None or not self.per_action:
            return None
        qs = [o.q(lam, eta, profile) for o in self.per_action.values()]
        return max(qs) - self.free_arm.outcome.q(lam, eta, profile)

    def split_key(self) -> str:
        return "|".join(
            [self.env, self.domain, self.scenario_template, self.tool_graph_signature]
        )

    def observation(self) -> dict[str, Any]:
        """The observation a policy may see."""
        return {
            "history": self.history,
            "legal_actions": [a.value for a in self.legal_actions],
            "domain": self.domain,
            "step_idx": self.step_idx,
        }

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["legal_actions"] = [a.value for a in self.legal_actions]
        d["u_true"] = self.u_true.as_list()
        d["per_action"] = {
            k: {**asdict(v), "action": v.action.value, "delta_u": v.delta_u.as_list()}
            for k, v in self.per_action.items()
        }
        if self.free_arm is not None:
            fa = self.free_arm
            d["free_arm"] = {
                **asdict(fa),
                "outcome": {
                    **asdict(fa.outcome),
                    "action": fa.outcome.action.value,
                    "delta_u": fa.outcome.delta_u.as_list(),
                },
            }
        return d


def state_hash(state: Any) -> str:
    """Hash of hidden simulator state."""
    blob = json.dumps(state, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@dataclass(frozen=True)
class CostProfile:
    name: str
    costs: dict[MetaAction, float]
    r_success: float = 1.0

    def cost_of(self, a: MetaAction) -> float:
        return self.costs[a]


def load_cost_profiles(path: Path | None = None) -> dict[str, CostProfile]:
    import yaml

    path = path or (_CONFIG_DIR / "cost_profiles.yaml")
    raw = yaml.safe_load(path.read_text())
    out: dict[str, CostProfile] = {}
    for name, spec in raw["profiles"].items():
        costs = {MetaAction(k): float(v) for k, v in spec["costs"].items()}
        missing = set(ALL_ACTIONS) - set(costs)
        if missing:
            raise ValueError(f"cost profile {name!r} missing {sorted(m.value for m in missing)}")
        out[name] = CostProfile(name=name, costs=costs, r_success=float(spec.get("r_success", 1.0)))
    return out


RISK_WEIGHTS: dict[str, float] = {"normal": 1.0, "risk_averse": 5.0}
