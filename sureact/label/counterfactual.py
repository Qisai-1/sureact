from __future__ import annotations

import copy
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sureact.schema import (
    ActionOutcome,
    Checkpoint,
    CostProfile,
    FreeArmOutcome,
    MetaAction,
    SourceVector,
    state_hash,
)

MANIFEST = Path(__file__).resolve().parent.parent.parent / "configs" / "tool_manifests"


@dataclass
class ToolTaxonomy:
    """Map ToolSandbox tools to meta-actions."""

    meta_of: dict[str, MetaAction]
    pure_computation: set[str] = field(default_factory=set)
    external_live_data: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path | None = None) -> ToolTaxonomy:
        raw = yaml.safe_load((path or (MANIFEST / "toolsandbox.yaml")).read_text())
        meta_of: dict[str, MetaAction] = {}
        pure: set[str] = set()
        live: set[str] = set()
        for name, spec in raw["tools"].items():
            meta_of[name] = MetaAction(spec["meta"])
            if spec.get("pure_computation"):
                pure.add(name)
            if spec.get("external_live_data"):
                live.add(name)
        return cls(meta_of=meta_of, pure_computation=pure, external_live_data=live)

    @staticmethod
    def _actual_name(name: str) -> str:
        try:
            from tool_sandbox.common.execution_context import get_current_context

            return get_current_context().get_execution_facing_tool_name(name)
        except Exception:  # noqa: BLE001 -- no context, or the name is already actual
            return name

    def meta_for(self, name: str) -> MetaAction | None:
        return self.meta_of.get(name) or self.meta_of.get(self._actual_name(name))

    def tools_for(self, action: MetaAction, available: dict[str, Any]) -> dict[str, Any]:
        return {
            n: t
            for n, t in available.items()
            if self.meta_for(n) is action
            and self._actual_name(n) not in self.external_live_data
        }

    def legal_actions(self, available: dict[str, Any]) -> list[MetaAction]:
        acts = [MetaAction.ASK]
        for a in (
            MetaAction.SEARCH,
            MetaAction.ACT_REVERSIBLE,
            MetaAction.COMMIT_IRREVERSIBLE,
            MetaAction.ABSTAIN_HANDOFF,
        ):
            if self.tools_for(a, available):
                acts.append(a)
        return acts


def make_forced_agent(base_cls: type, taxonomy: ToolTaxonomy) -> type:

    class ForcedAgent(base_cls):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.forced: MetaAction | None = None
            self._spent = False
            self._forcing_now: MetaAction | None = None

        def force(self, action: MetaAction) -> None:
            self.forced = action
            self._spent = False

        def model_inference(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            want = self._forcing_now
            self._forcing_now = None
            if want is not None and want is not MetaAction.ASK:
                self.openai_client._sureact_tool_choice = "required"
            try:
                return super().model_inference(*args, **kwargs)
            finally:
                self.openai_client._sureact_tool_choice = None

        def get_available_tools(self) -> dict[str, Any]:
            available = super().get_available_tools()
            if self.forced is None or self._spent:
                return available
            self._spent = True
            self._forcing_now = self.forced
            if self.forced is MetaAction.ASK:
                return {}
            return taxonomy.tools_for(self.forced, available)

    return ForcedAgent


EVAL_TIMEOUT_S = int(os.environ.get("SUREACT_EVAL_TIMEOUT", "120"))
STEP_TIMEOUT_S = int(os.environ.get("SUREACT_STEP_TIMEOUT", "120"))


def _with_alarm(seconds: int, fn: Any, *a: Any, **kw: Any) -> Any:
    import signal

    def _raise(signum: int, frame: Any) -> None:
        raise TimeoutError(f"step exceeded {seconds}s")

    prev = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        return fn(*a, **kw)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def _evaluate_with_timeout(scenario: Any, ctx: Any, seconds: int) -> Any:
    import signal

    def _raise(signum: int, frame: Any) -> None:
        raise TimeoutError(f"evaluate() exceeded {seconds}s")

    prev = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        return scenario.evaluation.evaluate(
            execution_context=ctx, max_turn_count=scenario.max_messages
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


@dataclass
class RolloutResult:
    success: float
    critical_error: float
    milestone: float
    minefield: float
    turns: int
    action_counts: dict[MetaAction, int]
    error: str | None = None
    first_action: MetaAction | None = None


def _agent_actions(ctx: Any, taxonomy: ToolTaxonomy, since_row: int) -> list[MetaAction]:
    from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType

    df = ctx.get_database(
        DatabaseNamespace.SANDBOX,
        drop_sandbox_message_index=False,
        get_all_history_snapshots=True,
    )
    out: list[MetaAction] = []
    for r in df.to_dicts()[since_row:]:
        if r.get("sender") != RoleType.AGENT:
            continue
        fn = r.get("openai_function_name")
        if fn:
            a = taxonomy.meta_for(fn)
            if a is not None:
                out.append(a)
        elif r.get("recipient") == RoleType.USER:
            out.append(MetaAction.ASK)
    return out


def _visible_history(ctx: Any, max_turns: int = 30) -> list[dict[str, Any]]:
    from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType

    df = ctx.get_database(
        DatabaseNamespace.SANDBOX,
        drop_sandbox_message_index=False,
        get_all_history_snapshots=True,
    )
    out: list[dict[str, Any]] = []
    for r in df.to_dicts():
        s, rc = r.get("sender"), r.get("recipient")
        if s == RoleType.SYSTEM and rc == RoleType.USER:
            continue
        if RoleType.USER in (s, rc) and RoleType.EXECUTION_ENVIRONMENT in (s, rc):
            continue
        content = r.get("content")
        if content is None:
            continue
        out.append(
            {
                "sender": str(s),
                "recipient": str(rc),
                "content": str(content)[:600],
                "tool": r.get("openai_function_name"),
            }
        )
    return out[-max_turns:]


def rollout_forced(
    scenario: Any,
    snapshot: Any,
    action: MetaAction | None,
    make_roles: Callable[[], dict],
    taxonomy: ToolTaxonomy,
    base_index: int,
    continue_for: int,
    since_row: int,
) -> RolloutResult:
    from tool_sandbox.common.execution_context import (
        DatabaseNamespace,
        RoleType,
        get_current_context,
        set_current_context,
    )

    set_current_context(copy.deepcopy(snapshot))
    roles = make_roles()
    if action is not None:
        roles[RoleType.AGENT].force(action)

    try:
        for _ in range(continue_for):
            db = get_current_context().get_database(
                DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False
            )
            if not db["conversation_active"][-1]:
                break
            if db["sandbox_message_index"][-1] >= scenario.max_messages + base_index:
                break
            _with_alarm(STEP_TIMEOUT_S, roles[db["recipient"][-1]].respond)
    except Exception as exc:  # noqa: BLE001
        return RolloutResult(0.0, 0.0, 0.0, 0.0, 0, {}, error=f"{type(exc).__name__}: {exc}")

    ctx = get_current_context()
    try:
        ev = _evaluate_with_timeout(scenario, ctx, EVAL_TIMEOUT_S)
    except TimeoutError:
        return RolloutResult(0.0, 0.0, 0.0, 0.0, 0, {}, error="evaluation_timeout")
    minefield = float(ev.minefield_similarity)
    milestone = float(ev.milestone_similarity)
    success = milestone
    df_len = len(
        ctx.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        )
    )
    taken = _agent_actions(ctx, taxonomy, since_row)
    counts: dict[MetaAction, int] = {}
    for a in taken:
        counts[a] = counts.get(a, 0) + 1
    return RolloutResult(
        success=success,
        critical_error=float(minefield != 0),
        milestone=milestone,
        minefield=minefield,
        turns=df_len - since_row,
        action_counts=counts,
        first_action=taken[0] if taken else None,
    )


def _aggregate(
    runs: list[RolloutResult], action: MetaAction, profile: CostProfile
) -> ActionOutcome:
    n = len(runs)
    counts: dict[str, float] = {}
    for r in runs:
        for a, c in r.action_counts.items():
            counts[a.value] = counts.get(a.value, 0.0) + c / n
    return ActionOutcome(
        action=action,
        success=sum(r.success for r in runs) / n,
        critical_error=sum(r.critical_error for r in runs) / n,
        cost=sum(
            sum(profile.cost_of(a) * c for a, c in r.action_counts.items()) for r in runs
        ) / n,
        steps=sum(r.turns for r in runs) / n,
        delta_u=SourceVector(),
        n_seeds=n,
        terminal_state_ok=sum(r.milestone for r in runs) / n,
        action_counts=counts,
    )


def _label_free_arm(
    scenario: Any,
    snapshot: Any,
    make_roles: Callable[[], dict],
    taxonomy: ToolTaxonomy,
    profile: CostProfile,
    base_index: int,
    continue_for: int,
    since_row: int,
    n_free: int,
    dropped: dict[str, str],
) -> FreeArmOutcome | None:
    if n_free <= 0:
        return None
    runs = [
        rollout_forced(
            scenario, snapshot, None, make_roles, taxonomy,
            base_index, continue_for, since_row,
        )
        for _ in range(n_free)
    ]
    ok = [r for r in runs if r.error is None]
    classified = [r for r in ok if r.first_action is not None]
    if not classified:
        dropped["__free__"] = runs[0].error or "no agent action after the fork"
        return None

    tally: dict[str, float] = {}
    for r in classified:
        k = r.first_action.value  # type: ignore[union-attr]
        tally[k] = tally.get(k, 0.0) + 1.0 / len(classified)
    modal = MetaAction(max(tally, key=tally.__getitem__))
    return FreeArmOutcome(
        outcome=_aggregate(ok, modal, profile),
        first_action_counts=tally,
        n_failed=len(runs) - len(ok),
        n_unclassified=len(ok) - len(classified),
    )


def label_checkpoint(
    scenario: Any,
    scenario_name: str,
    snapshot: Any,
    make_roles: Callable[[], dict],
    taxonomy: ToolTaxonomy,
    profile: CostProfile,
    base_index: int,
    step_idx: int,
    continue_for: int = 8,
    n_seeds: int = 1,
    u_true: SourceVector | None = None,
    n_free: int = 0,
) -> Checkpoint:
    from tool_sandbox.common.execution_context import (
        DatabaseNamespace,
        RoleType,
        get_current_context,
        set_current_context,
    )

    prev = get_current_context()
    set_current_context(snapshot)
    try:
        available = make_roles()[RoleType.AGENT].get_available_tools()
    finally:
        set_current_context(prev)

    legal = taxonomy.legal_actions(available)
    since_row = len(
        snapshot.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        )
    )

    per_action: dict[str, ActionOutcome] = {}
    dropped: dict[str, str] = {}
    for action in legal:
        runs = [
            rollout_forced(
                scenario, snapshot, action, make_roles, taxonomy,
                base_index, continue_for, since_row,
            )
            for _ in range(n_seeds)
        ]
        ok = [r for r in runs if r.error is None]
        if not ok:
            dropped[action.value] = runs[0].error or "unknown"
            continue
        per_action[action.value] = _aggregate(ok, action, profile)

    free_arm = _label_free_arm(
        scenario, snapshot, make_roles, taxonomy, profile, base_index, continue_for,
        since_row, n_free, dropped,
    )

    cp = Checkpoint(
        task_id=scenario_name,
        variant_id=f"{scenario_name}@{step_idx}",
        base_task_id=scenario_name,
        step_idx=step_idx,
        domain="toolsandbox",
        env="toolsandbox",
        history=_visible_history(snapshot),
        legal_actions=[MetaAction(a) for a in per_action],
        u_true=u_true if u_true is not None else SourceVector(),
        per_action=per_action,
        env_state_hash=state_hash(str(since_row)),
        scenario_template=scenario_name,
        tool_graph_signature=",".join(sorted(available)),
        is_native=u_true is None,
        free_arm=free_arm,
    )
    cp.dropped_actions = dropped
    if dropped:
        print(f"    dropped arms at {scenario_name}#{step_idx}: {dropped}")
    return cp
