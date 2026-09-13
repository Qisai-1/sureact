from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sureact.schema import MetaAction

MANIFEST = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "tool_manifests" / "tau2.yaml"
)


@dataclass
class Tau2Taxonomy:
    """Map tau2 tools to meta-actions."""

    domain: str
    meta_of: dict[str, MetaAction]
    disputable: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, domain: str, path: Path | None = None) -> Tau2Taxonomy:
        raw = yaml.safe_load((path or MANIFEST).read_text())["domains"][domain]
        return cls(
            domain=domain,
            meta_of={k: MetaAction(v["meta"]) for k, v in raw.items()},
            disputable={k for k, v in raw.items() if v.get("disputable")},
        )

    def cross_check(self, env: Any) -> dict[str, list[str]]:
        names = [t.name for t in env.get_tools()]
        theirs = {n for n in names if env._is_mutating_tool(n)}
        ours = {
            n
            for n in names
            if self.meta_of.get(n) in (MetaAction.ACT_REVERSIBLE, MetaAction.COMMIT_IRREVERSIBLE)
        }
        return {
            "agree_mutating": sorted(ours & theirs),
            "ours_only": sorted(ours - theirs),
            "theirs_only": sorted(theirs - ours),
            "unmapped": sorted(set(names) - set(self.meta_of)),
        }

    def tools_for(self, action: MetaAction, names: list[str]) -> list[str]:
        return [n for n in names if self.meta_of.get(n) is action]

    def legal_actions(self, names: list[str]) -> list[MetaAction]:
        """Legal actions; ASK is always legal."""
        acts = [MetaAction.ASK]
        for a in (
            MetaAction.SEARCH,
            MetaAction.ACT_REVERSIBLE,
            MetaAction.COMMIT_IRREVERSIBLE,
            MetaAction.ABSTAIN_HANDOFF,
        ):
            if self.tools_for(a, names):
                acts.append(a)
        return acts


def make_forced_tau2_agent(base_cls: type, taxonomy: Tau2Taxonomy) -> type:

    class ForcedTau2Agent(base_cls):  # type: ignore[misc,valid-type]
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, **kw)
            self._forced: MetaAction | None = None
            self._spent = False

        def force(self, action: MetaAction) -> None:
            self._forced = action
            self._spent = False

        def generate_next_message(self, *a: Any, **kw: Any):  # type: ignore[no-untyped-def]
            if self._forced is None or self._spent:
                return super().generate_next_message(*a, **kw)

            self._spent = True
            original = self.tools
            if self._forced is MetaAction.ASK:
                self.tools = []
            else:
                keep = set(taxonomy.tools_for(self._forced, [t.name for t in original]))
                self.tools = [t for t in original if t.name in keep]
            prev_args = dict(getattr(self, "llm_args", {}) or {})
            if self._forced is not MetaAction.ASK and self.tools:
                self.llm_args = {**prev_args, "tool_choice": "required"}
            try:
                return super().generate_next_message(*a, **kw)
            finally:
                self.tools = original
                self.llm_args = prev_args

    return ForcedTau2Agent


def classify_trajectory(
    messages: list[Any], taxonomy: Tau2Taxonomy
) -> dict[MetaAction, int]:
    """Count meta-actions; messages to the user count as ASK."""
    counts: dict[MetaAction, int] = {}
    for m in messages:
        tcs = getattr(m, "tool_calls", None) or []
        if tcs:
            for tc in tcs:
                a = taxonomy.meta_of.get(getattr(tc, "name", ""))
                if a is not None:
                    counts[a] = counts.get(a, 0) + 1
        elif getattr(m, "role", None) == "assistant" and getattr(m, "content", None):
            counts[MetaAction.ASK] = counts.get(MetaAction.ASK, 0) + 1
    return counts


def critical_error_from_trajectory(
    messages: list[Any], task: Any, taxonomy: Tau2Taxonomy
) -> float | None:
    got = critical_errors_from_trajectory(messages, task, taxonomy)
    return got["effective"]


def critical_errors_from_trajectory(
    messages: list[Any], task: Any, taxonomy: Tau2Taxonomy
) -> dict[str, float | None]:
    crit = getattr(task, "evaluation_criteria", None)
    gold = list(getattr(crit, "actions", None) or [])
    if not gold:
        return {"attempted": None, "effective": None}

    errored: dict[str, bool] = {}
    for m in messages:
        if getattr(m, "role", None) == "tool":
            mid = getattr(m, "id", None)
            if mid is not None:
                errored[str(mid)] = bool(getattr(m, "error", False))

    attempted = effective = 0.0
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            name = getattr(tc, "name", "")
            if taxonomy.meta_of.get(name) is not MetaAction.COMMIT_IRREVERSIBLE:
                continue
            sanctioned = False
            for g in gold:
                try:
                    if g.compare_with_tool_call(tc):
                        sanctioned = True
                        break
                except Exception:  # noqa: BLE001 - a comparison failure is not a sanction
                    continue
            if sanctioned:
                continue
            attempted = 1.0
            if not errored.get(str(getattr(tc, "id", "")), False):
                effective = 1.0
    return {"attempted": attempted, "effective": effective}
