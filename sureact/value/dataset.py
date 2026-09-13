from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sureact.schema import MetaAction

_FAMILY_SUFFIXES = re.compile(
    r"(_alt|_implicit|_multiple_user_turn|_insufficient_information"
    r"|_[0-9]+_distraction_tools|_tool_name_scrambled|_tool_description_scrambled"
    r"|_arg_type_scrambled|_arg_description_scrambled)+$"
)


def family_of(task_id: str) -> str:
    """Family key used for splits."""
    base = str(task_id).split("/")[0]
    base = base.split("#")[0]
    prev = None
    while prev != base:
        prev = base
        base = _FAMILY_SUFFIXES.sub("", base)
    return base


@dataclass
class ValueExample:
    """One (state, action) pair with its measured outcome."""

    task_id: str
    cp_key: str
    family: str
    action: str
    text: str
    success: float
    critical_error: float | None
    cost: float
    env: str

    def targets(self) -> list[float]:
        return [self.success, self.critical_error or 0.0, self.cost]

    def target_mask(self) -> list[float]:
        return [1.0, 1.0 if self.critical_error is not None else 0.0, 1.0]


def render_state(
    checkpoint: dict[str, Any],
    action: str,
    max_chars: int = 3000,
    backbone_hint: bool = False,
) -> str:
    parts = [f"[env] {checkpoint.get('env') or checkpoint.get('domain', '?')}"]
    legal = checkpoint.get("legal_actions") or list(checkpoint.get("per_action", {}))
    parts.append(f"[legal] {', '.join(str(a) for a in legal)}")
    hist = checkpoint.get("history") or []
    if hist:
        parts.append("[history]")
        for m in hist[-12:]:
            role = m.get("sender") or m.get("role") or "?"
            content = str(m.get("content") or "")[:300].replace("\n", " ")
            parts.append(f"  {role}: {content}")
    if backbone_hint:
        fa = (checkpoint.get("free_arm") or {}).get("outcome") or {}
        parts.append(f"[backbone would] {fa.get('action') or 'unknown'}")
    parts.append(f"[candidate action] {action}")
    text = "\n".join(parts)
    return text[-max_chars:]


def load_examples(
    paths: Iterable[Path], cost_profile: str = "medium", backbone_hint: bool = False
) -> list[ValueExample]:
    from sureact.baselines.scalar import checkpoint_key
    from sureact.schema import load_cost_profiles

    profile = load_cost_profiles()[cost_profile]

    def _cost(o: dict[str, Any]) -> float:
        counts = o.get("action_counts") or {}
        if not counts:
            return float(o.get("cost", 0.0))
        return sum(profile.cost_of(MetaAction(a)) * c for a, c in counts.items())

    out: list[ValueExample] = []
    for p in paths:
        blob = json.loads(Path(p).read_text())
        env = blob.get("env") or blob.get("kind") or Path(p).stem
        src = Path(p).stem
        for cp in blob.get("checkpoints", []):
            cp = dict(cp, _src=src)
            per = cp.get("per_action") or {}
            scored = {
                a: o for a, o in per.items()
                if a in MetaAction.__members__ and not o.get("unscorable_premature")
            }
            if len(scored) < 2:
                continue
            cp = dict(cp, legal_actions=sorted(scored))
            for action, o in scored.items():
                out.append(
                    ValueExample(
                        task_id=str(cp.get("task_id")),
                        cp_key=checkpoint_key(cp),
                        family=family_of(cp.get("task_id", "")),
                        action=action,
                        text=render_state(cp, action, backbone_hint=backbone_hint),
                        success=float(o.get("success", 0.0)),
                        critical_error=(
                            None if o.get("critical_error") is None
                            else float(o["critical_error"])
                        ),
                        cost=_cost(o),
                        env=str(env),
                    )
                )
    return out


def split_three_way(
    examples: list[ValueExample],
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    seed: int = 0,
) -> tuple[list[ValueExample], list[ValueExample], list[ValueExample]]:
    import random

    fams = sorted({e.family for e in examples})
    rng = random.Random(seed)
    rng.shuffle(fams)
    n_val = max(1, int(len(fams) * val_frac))
    n_test = max(1, int(len(fams) * test_frac))
    val_fams = set(fams[:n_val])
    test_fams = set(fams[n_val : n_val + n_test])
    train = [e for e in examples if e.family not in val_fams and e.family not in test_fams]
    val = [e for e in examples if e.family in val_fams]
    test = [e for e in examples if e.family in test_fams]
    return train, val, test


def split_by_family(
    examples: list[ValueExample], val_frac: float = 0.25, seed: int = 0
) -> tuple[list[ValueExample], list[ValueExample]]:
    import random

    fams = sorted({e.family for e in examples})
    rng = random.Random(seed)
    rng.shuffle(fams)
    n_val = max(1, int(len(fams) * val_frac))
    val_fams = set(fams[:n_val])
    train = [e for e in examples if e.family not in val_fams]
    val = [e for e in examples if e.family in val_fams]
    return train, val
