from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Load a label file."""
    blob = json.loads(path.read_text())
    env = blob.get("env") or blob.get("kind") or path.stem
    out = []
    for c in blob.get("checkpoints", []):
        pa = c.get("per_action") or {}
        if not pa:
            continue
        out.append(
            {
                "task_id": c.get("task_id"),
                "step_idx": c.get("step_idx", 0),
                "domain": c.get("domain", env),
                "per_action": {
                    a: {
                        "success": float(o.get("success", 0.0)),
                        "critical_error": (
                            None if o.get("critical_error") is None
                            else float(o["critical_error"])
                        ),
                        "action_counts": o.get("action_counts") or {},
                        "cost": float(o.get("cost", 0.0)),
                    }
                    for a, o in pa.items()
                },
                "dropped_actions": c.get("dropped_actions") or {},
                "free_arm": _free(c.get("free_arm")),
            }
        )
    return env, out


def _free(fa: dict[str, Any] | None) -> dict[str, Any] | None:
    if not fa:
        return None
    o = fa.get("outcome") or {}
    return {
        "action": o.get("action"),
        "success": float(o.get("success", 0.0)),
        "critical_error": (
            None if o.get("critical_error") is None
            else float(o["critical_error"])
        ),
        "action_counts": o.get("action_counts") or {},
        "cost": float(o.get("cost", 0.0)),
        "first_action_counts": fa.get("first_action_counts") or {},
        "n_failed": int(fa.get("n_failed", 0)),
        "n_unclassified": int(fa.get("n_unclassified", 0)),
    }


def q(outcome: dict[str, Any], lam: float, eta: float, profile: Any) -> float:
    from sureact.schema import MetaAction

    counts = outcome["action_counts"]
    cost = (
        sum(profile.cost_of(MetaAction(a)) * c for a, c in counts.items())
        if counts
        else outcome["cost"]
    )
    crit = outcome.get("critical_error")
    return outcome["success"] - lam * cost - eta * (crit or 0.0)


def risk_judgeable(outcome: dict[str, Any]) -> bool:
    return outcome.get("critical_error") is not None
