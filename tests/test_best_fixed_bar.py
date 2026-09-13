from types import SimpleNamespace

import pytest


from sureact.value.metrics import decision_metrics  # noqa: E402
from sureact.baselines import AlwaysAction, evaluate  # noqa: E402
from sureact.schema import load_cost_profiles  # noqa: E402

LAM = ETA = 1.0


def q_of(p):
    return p["success"] - LAM * p["cost"] - ETA * (p.get("critical_error") or 0.0)


ARMS = {
    "cp1": {"SEARCH": (0.90, 0.0), "ASK": (0.50, 0.0)},
    "cp2": {"ASK": (0.40, 0.0), "ACT_REVERSIBLE": (0.10, 0.0)},
    "cp3": {"SEARCH": (0.20, 0.0), "ASK": (0.00, 1.0)},
}


def _groups():
    return {
        k: [SimpleNamespace(action=a, success=s, cost=0.0, critical_error=c, text="")
            for a, (s, c) in arms.items()]
        for k, arms in ARMS.items()
    }


def _checkpoints():
    return [
        {"task_id": k, "per_action": {a: {"action": a, "success": s, "cost": 0.0,
                                          "critical_error": c, "action_counts": {}}
                                      for a, (s, c) in arms.items()},
         "legal_actions": sorted(arms)}
        for k, arms in ARMS.items()
    ]


def _preds(groups):
    return [{"success": e.success, "cost": e.cost, "critical_error": e.critical_error}
            for g in groups.values() for e in g]


def test_bar_charges_the_checkpoints_own_worst_arm():
    groups = _groups()
    m = decision_metrics(groups, _preds(groups), q_of)
    assert m["best_fixed"] == pytest.approx(0.40, abs=1e-9)
    assert m["best_fixed"] > 0.0333 + 1e-6


def test_the_trainer_path_and_the_baseline_harness_agree_on_the_bar():
    """Trainer and baseline harness use the same bar."""
    groups = _groups()
    trainer_bar = decision_metrics(groups, _preds(groups), q_of)["best_fixed"]
    profile = load_cost_profiles()["medium"]
    cps = _checkpoints()
    harness_bar = max(
        evaluate(AlwaysAction(a), cps, LAM, ETA, profile).utility
        for a in sorted({a for c in cps for a in c["per_action"]})
    )
    assert trainer_bar == pytest.approx(harness_bar, abs=1e-9)
