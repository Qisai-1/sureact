from __future__ import annotations


import pytest


from sureact.baselines.learned import ValueController  # noqa: E402
from sureact.baselines.scalar import checkpoint_key  # noqa: E402
from sureact.schema import load_cost_profiles  # noqa: E402

PROFILE = load_cost_profiles()["medium"]
CP = {
    "env": "tau2",
    "domain": "retail",
    "task_id": 7,
    "fork_at": 4,
    "legal_actions": ["ASK", "SEARCH", "COMMIT_IRREVERSIBLE"],
    "per_action": {"ASK": {}, "SEARCH": {}, "COMMIT_IRREVERSIBLE": {}},
}


def _controller(preds: dict[str, dict[str, float]], lam: float = 1.0, eta: float = 1.0):
    c = ValueController(model=None, tok=None, lam=lam, eta=eta, profile=PROFILE)
    base = checkpoint_key(CP)
    c._cache = {f"{base}|{a}": p for a, p in preds.items()}
    return c


def test_picks_the_highest_predicted_utility():
    c = _controller({
        "ASK": {"success": 0.9, "critical_error": 0.0, "cost": 0.05},
        "SEARCH": {"success": 0.4, "critical_error": 0.0, "cost": 0.02},
        "COMMIT_IRREVERSIBLE": {"success": 0.3, "critical_error": 0.6, "cost": 0.02},
    })
    assert c.choose(CP) == "ASK"


def test_eta_participates_in_the_choice():
    preds = {
        "ASK": {"success": 0.60, "critical_error": 0.00, "cost": 0.05},
        "SEARCH": {"success": 0.10, "critical_error": 0.00, "cost": 0.02},
        "COMMIT_IRREVERSIBLE": {"success": 0.98, "critical_error": 0.50, "cost": 0.02},
    }
    assert _controller(preds, eta=0.1).choose(CP) == "COMMIT_IRREVERSIBLE"
    assert _controller(preds, eta=5.0).choose(CP) == "ASK"


def test_lambda_participates_in_the_choice():
    preds = {
        "ASK": {"success": 0.80, "critical_error": 0.0, "cost": 0.90},
        "SEARCH": {"success": 0.55, "critical_error": 0.0, "cost": 0.01},
        "COMMIT_IRREVERSIBLE": {"success": 0.10, "critical_error": 0.0, "cost": 0.01},
    }
    assert _controller(preds, lam=0.0).choose(CP) == "ASK"
    assert _controller(preds, lam=1.0).choose(CP) == "SEARCH"


def test_never_chooses_outside_the_menu():
    c = _controller({
        "ASK": {"success": 0.1, "critical_error": 0.0, "cost": 0.0},
        "SEARCH": {"success": 0.2, "critical_error": 0.0, "cost": 0.0},
        "COMMIT_IRREVERSIBLE": {"success": 0.3, "critical_error": 0.0, "cost": 0.0},
    })
    cp = dict(CP, legal_actions=["ASK", "SEARCH"])
    assert c.choose(cp) in {"ASK", "SEARCH"}


def test_missing_prediction_raises_rather_than_skipping():
    c = _controller({"ASK": {"success": 0.9, "critical_error": 0.0, "cost": 0.0}})
    with pytest.raises(KeyError, match="prime"):
        c.choose(CP)


def test_fork_depths_are_separate_states():
    c = _controller({
        "ASK": {"success": 0.9, "critical_error": 0.0, "cost": 0.0},
        "SEARCH": {"success": 0.1, "critical_error": 0.0, "cost": 0.0},
        "COMMIT_IRREVERSIBLE": {"success": 0.1, "critical_error": 0.0, "cost": 0.0},
    })
    deeper = dict(CP, fork_at=8)
    with pytest.raises(KeyError):
        c.choose(deeper)
