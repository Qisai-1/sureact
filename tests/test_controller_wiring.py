from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "labels"

from sureact.baselines import evaluate  # noqa: E402
from sureact.baselines.learned import ValueController  # noqa: E402
from sureact.baselines.scalar import checkpoint_key  # noqa: E402
from sureact.schema import load_cost_profiles  # noqa: E402

PROFILE = load_cost_profiles()["medium"]


def _load(pattern: str) -> list[dict]:
    cps = []
    for p in map(Path, glob.glob(str(DATA / pattern))):
        blob = json.loads(p.read_text())
        for cp in blob.get("checkpoints", []):
            cp.setdefault("env", blob.get("env"))
            cp["per_action"] = {
                a: o for a, o in (cp.get("per_action") or {}).items()
                if not o.get("unscorable_premature")
            }
            if len(cp["per_action"]) >= 2:
                cp["legal_actions"] = sorted(cp["per_action"])
                cps.append(cp)
    return cps


@pytest.fixture(scope="module")
def checkpoints():
    cps = _load("labels_tau2_retail_v6.json")
    if not cps:
        pytest.skip("retail v6 labels not present")
    return cps[:30]


def _stub(controller, cps, fn):
    """Fill the prediction cache."""
    for cp in cps:
        base = checkpoint_key(cp)
        for a in cp["legal_actions"]:
            controller._cache[f"{base}|{a}"] = fn(cp, a)


def test_a_perfect_model_reaches_the_oracle(checkpoints):
    from sureact.scoring import q

    from sureact.schema import MetaAction

    def true_cost(o):
        counts = o.get("action_counts") or {}
        if not counts:
            return o.get("cost") or 0.0
        return sum(PROFILE.cost_of(MetaAction(a)) * c for a, c in counts.items())

    c = ValueController(None, None, lam=1.0, eta=1.0, profile=PROFILE)
    _stub(c, checkpoints, lambda cp, a: {
        "success": cp["per_action"][a].get("success") or 0.0,
        "critical_error": cp["per_action"][a].get("critical_error") or 0.0,
        "cost": true_cost(cp["per_action"][a]),
    })
    got = evaluate(c, checkpoints, 1.0, 1.0, PROFILE)
    oracle = sum(
        max(q(o, 1.0, 1.0, PROFILE) for o in cp["per_action"].values()) for cp in checkpoints
    ) / len(checkpoints)
    assert got.utility == pytest.approx(oracle, abs=1e-9)
    assert got.n == len(checkpoints)
    assert got.unlabelled == 0


def test_a_constant_model_degenerates_to_a_fixed_policy(checkpoints):
    from sureact.baselines import AlwaysAction

    c = ValueController(None, None, lam=1.0, eta=1.0, profile=PROFILE)
    _stub(c, checkpoints, lambda cp, a: {
        "success": 1.0 if a == "ABSTAIN_HANDOFF" else 0.0,
        "critical_error": 0.0,
        "cost": 0.0,
    })
    got = evaluate(c, checkpoints, 1.0, 1.0, PROFILE)
    fixed = evaluate(AlwaysAction("ABSTAIN_HANDOFF"), checkpoints, 1.0, 1.0, PROFILE)
    assert got.utility == pytest.approx(fixed.utility, abs=1e-9)


def test_an_inverted_model_lands_below_every_fixed_policy(checkpoints):
    from sureact.baselines import AlwaysAction

    c = ValueController(None, None, lam=1.0, eta=1.0, profile=PROFILE)
    _stub(c, checkpoints, lambda cp, a: {
        "success": -(cp["per_action"][a].get("success") or 0.0),
        "critical_error": 0.0,
        "cost": 0.0,
    })
    got = evaluate(c, checkpoints, 1.0, 1.0, PROFILE)
    best_fixed = max(
        evaluate(AlwaysAction(a), checkpoints, 1.0, 1.0, PROFILE).utility
        for a in sorted({a for cp in checkpoints for a in cp["per_action"]})
    )
    assert got.utility < best_fixed


def test_priming_covers_every_action_the_evaluation_will_ask_for(checkpoints):
    c = ValueController(None, None, lam=1.0, eta=1.0, profile=PROFILE)
    _stub(c, checkpoints, lambda cp, a: {"success": 0.5, "critical_error": 0.0, "cost": 0.0})
    for cp in checkpoints:
        c.choose(cp)


def test_training_examples_exclude_premature_arms_and_match_the_scored_menu():
    import glob

    from sureact.value.dataset import load_examples

    paths = [Path(p) for p in glob.glob(str(DATA / "labels_tau2_airline_v6.json"))]
    if not paths:
        pytest.skip("airline v6 labels not present")
    ex = load_examples(paths)
    assert ex

    raw = json.loads(paths[0].read_text())
    scored = {}
    for cp in raw["checkpoints"]:
        key = (str(cp.get("task_id")), cp.get("fork_at"))
        scored[key] = {
            a for a, o in (cp.get("per_action") or {}).items()
            if not o.get("unscorable_premature")
        }

    for e in ex:
        assert any(e.action in v for v in scored.values())
        menu = e.text.split("[legal] ", 1)[1].split("\n", 1)[0]
        listed = {t.strip() for t in menu.split(",") if t.strip()}
        assert e.action in listed
        assert len(listed) >= 2


def test_baselines_and_the_value_model_see_the_same_state():
    import glob

    from sureact.value.dataset import load_examples, render_state

    paths = [Path(p) for p in glob.glob(str(DATA / "labels_tau2_airline_v6.json"))]
    if not paths:
        pytest.skip("airline v6 labels not present")
    blob = json.loads(paths[0].read_text())
    ex = {e.task_id: e for e in load_examples(paths)}

    checked = 0
    for cp in blob["checkpoints"]:
        cp.setdefault("env", blob.get("env"))
        scored = sorted(
            a for a, o in (cp.get("per_action") or {}).items()
            if not o.get("unscorable_premature")
        )
        if len(scored) < 2 or str(cp.get("task_id")) not in ex:
            continue
        conf = render_state(dict(cp, legal_actions=scored), "(deciding)")
        val = ex[str(cp.get("task_id"))].text
        menu = lambda t: t.split("[legal] ", 1)[1].split("\n", 1)[0]  # noqa: E731
        assert menu(conf) == menu(val)
        checked += 1
    assert checked > 0


def test_q_refuses_the_silently_cost_blind_call():
    from sureact.schema import ActionOutcome, MetaAction, SourceVector, load_cost_profiles

    tau2_like = ActionOutcome(
        MetaAction.ASK, success=0.8, critical_error=0.0, cost=0.0, steps=5,
        delta_u=SourceVector(), action_counts={"ASK": 2.0, "SEARCH": 3.0},
    )
    with pytest.raises(ValueError, match="cost profile"):
        tau2_like.q(1.0, 1.0)
    assert tau2_like.q(1.0, 1.0, load_cost_profiles()["medium"]) < 0.8

    ts_like = ActionOutcome(
        MetaAction.ASK, success=0.8, critical_error=0.0, cost=0.25, steps=5,
        delta_u=SourceVector(),
    )
    assert ts_like.q(1.0, 1.0) == pytest.approx(0.55)
