#!/usr/bin/env python

from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sureact.scoring import load, q, risk_judgeable  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()

    from sureact.schema import RISK_WEIGHTS, load_cost_profiles

    profiles = load_cost_profiles()

    for p in args.paths:
        path = Path(p)
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            continue
        env, cps = load(path)
        if not cps:
            print(f"\n=== {path.name}: no labelled checkpoints ===")
            continue

        print(f"\n=== {path.name} | {env} | {len(cps)} checkpoints ===")

        best = collections.Counter()
        per_task = collections.defaultdict(list)
        spreads = []
        for c in cps:
            qs = {a: q(o, args.lam, 1.0, profiles["medium"]) for a, o in c["per_action"].items()}
            b = max(qs, key=qs.__getitem__)
            best[b] += 1
            per_task[c["task_id"]].append(b)
            spreads.append(max(qs.values()) - min(qs.values()))
        multi = [k for k, v in per_task.items() if len(set(v)) > 1]
        act_legal = [c for c in cps if "ACT_REVERSIBLE" in c["per_action"]]
        notact = sum(
            1
            for c in act_legal
            if max(
                (qq := {a: q(o, args.lam, 1.0, profiles["medium"]) for a, o in c["per_action"].items()}),
                key=qq.__getitem__,
            )
            != "ACT_REVERSIBLE"
        )
        ncrit = sum(
            1
            for c in cps
            for o in c["per_action"].values()
            if (o["critical_error"] or 0) > 0
        )
        dropped = sum(1 for c in cps if c["dropped_actions"])

        print(f"  optimal action        {dict(best)}")
        if act_legal:
            print(f"  acting NOT optimal    {notact}/{len(act_legal)} = {notact / len(act_legal):.0%}"
                  f"   (of checkpoints where ACT is legal)")
        else:
            print("  acting NOT optimal    n/a -- ACT_REVERSIBLE is not legal anywhere here")
        print(
            f"  optimal changes within a task  {len(multi)}/{len(per_task)}"
            + ("  (single checkpoint per task)" if len(per_task) == len(cps) else "")
        )
        print(f"  median best-worst spread       {statistics.median(spreads):.3f}")
        flips = 0
        for c in cps:
            q1 = {a: q(o, args.lam, 1.0, profiles["medium"]) for a, o in c["per_action"].items()}
            q5 = {a: q(o, args.lam, 5.0, profiles["medium"]) for a, o in c["per_action"].items()}
            if max(q1, key=q1.__getitem__) != max(q5, key=q5.__getitem__):
                flips += 1
        print(f"  arms with a critical error     {ncrit}")
        judgeable = [o for c in cps for o in c["per_action"].values() if risk_judgeable(o)]
        cells = collections.Counter(
            ((o.get("critical_error") or 0) > 0, (o.get("success") or 0) > 0)
            for o in judgeable
        )
        print(
            f"  risk x success (judgeable arms, n={len(judgeable)}):"
            f"  risky+success {cells[(True, True)]}"
            f" | risky+fail {cells[(True, False)]}"
            f" | safe+success {cells[(False, True)]}"
            f" | safe+fail {cells[(False, False)]}"
        )
        if not cells[(True, True)]:
            print(
                "   <- no arm is both risky and successful"
            )
        print(
            f"  checkpoints where eta 1->5 changes the optimal action  {flips}/{len(cps)}"
            + (
                "   <- eta never changes the argmax"
                if flips == 0
                else "   <- risk axis flips the answer"
            )
        )
        if dropped:
            print(f"  ⚠ checkpoints with dropped arms {dropped}")

        def fixed(cp: dict[str, Any], prefs: list[str]) -> str:
            for a in prefs:
                if a in cp["per_action"]:
                    return a
            return next(iter(cp["per_action"]))

        POLICIES = {
            "oracle_q": None,
            "always_act": ["ACT_REVERSIBLE", "COMMIT_IRREVERSIBLE", "SEARCH", "ASK"],
            "verify_before_commit": ["SEARCH", "ACT_REVERSIBLE", "COMMIT_IRREVERSIBLE"],
            "never_commit": ["ACT_REVERSIBLE", "SEARCH", "ASK", "ABSTAIN_HANDOFF"],
            "always_ask": ["ASK", "SEARCH", "ACT_REVERSIBLE"],
            "always_abstain": ["ABSTAIN_HANDOFF", "ASK", "SEARCH"],
        }
        for pname, profile in profiles.items():
            for rname, eta in RISK_WEIGHTS.items():
                util = {}
                for pol, prefs in POLICIES.items():
                    tot = 0.0
                    for c in cps:
                        qs = {a: q(o, args.lam, eta, profile) for a, o in c["per_action"].items()}
                        chosen = max(qs, key=qs.__getitem__) if prefs is None else fixed(c, prefs)
                        tot += qs[chosen]
                    util[pol] = tot / len(cps)
                orc = util["oracle_q"]
                bestfix = max((k for k in util if k != "oracle_q"), key=util.__getitem__)
                print(
                    f"  cost={pname:<6} risk={rname:<11} oracle={orc:>6.3f}  "
                    f"best={bestfix}({util[bestfix]:.3f})  gap={orc - util[bestfix]:+.3f}"
                )

        report_free_arm(cps, args.lam, profiles)
    return 0


def report_free_arm(cps: list[dict[str, Any]], lam: float, profiles: dict[str, Any]) -> None:
    from sureact.schema import RISK_WEIGHTS

    have = [c for c in cps if c["free_arm"]]
    if not have:
        print("  free arm              NOT MEASURED -- rerun with --n-free >= 1")
        return

    chose = collections.Counter(c["free_arm"]["action"] for c in have)
    print(f"\n  --- unforced backbone ({len(have)}/{len(cps)} checkpoints) ---")
    print(f"  chose                 {dict(chose)}")

    agree = [
        (c["free_arm"]["success"] - c["per_action"][c["free_arm"]["action"]]["success"])
        for c in have
        if c["free_arm"]["action"] in c["per_action"]
    ]
    off_menu = [c for c in have if c["free_arm"]["action"] not in c["per_action"]]
    if agree:
        print(
            f"  forcing footprint     median |free - forced(same action)| = "
            f"{statistics.median(abs(d) for d in agree):.3f}  (n={len(agree)}, "
            f"max {max(abs(d) for d in agree):.3f})"
        )
    if off_menu:
        print(
            f"  ⚠ chose an unlabelled action at {len(off_menu)} checkpoint(s): "
            f"{sorted({c['free_arm']['action'] for c in off_menu})}"
        )

    for pname, profile in profiles.items():
        for rname, eta in RISK_WEIGHTS.items():
            regrets, crit = [], 0.0
            for c in have:
                qs = [q(o, lam, eta, profile) for o in c["per_action"].values()]
                regrets.append(max(qs) - q(c["free_arm"], lam, eta, profile))
                crit += c["free_arm"]["critical_error"] or 0.0
            worse = sum(1 for r in regrets if r > 1e-9)
            print(
                f"  cost={pname:<6} risk={rname:<11} free_regret mean="
                f"{statistics.mean(regrets):>6.3f} median={statistics.median(regrets):>6.3f}"
                f"  suboptimal at {worse}/{len(regrets)}"
                f"  critical={crit / len(have):.3f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
