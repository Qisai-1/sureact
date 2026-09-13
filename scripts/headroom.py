#!/usr/bin/env python

from __future__ import annotations

import argparse
import collections
import glob
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.05, help="scalars within this are 'tied'")
    args = ap.parse_args()

    from sureact.scoring import load, q, risk_judgeable
    from sureact.schema import load_cost_profiles

    profile = load_cost_profiles()["medium"]
    paths = [Path(p) for pat in args.paths for p in glob.glob(pat)]
    cps: list[dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            continue
        _, rows = load(p)
        for c in rows:
            c["per_action"] = {
                a: o
                for a, o in c["per_action"].items()
                if not o.get("unscorable_premature")
            }
        cps.extend(c for c in rows if len(c["per_action"]) >= 2)

    if len(cps) < 10:
        print(f"only {len(cps)} usable checkpoints -- too few", file=sys.stderr)
        return 2

    for c in cps:
        qs = {a: q(o, args.lam, args.eta, profile) for a, o in c["per_action"].items()}
        c["_q"] = qs
        c["_best"] = max(qs, key=qs.__getitem__)
        c["_scalar"] = max(o["success"] for o in c["per_action"].values())

    print(f"=== headroom ===  {len(cps)} checkpoints from {len(paths)} file(s)")
    print(f"optimal actions: {dict(collections.Counter(c['_best'] for c in cps))}")

    if args.eta:
        n_arms = sum(len(c["per_action"]) for c in cps)
        n_unjudge = sum(
            1 for c in cps for o in c["per_action"].values() if not risk_judgeable(o)
        )
        if n_unjudge:
            print(
                f"note: {n_unjudge}/{n_arms} arms have unjudgeable risk"
            )

    actions = sorted({a for c in cps for a in c["per_action"]})
    oracle = statistics.mean(c["_q"][c["_best"]] for c in cps)

    def util_of_rule(thresholds: list[float], mapping: list[str]) -> float:
        tot = 0.0
        for c in cps:
            band = sum(1 for t in thresholds if c["_scalar"] >= t)
            want = mapping[band]
            tot += c["_q"].get(want, min(c["_q"].values()))
        return tot / len(cps)

    best_flat = max(util_of_rule([], [a]) for a in actions)
    grid = [i / 20 for i in range(21)]
    best_2band = max(
        util_of_rule([t], [a0, a1]) for t in grid for a0 in actions for a1 in actions
    )
    print("\nA. what the BEST possible scalar rule could achieve")
    print(f"   per-checkpoint oracle            {oracle:.3f}")
    print(f"   best 2-band scalar threshold     {best_2band:.3f}   gap {oracle - best_2band:+.3f}")
    print(f"   best fixed single action         {best_flat:.3f}   gap {oracle - best_flat:+.3f}")
    print("   (scalar = the checkpoint's best achievable success)")

    ties = disagree = 0
    examples = []
    for i, a in enumerate(cps):
        for b in cps[i + 1 :]:
            if abs(a["_scalar"] - b["_scalar"]) > args.tol:
                continue
            ties += 1
            if a["_best"] != b["_best"]:
                disagree += 1
                if len(examples) < 4 and a["task_id"] != b["task_id"]:
                    examples.append((a, b))
    frac = disagree / ties if ties else 0.0
    print(f"\nB. checkpoints with near-identical scalars (within {args.tol})")
    print(f"   tied pairs                       {ties}")
    print(f"   ...wanting DIFFERENT actions     {disagree}  = {frac:.1%}")
    verdict = "PASS" if frac >= 0.15 else "FAIL"
    print(f"   target >=15%                     [{verdict}]")

    default = max(actions, key=lambda a: sum(c["_q"].get(a, min(c["_q"].values())) for c in cps))
    off = [c for c in cps if c["_best"] != default]
    if off:
        print(f"\nC. where the best fixed action ({default}) is NOT optimal: {len(off)}/{len(cps)}")
        means = {
            a: statistics.mean(c["_q"].get(a, min(c["_q"].values())) for c in off)
            for a in actions
        }
        for a in sorted(means, key=lambda x: -means[x]):
            mark = "  <- the default" if a == default else ""
            print(f"     mean utility of always-{a:<22} {means[a]:+.3f}{mark}")
        orc = statistics.mean(c["_q"][c["_best"]] for c in off)
        print(f"     per-checkpoint oracle                          {orc:+.3f}")
        print(f"     optimal action there: {dict(collections.Counter(c['_best'] for c in off))}")
        if max(means, key=means.__getitem__) == default:
            print(f"     no single substitute beats the default; "
                  f"{orc - means[default]:+.3f} needs a per-checkpoint choice")

    for a, b in examples:
        print(
            f"\n   same scalar ~{a['_scalar']:.2f}, opposite answers:"
            f"\n     {a['task_id'][:44]:<44} -> {a['_best']}"
            f"\n     {b['task_id'][:44]:<44} -> {b['_best']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
