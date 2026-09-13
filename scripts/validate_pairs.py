#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INTENDED = {"goal": "ASK", "observation": "SEARCH"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--cost-profile", default="medium")
    ap.add_argument("--expect-pairs", type=int, default=0,
                    help="expected number of pairs; warns if fewer are present")
    ap.add_argument("--min-pairs", type=int, default=8,
                    help="fail if fewer complete pairs validate than this")
    ap.add_argument("--by-success", action="store_true",
                    help="rank by raw success instead of utility (matches validate_pair)")
    args = ap.parse_args()

    from sureact.scoring import q
    from sureact.schema import load_cost_profiles

    profile = load_cost_profiles()[args.cost_profile]

    pairs: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for p in args.paths:
        blob = json.loads(Path(p).read_text())
        for cp in blob.get("checkpoints", []):
            name = str(cp.get("scenario") or cp.get("name") or cp.get("task_id") or "")
            variant = next((v for v in ("goal", "observation") if name.endswith("/" + v)), None)
            if variant is None:
                continue
            pa = {a: o for a, o in (cp.get("per_action") or {}).items()
                  if not o.get("unscorable_premature")}
            if not pa:
                continue
            scores = ({a: float(o.get("success") or 0.0) for a, o in pa.items()}
                      if args.by_success
                      else {a: q(o, args.lam, args.eta, profile) for a, o in pa.items()})
            pairs[name[: -len("/" + variant)]][variant] = scores

    complete = {k: v for k, v in pairs.items() if len(v) == 2}
    partial = {k: v for k, v in pairs.items() if len(v) == 1}
    if args.expect_pairs and len(complete) + len(partial) < args.expect_pairs:
        print(f"warning: {len(complete) + len(partial)} pairs found, "
              f"expected {args.expect_pairs}", file=sys.stderr)
    if not complete:
        print("no complete pairs found -- check the label file's scenario naming",
              file=sys.stderr)
        return 2

    ok_pairs, rows = 0, []
    for name, v in sorted(complete.items()):
        verdict = {}
        for variant, scores in v.items():
            want = INTENDED[variant]
            best = max(scores, key=scores.__getitem__)
            hit = want in scores and scores[want] >= max(scores.values()) - 1e-9
            verdict[variant] = (hit, best, scores.get(want))
        both = verdict["goal"][0] and verdict["observation"][0]
        ok_pairs += bool(both)
        rows.append((name, verdict, both))

    print(f"pair validation  ({len(complete)} complete pairs"
          f"{f', {len(partial)} incomplete' if partial else ''})")
    print(f"  {'pair':28} {'goal->ASK':>18} {'obs->SEARCH':>20}  flip")
    for name, v, both in rows:
        g_hit, g_best, _ = v["goal"]
        o_hit, o_best, _ = v["observation"]
        print(f"  {name:28} {('yes' if g_hit else 'no (' + g_best + ')'):>18} "
              f"{('yes' if o_hit else 'no (' + o_best + ')'):>20}  {'YES' if both else '--'}")

    print(f"\n  pairs flipping in BOTH directions: {ok_pairs}/{len(complete)} "
          f"({ok_pairs / len(complete):.0%})")
    if partial:
        print(f"  {len(partial)} pairs have only one variant labelled and are excluded:")
        for k in sorted(partial):
            print(f"    {k} (only {next(iter(partial[k]))})")
    if ok_pairs < args.min_pairs:
        print(f"\n  fewer than --min-pairs={args.min_pairs} pairs validated", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
