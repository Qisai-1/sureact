#!/usr/bin/env python

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--confidence", help="output of scripts/compute_confidence.py")
    ap.add_argument(
        "--value-model",
        help="directory written by scripts/train_value.py; adds the SURE-Act controller",
    )
    ap.add_argument("--value-base", default="Qwen/Qwen3-8B", help="encoder for --value-model")
    ap.add_argument(
        "--value-model-pattern",
        help="controller directory per split, e.g. results/value_s{seed}",
    )
    ap.add_argument(
        "--value-backbone-hint",
        action="store_true",
        help="set if the value model was trained with --backbone-hint",
    )
    ap.add_argument(
        "--require-scorable-without", action="append", default=None,
        help="keep only checkpoints that still have two arms without this action")
    ap.add_argument(
        "--drop-action", action="append", default=None,
        help="remove this action from every checkpoint's menu (repeatable)")
    ap.add_argument(
        "--only",
        help="evaluate only checkpoints whose env/domain contains this string",
    )
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--cost-profile", default="medium")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument(
        "--test-frac", type=float, default=0.0,
        help="score on the test slice of the train_value.py family split",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output JSON")
    ap.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="average over this many family splits and report the spread",
    )
    args = ap.parse_args()

    from sureact.scoring import q
    from sureact.baselines import (
        AlwaysAction,
        checkpoint_key,
        AskOrAct,
        RandomLegal,
        evaluate,
        fit_threshold_policy,
    )
    from sureact.baselines.costaware import (
        fit_cost_aware_policy,
    )
    from sureact.schema import load_cost_profiles
    from sureact.value.dataset import family_of

    profile = load_cost_profiles()[args.cost_profile]

    cps: list[dict[str, Any]] = []
    for pat in args.paths:
        for p in map(Path, glob.glob(pat)):
            blob = json.loads(p.read_text())
            for cp in blob.get("checkpoints", []):
                cp.setdefault("env", blob.get("env"))
                cp["_src"] = p.stem
                cp["per_action"] = {
                    a: o
                    for a, o in (cp.get("per_action") or {}).items()
                    if not o.get("unscorable_premature")
                }
                for a_ in (args.require_scorable_without or []):
                    if len({x for x in cp["per_action"] if x != a_}) < 2:
                        cp["per_action"] = {}
                for a_ in (args.drop_action or []):
                    cp["per_action"].pop(a_, None)
                if len(cp["per_action"]) >= 2:
                    cp["legal_actions"] = sorted(cp["per_action"])
                    cps.append(cp)

    if len(cps) < 8:
        print(f"only {len(cps)} usable checkpoints -- too few to compare", file=sys.stderr)
        return 2

    fams = sorted({family_of(str(cp.get("task_id", ""))) for cp in cps})
    import random

    conf = json.loads(Path(args.confidence).read_text()) if args.confidence else None
    label = {
        "verbalized": "B3_verbalized",
        "self_consistency": "B4_self_consistency",
        "entropy": "B5_entropy",
    }
    actions = sorted({a for cp in cps for a in cp["per_action"]})

    vc = None
    if args.value_model and args.value_model_pattern:
        print("  pass --value-model OR --value-model-pattern, not both", file=sys.stderr)
        return 2
    if args.value_model:
        from sureact.baselines.learned import ValueController
        from sureact.value.model import load_value_model

        vm, vtok = load_value_model(args.value_model, args.value_base)
        vc = ValueController(
            vm, vtok, args.lam, args.eta, profile,
            backbone_hint=args.value_backbone_hint,
        )

    fitted_margins: dict[str, list[float]] = {}
    fitted_defaults: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    per_seed: dict[str, list[float]] = {}
    oracles: list[float] = []
    extras: dict[str, Any] = {}
    dev_totals: dict[str, dict] = {}
    const_splits: dict[str, int] = {}
    n_eval = 0
    for seed in range(args.seed, args.seed + args.seeds):
        rng = random.Random(seed)
        shuffled = list(fams)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * args.val_frac))
        if args.test_frac > 0:
            n_test = max(1, int(len(shuffled) * args.test_frac))
            val_fams = set(shuffled[:n_val])
            test_fams = set(shuffled[n_val : n_val + n_test])
            train = [c for c in cps
                     if family_of(str(c.get("task_id", ""))) not in val_fams
                     and family_of(str(c.get("task_id", ""))) not in test_fams]
            evalset = [c for c in cps if family_of(str(c.get("task_id", ""))) in test_fams]
        else:
            val_fams = set(shuffled[:n_val])
            train = [c for c in cps if family_of(str(c.get("task_id", ""))) not in val_fams]
            evalset = [c for c in cps if family_of(str(c.get("task_id", ""))) in val_fams]
        if args.only:
            evalset = [
                c for c in evalset
                if args.only in str(c.get("env", "")) or args.only in str(c.get("domain", ""))
            ]
        if not train or not evalset:
            continue
        n_eval = len(evalset)

        print(f"  split {seed} ({len(train)} train / {len(evalset)} eval checkpoints)",
              file=sys.stderr, flush=True)
        controllers: list[Any] = [RandomLegal(seed=seed), AskOrAct()]
        controllers += [AlwaysAction(a) for a in actions]
        if conf:
            for method, scores in (conf.get("scores") or {}).items():
                if not scores:
                    continue
                covered = sum(1 for c in cps if checkpoint_key(c) in scores)
                if covered < 0.5 * len(cps):
                    skipped[label.get(method, method)] = f"{covered}/{len(cps)}"
                    continue
                controllers.append(
                    fit_threshold_policy(
                        label.get(method, method), train, scores,
                        args.lam, args.eta, profile,
                    )
                )
                controllers.append(
                    fit_cost_aware_policy(
                        f"B6_cost_aware_{method}", train, scores,
                        args.lam, args.eta, profile,
                    )
                )
        if args.value_model_pattern:
            from sureact.baselines.learned import ValueController
            from sureact.value.model import load_value_model

            run_dir = args.value_model_pattern.format(seed=seed)
            if not Path(f"{run_dir}/head.pt").exists():
                skipped[f"SUREACT_value_s{seed}"] = "no artefacts"
                vc = None
            else:
                vm, vtok = load_value_model(run_dir, args.value_base)
                vc = ValueController(vm, vtok, args.lam, args.eta, profile,
                                     backbone_hint=bool(args.value_backbone_hint))
        if vc is not None:
            vc.prime(train)
            vc.prime(evalset)
            best_fixed_name = max(
                actions,
                key=lambda a: statistics.mean(
                    q(c["per_action"][a], args.lam, args.eta, profile)
                    if a in c["per_action"]
                    else min(q(o, args.lam, args.eta, profile) for o in c["per_action"].values())
                    for c in train
                ),
            )
            vc._default = best_fixed_name
            grid = [i / 40 for i in range(0, 41)]

            def fit_margin(on: list[dict[str, Any]]) -> float:
                """Grid-search the deviation margin."""
                best_m, best_u = 0.0, float("-inf")
                for m in grid:
                    vc._margin = m
                    u = evaluate(vc, on, args.lam, args.eta, profile).utility
                    if u > best_u:
                        best_m, best_u = m, u
                return best_m

            best_m = fit_margin(train)
            vc._margin = best_m
            vc.name = "SUREACT_value"
            fitted_margins.setdefault(vc.name, []).append(best_m)
            fitted_defaults.setdefault(vc.name, []).append(best_fixed_name)
            controllers.append(vc)
        for c in controllers:
            r = evaluate(c, evalset, args.lam, args.eta, profile)
            per_seed.setdefault(r.name, []).append(r.utility)
            extras.setdefault(r.name, r)
            agg = dev_totals.setdefault(r.name, {})
            for a, n in r.chosen.items():
                agg[a] = agg.get(a, 0) + n
            top_a = max(r.chosen.items(), key=lambda kv: kv[1]) if r.chosen else None
            if top_a and top_a[1] == sum(r.chosen.values()):
                const_splits[r.name] = const_splits.get(r.name, 0) + 1
        oracles.append(
            statistics.mean(
                max(q(o, args.lam, args.eta, profile) for o in cp["per_action"].values())
                for cp in evalset
            )
        )

    if not per_seed:
        print("every split produced an empty side -- too few families", file=sys.stderr)
        return 2

    print(
        f"=== baselines ===  {len(cps)} checkpoints, {len(fams)} families"
        f"  |  {args.seeds} split(s), ~{n_eval} eval checkpoints each"
        f"  (lam={args.lam} eta={args.eta} cost={args.cost_profile})"
    )
    if not args.confidence:
        print("(no --confidence: confidence baselines omitted)")
    for name, cov in skipped.items():
        print(f"  skipped {name}: only {cov} checkpoints have a score")

    def fmt(vals: list[float]) -> str:
        m = statistics.mean(vals)
        if len(vals) < 2:
            return f"{m:+.3f}"
        return f"{m:+.3f} +/-{statistics.stdev(vals):.3f}"

    print(f"\n  {'ORACLE (upper bound)':26} U={fmt(oracles)}")
    rows = sorted(per_seed.items(), key=lambda kv: -statistics.mean(kv[1]))
    for name, vals in rows:
        r = extras[name]
        crit = "  n/a" if r.critical_error is None else f"{r.critical_error:5.3f}"
        flag = f"  [{r.unlabelled} off-menu]" if r.unlabelled else ""
        print(
            f"  {name:26} U={fmt(vals)}  succ={r.success:.3f}"
            f"  cost={r.cost:.3f}  crit={crit}{flag}"
        )
    print("  (succ/cost/crit are from the first split; U is averaged over all splits)")

    print(f"\n  chosen actions, pooled over all {args.seeds} splits:")
    for name, _ in rows:
        agg = dev_totals.get(name, {})
        tot = sum(agg.values()) or 1
        top = sorted(agg.items(), key=lambda kv: -kv[1])
        share = ", ".join(f"{a}={n / tot:.1%}" for a, n in top[:3])
        cs = const_splits.get(name, 0)
        flag = f"   <- constant on {cs}/{args.seeds} splits" if cs else ""
        print(f"    {name:26} {share}{flag}")

    print("\n  chosen actions (first split):")
    for name, _ in rows:
        ch = extras[name].chosen
        tot = sum(ch.values()) or 1
        top = sorted(ch.items(), key=lambda kv: -kv[1])
        share = ", ".join(f"{a}={n} ({n/tot:.0%})" for a, n in top[:4])
        flag = "   <- CONSTANT" if top and top[0][1] == tot else ""
        print(f"    {name:26} {share}{flag}")

    fixed_names = [n for n in per_seed if n.startswith("B2_")]
    if fixed_names and args.seeds > 1:
        ref = max(fixed_names, key=lambda n: statistics.mean(per_seed[n]))
        print(f"\n  paired difference vs {ref}, same splits:")
        for name, vals in rows:
            if name == ref or len(vals) != len(per_seed[ref]):
                continue
            d = [a - b for a, b in zip(vals, per_seed[ref])]
            m = statistics.mean(d)
            se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
            wins = sum(1 for x in d if x > 1e-9)
            sig = "" if se != se or abs(m) < 2 * se else "  *"
            print(f"    {name:26} {m:+.4f} +/-{se:.4f} (se)  better on {wins}/{len(d)}{sig}")
        print("    (* = at least 2 standard errors from zero)")

        if not args.out:
            print("\n  (pass --out to save results)")
            return 0
        Path(args.out).write_text(json.dumps({
            "ref": ref,
            "seeds": len(per_seed[ref]),
            "cost_profile": args.cost_profile,
            "lam": args.lam, "eta": args.eta,
            "test_frac": args.test_frac,
            "skipped": skipped,
            "fitted_margins": {k: {"n": len(v), "mean": statistics.mean(v),
                                   "min": min(v), "max": max(v),
                                   "zero_splits": sum(1 for x in v if x == 0.0)}
                               for k, v in fitted_margins.items()},
            "fitted_defaults": {k: {d: v.count(d) for d in sorted(set(v))}
                                for k, v in fitted_defaults.items()},
            "per_seed": {n: v for n, v in per_seed.items()},
            "deviation": {n: {"top_share": max(a.values()) / (sum(a.values()) or 1),
                              "constant_splits": const_splits.get(n, 0),
                              "n_decisions": sum(a.values()),
                              "shares": {a_: c / (sum(a.values()) or 1)
                                         for a_, c in sorted(a.items(), key=lambda kv: -kv[1])},
                              "counts": dict(sorted(a.items(), key=lambda kv: -kv[1]))}
                          for n, a in dev_totals.items()},
            "paired": {
                name: {
                    "mean": statistics.mean([a - b for a, b in zip(vals, per_seed[ref])]),
                    "se": (statistics.stdev([a - b for a, b in zip(vals, per_seed[ref])])
                           / len(vals) ** 0.5) if len(vals) > 1 else None,
                    "wins": sum(1 for a, b in zip(vals, per_seed[ref]) if a - b > 1e-9),
                    "n": len(vals),
                }
                for name, vals in rows
                if name != ref and len(vals) == len(per_seed[ref])
            },
        }, indent=2))
        print(f"\n  wrote {args.out}")

    fixed = [(n, v) for n, v in per_seed.items() if n.startswith("B2_")]
    if fixed:
        bn, bv = max(fixed, key=lambda kv: statistics.mean(kv[1]))
        gap = statistics.mean(oracles) - statistics.mean(bv)
        print(f"\n  bar to beat: {bn} at U={fmt(bv)}   headroom to oracle {gap:+.3f}")
        for name, vals in rows:
            if name.startswith(("B1_", "B2_")):
                continue
            d = statistics.mean(vals) - statistics.mean(bv)
            verdict = "beats" if d > 0 else "does NOT beat"
            print(f"    {name:26} {verdict} the best fixed action ({d:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
