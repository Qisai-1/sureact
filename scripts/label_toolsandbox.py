#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ALL_CATEGORIES = [
    "SINGLE_TOOL_CALL",
    "MULTIPLE_TOOL_CALL",
    "SINGLE_USER_TURN",
    "MULTIPLE_USER_TURN",
    "STATE_DEPENDENCY",
    "CANONICALIZATION",
    "INSUFFICIENT_INFORMATION",
    "NO_DISTRACTION_TOOLS",
    "TEN_DISTRACTION_TOOLS",
    "ALL_TOOLS_AVAILABLE",
    "TOOL_NAME_SCRAMBLED",
    "TOOL_DESCRIPTION_SCRAMBLED",
    "ARG_TYPE_SCRAMBLED",
    "ARG_DESCRIPTION_SCRAMBLED",
]
CATEGORIES = ["INSUFFICIENT_INFORMATION", "MULTIPLE_USER_TURN", "STATE_DEPENDENCY"]


def _label_one(job: dict[str, Any]) -> list[dict[str, Any]]:
    from tool_sandbox.common.execution_context import (
        RoleType,
        get_current_context,
        set_current_context,
    )
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.roles.execution_environment import ExecutionEnvironment
    from tool_sandbox.scenarios import named_scenarios

    from sureact.envs.toolsandbox_fork import setup, step_n
    from sureact.envs.toolsandbox_roles import LocalAgent, LocalUser
    from sureact.label.counterfactual import (
        ToolTaxonomy,
        label_checkpoint,
        make_forced_agent,
    )
    from sureact.label.perturb import (
        build_contact_suite,
        build_messaging_suite,
        build_remove_suite,
    )
    from sureact.schema import SourceVector, load_cost_profiles

    taxonomy = ToolTaxonomy.load()
    Forced = make_forced_agent(LocalAgent, taxonomy)

    def make_roles() -> dict:
        return {
            RoleType.USER: LocalUser(),
            RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
            RoleType.AGENT: Forced(),
        }

    scenarios = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
    profile = load_cost_profiles()["medium"]

    if job["kind"] == "native":
        scenario, tag, u_true = scenarios[job["name"]], job["name"], None
    else:
        suite = {f"{p['name']}/{k}": p[k]
                 for p in (build_messaging_suite(scenarios) + build_contact_suite(scenarios) + build_remove_suite(scenarios))
                 for k in ("goal", "observation")}
        scenario = suite[job["name"]]
        tag = job["name"]
        u_true = (
            SourceVector(goal=1.0) if job["name"].endswith("/goal")
            else SourceVector(observation=1.0)
        )

    out: list[dict[str, Any]] = []
    try:
        roles = make_roles()
        _, base = setup(scenario, roles)
        step_n(roles, job["first_fork_at"], base, scenario.max_messages)
        snapshot = None
        for ci in range(job["checkpoints"]):
            if ci and snapshot is not None:
                set_current_context(snapshot)
                roles = make_roles()
                if step_n(roles, job["stride"], base, scenario.max_messages) == 0:
                    break
            snapshot = copy.deepcopy(get_current_context())
            cp = label_checkpoint(
                scenario, tag, snapshot, make_roles, taxonomy, profile, base,
                step_idx=ci, continue_for=job["continue_for"],
                n_seeds=job["n_seeds"], u_true=u_true, n_free=job["n_free"],
            )
            if len(cp.per_action) >= 2:
                out.append(cp.to_json())
    except Exception as exc:  # noqa: BLE001
        return [{"__error__": f"{type(exc).__name__}: {exc}", "task_id": tag}]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=40)
    ap.add_argument(
        "--categories",
        default=",".join(CATEGORIES),
        help="comma-separated ToolSandbox categories; see ALL_CATEGORIES",
    )
    ap.add_argument("--checkpoints", type=int, default=2)
    ap.add_argument("--first-fork-at", type=int, default=4)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--continue-for", type=int, default=12)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument(
        "--n-free", type=int, default=1,
        help="unforced rollouts per checkpoint (0 disables)",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--perturbed", action="store_true", help="label the messaging suite")
    ap.add_argument(
        "--task-timeout", type=float, default=900.0,
        help="seconds per scenario before it is abandoned; one hang must not wedge the run",
    )
    ap.add_argument("--out", default="results/labels_toolsandbox.json")
    args = ap.parse_args()

    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.scenarios import named_scenarios

    from sureact.label.perturb import (
        build_contact_suite,
        build_messaging_suite,
        build_remove_suite,
    )

    scenarios = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)

    if args.perturbed:
        names = [
            f"{p['name']}/{k}"
            for p in (build_messaging_suite(scenarios) + build_contact_suite(scenarios) + build_remove_suite(scenarios))
            for k in ("goal", "observation")
        ]
        kind = "perturbed"
    else:
        wanted = [x.strip() for x in args.categories.split(",") if x.strip()]
        unknown = [x for x in wanted if x not in ALL_CATEGORIES]
        if unknown:
            print(f"unknown categories: {unknown}; valid: {ALL_CATEGORIES}", file=sys.stderr)
            return 2
        names = [
            n
            for n, s in scenarios.items()
            if any(str(c).split(".")[-1] in wanted for c in getattr(s, "categories", []))
        ][: args.scenarios]
        kind = "native"

    jobs = [
        {
            "kind": kind, "name": n, "checkpoints": args.checkpoints,
            "first_fork_at": args.first_fork_at, "stride": args.stride,
            "continue_for": args.continue_for, "n_seeds": args.n_seeds,
            "n_free": args.n_free,
        }
        for n in names
    ]
    print(f"{len(jobs)} {kind} scenario(s), {args.workers} workers", flush=True)

    started = time.time()
    cps: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    done = 0

    def flush() -> None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "kind": kind, "n_checkpoints": len(cps),
            "n_scenarios": len({c["task_id"] for c in cps}),
            "n_seeds": args.n_seeds, "n_free": args.n_free, "workers": args.workers,
            "elapsed_s": round(time.time() - started, 1),
            "complete": False, "errors": errors, "checkpoints": cps,
        }, indent=2, default=str))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_label_one, j): j["name"] for j in jobs}
        budget = max(args.task_timeout, 600) * max(1, len(jobs) // max(1, args.workers))
        try:
            for fut in as_completed(futures, timeout=budget):
                name = futures[fut]
                done += 1
                try:
                    res = fut.result(timeout=args.task_timeout)
                except Exception as exc:  # noqa: BLE001
                    errors.append({"task_id": name, "__error__": repr(exc)[:200]})
                    print(f"  [{done}/{len(jobs)}] {name}: ABANDONED {type(exc).__name__}", flush=True)
                    flush()
                    continue
                bad = [r for r in res if "__error__" in r]
                good = [r for r in res if "__error__" not in r]
                errors.extend(bad)
                cps.extend(good)
                if bad:
                    summary = bad[0]["__error__"][:60]
                else:
                    first = good[0] if good else {}
                    summary = " ".join(
                        f"{a[:4]}={o['success']:.2f}" for a, o in (first.get("per_action") or {}).items()
                    )
                    fa = first.get("free_arm")
                    if fa:
                        summary += (
                            f"  | free={fa['outcome']['action'][:4]}({fa['outcome']['success']:.2f})"
                        )
                print(f"  [{done}/{len(jobs)}] {name[:46]:<46} {len(good)} cps  {summary}", flush=True)
                flush()
    
        except TimeoutError:
            left = [f for f in futures if not f.done()]
            print(
                f"  abandoning {len(left)} task(s) after {budget}s without progress",
                flush=True,
            )
            for f in left:
                f.cancel()
            errors.extend({"task_id": futures[f], "__error__": "hung, abandoned"} for f in left)
            flush()
    elapsed = time.time() - started
    out = {
        "kind": kind, "n_checkpoints": len(cps),
        "n_scenarios": len({c["task_id"] for c in cps}),
        "n_seeds": args.n_seeds, "n_free": args.n_free, "workers": args.workers,
        "elapsed_s": round(elapsed, 1), "complete": True,
        "errors": errors, "checkpoints": cps,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(
        f"\n{len(cps)} checkpoints from {out['n_scenarios']} scenarios in {elapsed/60:.1f} min "
        f"({len(errors)} errors) -> {args.out}",
        flush=True,
    )
    dropped = [(c["task_id"], c.get("dropped_actions")) for c in cps if c.get("dropped_actions")]
    if dropped:
        print(f"⚠ {len(dropped)} checkpoint(s) with dropped arms: {dropped[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
