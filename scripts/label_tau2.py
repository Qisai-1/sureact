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


def _build(domain: str, task: Any, model: str) -> Any:
    """Build an orchestrator with a forcible agent."""
    import os

    from tau2.agent.llm_agent import LLMAgent
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry
    from tau2.user.user_simulator import UserSimulator

    from sureact.label.tau2_cf import Tau2Taxonomy, make_forced_tau2_agent

    env = registry.get_env_constructor(domain)()
    taxonomy = Tau2Taxonomy.load(domain)
    gaps = taxonomy.cross_check(env)
    if gaps["unmapped"]:
        raise SystemExit(
            f"{domain}: {len(gaps['unmapped'])} tools are missing from "
            f"configs/tool_manifests/tau2.yaml: {gaps['unmapped']}. "
            "Declare them (with tau2's own _is_mutating_tool as the cross-check) before labelling."
        )
    Forced = make_forced_tau2_agent(LLMAgent, taxonomy)
    llm_args = {
        "temperature": 0.0,
        "api_key": os.environ.get("HOSTED_VLLM_API_KEY", "EMPTY"),
        "api_base": os.environ.get("HOSTED_VLLM_API_BASE"),
        "timeout": float(os.environ.get("SUREACT_LLM_TIMEOUT", "180")),
    }
    if os.environ.get("SUREACT_THINK", "0") != "1":
        llm_args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    agent = Forced(
        tools=env.get_tools(), domain_policy=env.get_policy(), llm=model,
        llm_args=dict(llm_args),
    )
    user = UserSimulator(
        instructions=task.user_scenario.instructions, llm=model,
        llm_args=dict(llm_args),
    )
    orch = Orchestrator(
        domain=domain, agent=agent, user=user, environment=env, task=task,
        max_steps=40, seed=0,
    )
    orch.initialize()
    return orch, taxonomy


def _advance(orch: Any, n: int) -> int:
    took = 0
    for _ in range(n):
        if orch.done:
            break
        orch.step()
        took += 1
    return took


def _settle(orch: Any, max_extra: int = 6) -> None:
    from tau2.orchestrator.orchestrator import Role

    for _ in range(max_extra):
        if orch.done or orch.to_role != Role.ENV:
            return
        orch.step()


def _finalize_for_grading(orch: Any) -> Any:
    import time as _time

    from tau2.data_model.simulation import TerminationReason
    from tau2.utils.utils import get_now

    if getattr(orch, "_run_start_time", None) is None:
        orch._run_start_time = get_now()
    if getattr(orch, "_run_start_perf", None) is None:
        orch._run_start_perf = _time.perf_counter()
    if getattr(orch, "termination_reason", None) is None:
        orch.termination_reason = TerminationReason.MAX_STEPS
        orch._sureact_truncated = True
    return orch._finalize()


def _eval_type_for(task: Any) -> Any:
    from tau2.data_model.tasks import RewardType
    from tau2.evaluator.evaluator import EvaluationType

    if task.evaluation_criteria is None:
        return EvaluationType.ALL
    basis = set(task.evaluation_criteria.reward_basis or [])
    if RewardType.NL_ASSERTION not in basis:
        return EvaluationType.ALL
    programmatic = basis - {RewardType.NL_ASSERTION}
    if programmatic <= {RewardType.DB, RewardType.ENV_ASSERTION}:
        return EvaluationType.ENV
    raise ValueError(
        f"unsupported reward_basis {sorted(str(b) for b in basis)}"
    )


def _first_action(msgs: list[Any], since: int, taxonomy: Any) -> Any:
    from sureact.schema import MetaAction

    for m in msgs[since:]:
        if getattr(m, "role", None) != "assistant":
            continue
        for tc in getattr(m, "tool_calls", None) or []:
            a = taxonomy.meta_of.get(getattr(tc, "name", ""))
            if a is not None:
                return a
        if getattr(m, "content", None):
            return MetaAction.ASK
    return None


def _call_sig(m: Any) -> str | None:
    if getattr(m, "role", None) != "assistant":
        return None
    for tc in getattr(m, "tool_calls", None) or []:
        args = getattr(tc, "arguments", None)
        return f"{getattr(tc, 'name', '?')}({json.dumps(args, sort_keys=True, default=str)[:200]})"
    content = getattr(m, "content", None)
    return f"say:{str(content)[:120]}" if content else None


def _loop_signature(branch: list[Any]) -> dict[str, Any]:
    sigs = [s for s in (_call_sig(m) for m in branch) if s]
    if not sigs:
        return {"n_calls": 0, "max_run": 0, "repeat_frac": 0.0}
    run = best = 1
    for a, b in zip(sigs, sigs[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return {
        "n_calls": len(sigs),
        "max_run": best,
        "repeat_frac": round(1 - len(set(sigs)) / len(sigs), 3),
    }


def _tail_digest(branch: list[Any], n: int = 12) -> list[str]:
    """Summarise the last few turns."""
    out = []
    for m in branch[-n:]:
        role = str(getattr(m, "role", "?"))
        sig = _call_sig(m)
        if sig is None:
            content = getattr(m, "content", None)
            sig = str(content)[:120] if content else ""
        out.append(f"{role}: {sig}"[:240])
    return out


def _label_task(job: dict[str, Any]) -> dict[str, Any]:
    """Label one task by forcing every legal action."""
    from tau2.registry import registry

    from sureact.label.tau2_cf import (
        classify_trajectory,
        critical_errors_from_trajectory,
    )

    domain, model = job["domain"], job["model"]
    tasks = registry.get_tasks_loader(domain)()
    task = next(t for t in tasks if str(t.id) == str(job["task_id"]))

    try:
        orch, taxonomy = _build(domain, task, model)
        snapshot, fork_at = copy.deepcopy(orch), 0
        for step in range(job["fork_at"]):
            if orch.done:
                break
            orch.step()
            if orch.done:
                break
            snapshot, fork_at = copy.deepcopy(orch), step + 1
        if fork_at < job["fork_at"]:
            print(
                f"    {job['task_id']}: episode ended early, forking at {fork_at} "
                f"instead of {job['fork_at']}",
                flush=True,
            )
        history = [
            {
                "sender": getattr(m, "role", "?"),
                "recipient": "agent" if getattr(m, "role", "") == "user" else "user",
                "content": str(getattr(m, "content", "") or "")[:600],
                "tool": ",".join(
                    getattr(tc, "name", "") for tc in (getattr(m, "tool_calls", None) or [])
                )
                or None,
            }
            for m in snapshot.get_messages()[-30:]
        ]
        tool_names = [t.name for t in snapshot.environment.get_tools()]
        legal = taxonomy.legal_actions(tool_names)

        since = len(snapshot.get_messages())

        def run(action: Any) -> dict[str, Any]:
            branch = copy.deepcopy(snapshot)
            if action is not None:
                branch.agent.force(action)
            _advance(branch, job["continue_for"])
            _settle(branch)
            msgs = branch.get_messages()
            from tau2.evaluator.evaluator import (
                EvaluationType,
                evaluate_simulation,
            )

            sim = _finalize_for_grading(branch)
            eval_type = _eval_type_for(task)
            ri = evaluate_simulation(
                simulation=sim, task=task, evaluation_type=eval_type,
                solo_mode=False, domain=domain,
            )
            counts = classify_trajectory(msgs, taxonomy)
            taken = _first_action(msgs, since, taxonomy)
            chosen = action or taken
            term = getattr(sim, "termination_reason", None)
            term = getattr(term, "value", term)
            note = str((getattr(ri, "info", None) or {}).get("note", ""))
            premature = "terminated prematurely" in note.lower()
            _crit = critical_errors_from_trajectory(msgs[since:], task, taxonomy)
            out = {
                "action": chosen.value if chosen is not None else None,
                "success": float(getattr(ri, "reward", 0.0) or 0.0),
                "termination_reason": term,
                "eval_type": eval_type.value,
                "unscorable_premature": premature,
                "critical_error": _crit["effective"],
                "critical_error_attempted": _crit["attempted"],
                "steps": len(msgs),
                "action_counts": {a.value: c for a, c in counts.items()},
                "first_action": taken.value if taken is not None else None,
            }
            out.update(_loop_signature(msgs[since:]))
            if premature:
                out["tail"] = _tail_digest(msgs[since:])
            return out

        per_action: dict[str, Any] = {}
        dropped: dict[str, str] = {}
        for action in legal:
            try:
                per_action[action.value] = run(action)
            except Exception as exc:  # noqa: BLE001
                dropped[action.value] = f"{type(exc).__name__}: {exc}"[:160]

        free_arm = None
        if job.get("n_free", 0) > 0:
            try:
                got = run(None)
                if got["first_action"] is None:
                    dropped["__free__"] = "no agent action after the fork"
                else:
                    free_arm = {
                        "outcome": got,
                        "first_action_counts": {got["first_action"]: 1.0},
                        "n_failed": 0,
                        "n_unclassified": 0,
                    }
            except Exception as exc:  # noqa: BLE001
                dropped["__free__"] = f"{type(exc).__name__}: {exc}"[:160]

        return {
            "task_id": job["task_id"], "domain": domain, "fork_at": fork_at,
            "fork_at_requested": job["fork_at"],
            "env": "tau2", "history": history,
            "legal_actions": [a.value for a in legal], "per_action": per_action,
            "dropped_actions": dropped, "free_arm": free_arm,
        }
    except Exception as exc:  # noqa: BLE001
        return {"task_id": job["task_id"], "__error__": f"{type(exc).__name__}: {exc}"[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="airline")
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--fork-at", type=int, default=6)
    ap.add_argument(
        "--continue-for", type=int, default=30,
        help=(
            "turns to run a branch after the forced action; branches hitting the cap score 0"
        ),
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--n-free", type=int, default=1,
        help="unforced rollouts per checkpoint (0 disables)",
    )
    ap.add_argument("--model", default=None, help="LiteLLM model string")
    ap.add_argument("--out", default="results/labels_tau2.json")
    args = ap.parse_args()

    import os

    from tau2.registry import registry

    model = args.model or (
        f"hosted_vllm/{os.environ.get('SUREACT_LLM_MODEL', 'Qwen/Qwen3-8B')}"
    )
    tasks = registry.get_tasks_loader(args.domain)()[: args.tasks]
    jobs = [
        {
            "domain": args.domain, "task_id": str(t.id), "model": model,
            "fork_at": args.fork_at, "continue_for": args.continue_for,
            "n_free": args.n_free,
        }
        for t in tasks
    ]
    print(f"{len(jobs)} {args.domain} task(s), {args.workers} workers, model={model}", flush=True)

    started = time.time()
    rows, errors, done = [], [], 0

    def flush(complete: bool = False) -> None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "env": "tau2", "domain": args.domain, "n_checkpoints": len(rows),
            "model": model, "elapsed_s": round(time.time() - started, 1),
            "complete": complete, "errors": errors, "checkpoints": rows,
        }, indent=2, default=str))

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futs = {pool.submit(_label_task, j): j["task_id"] for j in jobs}
        for fut in as_completed(futs):
            done += 1
            tid = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001
                errors.append({"task_id": tid, "__error__": repr(exc)})
                print(f"  [{done}/{len(jobs)}] {tid}: WORKER DIED {exc}", flush=True)
                flush()
                continue
            if "__error__" in r or "__skip__" in r:
                errors.append(r)
                print(f"  [{done}/{len(jobs)}] {tid}: {r.get('__error__') or r.get('__skip__')}", flush=True)
                flush()
                continue
            rows.append(r)
            pa = r["per_action"]
            summary = " ".join(f"{a[:4]}={o['success']:.2f}" for a, o in pa.items())
            if r.get("free_arm"):
                fo = r["free_arm"]["outcome"]
                summary += f"  | free={fo['action'][:4]}({fo['success']:.2f})"
            judged = [o["critical_error"] for o in pa.values() if o["critical_error"] is not None]
            crit = sum(judged)
            n_unjudged = len(pa) - len(judged)
            print(
                f"  [{done}/{len(jobs)}] task {tid:<4} {summary}"
                + (f"  crit={crit:.0f}" if crit else "")
                + (f"  unjudgeable={n_unjudged}" if n_unjudged else "")
                + (f"  DROPPED {list(r['dropped_actions'])}" if r["dropped_actions"] else ""),
                flush=True,
            )
            flush()

    arms = [o for r in rows for o in r["per_action"].values()]
    n_prem = sum(1 for o in arms if o.get("unscorable_premature"))
    if arms and n_prem / len(arms) > 0.1:
        print(
            f"\nwarning: {n_prem}/{len(arms)} arms hit --continue-for={args.continue_for}",
            flush=True,
        )

    out = {
        "env": "tau2", "domain": args.domain, "n_checkpoints": len(rows),
        "n_arms": len(arms), "n_unscorable_premature": n_prem,
        "model": model, "elapsed_s": round(time.time() - started, 1),
        "errors": errors, "checkpoints": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    all_arms = [o for r in rows for o in r["per_action"].values()]
    ncrit = sum(1 for o in all_arms if (o["critical_error"] or 0) > 0)
    nattempt = sum(1 for o in all_arms if (o.get("critical_error_attempted") or 0) > 0)
    nunjudged = sum(1 for o in all_arms if o["critical_error"] is None)
    print(
        f"\n{len(rows)} checkpoints in {out['elapsed_s'] / 60:.1f} min "
        f"({len(errors)} errors) | critical: {ncrit} effective, {nattempt} attempted, "
        f"{nunjudged} unjudgeable -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
