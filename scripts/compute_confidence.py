#!/usr/bin/env python

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROMPT_VERBALIZED = (
    "You are an agent deciding what to do next.\n\n{state}\n\n"
    "How confident are you that you could complete the user's request CORRECTLY right now, "
    "without gathering any more information?\n"
    "Answer with a single integer from 0 to 100 and nothing else."
)

PROMPT_NEXT_ACTION = (
    "You are an agent deciding what to do next.\n\n{state}\n\n"
    "Which ONE of these should you do next? Answer with exactly one option and nothing "
    "else.\nOptions: {legal}"
)


def _client(base_url: str, api_key: str) -> Any:
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)


def _verbalized(client: Any, model: str, state: str) -> float | None:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT_VERBALIZED.format(state=state)}],
        temperature=0.0,
        max_tokens=8,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = (r.choices[0].message.content or "").strip()
    return _parse_confidence(text)


def _parse_confidence(text: str) -> float | None:
    import re

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|/\s*(\d+)|out\s+of\s+(\d+))?", text)
    if not m:
        return None
    val = float(m.group(1))
    denom = m.group(2) or m.group(3)
    if denom:
        d = float(denom)
        return max(0.0, min(1.0, val / d)) if d > 0 else None
    return max(0.0, min(1.0, val if val <= 1.0 and "." in m.group(1) else val / 100.0))


def _self_consistency(
    client: Any, model: str, state: str, legal: list[str], k: int
) -> float | None:
    prompt = PROMPT_NEXT_ACTION.format(state=state, legal=", ".join(legal))
    votes: list[str] = []
    for i in range(k):
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            seed=i,
            max_tokens=12,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = (r.choices[0].message.content or "").strip().upper()
        for a in legal:
            if a in text:
                votes.append(a)
                break
    if not votes:
        return None
    return collections.Counter(votes).most_common(1)[0][1] / k


def _entropy(client: Any, model: str, state: str, legal: list[str]) -> float | None:
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": PROMPT_NEXT_ACTION.format(state=state, legal=", ".join(legal))}
        ],
        temperature=0.0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=20,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    lp = r.choices[0].logprobs
    if not lp or not lp.content:
        return None
    tops = lp.content[0].top_logprobs
    if not tops:
        return None
    ps = [math.exp(t.logprob) for t in tops]
    total = sum(ps) or 1.0
    ps = [p / total for p in ps]
    h = -sum(p * math.log(p) for p in ps if p > 0)
    h_max = math.log(len(ps)) or 1.0
    return 1.0 - h / h_max


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5, help="samples for self-consistency")
    ap.add_argument(
        "--methods",
        default="verbalized,self_consistency,entropy",
        help="comma-separated subset to compute",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--backbone-hint",
        action="store_true",
        help="render the backbone's proposed action, as the value model sees it",
    )
    args = ap.parse_args()

    from sureact.baselines.scalar import checkpoint_key
    from sureact.value.dataset import render_state

    base_url = os.environ.get("SUREACT_LLM_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("SUREACT_LLM_MODEL", "Qwen/Qwen3-8B")
    client = _client(base_url, os.environ.get("HOSTED_VLLM_API_KEY", "EMPTY"))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    cps: list[dict[str, Any]] = []
    for pat in args.paths:
        for p in map(Path, glob.glob(pat)):
            blob = json.loads(p.read_text())
            for cp in blob.get("checkpoints", []):
                cp.setdefault("env", blob.get("env"))
                cp["_src"] = p.stem
                cps.append(cp)
    if args.limit:
        cps = cps[: args.limit]

    keys = [checkpoint_key(cp) for cp in cps]
    dupes = collections.Counter(keys)
    clashing = {k: n for k, n in dupes.items() if n > 1}
    if clashing:
        sample = list(clashing.items())[:3]
        print(
            f"duplicate checkpoint keys: {len(clashing)}, e.g. {sample}",
            file=sys.stderr,
        )
        return 2
    print(f"{len(cps)} checkpoints | model={model} | methods={methods}", flush=True)

    out: dict[str, dict[str, float]] = {m: {} for m in methods}
    fails: collections.Counter = collections.Counter()
    for i, cp in enumerate(cps, 1):
        key = checkpoint_key(cp)
        legal = sorted(
            a for a, o in (cp.get("per_action") or {}).items()
            if not o.get("unscorable_premature")
        ) or list(cp.get("legal_actions") or [])
        if len(legal) < 2:
            continue
        cp = dict(cp, legal_actions=legal)
        state = render_state(cp, "(deciding)", backbone_hint=args.backbone_hint)
        try:
            if "verbalized" in methods:
                v = _verbalized(client, model, state)
                if v is None:
                    fails["verbalized"] += 1
                else:
                    out["verbalized"][key] = v
            if "self_consistency" in methods:
                v = _self_consistency(client, model, state, legal, args.k)
                if v is None:
                    fails["self_consistency"] += 1
                else:
                    out["self_consistency"][key] = v
            if "entropy" in methods:
                v = _entropy(client, model, state, legal)
                if v is None:
                    fails["entropy"] += 1
                else:
                    out["entropy"][key] = v
        except Exception as exc:  # noqa: BLE001
            fails[f"error:{type(exc).__name__}"] += 1
        if i % 20 == 0 or i == len(cps):
            Path(args.out).write_text(json.dumps({"model": model, "scores": out}, indent=1))
            print(f"  [{i}/{len(cps)}] " + " ".join(f"{m}={len(out[m])}" for m in methods), flush=True)

    Path(args.out).write_text(json.dumps({"model": model, "scores": out}, indent=1))
    print(f"wrote {args.out}")
    for m in methods:
        cov = len(out[m]) / len(cps) if cps else 0
        print(f"  {m:18} coverage {len(out[m])}/{len(cps)} = {cov:.0%}")
    if fails:
        print(f"  failures: {dict(fails)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
