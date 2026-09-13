#!/usr/bin/env python

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _group(examples: list[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = defaultdict(list)
    for e in examples:
        out[e.cp_key].append(e)
    return {k: v for k, v in out.items() if len(v) >= 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["data/labels/labels_native_v6.json", "data/labels/labels_native_q14b.json"])
    ap.add_argument(
        "--eval-labels", nargs="+", default=None,
        help="evaluate on a different label set, held out by family",
    )
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=50, help="steps between evaluations")
    ap.add_argument("--patience", type=int, default=6, help="evals without improvement")
    ap.add_argument("--eval-batch", type=int, default=32, help="batch size for evaluation")
    ap.add_argument("--out", default="results/value")
    ap.add_argument(
        "--val-frac", type=float, default=0.20,
        help="family fraction used for early stopping and checkpoint selection",
    )
    ap.add_argument(
        "--test-frac", type=float, default=0.20,
        help="fraction of families held out for test",
    )
    ap.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="family split seed",
    )
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="trade compute for memory; needed to fit an 8B LoRA on a 24GB card")
    ap.add_argument(
        "--backbone-hint",
        action="store_true",
        help="include the unforced backbone's action in the rendered state",
    )
    ap.add_argument(
        "--cost-profile", default="medium",
        help="cost profile name from configs/cost_profiles.yaml",
    )
    ap.add_argument(
        "--scalar-target", action="store_true",
        help="regress the combined Q instead of the three heads",
    )
    ap.add_argument(
        "--rank-weight",
        type=float,
        default=1.0,
        help="weight on the within-checkpoint pairwise ranking loss (0 = pure regression)",
    )
    ap.add_argument(
        "--centre-targets",
        action="store_true",
        help="subtract each checkpoint's mean from targets and predictions",
    )
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    from sureact.schema import load_cost_profiles
    from sureact.value.dataset import load_examples, split_by_family, split_three_way
    from sureact.value.model import ValueHead

    profile = load_cost_profiles()[args.cost_profile]
    paths = [Path(p) for pat in args.labels for p in glob.glob(pat)]
    paths = [p for p in paths if p.exists()]
    examples = load_examples(paths, cost_profile=args.cost_profile,
                             backbone_hint=args.backbone_hint)
    if not examples:
        print("no examples found", file=sys.stderr)
        return 2
    if args.test_frac > 0:
        train, val, test = split_three_way(
            examples, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.split_seed
        )
    else:
        train, val = split_by_family(examples, val_frac=args.val_frac, seed=args.split_seed)
        test = []
    val_groups = _group(val)
    test_groups = _group(test) if test else {}
    if args.eval_labels:
        ev = load_examples([Path(p) for p in args.eval_labels],
                           cost_profile=args.cost_profile,
                           backbone_hint=args.backbone_hint)
        train_fams = {e.family for e in train}
        ev = [e for e in ev if e.family not in train_fams]
        test_groups = _group(ev)
        print(f"transfer eval: {len(ev)} examples from {len(args.eval_labels)} file(s), "
              f"{len(test_groups)} checkpoints, {len({e.family for e in ev})} families "
              f"(training families excluded)")

    print(f"examples {len(examples)} from {len(paths)} files | "
          f"families {len({e.family for e in examples})} | "
          f"train {len(train)} / val {len(val)} | val checkpoints {len(val_groups)}")

    leaks = [e for e in examples
             if "u_true" in e.text or "per_action" in e.text or "critical_error" in e.text]
    if leaks:
        print(f"FATAL: {len(leaks)} examples contain label text", file=sys.stderr)
        return 1
    overlap = {e.family for e in train} & {e.family for e in val}
    if overlap:
        print(f"FATAL: {len(overlap)} families on both sides of the split", file=sys.stderr)
        return 1

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation="sdpa")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "v_proj"])
    peft_model = get_peft_model(base, lora)
    if args.grad_checkpointing:
        peft_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        peft_model.enable_input_require_grads()
    model = ValueHead(peft_model, base.config.hidden_size).cuda()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def q_of(p: dict[str, float]) -> float:
        return p["success"] - args.lam * p["cost"] - args.eta * p["critical_error"]

    def evaluate(groups: dict | None = None) -> dict[str, float]:
        """Evaluate decisions on held-out families."""
        from sureact.value.metrics import decision_metrics
        from sureact.value.model import predict

        groups = val_groups if groups is None else groups
        model.eval()
        texts = [e.text for g in groups.values() for e in g]
        preds = predict(model, tok, texts, max_len=args.max_len, batch=args.eval_batch)
        model.train()

        return decision_metrics(groups, preds, q_of)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    best_util, since_best, step, history = float("-inf"), 0, 0, []
    best_adapter: dict = {}

    train_groups = _group(train)
    group_keys = list(train_groups)

    for epoch in range(args.epochs):
        import random
        random.Random(epoch).shuffle(group_keys)
        batches, cur = [], []
        for k in group_keys:
            g = train_groups[k]
            if cur and len(cur) + len(g) > args.batch:
                batches.append(cur)
                cur = []
            cur.extend(g)
        if cur:
            batches.append(cur)
        for batch in batches:
            enc = tok([b.text for b in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_len).to("cuda")
            y = torch.tensor([b.targets() for b in batch], dtype=torch.float32).cuda()
            w = torch.tensor([b.target_mask() for b in batch], dtype=torch.float32).cuda()
            pred = model(enc["input_ids"], enc["attention_mask"])

            pred_r, y_r = pred.float(), y
            if args.centre_targets:
                by_cp_c: dict[str, list[int]] = defaultdict(list)
                for i, b in enumerate(batch):
                    by_cp_c[b.cp_key].append(i)
                pred_r, y_r = pred.float().clone(), y.clone()
                for idx in by_cp_c.values():
                    if len(idx) < 2:
                        continue
                    ii = torch.tensor(idx, device=pred.device)
                    wm = w[ii]
                    denom = wm.sum(dim=0, keepdim=True).clamp(min=1.0)
                    y_r[ii] = y[ii] - (wm * y[ii]).sum(dim=0, keepdim=True) / denom
                    pf = pred.float()[ii]
                    pred_r[ii] = pf - (wm * pf).sum(dim=0, keepdim=True) / denom
            if args.scalar_target:
                q_p = (pred_r[:, 0] - args.lam * pred_r[:, 2] - args.eta * pred_r[:, 1])
                q_t = y_r[:, 0] - args.lam * y_r[:, 2] - args.eta * y_r[:, 1]
                loss = ((q_p - q_t) ** 2).mean()
            else:
                loss = (w * (pred_r - y_r) ** 2).sum() / w.sum().clamp(min=1.0)

            if args.rank_weight:
                by_cp: dict[str, list[int]] = defaultdict(list)
                for i, b in enumerate(batch):
                    by_cp[b.cp_key].append(i)
                q_pred = (
                    pred[:, 0].float()
                    - args.lam * pred[:, 2].float()
                    - args.eta * pred[:, 1].float()
                )
                q_true = y[:, 0] - args.lam * y[:, 2] - args.eta * y[:, 1]
                pairs = []
                for idx in by_cp.values():
                    for a in range(len(idx)):
                        for b_ in range(a + 1, len(idx)):
                            pairs.append((idx[a], idx[b_]))
                if pairs:
                    ia = torch.tensor([p[0] for p in pairs], device=pred.device)
                    ib = torch.tensor([p[1] for p in pairs], device=pred.device)
                    dt = q_true[ia] - q_true[ib]
                    dp = q_pred[ia] - q_pred[ib]
                    rank = torch.relu(dt.abs() - torch.sign(dt) * dp).mean()
                    loss = loss + args.rank_weight * rank
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1

            if step % args.eval_every == 0:
                m = evaluate()
                history.append({"step": step, "loss": float(loss), **m})
                gap = m["oracle"] - m["best_fixed"]
                got = m["decision_util"] - m["best_fixed"]
                print(f"  step {step:5} loss {float(loss):.4f} | decision_util "
                      f"{m['decision_util']:+.3f}  vs best_fixed {m['best_fixed']:+.3f} "
                      f"({got:+.3f} of {gap:+.3f} available)  top1 {m['top1']:.2f}", flush=True)
                if m["decision_util"] > best_util:
                    best_util, since_best = m["decision_util"], 0
                    model.encoder.save_pretrained(outdir / "lora")
                    torch.save(model.head.state_dict(), outdir / "head.pt")
                    best_adapter = {k: v.detach().to("cpu", copy=True)
                                    for k, v in model.encoder.state_dict().items()
                                    if "lora" in k.lower()}
                else:
                    since_best += 1
                    if since_best >= args.patience:
                        print(f"  early stop: {args.patience} evals without improvement")
                        epoch = args.epochs
                        break
        if epoch >= args.epochs:
            break

    final = history[-1] if history else {}

    test_metrics = {}
    if test_groups:
        head_path = outdir / "head.pt"
        if head_path.exists():
            model.head.load_state_dict(torch.load(head_path, map_location="cpu"))
            model.head.to(next(model.encoder.parameters()).device)
        if best_adapter:
            dev = next(model.encoder.parameters()).device
            enc = model.encoder.state_dict()
            missing = [k for k in best_adapter if k not in enc]
            if missing:
                print(f"  WARNING: {len(missing)} saved adapter tensors have no home in the "
                      "encoder; NOT restoring, and this test figure is the hybrid one")
            else:
                model.encoder.load_state_dict(
                    {**enc, **{k: v.to(dev) for k, v in best_adapter.items()}}, strict=False)
                print(f"  restored the val-selected adapter ({len(best_adapter)} tensors) "
                      "before the test evaluation")
        test_metrics = evaluate(test_groups)
        tgap = test_metrics["oracle"] - test_metrics["best_fixed"]
        tgot = test_metrics["decision_util"] - test_metrics["best_fixed"]
        print(f"\nTEST (never used for stopping or selection, {len(test_groups)} checkpoints):")
        print(f"  decision_util {test_metrics['decision_util']:+.3f}  "
              f"best_fixed {test_metrics['best_fixed']:+.3f}  oracle {test_metrics['oracle']:+.3f}")
        print(f"  gain {tgot:+.4f} of {tgap:+.4f} available"
              + (f" = {tgot / tgap:.0%}" if tgap > 1e-9 else "  (no headroom on this split)"))

    (outdir / "training.json").write_text(json.dumps(
        {"args": vars(args), "history": history, "best_decision_util": best_util,
         "test": test_metrics}, indent=2))
    if history:
        best = max(history, key=lambda h: h["decision_util"])
        gap = best["oracle"] - best["best_fixed"]
        got = best["decision_util"] - best["best_fixed"]
        print(f"\nbest: decision_util {best['decision_util']:+.3f}  "
              f"best_fixed {best['best_fixed']:+.3f}  oracle {best['oracle']:+.3f}")
        print(f"captured {got:+.3f} of {gap:+.3f} available"
              f" = {got / gap:.0%}" if gap > 1e-9 else "  (no headroom on this split)")
        print(f"  MAE success {best.get('mae_success', float('nan')):.3f} "
              f"critical {best.get('mae_critical_error', float('nan')):.3f} "
              f"cost {best.get('mae_cost', float('nan')):.3f}")
    print(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
