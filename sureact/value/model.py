from __future__ import annotations

from pathlib import Path
from typing import Any

TARGETS = ("success", "critical_error", "cost")


class ValueHead:

    def __new__(cls, encoder: Any, hidden: int) -> Any:  # noqa: D102
        import torch

        class _ValueHead(torch.nn.Module):
            def __init__(self, enc: Any, h: int) -> None:
                super().__init__()
                self.encoder = enc
                self.head = torch.nn.Linear(h, 3)

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                h = out.last_hidden_state
                mask = attention_mask.unsqueeze(-1).to(h.dtype)
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
                return self.head(pooled.to(self.head.weight.dtype))

        return _ValueHead(encoder, hidden)


def load_value_model(outdir: str | Path, base_model: str, device: str = "cuda") -> Any:
    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    outdir = Path(outdir)
    head_path = outdir / "head.pt"
    if not head_path.exists():
        raise FileNotFoundError(
            f"{head_path} not found -- train first with scripts/train_value.py. "
            "Refusing to score with an untrained head."
        )

    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModel.from_pretrained(base_model, dtype=torch.bfloat16,
        attn_implementation="sdpa")
    enc = PeftModel.from_pretrained(base, str(outdir / "lora"))
    model = ValueHead(enc, base.config.hidden_size)
    model.head.load_state_dict(torch.load(head_path, map_location="cpu"))
    model.to(device).eval()
    return model, tok


def predict(model: Any, tok: Any, texts: list[str], max_len: int = 1024,
            batch: int = 16, device: str = "cuda") -> list[dict[str, float]]:
    import torch

    out: list[dict[str, float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            encd = tok(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len
            ).to(device)
            pred = model(encd["input_ids"], encd["attention_mask"]).float().cpu()
            for row in pred:
                out.append({k: float(v) for k, v in zip(TARGETS, row.tolist())})
    return out
