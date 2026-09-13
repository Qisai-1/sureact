# SURE-Act

Counterfactual labelling of intervention decisions for tool-using language agents, and a learned
controller that picks among them.

At a decision point the environment is snapshotted; for each legal meta-action (`ASK`, `SEARCH`,
`ACT_REVERSIBLE`, `COMMIT_IRREVERSIBLE`, `ABSTAIN_HANDOFF`) the snapshot is restored, that action
is forced for one turn, the unmodified agent continues to termination, and the branch is scored
by the environment's own evaluator:

```
q = success - lambda * cost - eta * critical_error
```

A Qwen3-1.7B + LoRA value model is trained on these labels and scored against fixed-action and
scalar-confidence baselines on family-held-out splits.

## Layout

```
configs/      cost profiles and tool -> meta-action manifests
sureact/
  schema.py         meta-actions, cost profiles
  scoring.py        label loading and the utility q
  envs/             ToolSandbox role adapters and fork helpers
  label/            counterfactual labelling (ToolSandbox, tau2), matched perturbations
  value/            dataset rendering, value model, decision metrics
  baselines/        fixed, scalar-confidence, cost-aware and learned controllers
scripts/      entry points (below)
slurm/        batch templates; each starts its own vLLM server and stops it on exit
tests/
data/
  labels/           counterfactual label files
  confidence_headline.json
```

## Install

```bash
pip install -e ".[train,label,dev]"

git clone https://github.com/apple/ToolSandbox.git third_party/ToolSandbox
git -C third_party/ToolSandbox checkout 165848b9a78cead7ca7fe7c89c688b58e6501219
pip install -e third_party/ToolSandbox

git clone https://github.com/sierra-research/tau2-bench.git third_party/tau2-bench
git -C third_party/tau2-bench checkout 363133ada1936491fb5bcec33cd62c3518a99f65
pip install -e third_party/tau2-bench
```

Labelling and confidence scoring need an OpenAI-compatible server (vLLM). The scripts read:

| variable | meaning |
|---|---|
| `SUREACT_LLM_BASE_URL` | server URL, e.g. `http://localhost:8000/v1` |
| `SUREACT_LLM_MODEL` | served model, e.g. `Qwen/Qwen3-8B` |
| `SUREACT_TEMPERATURE`, `SUREACT_SEED` | set to `0` so forked branches are deterministic |
| `HOSTED_VLLM_API_BASE`, `HOSTED_VLLM_API_KEY` | the same server, for tau2 (LiteLLM) |

`slurm/vllm.sh` starts the server and sets all of them.

## Pipeline

```bash
# 1. counterfactual labels
python scripts/label_toolsandbox.py --scenarios 80 --continue-for 30 --workers 12 --out results/labels_toolsandbox.json
python scripts/label_tau2.py --domain retail --tasks 114 --fork-at 4 --continue-for 45 --out results/labels_tau2_retail.json

# 2. inspect labels: optimal-action distribution and headroom over the best fixed action
python scripts/score_labels.py data/labels/labels_native_v6.json
python scripts/headroom.py data/labels/labels_native_v6.json data/labels/labels_native_q14b.json

# 3. scalar confidence signals for the baselines
python scripts/compute_confidence.py data/labels/labels_native_v6.json data/labels/labels_native_q14b.json --out results/confidence.json

# 4. train one controller per split seed
python scripts/train_value.py \
  --labels data/labels/labels_native_v6.json data/labels/labels_native_q14b.json \
  --model Qwen/Qwen3-1.7B --epochs 4 --batch 8 --max-len 1024 --lr 1e-4 \
  --val-frac 0.20 --test-frac 0.20 --split-seed 0 --grad-checkpointing \
  --centre-targets --rank-weight 0 --out results/value_s0

# 5. controller and all baselines, one harness, same splits
python scripts/run_baselines.py data/labels/labels_native_v6.json data/labels/labels_native_q14b.json \
  --confidence data/confidence_headline.json --test-frac 0.20 --seeds 120 \
  --value-model-pattern 'results/value_s{seed}' --value-base Qwen/Qwen3-1.7B --out results/headtohead.json
```

On Slurm: `sbatch slurm/train_value.sbatch` (array over seeds 0-119, resumable), then
`sbatch slurm/run_baselines.sbatch`. Pass `--account` / `--partition` on the command line.

## Data

| file | contents |
|---|---|
| `labels_native_v6.json`, `labels_native_q14b.json` | ToolSandbox labels, Qwen3-8B and Qwen3-14B agents; training data for the reported results |
| `labels_native_v7big.json` | independent ToolSandbox label set (748 checkpoints, 114 families) |
| `labels_tau2_retail_v6.json`, `labels_tau2_airline_v6.json` | tau2-bench labels |
| `confidence_headline.json` | verbalized / self-consistency / entropy signals for the two training label files |

## Tests

```bash
pytest -q
```
