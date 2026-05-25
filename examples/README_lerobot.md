# Reproducing KeyStone on LeRobot (SmolVLA & π0.5, LIBERO)

The SmolVLA / π0.5 integration lives in a LeRobot fork. It overrides each
policy's action sampler to draw K candidates in one batched forward and select
one with KeyStone (Pattern A in [`README_custom.md`](README_custom.md)). The
selector itself is imported from this `keystone` package.

- **Fork:** [`dywsjtu/lerobot-self-consistency`](https://github.com/dywsjtu/lerobot-self-consistency) @ branch **`keystone`** — forked from upstream [`huggingface/lerobot`](https://github.com/huggingface/lerobot)
- **Benchmark:** LIBERO (`libero_spatial`, `libero_object`, `libero_10`, `libero_goal`)

| Model | `--policy.path` | `n_action_steps` | Selector in the paper |
|---|---|---|---|
| SmolVLA | `HuggingFaceVLA/smolvla_libero` | 25 | `cluster_medoid`, K=16 |
| π0.5 | `lerobot/pi05_libero_finetuned` | 50 | `cluster_medoid_guarded`, K=16 |

## Setup

```bash
git clone -b keystone git@github.com:dywsjtu/lerobot-self-consistency.git
cd lerobot-self-consistency
uv sync --all-extras                 # installs LIBERO + SmolVLA + π0.5 deps
source .venv/bin/activate

# KeyStone's selector lives in the standalone package — install it into this env:
uv pip install -e /path/to/keystone  # or: pip install -e /path/to/keystone
```

`uv sync --all-extras` reproduces the environment from the fork's lockfile. For
base LeRobot installation notes and troubleshooting (system deps, CUDA, MuJoCo),
see upstream: https://huggingface.co/docs/lerobot/installation.

## Run

The fork ships ready-made scripts that sweep the four LIBERO suites:

```bash
bash scripts/run_libero_smolvla.sh   # SmolVLA, K=16 cluster_medoid
bash scripts/run_libero_pi05.sh      # π0.5,   K=16 cluster_medoid_guarded
```

Each invokes `lerobot-eval`. The self-consistency knobs are just policy flags,
so you can run one cell directly:

```bash
# KeyStone (K=16, cluster-medoid)
lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --policy.n_action_steps=25 \
  --policy.self_consistency.num_samples=16 \
  --policy.self_consistency.aggregation=cluster_medoid \
  --env.type=libero --env.task=libero_spatial \
  --eval.n_episodes=10 --eval.use_async_envs=false \
  --seed=0 \
  --output_dir=outputs/eval/smolvla_libero_spatial_cluster_medoid_k16_seed0
```

```bash
# K=1 baseline — simply omit the self_consistency flags
lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --policy.n_action_steps=25 \
  --env.type=libero --env.task=libero_spatial \
  --eval.n_episodes=10 --eval.use_async_envs=false \
  --seed=0 \
  --output_dir=outputs/eval/smolvla_libero_spatial_baseline_k1_seed0
```

Relevant flags:

| Flag | Meaning |
|---|---|
| `--policy.self_consistency.num_samples` | K, candidate chunks per round. |
| `--policy.self_consistency.aggregation` | `cluster_medoid_guarded` (default), `medoid`, or `cluster_medoid`. |
| `--policy.n_action_steps` | Executed steps before the next replan (selection runs over these). |

## Parse results

```bash
python scripts/parse_results.py outputs/eval     # mean ± std across seeds, Δ vs baseline
```

## Where the integration lives in the fork

- `src/lerobot/policies/{smolvla,pi05}/modeling_*.py` →
  `_sample_actions_self_consistent`: encode prefix once, expand the KV cache to
  K, run one batched denoise, then call `keystone.aggregate_actions`.
- `src/lerobot/policies/self_consistency.py` → thin re-export shim importing
  `SelfConsistencyConfig`, `aggregate_actions`, `expand_kv_cache` from `keystone`.
