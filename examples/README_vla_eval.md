# Reproducing KeyStone on the VLA evaluation harness (SimplerEnv)

This harness serves a model behind a `generate_actions` endpoint and runs a
sharded benchmark against it. KeyStone is wired in by monkey-patching that
endpoint to draw K candidates in one batched forward and select one (Pattern B
in [`README_custom.md`](README_custom.md)). The selector is imported from this
`keystone` package.

- **Fork:** [`dywsjtu/vla-evaluation-harness`](https://github.com/dywsjtu/vla-evaluation-harness) @ branch **`keystone`** — forked from upstream [`allenai/vla-evaluation-harness`](https://github.com/allenai/vla-evaluation-harness)
- **Benchmark:** SimplerEnv (WidowX, Google Robot)

| Model | Benchmark | K in the paper |
|---|---|---|
| X-VLA (bf16) | SimplerEnv WidowX | 4 |
| GR00T N1.6 | SimplerEnv WidowX | 4 |
| GR00T N1.6 | SimplerEnv Google Robot | 4 |
| StarVLA Qwen3-GR00T | SimplerEnv WidowX | 16 |

## Setup

```bash
git clone -b keystone git@github.com:dywsjtu/vla-evaluation-harness.git
cd vla-evaluation-harness
uv sync --python 3.11 --all-extras --dev
source .venv/bin/activate

# KeyStone's selector lives in the standalone package — install it into this env:
uv pip install -e /path/to/keystone  # or: pip install -e /path/to/keystone

# The runners reference repos + HF cache via ${WORKSPACE} — set it first:
export WORKSPACE=$HOME/projects       # parent dir holding this repo + cache/
```

For the base harness setup that this fork inherits — building the SimplerEnv
benchmark Docker images, fetching model weights, and the `vla-eval serve` / `run`
mechanics — follow the upstream README:
https://github.com/allenai/vla-evaluation-harness. The steps below are the
KeyStone-specific additions on top of that.

GR00T / StarVLA need their upstream caches under `~/.cache/vla-eval/` patched for
K-noise sampling; each server reapplies the one-line noise-tiling patch
automatically on load (`_install_k_noise_patch`).

## How K is configured

A K=1 baseline is the unmodified server config. A KeyStone cell adds a
`self_consistency:` block — these YAMLs are already in the repo at
`configs/model_servers/<model>/<suite>_sc_k<K>_<agg>.yaml`:

```yaml
extends: simpler_google_robot.yaml
args:
  self_consistency:
    num_samples: 16             # K
    aggregation: "cluster_medoid"   # or medoid / cluster_medoid_auto
    cluster_medoid_num_clusters: 2
    distance: "l2"
```

## Run

Per-pair sweep — starts the server, runs the sharded benchmark, tears the
server down, advances to the next cell:

```bash
./scripts/run_sc_sweep.sh \
  --pair xvla-simpler-widowx-bf16 \
  --server-config configs/model_servers/xvla/simpler_widowx_bf16.yaml \
  --bench-config configs/simpler_xvla_tasks.yaml \
  --shards 12 \
  --ks "1 4 8 16"
```

All pairs in one shot, sequential on one GPU:

```bash
./scripts/run_all_sc_sweeps.sh                              # GPU 0, port 8000
GPU=1 PORT=8001 ./scripts/run_all_sc_sweeps.sh xvla starvla # filtered, parallel-safe
KS=1 ./scripts/run_all_sc_sweeps.sh                         # K=1 baseline only
```

Output: `results/<pair>_sc/<cell>/merged_*.json`; logs at `results/logs/sc-<pair>.log`.

## Parse results

`scripts/parse_sc_results.py` recognises the `<pair>_sc[-N]` directory pattern
and prints per-pair tables; `--merge` collapses repeated runs into mean ± std.

```bash
python scripts/parse_sc_results.py --results-dir results --merge
python scripts/parse_sc_results.py --pair xvla --pair starvla --json out.json
```

## Latency benchmark (optional)

```bash
python scripts/benchmark_sc_latency.py \
  --server-config configs/model_servers/xvla/simpler_widowx_bf16.yaml \
  --suite-tag xvla-simpler_widowx-bf16 \
  --ks 1 4 8 16 --warmup 3 --iters 10
python scripts/plot_sc_latency.py -o scripts/k-noise-latency.pdf
```

## Where the integration lives in the fork

- Each server's `_install_k_noise_patch` monkey-patches its `generate_actions`
  to do K-noise + prefix reuse, then calls `keystone.aggregate_actions`.
- `src/vla_eval/model_servers/self_consistency.py` → thin re-export shim
  importing `SelfConsistencyConfig`, `aggregate_actions`, `expand_kv_cache` from
  `keystone`.
