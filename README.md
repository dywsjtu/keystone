# KeyStone

**Geometry-Guided Self-Consistency for Physical AI**

KeyStone is a drop-in, **inference-time, model-free** wrapper for diffusion- and
flow-matching physical-AI policies (VLAs and world-action models). At each round
of the standard sample-and-execute loop it:

1. draws **K** candidate action chunks in parallel from a *shared* model context,
2. clusters them in continuous action space, and
3. executes the **medoid of the largest cluster** — a real sampled chunk, not a synthesized average.

No extra model, no training, no learned judge. Two properties make this practical:

- **Latency-preserving.** Action diffusion runs over compact trajectories and is
  memory-bandwidth bound, so the shared context is computed once and the K chains
  run as a single batched forward — adding the K candidates is nearly free.
- **Judge-free selection.** Action chunks are ordered sequences of bounded robot
  commands, so Euclidean distance between chunks reflects *physical* similarity.
  The largest dense region is a principled, model-free notion of consensus.

Across diverse VLAs and WAMs, KeyStone improves task success rates by up to
**13.3%** over single-trajectory sampling with negligible latency overhead, on
par with model-based selectors at no training cost.

> Paper: [Geometry Guided Self-Consistency for Physical AI](https://arxiv.org/abs/2605.08638)

## Install

```bash
pip install -e .          # from this repo
# or, for development:
pip install -e ".[dev]"
```

The core library depends only on PyTorch.

## Quickstart

```python
import torch
from keystone import cluster_medoid_select

# K candidate action chunks, e.g. drawn from K independent noises.
candidates = torch.randn(16, 50, 7)        # (K, T, action_dim)

# Returns ONE real chunk — the medoid of the dominant mode.
chunk = cluster_medoid_select(candidates)  # (T, action_dim)
```

Run the end-to-end demo (no model or GPU required):

```bash
python examples/minimal_quickstart.py
```

## API

| Symbol | Purpose |
|---|---|
| `cluster_medoid_select(candidates, num_clusters=2, distance="l2", auto=False, ...)` | Convenience selector. Returns one real chunk. Accepts `(K, T, D)` or `(K, B, T, D)`. Pass `auto=True` for the auto cluster-count variant. |
| `SelfConsistencyConfig` | All knobs: `num_samples` (K), `aggregation`, `cluster_medoid_num_clusters` (C), `cluster_medoid_auto_min_gap`, `distance`, `action_dim`. |
| `aggregate_actions(actions, config, executed_steps=None)` | Lower-level engine over `(K, B, T, D)`. |
| `select_chunk(candidates, config, ...)` | Like `aggregate_actions` but also accepts unbatched `(K, T, D)`. |
| `expand_kv_cache(kv, K)`, `repeat_batch(t, K)` | Latency-preserving K-batch expansion of a transformer KV cache / prefix tensors. |
| `KeystonePolicy(base_policy, config)` | Generic wrapper implementing the paper's Listing 1 pattern. |

### Selection methods

Both return a **real** sampled chunk (the medoid of the dominant cluster) — never
a synthesized average that could land off the model's action manifold.

| `aggregation` | Notes |
|---|---|
| `medoid` | The global medoid — the chunk minimizing total distance to all others (vector generalization of the median). No clustering. |
| `cluster_medoid` *(default)* | k-means with a fixed cluster count C (`cluster_medoid_num_clusters`, default 2); return the medoid of the largest cluster. |
| `cluster_medoid_auto` | Cluster count detected automatically via a single-linkage largest-gap cut; falls back to the global medoid when the candidates look unimodal (`cluster_medoid_auto_min_gap`). |

`distance` ∈ `{l1, l2, cosine}`. Set `action_dim` to ignore zero-padding beyond
the real action dimensions. Pass `executed_steps` to select over only the
timesteps that actually run before the next replan.

## How it works

```
                      ┌─ noise ε₁ ─┐
   observation ──► encode context ─┤  ...  ├─► K candidate chunks ─► cluster-medoid ─► aₜ*
   (shared, once)      (expand ×K)  └─ noise ε_K ┘   (one batched forward)   selection
```

The selector flattens each candidate chunk and computes one `K×K` pairwise
distance matrix. `cluster_medoid` runs k-means with a fixed cluster count, takes
the largest cluster, and returns its medoid (the within-cluster chunk closest to
the others). `cluster_medoid_auto` instead infers the cluster count from the
largest gap in the sorted pairwise distances, and returns the global medoid when
no clear structure exists. Every step is a batched on-GPU tensor op over only K
compact chunks (≤ a few hundred µs at K≤16).

## Adding KeyStone to your policy

Integration always reduces to *encode-once → expand-to-K → batched-sample →
select*, in one of two styles. The framework-agnostic guide is
[`examples/README_custom.md`](examples/README_custom.md); each style has a
runnable template:

- **Override the sampler** — subclass a policy and replace its action generator
  ([`examples/flow_matching_policy.py`](examples/flow_matching_policy.py); the
  LeRobot SmolVLA / π0.5 pattern).
- **Monkey-patch generate** — intercept a served model's `generate_actions`
  without touching the simulator
  ([`examples/server_monkeypatch.py`](examples/server_monkeypatch.py); the
  vla-eval GR00T / X-VLA / StarVLA pattern).

## Reproducing the paper

The benchmark harnesses used in the paper **import this package directly** —
their policy / server code calls `aggregate_actions` and `expand_kv_cache` from
`keystone`, so the selector is maintained in exactly one place. Each has a
self-contained, step-by-step guide (clone the fork, install `keystone`, run):

- **[`examples/README_lerobot.md`](examples/README_lerobot.md)** — SmolVLA & π0.5
  on LIBERO (LeRobot fork).
- **[`examples/README_vla_eval.md`](examples/README_vla_eval.md)** — GR00T N1.6,
  X-VLA, StarVLA on SimplerEnv (vla-eval fork).

## Tests

```bash
pytest
```

## Citation

```bibtex
@misc{keystone,
      title={Geometry Guided Self-Consistency for Physical AI},
      author={Yinwei Dai and Zhuofu Chen and Lijie Yang and Ravi Netravali},
      year={2026},
      eprint={2605.08638},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.08638},
}
```

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) — the selector was
developed alongside forks of [LeRobot](https://github.com/huggingface/lerobot)
and [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness),
both Apache 2.0.
