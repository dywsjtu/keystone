# Adding KeyStone to your own policy

KeyStone is framework-agnostic: it only needs the K candidate action chunks as a
tensor. Integration always reduces to the same four steps —

1. **Encode the shared context once** (VLM features / KV cache / world embedding).
2. **Expand it to a K-batch** without materializing K copies
   (`expand_kv_cache` / `repeat_interleave` / `torch.expand`).
3. **Sample K candidates** by running the denoising / flow loop as one batched
   forward over `K*B` rows with K independent noises.
4. **Select** with `aggregate_actions(...)` / `cluster_medoid_select(...)` and
   return one chunk in the policy's original output format.

The only model-specific work is steps 1–3 (you already have a single-sample
version of these). Step 4 is one call into `keystone`.

## Two patterns, two templates

Pick by how much access you have to the policy:

| | Pattern A — override the sampler | Pattern B — monkey-patch generate |
|---|---|---|
| **When** | You can subclass the policy (own training code, LeRobot-style). | The model is served / black-box behind a `generate_actions`. |
| **Template** | [`flow_matching_policy.py`](flow_matching_policy.py) | [`server_monkeypatch.py`](server_monkeypatch.py) |
| **Used by** | LeRobot SmolVLA / π0.5 | vla-eval GR00T / X-VLA / StarVLA |
| **Expansion** | `torch.cat` / `.repeat` (**block** layout) | `repeat_interleave` (**interleaved** layout) |
| **Reshape** | `result.reshape(K, B, ...)` | `result.view(B, K, ...).permute(1, 0, ...)` |

> **Batch layout must match.** The reshape back to `(K, B, T, D)` has to mirror
> how you expanded the batch. Block expansion (`cat`/`repeat`) → `reshape(K, B, …)`;
> interleaved expansion (`repeat_interleave`) → `view(B, K, …).permute(1, 0, …)`.
> Mismatching them silently mixes candidates across batch elements. The helpers
> in [`../src/keystone/parallel.py`](../src/keystone/parallel.py) use the block
> layout.

## The selection call

Whatever you do in steps 1–3, step 4 is one of:

```python
from keystone import SelfConsistencyConfig, aggregate_actions, cluster_medoid_select

# Convenience: unbatched (K, T, D) or batched (K, B, T, D) in, one chunk out.
chunk = cluster_medoid_select(candidates, num_clusters=2, distance="l2")

# Full control via a config (padding-aware action_dim, executed_steps, method):
cfg = SelfConsistencyConfig(num_samples=K, aggregation="cluster_medoid", action_dim=7)
chunk = aggregate_actions(candidates_KBTD, cfg, executed_steps=n_action_steps)
```

- `aggregation`: `medoid` (global medoid), `cluster_medoid` (fixed-C k-means,
  default), or `cluster_medoid_auto` (auto cluster count with a unimodality
  fall-back). All return a **real** sampled chunk, never an average.
- `action_dim`: set this if your chunks are zero-padded beyond the real action
  dims, so distances ignore the padding.
- `executed_steps`: select over only the steps that actually run before the next
  replan (the tail is discarded anyway).

See [`minimal_quickstart.py`](minimal_quickstart.py) for a runnable end-to-end
version, and the full API table in the [top-level README](../README.md#api).
