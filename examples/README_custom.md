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
chunk = cluster_medoid_select(candidates, C=2, tau=0.3, distance="l2")

# Full control via a config (padding-aware action_dim, executed_steps, method):
cfg = SelfConsistencyConfig(num_samples=K, aggregation="cluster_medoid_guarded", action_dim=7)
chunk = aggregate_actions(candidates_KBTD, cfg, executed_steps=n_action_steps)
```

## Customizing selection

### Tuning the built-in selector

Every knob lives on `SelfConsistencyConfig`; the most common ones are also direct
arguments to `cluster_medoid_select`.

| Knob | Config field | `cluster_medoid_select` arg | What it does |
|---|---|---|---|
| **K** | `num_samples` | *(inferred from input)* | Candidate chunks drawn per round. Higher K → more consensus signal (and more parallel compute, but no extra latency while bandwidth-bound). |
| **Method** | `aggregation` | *(use the config)* | `cluster_medoid_guarded` (default, the paper's τ-guard), `medoid` (global medoid only), or `cluster_medoid` (fixed-C k-means, no guard). |
| **τ** | `unimodal_tau` | `tau` | Unimodality-guard threshold (default 0.3). Lower → cluster more readily; higher → return the global medoid more often. Used only by `cluster_medoid_guarded`. |
| **C** | `cluster_medoid_num_clusters` | `C` | Number of k-means clusters (default 2) for the clustering branch. |
| **Distance** | `distance` | `distance` | `l1`, `l2` (default), or `cosine`. |
| **Padding** | `action_dim` | `action_dim` | Real action dims before zero-padding, so distances ignore padding. |
| **Executed steps** | *(arg to `aggregate_actions`/`select_chunk`)* | `executed_steps` | Select over only the timesteps that actually run before the next replan; the tail is discarded anyway. |

```python
from keystone import SelfConsistencyConfig
cfg = SelfConsistencyConfig(
    num_samples=16,
    aggregation="cluster_medoid_guarded",   # or "medoid" / "cluster_medoid"
    unimodal_tau=0.3,
    cluster_medoid_num_clusters=2,
    distance="l2",
    action_dim=7,                           # ignore padding beyond 7 real dims
)
```

The benchmark harnesses surface the same knobs without code:

- **LeRobot** — CLI flags, e.g. `--policy.self_consistency.num_samples=16
  --policy.self_consistency.aggregation=cluster_medoid_guarded
  --policy.self_consistency.unimodal_tau=0.3`.
- **vla-eval** — a `self_consistency:` block in the server-config YAML
  (`num_samples`, `aggregation`, `cluster_medoid_num_clusters`, `unimodal_tau`,
  `distance`).

### Writing your own selector

Selection is just a pure function of the candidate chunks — KeyStone is not
locked to cluster-medoid. After step 3 you hold a `(K, B, T, D)` (or `(K, T, D)`)
tensor of candidates; **anywhere the templates call `aggregate_actions(...)` /
`cluster_medoid_select(...)`, call your own selector instead.** It only needs to
map the candidates to one chunk per batch element:

```python
import torch

def smoothest_select(candidates):            # (K, T, D) -> (T, D)
    """Pick the lowest-acceleration (smoothest) candidate chunk."""
    accel = candidates.diff(dim=1).diff(dim=1)        # (K, T-2, D)
    energy = accel.flatten(1).pow(2).sum(dim=1)       # (K,)
    return candidates[energy.argmin()]               # a real sampled chunk

# ... in your sample path, replacing the keystone selection call:
chunk = smoothest_select(candidates)         # candidates: (K, T, D)
```

Two things worth preserving from the paper's design, though they're your call:

- **Return a *real* sampled chunk** (index into `candidates`) rather than a
  synthesized/averaged one, so the executed action stays on the model's action
  manifold.
- **Keep it judge-free and cheap** — operate on the candidate geometry (distances,
  smoothness, agreement) so selection stays a sub-millisecond tensor op with no
  extra model.

You can also compose: call `cluster_medoid_select(...)` first and post-filter, or
reuse the public `aggregate_actions` engine for the heavy lifting and only swap
the final rule. (For reference, the built-ins live in
[`../src/keystone/selection.py`](../src/keystone/selection.py).)

See [`minimal_quickstart.py`](minimal_quickstart.py) for a runnable end-to-end
version, and the full API table in the [top-level README](../README.md#api).
