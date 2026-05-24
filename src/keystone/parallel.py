"""Utilities for latency-preserving K-parallel candidate sampling.

Keystone's efficiency comes from computing the (expensive) shared context once
and running the K denoising chains as a single batched forward pass. These
helpers expand shared prefix tensors and the transformer KV cache to width K
*without materializing K copies* of the context: they use ``torch.expand``, so
the K candidates share the same underlying storage.

Two batch layouts are common; pick the matching reshape after sampling:

- **Block / tiled** (``repeat_batch``, ``torch.repeat``): row order is
  ``[k0_b0, k0_b1, ..., k1_b0, ...]``. Reshape candidates back with
  ``out.reshape(K, B, T, D)``.
- **Interleaved** (``torch.repeat_interleave(K, dim=0)``): row order is
  ``[b0_k0, b0_k1, ..., b1_k0, ...]``. Reshape back with
  ``out.view(B, K, T, D).permute(1, 0, 2, 3)``.

Both are correct as long as the post-sampling reshape matches the expansion.
"""

from __future__ import annotations

import torch
from torch import Tensor


def repeat_batch(t: Tensor, K: int) -> Tensor:
    """Tile a tensor K times along the batch dim (dim=0) via expand + reshape.

    (B, ...) -> (K*B, ...) in *block* layout: rows ``[0:B]`` are replica 0,
    ``[B:2B]`` replica 1, etc. Uses ``expand`` (no copy) then ``reshape``.
    """
    shape = t.shape
    return t.unsqueeze(0).expand(K, *shape).reshape(K * shape[0], *shape[1:])


def expand_kv_cache(past_key_values, K: int):
    """Expand a transformer KV cache by tiling each entry K times along batch.

    All K candidates attend to the same cached prefix, so the cache batch dim
    must become K*B (block layout, matching :func:`repeat_batch`). Supports the
    common cache containers across transformers versions.

    Args:
        past_key_values: KV cache (DynamicCache, dict, list, or tuple).
        K: number of candidates.

    Returns:
        The expanded KV cache, same container type as the input.
    """
    if hasattr(past_key_values, "layers"):
        # transformers >=5.x DynamicCache: layers[i].keys / .values
        from transformers.cache_utils import DynamicCache

        expanded = DynamicCache()
        for layer in past_key_values.layers:
            expanded.update(
                repeat_batch(layer.keys, K),
                repeat_batch(layer.values, K),
                layer_idx=len(expanded),
            )
        return expanded
    elif hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        # transformers <5.x DynamicCache: key_cache / value_cache lists
        from transformers.cache_utils import DynamicCache

        expanded = DynamicCache()
        for i in range(len(past_key_values.key_cache)):
            expanded.key_cache.append(repeat_batch(past_key_values.key_cache[i], K))
            expanded.value_cache.append(repeat_batch(past_key_values.value_cache[i], K))
        return expanded
    elif isinstance(past_key_values, dict):
        # Dict format: {layer_idx: {"key_states": ..., "value_states": ...}}
        return {
            layer_idx: {
                "key_states": repeat_batch(layer_kv["key_states"], K),
                "value_states": repeat_batch(layer_kv["value_states"], K),
            }
            for layer_idx, layer_kv in past_key_values.items()
        }
    elif isinstance(past_key_values, (list, tuple)):
        new_cache = []
        for layer_cache in past_key_values:
            if isinstance(layer_cache, tuple) and len(layer_cache) == 2:
                k, v = layer_cache
                new_cache.append((repeat_batch(k, K), repeat_batch(v, K)))
            else:
                new_cache.append(layer_cache)
        return type(past_key_values)(new_cache) if isinstance(past_key_values, tuple) else new_cache
    else:
        raise TypeError(f"Unsupported KV cache type: {type(past_key_values)}")
