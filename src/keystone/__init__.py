"""Keystone: geometry-guided self-consistency for diffusion/flow-matching physical AI.

Keystone is a drop-in, model-free inference-time wrapper. At each round of the
sample-and-execute loop it draws K candidate action chunks in parallel from a
shared model context, clusters them in continuous action space, and executes the
medoid of the largest cluster -- a real sampled chunk, no extra model required.

Quickstart
----------
>>> import torch
>>> from keystone import cluster_medoid_select
>>> candidates = torch.randn(16, 50, 7)        # (K, T, action_dim)
>>> chunk = cluster_medoid_select(candidates)  # (T, action_dim) -- one real chunk

Lower-level API
---------------
>>> from keystone import SelfConsistencyConfig, aggregate_actions
>>> cfg = SelfConsistencyConfig(num_samples=16, aggregation="cluster_medoid")
>>> chunk = aggregate_actions(candidates.unsqueeze(1), cfg)  # (B, T, D)
"""

from .config import SelfConsistencyConfig
from .parallel import expand_kv_cache, repeat_batch
from .selection import aggregate_actions, cluster_medoid_select, select_chunk
from .wrapper import KeystonePolicy, SamplablePolicy

__all__ = [
    "SelfConsistencyConfig",
    "aggregate_actions",
    "cluster_medoid_select",
    "select_chunk",
    "expand_kv_cache",
    "repeat_batch",
    "KeystonePolicy",
    "SamplablePolicy",
]

__version__ = "0.1.0"
