"""A drop-in wrapper that turns a single-sample policy into a Keystone policy.

This is the generic integration pattern from the paper (Listing 1). It does not
modify model weights, retrain anything, or change the downstream control loop:
at each round it encodes the shared context once, expands it to a K-batch,
samples K candidate chunks in one batched forward, and returns the cluster-medoid
chunk in the policy's original output format.

Most real policies fuse the encode and denoise steps and manage their own KV
cache, so production integrations override the policy's action-generation method
directly (see ``examples/``). Use this wrapper when your policy already exposes
clean ``encode`` / ``sample_actions`` hooks, or as a reference for the pattern.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import Tensor

from .config import SelfConsistencyConfig
from .selection import aggregate_actions


class SamplablePolicy(Protocol):
    """Minimal interface a base policy must expose to be wrapped by Keystone."""

    noise_shape: tuple[int, ...]
    """Shape of the noise tensor for a single candidate, excluding the batch
    dim, e.g. ``(chunk_size, action_dim)``."""

    def encode(self, observation: Any) -> Any:
        """Encode the observation into the shared context (features / KV cache /
        world embedding). Called once per round."""
        ...

    def sample_actions(self, context: Any, noise: Tensor) -> Tensor:
        """Run the batched denoising / flow loop. ``noise`` is ``(K*B, *noise_shape)``;
        returns candidate chunks of shape ``(K*B, T, D)`` or ``(K, B, T, D)``."""
        ...

    def expand_context(self, context: Any, K: int) -> Any:
        """Tile the encoded context to a K-batch (block layout) without copying;
        see :mod:`keystone.parallel`."""
        ...


class KeystonePolicy:
    """Wrap a base policy with K-sample parallel inference + cluster-medoid selection.

    Args:
        base_policy: a policy implementing the :class:`SamplablePolicy` protocol.
        config: Keystone configuration (K, aggregation method, etc.). If omitted,
            uses the paper defaults (``cluster_medoid_guarded`` selection, K from
            ``num_samples``).
        executed_steps: optional chunk truncation before selection (the policy's
            ``n_action_steps``).
    """

    def __init__(
        self,
        base_policy: SamplablePolicy,
        config: SelfConsistencyConfig | None = None,
        executed_steps: int | None = None,
    ):
        self.base = base_policy
        self.config = config or SelfConsistencyConfig()
        self.executed_steps = executed_steps

    @torch.no_grad()
    def predict_action(self, observation: Any, batch_size: int = 1) -> Tensor:
        """Return a single action chunk ``(B, T, D)`` selected from K candidates."""
        K = self.config.num_samples

        # 1. Encode the shared context once.
        context = self.base.encode(observation)

        # 2. Expand to a K-batch without materializing K copies.
        context_K = self.base.expand_context(context, K)

        # 3. Run K denoising / flow chains as one batched forward.
        noise = torch.randn(K * batch_size, *self.base.noise_shape)
        candidates = self.base.sample_actions(context_K, noise)  # (K*B, T, D) or (K, B, T, D)

        # 4. Reshape to (K, B, T, D) if needed, then cluster-medoid select.
        if candidates.ndim == 3:
            candidates = candidates.reshape(K, batch_size, *candidates.shape[1:])
        return aggregate_actions(candidates, self.config, executed_steps=self.executed_steps)
