"""Integration pattern A: override a flow-matching policy's action sampler.

This is how Keystone is wired into LeRobot-style policies (SmolVLA, pi0.5), where
the policy encodes a VLM prefix into a KV cache once and then runs an Euler /
flow-matching denoising loop. The Keystone change is small and self-contained:

    encode prefix once  ->  expand KV cache + masks to K-batch
                        ->  draw K independent noises, run ONE batched denoise
                        ->  reshape to (K, B, T, D), aggregate_actions(...)

The snippet below mirrors that integration verbatim, with the upstream pieces
(``embed_prefix``, ``_denoise_loop``) left as stubs you replace with your model's.

To reproduce the paper numbers on SmolVLA / pi0.5 directly, the matching
override already lives in the LeRobot fork's
``policies/{smolvla,pi05}/modeling_*.py`` -- this file is the portable template.
"""

import torch
from torch import Tensor

from keystone import SelfConsistencyConfig, aggregate_actions, expand_kv_cache


class FlowMatchingPolicyWithKeystone:
    """A flow-matching action expert with an optional Keystone sampling path."""

    def __init__(self, config, self_consistency: SelfConsistencyConfig | None = None):
        self.config = config                      # has chunk_size, max_action_dim, n_action_steps
        self.self_consistency = self_consistency  # None => unchanged single-sample behavior

    # ---- model-specific pieces you already have (stubs) ----------------------
    def embed_prefix(self, observation) -> tuple:
        """Encode images/language/state -> (prefix_embs, prefix_pad_masks, prefix_att_masks)."""
        raise NotImplementedError

    def forward_vlm(self, prefix_embs, prefix_pad_masks, prefix_att_masks):
        """Run the VLM once, return its KV cache (past_key_values)."""
        raise NotImplementedError

    def sample_noise(self, shape, device) -> Tensor:
        return torch.randn(shape, device=device)

    def _denoise_loop(self, bsize, device, noise, prefix_pad_masks, past_key_values) -> Tensor:
        """Run the flow-matching ODE loop, return (bsize, T, D). Works for any
        batch size, so it handles the K*B batch unchanged."""
        raise NotImplementedError

    # ---- the action sampler: single-sample vs Keystone -----------------------
    def sample_actions(self, observation) -> Tensor:
        device = observation.device
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(observation)
        bsize = prefix_pad_masks.shape[0]

        # Encode the shared prefix ONCE.
        past_key_values = self.forward_vlm(prefix_embs, prefix_pad_masks, prefix_att_masks)

        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        noise = self.sample_noise(actions_shape, device)

        sc = self.self_consistency
        if sc is not None and sc.num_samples > 1:
            return self._sample_actions_self_consistent(
                bsize, device, prefix_pad_masks, past_key_values, noise, sc
            )

        # Unchanged single-sample path.
        return self._denoise_loop(bsize, device, noise, prefix_pad_masks, past_key_values)

    def _sample_actions_self_consistent(
        self, bsize, device, prefix_pad_masks, past_key_values, noise, sc
    ) -> Tensor:
        """Run K denoising passes in parallel via batch expansion, then aggregate."""
        # Distances should ignore zero-padding beyond the real action dims.
        if sc.action_dim is None and hasattr(self.config, "action_feature"):
            sc.action_dim = self.config.action_feature.shape[0]

        K = sc.num_samples
        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)

        # K independent noises, concatenated on the batch dim (block layout).
        noises = [noise] + [self.sample_noise(actions_shape, device) for _ in range(K - 1)]
        x_t = torch.cat(noises, dim=0)                       # (K*B, T, D)

        # Reuse the shared prefix for all K candidates -- no extra VLM passes.
        expanded_masks = prefix_pad_masks.repeat(K, 1)       # block layout matches torch.cat
        expanded_kv = expand_kv_cache(past_key_values, K)

        result = self._denoise_loop(K * bsize, device, x_t, expanded_masks, expanded_kv)

        # (K*B, T, D) -> (K, B, T, D); select over only the executed steps.
        result = result.reshape(K, bsize, *result.shape[1:])
        return aggregate_actions(result, sc, executed_steps=self.config.n_action_steps)
