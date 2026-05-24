"""Tests for Keystone candidate selection."""

import pytest
import torch

from keystone import (
    SelfConsistencyConfig,
    aggregate_actions,
    cluster_medoid_select,
    select_chunk,
)
from keystone.selection import _pairwise_distances

METHODS = ["medoid", "cluster_medoid", "cluster_medoid_auto"]
REMOVED_METHODS = ["mean", "median", "mode", "plurality", "smoothness", "auto"]


def _is_one_of(chunk: torch.Tensor, candidates: torch.Tensor) -> bool:
    """True iff `chunk` (T, D) exactly equals one of the K candidates (K, T, D)."""
    return any((candidates[k] - chunk).abs().max().item() < 1e-6 for k in range(candidates.shape[0]))


@pytest.mark.parametrize("method", METHODS)
def test_output_shape(method):
    K, B, T, D = 8, 3, 50, 7
    actions = torch.randn(K, B, T, D)
    cfg = SelfConsistencyConfig(num_samples=K, aggregation=method)
    out = aggregate_actions(actions, cfg)
    assert out.shape == (B, T, D)


@pytest.mark.parametrize("method", METHODS)
def test_methods_return_a_sampled_chunk(method):
    """Both selectors must return an actual candidate, never a synthetic one."""
    K, T, D = 12, 40, 7
    candidates = torch.randn(K, T, D)
    cfg = SelfConsistencyConfig(num_samples=K, aggregation=method)
    out = aggregate_actions(candidates.unsqueeze(1), cfg)[0]  # (T, D)
    assert _is_one_of(out, candidates), f"{method} returned an off-manifold chunk"


@pytest.mark.parametrize("method", REMOVED_METHODS)
def test_removed_methods_raise(method):
    """Only cluster_medoid / cluster_medoid_auto are supported now."""
    cfg = SelfConsistencyConfig(num_samples=4, aggregation=method)
    with pytest.raises(ValueError):
        aggregate_actions(torch.randn(4, 1, 10, 7), cfg)


def test_executed_steps_truncates():
    K, B, T, D = 6, 2, 50, 7
    actions = torch.randn(K, B, T, D)
    cfg = SelfConsistencyConfig(num_samples=K, aggregation="cluster_medoid")
    out = aggregate_actions(actions, cfg, executed_steps=10)
    assert out.shape == (B, 10, D)


def test_auto_unimodality_fallback_to_global_medoid():
    """A tight unimodal cloud: cluster_medoid_auto must return the global medoid."""
    K, T, D = 16, 30, 7
    g = torch.Generator().manual_seed(0)
    center = torch.randn(T, D, generator=g)
    candidates = torch.stack([center + 0.01 * torch.randn(T, D, generator=g) for _ in range(K)])

    dists = _pairwise_distances(candidates.unsqueeze(1), "l2")[0]  # (K, K)
    global_medoid_idx = int(dists.sum(dim=-1).argmin().item())

    out = cluster_medoid_select(candidates, auto=True)
    assert torch.allclose(out, candidates[global_medoid_idx], atol=1e-6)


@pytest.mark.parametrize("auto", [False, True])
def test_bimodal_selects_dominant_mode(auto):
    """70/30 bimodal cloud: the chosen chunk must come from the larger mode."""
    K, T, D = 20, 30, 7
    g = torch.Generator().manual_seed(1)
    mode_a = torch.randn(T, D, generator=g)         # dominant
    mode_b = torch.randn(T, D, generator=g) + 10.0  # far minority
    cands, in_a = [], []
    for i in range(K):
        is_a = i % 10 < 7
        base = mode_a if is_a else mode_b
        cands.append(base + 0.05 * torch.randn(T, D, generator=g))
        in_a.append(is_a)
    candidates = torch.stack(cands)

    out = cluster_medoid_select(candidates, num_clusters=2, auto=auto)
    chosen_idx = next(k for k in range(K) if (candidates[k] - out).abs().max() < 1e-6)
    assert in_a[chosen_idx], f"(auto={auto}) selected a chunk from the minority mode"


@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
def test_distance_metrics(metric):
    K, T, D = 10, 25, 7
    candidates = torch.randn(K, T, D)
    out = cluster_medoid_select(candidates, distance=metric)
    assert out.shape == (T, D)
    assert _is_one_of(out, candidates)


def test_action_dim_ignores_padding():
    """Padding dims must not influence selection."""
    K, T, real_D, pad = 10, 20, 7, 5
    g = torch.Generator().manual_seed(2)
    real = torch.randn(K, T, real_D, generator=g)
    # Identical real dims, wildly different padding -> selection must match real-only.
    noisy_pad = torch.cat([real, 100.0 * torch.randn(K, T, pad, generator=g)], dim=-1)

    cfg_real = SelfConsistencyConfig(num_samples=K, aggregation="cluster_medoid")
    cfg_padded = SelfConsistencyConfig(num_samples=K, aggregation="cluster_medoid", action_dim=real_D)

    assert _argselect(real, cfg_real) == _argselect(noisy_pad, cfg_padded)


def _argselect(candidates, cfg):
    out = aggregate_actions(candidates.unsqueeze(1), cfg)[0]
    ad = cfg.action_dim or candidates.shape[-1]
    for k in range(candidates.shape[0]):
        if (candidates[k, :, :ad] - out[:, :ad]).abs().max() < 1e-6:
            return k
    raise AssertionError("selected chunk not found among candidates")


def test_select_chunk_unbatched_and_batched():
    K, T, D = 8, 20, 7
    cfg = SelfConsistencyConfig(num_samples=K, aggregation="cluster_medoid")
    assert select_chunk(torch.randn(K, T, D), cfg).shape == (T, D)
    assert select_chunk(torch.randn(K, 4, T, D), cfg).shape == (4, T, D)


def test_cluster_medoid_select_auto_flag_shape():
    K, T, D = 12, 30, 7
    candidates = torch.randn(K, T, D)
    out = cluster_medoid_select(candidates, auto=True)
    assert out.shape == (T, D)
    assert _is_one_of(out, candidates)


def test_k_equals_one_is_identity():
    T, D = 30, 7
    candidates = torch.randn(1, T, D)
    assert torch.allclose(cluster_medoid_select(candidates), candidates[0], atol=1e-6)
    assert torch.allclose(cluster_medoid_select(candidates, auto=True), candidates[0], atol=1e-6)


def test_unknown_method_raises():
    cfg = SelfConsistencyConfig(num_samples=4, aggregation="nonsense")
    with pytest.raises(ValueError):
        aggregate_actions(torch.randn(4, 1, 10, 7), cfg)


def test_determinism():
    K, T, D = 12, 30, 7
    candidates = torch.randn(K, T, D)
    assert torch.equal(cluster_medoid_select(candidates), cluster_medoid_select(candidates))
    assert torch.equal(
        cluster_medoid_select(candidates, auto=True), cluster_medoid_select(candidates, auto=True)
    )
