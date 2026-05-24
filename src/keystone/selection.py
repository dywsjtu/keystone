"""Keystone candidate selection over K sampled action chunks.

Given K action chunks drawn from a shared model context, ``aggregate_actions``
returns a single chunk to execute. Selection uses only the geometry of the
sampled chunks in continuous action space -- no learned judge, no extra model.
Action chunks are ordered sequences of bounded robot commands, so Euclidean
distance between flattened chunks reflects physical similarity, and the largest
dense region is a natural model-free notion of consensus.

Three selectors are provided, all returning a *real* sampled chunk (never an
interpolation between modes):

- ``medoid``: the global medoid -- the chunk minimizing total distance to all
  others (the vector generalization of the median).
- ``cluster_medoid``: k-means with a fixed number of clusters C; return the
  medoid of the largest cluster.
- ``cluster_medoid_auto``: cluster count detected automatically, with a
  fall-back to the global medoid when the candidates look unimodal.

Both are batched on-GPU tensor ops over only K compact chunks and reuse a single
K x K pairwise distance matrix; the entire selection adds at most a few hundred
microseconds per round at the K values used in the paper (<= 16).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import SelfConsistencyConfig


def aggregate_actions(
    actions: Tensor,
    config: SelfConsistencyConfig,
    executed_steps: int | None = None,
) -> Tensor:
    """Aggregate K action chunks into a single chunk to execute.

    Args:
        actions: (K, B, T, D) tensor of K denoised chunks. D may include
            zero-padding beyond the real action dimensions.
        config: Keystone configuration. If ``config.action_dim`` is set, only
            the first ``action_dim`` dims are used for distance computation.
        executed_steps: If set, the chunk is truncated to its first
            ``executed_steps`` timesteps before selection -- typically the
            policy's ``n_action_steps``, since the tail is discarded before the
            next replan anyway. Distances are then computed over (and only over)
            the steps that actually run.

    Returns:
        (B, T', D) selected chunk, where T' = executed_steps if set else T.
    """
    if executed_steps is not None and executed_steps < actions.shape[2]:
        actions = actions[:, :, :executed_steps, :]

    ad = config.action_dim
    if config.aggregation == "medoid":
        return _medoid_aggregation(actions, config.distance, ad)
    elif config.aggregation == "cluster_medoid":
        return _cluster_medoid_aggregation(
            actions, config.distance, ad, config.cluster_medoid_num_clusters,
        )
    elif config.aggregation == "cluster_medoid_auto":
        return _cluster_medoid_auto_aggregation(
            actions, config.distance, ad, config.cluster_medoid_auto_min_gap,
        )
    else:
        raise ValueError(
            f"Unknown aggregation method: {config.aggregation!r} "
            "(expected 'medoid', 'cluster_medoid', or 'cluster_medoid_auto')"
        )


def _pairwise_distances(actions: Tensor, metric: str, action_dim: int | None = None) -> Tensor:
    """Pairwise distance matrix between the K chunks, per batch element.

    Each chunk is flattened to a single vector (T*ad) so the distance captures
    the full chunk shape. Only the first ``action_dim`` dims are used.

    Returns:
        (B, K, K) distance matrix.
    """
    K, B, T, D = actions.shape
    ad = action_dim or D
    flat = actions[:, :, :, :ad].reshape(K, B, T * ad).permute(1, 0, 2)  # (B, K, T*ad)

    if metric == "cosine":
        flat_norm = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        sim = torch.bmm(flat_norm, flat_norm.transpose(1, 2))  # (B, K, K)
        return 1.0 - sim
    elif metric == "l1":
        return torch.cdist(flat, flat, p=1)
    elif metric == "l2":
        return torch.cdist(flat, flat, p=2)
    else:
        raise ValueError(f"Unknown distance metric: {metric} (expected 'l1', 'l2', or 'cosine')")


def _medoid_aggregation(actions: Tensor, metric: str = "l2", action_dim: int | None = None) -> Tensor:
    """Select the real chunk minimizing total distance to all others (the medoid)."""
    K, B, T, D = actions.shape
    dists = _pairwise_distances(actions, metric, action_dim)  # (B, K, K)
    medoid_idx = dists.sum(dim=-1).argmin(dim=-1)  # (B,)
    batch_indices = torch.arange(B, device=actions.device)
    return actions[medoid_idx, batch_indices]  # (B, T, D)


def _kmeans_assignments(points: Tensor, num_clusters: int, max_iters: int = 10) -> Tensor:
    """Simple k-means; returns (K,) cluster ids. Deterministic init (first C points)."""
    K, _D = points.shape
    num_clusters = min(num_clusters, K)
    if num_clusters <= 1:
        return torch.zeros(K, dtype=torch.long, device=points.device)

    centroids = points[:num_clusters].clone()
    for _ in range(max_iters):
        dists = torch.cdist(points.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
        assignments = dists.argmin(dim=1)
        new_centroids = torch.zeros_like(centroids)
        for c in range(num_clusters):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = points[mask].mean(dim=0)
            else:
                new_centroids[c] = centroids[c]
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    dists = torch.cdist(points.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
    return dists.argmin(dim=1)


def _cluster_medoid_aggregation(
    actions: Tensor,
    metric: str = "l2",
    action_dim: int | None = None,
    num_clusters: int = 2,
) -> Tensor:
    """Cluster the K chunks (fixed C), return the medoid of the largest cluster.

    The chosen chunk comes from the dominant mode and is a real denoised sample
    (the within-cluster medoid), never an interpolation between modes.
    """
    K, B, T, D = actions.shape
    ad = action_dim or D
    flat = actions[:, :, :, :ad].reshape(K, B, T * ad).permute(1, 0, 2)  # (B, K, T*ad)
    dists = _pairwise_distances(actions, metric, action_dim)  # (B, K, K)

    winners = torch.empty(B, dtype=torch.long, device=actions.device)
    for b in range(B):
        assigns = _kmeans_assignments(flat[b], num_clusters)
        counts = torch.bincount(assigns, minlength=num_clusters)
        members = (assigns == int(counts.argmax().item())).nonzero(as_tuple=True)[0]
        sub = dists[b].index_select(0, members).index_select(1, members)
        winners[b] = members[sub.sum(dim=-1).argmin()]

    batch_indices = torch.arange(B, device=actions.device)
    return actions[winners, batch_indices]


def _single_linkage_auto_clusters(dists: Tensor, min_relative_gap: float) -> Tensor:
    """Single-linkage clustering with an automatic cut at the largest distance gap.

    Sort the unique pairwise distances, cut at the largest consecutive gap, and
    union endpoints of every edge below the cut. Falls back to a single cluster
    when the largest gap is small relative to the median distance (no clear
    structure). Cost ~O(K^2 log K) per batch element; no sweep over K.

    Returns:
        (K,) long tensor of densely-relabeled cluster ids.
    """
    K = dists.shape[0]
    if K <= 2:
        return torch.zeros(K, dtype=torch.long, device=dists.device)

    iu, ju = torch.triu_indices(K, K, offset=1, device=dists.device)
    edges = dists[iu, ju]
    sorted_edges, order = edges.sort()
    gaps = sorted_edges[1:] - sorted_edges[:-1]
    cut_idx = int(gaps.argmax().item())
    median_dist = sorted_edges.median()

    if gaps[cut_idx].item() < min_relative_gap * (median_dist.item() + 1e-8):
        return torch.zeros(K, dtype=torch.long, device=dists.device)

    parent = list(range(K))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    iu_l, ju_l, order_l = iu.tolist(), ju.tolist(), order.tolist()
    for rank in range(cut_idx + 1):
        e_idx = order_l[rank]
        i, j = iu_l[e_idx], ju_l[e_idx]
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    roots = torch.tensor([find(i) for i in range(K)], device=dists.device, dtype=torch.long)
    _, assigns = torch.unique(roots, return_inverse=True)
    return assigns


def _cluster_medoid_auto_aggregation(
    actions: Tensor,
    metric: str = "l2",
    action_dim: int | None = None,
    min_relative_gap: float = 0.3,
) -> Tensor:
    """Like cluster_medoid, but the number of clusters is auto-detected.

    Returns the medoid of the largest auto-detected cluster; when no clear
    cluster structure exists, returns the global medoid.
    """
    K, B, T, D = actions.shape
    dists = _pairwise_distances(actions, metric, action_dim)  # (B, K, K)

    winners = torch.empty(B, dtype=torch.long, device=actions.device)
    for b in range(B):
        assigns = _single_linkage_auto_clusters(dists[b], min_relative_gap)
        num_found = int(assigns.max().item()) + 1
        counts = torch.bincount(assigns, minlength=num_found)
        members = (assigns == int(counts.argmax().item())).nonzero(as_tuple=True)[0]
        sub = dists[b].index_select(0, members).index_select(1, members)
        winners[b] = members[sub.sum(dim=-1).argmin()]

    batch_indices = torch.arange(B, device=actions.device)
    return actions[winners, batch_indices]


def select_chunk(
    candidates: Tensor,
    config: SelfConsistencyConfig,
    executed_steps: int | None = None,
) -> Tensor:
    """Select one chunk from K candidates, accepting unbatched or batched input.

    Thin wrapper over :func:`aggregate_actions` that accepts either
    ``(K, T, D)`` (single observation) or ``(K, B, T, D)`` (batched). Returns
    ``(T, D)`` or ``(B, T, D)`` respectively.
    """
    if candidates.ndim == 3:
        return aggregate_actions(candidates.unsqueeze(1), config, executed_steps)[0]
    if candidates.ndim == 4:
        return aggregate_actions(candidates, config, executed_steps)
    raise ValueError(f"Expected candidates of shape (K, T, D) or (K, B, T, D), got {tuple(candidates.shape)}")


def cluster_medoid_select(
    candidates: Tensor,
    num_clusters: int = 2,
    distance: str = "l2",
    action_dim: int | None = None,
    executed_steps: int | None = None,
    auto: bool = False,
    auto_min_gap: float = 0.3,
) -> Tensor:
    """Keystone's cluster-medoid selector -- convenience entry point.

    Clusters the K candidate chunks in action space and returns the medoid of
    the largest cluster (a real sampled chunk).

    Args:
        candidates: ``(K, T, D)`` or ``(K, B, T, D)`` sampled action chunks.
        num_clusters: C, number of k-means clusters (default 2). Ignored when
            ``auto=True``.
        distance: 'l1', 'l2', or 'cosine'.
        action_dim: real action dims before padding (None = use all D).
        executed_steps: truncate the chunk to its first ``executed_steps``
            timesteps before selection (the steps that actually run).
        auto: if True, use ``cluster_medoid_auto`` (auto cluster-count detection
            with a unimodality fall-back to the global medoid) instead of
            fixed-C ``cluster_medoid``.
        auto_min_gap: unimodality threshold used only when ``auto=True``.

    Returns:
        ``(T, D)`` or ``(B, T, D)`` -- a single selected real chunk.
    """
    if auto:
        config = SelfConsistencyConfig(
            num_samples=candidates.shape[0],
            aggregation="cluster_medoid_auto",
            cluster_medoid_auto_min_gap=auto_min_gap,
            distance=distance,
            action_dim=action_dim,
        )
    else:
        config = SelfConsistencyConfig(
            num_samples=candidates.shape[0],
            aggregation="cluster_medoid",
            cluster_medoid_num_clusters=num_clusters,
            distance=distance,
            action_dim=action_dim,
        )
    return select_chunk(candidates, config, executed_steps)
