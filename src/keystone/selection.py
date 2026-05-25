"""Keystone candidate selection over K sampled action chunks.

Given K action chunks drawn from a shared model context, ``aggregate_actions``
returns a single chunk to execute. Selection uses only the geometry of the
sampled chunks in continuous action space -- no learned judge, no extra model.
Action chunks are ordered sequences of bounded robot commands, so Euclidean
distance between flattened chunks reflects physical similarity, and the largest
dense region is a natural model-free notion of consensus.

Three selectors are provided, all returning a *real* sampled chunk (never an
interpolation between modes):

- ``cluster_medoid_guarded`` *(the paper's method, Listing 2)*: a unimodality
  guard first decides whether clustering is warranted -- if the sample mean lies
  close to the global medoid relative to the median pairwise distance
  (``spread = ||mean - medoid|| / (median dist + eps)``, below ``tau``) the
  global medoid is returned; otherwise k-means is run and the largest cluster's
  medoid is returned.
- ``medoid``: the global medoid alone -- the chunk minimizing total distance to
  all others (the vector generalization of the median).
- ``cluster_medoid``: k-means with a fixed number of clusters C; return the
  medoid of the largest cluster (no guard).

All are batched on-GPU tensor ops over only K compact chunks and reuse a single
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
    if config.aggregation == "cluster_medoid_guarded":
        return _cluster_medoid_guarded_aggregation(
            actions, config.distance, ad,
            config.cluster_medoid_num_clusters, config.unimodal_tau,
        )
    elif config.aggregation == "medoid":
        return _medoid_aggregation(actions, config.distance, ad)
    elif config.aggregation == "cluster_medoid":
        return _cluster_medoid_aggregation(
            actions, config.distance, ad, config.cluster_medoid_num_clusters,
        )
    else:
        raise ValueError(
            f"Unknown aggregation method: {config.aggregation!r} "
            "(expected 'cluster_medoid_guarded', 'medoid', or 'cluster_medoid')"
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


def _cluster_medoid_guarded_aggregation(
    actions: Tensor,
    metric: str = "l2",
    action_dim: int | None = None,
    num_clusters: int = 2,
    tau: float = 0.3,
) -> Tensor:
    """The paper's guarded selector (Listing 2): a per-batch switch between the
    global medoid and cluster-medoid based on a unimodality check.

    Multimodality is detected from the gap between the sample mean and the
    medoid, normalized by the typical pairwise spread:

        spread = ||mean(samples) - medoid(samples)|| / median_pairwise_distance

    Unimodal cloud -> mean ~= medoid -> spread ~= 0 -> use the global medoid
    (all K samples, for stability). Multimodal cloud -> the mean falls between
    modes while the medoid sits inside one -> spread is large -> use
    cluster_medoid (commit to the dominant mode). ``spread > tau`` triggers
    clustering.

    Returns:
        (B, T, D) -- chosen real chunk per batch element.
    """
    K, B, T, D = actions.shape
    if K == 1:
        return actions[0]

    ad = action_dim or D
    flat = actions[:, :, :, :ad].reshape(K, B, T * ad).permute(1, 0, 2)  # (B, K, T*ad)
    dists = _pairwise_distances(actions, metric, action_dim)  # (B, K, K)

    medoid_idx = dists.sum(dim=-1).argmin(dim=-1)  # (B,)
    batch_indices = torch.arange(B, device=actions.device)

    # Spread heuristic uses L2 in flattened-trajectory space regardless of
    # `metric`, so the threshold is interpretable across distance choices.
    mean_flat = flat.mean(dim=1)  # (B, T*ad)
    medoid_flat = flat[batch_indices, medoid_idx]  # (B, T*ad)
    gap = (mean_flat - medoid_flat).norm(dim=-1)  # (B,)

    iu, ju = torch.triu_indices(K, K, offset=1, device=actions.device)
    median_dist = dists[:, iu, ju].median(dim=-1).values  # (B,)
    spread = gap / (median_dist + 1e-8)
    use_cluster = spread > tau  # (B,)

    winners = medoid_idx.clone()
    for b in use_cluster.nonzero(as_tuple=True)[0].tolist():
        assigns = _kmeans_assignments(flat[b], num_clusters)
        counts = torch.bincount(assigns, minlength=num_clusters)
        members = (assigns == int(counts.argmax().item())).nonzero(as_tuple=True)[0]
        sub = dists[b].index_select(0, members).index_select(1, members)
        winners[b] = members[sub.sum(dim=-1).argmin()]

    return actions[winners, batch_indices]  # (B, T, D)


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
    C: int = 2,
    tau: float = 0.3,
    distance: str = "l2",
    action_dim: int | None = None,
    executed_steps: int | None = None,
) -> Tensor:
    """KeyStone's selector -- the paper's ``cluster_medoid_select`` (Listing 2).

    Runs the unimodality-guarded cluster-medoid rule over K candidate chunks and
    returns a single real sampled chunk: the global medoid when the candidates
    look unimodal (sample mean close to the medoid relative to the median
    pairwise distance, ``spread < tau``), otherwise the medoid of the largest of
    ``C`` k-means clusters.

    Args:
        candidates: ``(K, T, D)`` or ``(K, B, T, D)`` sampled action chunks.
        C: number of k-means clusters used when the guard triggers clustering
            (default 2).
        tau: unimodality-guard threshold (default 0.3); below this the global
            medoid is returned with no clustering.
        distance: 'l1', 'l2', or 'cosine'.
        action_dim: real action dims before padding (None = use all D).
        executed_steps: truncate the chunk to its first ``executed_steps``
            timesteps before selection (the steps that actually run).

    Returns:
        ``(T, D)`` or ``(B, T, D)`` -- a single selected real chunk.
    """
    config = SelfConsistencyConfig(
        num_samples=candidates.shape[0],
        aggregation="cluster_medoid_guarded",
        cluster_medoid_num_clusters=C,
        unimodal_tau=tau,
        distance=distance,
        action_dim=action_dim,
    )
    return select_chunk(candidates, config, executed_steps)
