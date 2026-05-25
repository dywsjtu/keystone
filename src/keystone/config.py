"""Configuration for Keystone self-consistent action selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SelfConsistencyConfig:
    """Configuration for Keystone K-sample self-consistent action generation.

    Keystone draws ``num_samples`` (K) candidate action chunks from a shared
    model context and selects one to execute by clustering the candidates in
    continuous action space and returning the medoid of the largest cluster --
    a real sampled chunk, never an interpolation between modes.
    """

    num_samples: int = 5
    """Number of independent noise samples (K)."""

    aggregation: str = "cluster_medoid_guarded"
    """Selection method applied to the K candidate chunks (all return a real
    sampled chunk, never an interpolation):

    - ``cluster_medoid_guarded`` *(default)*: the paper's method
      (Geometry-Guided Self-Consistency, Listing 2). It first checks whether
      clustering is needed via a unimodality guard -- it computes the global
      medoid and the spread of the sample mean from that medoid, normalized by
      the median pairwise distance,
      ``s = ||mean - medoid|| / (median_{i<j} dist_ij + eps)``. If
      ``s < unimodal_tau`` the candidates are treated as one compact mode and
      the global medoid is returned; otherwise it runs k-means with
      ``cluster_medoid_num_clusters`` clusters and returns the largest cluster's
      medoid.
    - ``medoid``: the global medoid alone -- the chunk minimizing total distance
      to all others (the vector generalization of the median); the guard's
      unimodal branch, with no clustering.
    - ``cluster_medoid``: k-means over whole chunks with a fixed number of
      clusters C (``cluster_medoid_num_clusters``), returning the medoid of the
      largest cluster; the guard's multimodal branch, with no unimodality check.
    """

    cluster_medoid_num_clusters: int = 2
    """Number of clusters C for the k-means step of ``cluster_medoid`` and the
    multimodal branch of ``cluster_medoid_guarded`` (trajectory-level k-means).
    Default 2: most manipulation decisions have one dominant valid mode plus
    scattered failure samples."""

    unimodal_tau: float = 0.3
    """Unimodality-guard threshold tau for ``cluster_medoid_guarded`` (paper
    default 0.3). Clustering is suppressed -- the global medoid is returned --
    when ``||mean - medoid|| / (median_pairwise_distance + eps) < unimodal_tau``,
    i.e. when the sample mean and the medoid nearly agree relative to the typical
    candidate spread (the regime where a cluster split is least informative)."""

    distance: str = "l2"
    """Distance metric for clustering / medoid selection: 'l1', 'l2', or 'cosine'."""

    action_dim: int | None = None
    """Real action dimension (before padding). If None, uses the full D dim.
    Many policies output (B, T, max_action_dim) where only the first
    ``action_dim`` dims are meaningful and the rest is zero-padding. Distances
    should only consider the real dims to avoid padding distortion."""
