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

    aggregation: str = "cluster_medoid"
    """Selection method applied to the K candidate chunks (all return a real
    sampled chunk, never an interpolation):

    - ``medoid``: the global medoid -- the chunk minimizing total distance to
      all others (the vector generalization of the median).
    - ``cluster_medoid`` *(default)*: k-means over whole chunks with a fixed
      number of clusters C (``cluster_medoid_num_clusters``); return the medoid
      of the largest cluster. The chunk comes from the dominant mode.
    - ``cluster_medoid_auto``: same output type, but the number of clusters is
      detected automatically from the data via single-linkage clustering with a
      largest-gap cut. When the candidates show no clear cluster structure
      (largest gap small relative to the median pairwise distance, controlled by
      ``cluster_medoid_auto_min_gap``) it falls back to the global medoid.
    """

    cluster_medoid_num_clusters: int = 2
    """Number of clusters C for ``cluster_medoid`` (trajectory-level k-means).
    Default 2: most manipulation decisions have one dominant valid mode plus
    scattered failure samples."""

    cluster_medoid_auto_min_gap: float = 0.3
    """Threshold for ``cluster_medoid_auto``. The largest distance gap must
    exceed ``cluster_medoid_auto_min_gap`` x median_pairwise_distance to trigger
    a split; otherwise the candidates are treated as unimodal and the global
    medoid is returned."""

    distance: str = "l2"
    """Distance metric for clustering / medoid selection: 'l1', 'l2', or 'cosine'."""

    action_dim: int | None = None
    """Real action dimension (before padding). If None, uses the full D dim.
    Many policies output (B, T, max_action_dim) where only the first
    ``action_dim`` dims are meaningful and the rest is zero-padding. Distances
    should only consider the real dims to avoid padding distortion."""
