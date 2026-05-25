"""Minimal, runnable end-to-end demo of Keystone selection.

No real model or GPU required. We fake a bimodal flow-matching policy whose
samples cluster around two distinct action chunks (a "good" dominant mode and a
"failure" minority mode), then show that Keystone recovers the dominant mode and
returns a real sampled chunk -- unlike the mean, which interpolates between the
two modes and lands off the manifold.

Run:
    python examples/minimal_quickstart.py
"""

import torch

from keystone import SelfConsistencyConfig, aggregate_actions, cluster_medoid_select


def fake_policy_samples(K: int, T: int = 50, action_dim: int = 7, seed: int = 0) -> torch.Tensor:
    """Draw K candidate chunks from a bimodal distribution: ~70% mode A, ~30% mode B."""
    g = torch.Generator().manual_seed(seed)
    mode_a = torch.randn(T, action_dim, generator=g)        # dominant (good) mode
    mode_b = torch.randn(T, action_dim, generator=g) + 4.0  # minority (failure) mode
    out = []
    for i in range(K):
        base = mode_a if i % 10 < 7 else mode_b
        out.append(base + 0.1 * torch.randn(T, action_dim, generator=g))
    return torch.stack(out)  # (K, T, action_dim)


def main() -> None:
    K, T, D = 16, 50, 7
    candidates = fake_policy_samples(K, T, D)  # (K, T, D)

    # KeyStone selector (the paper's guarded cluster-medoid rule, Listing 2): the
    # unimodality guard sees a clear bimodal split here, so it runs k-means and
    # returns the medoid of the dominant mode -- one real sampled chunk.
    chosen = cluster_medoid_select(candidates)  # (T, D)
    chosen_idx = next(k for k in range(K) if (candidates[k] - chosen).abs().max() < 1e-6)

    # ... whereas the mean is a synthetic chunk far from every candidate
    # (it interpolates the two modes, landing off the action manifold).
    mean_chunk = candidates.mean(dim=0)
    mean_min_dist = min((candidates[k] - mean_chunk).norm().item() for k in range(K))

    print(f"K={K} candidates, chunk shape (T={T}, D={D})")
    print(f"cluster_medoid_guarded -> candidate #{chosen_idx} (real chunk, dominant mode)")
    print(f"  min distance from the *mean* chunk to any candidate: {mean_min_dist:.2f}  "
          f"(off-manifold: interpolates the two modes)")

    # On a (near-)unimodal cloud the guard skips clustering and returns the global
    # medoid -- the same rule, its other branch.
    g = torch.Generator().manual_seed(7)
    center = torch.randn(T, D, generator=g)
    unimodal = torch.stack([center + 0.02 * torch.randn(T, D, generator=g) for _ in range(K)])
    flat = unimodal.flatten(1)
    global_medoid_idx = int(torch.cdist(flat, flat).sum(dim=-1).argmin())
    guarded_unimodal = cluster_medoid_select(unimodal)
    same = (unimodal[global_medoid_idx] - guarded_unimodal).abs().max() < 1e-6
    print(f"unimodal cloud         -> guard returns the global medoid (#{global_medoid_idx}): {bool(same)}")

    # The lower-level config API adds the padding-aware action_dim, batched
    # (K, B, T, D) inputs, and executed_steps.
    cfg = SelfConsistencyConfig(num_samples=K, aggregation="cluster_medoid_guarded", distance="l2")
    batched = candidates.unsqueeze(1)                          # (K, B=1, T, D)
    chunk = aggregate_actions(batched, cfg, executed_steps=10)  # (B=1, 10, D)
    print(f"\naggregate_actions(cluster_medoid_guarded, executed_steps=10) -> {tuple(chunk.shape)}")


if __name__ == "__main__":
    main()
