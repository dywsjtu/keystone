"""Tests for the K-parallel expansion utilities."""

import torch

from keystone import expand_kv_cache, repeat_batch


def test_repeat_batch_block_layout():
    B, H, W = 2, 3, 4
    x = torch.arange(B * H * W).reshape(B, H, W).float()
    K = 3
    out = repeat_batch(x, K)
    assert out.shape == (K * B, H, W)
    # Block layout: rows [0:B] == replica 0 == original x, [B:2B] == replica 1, ...
    for k in range(K):
        assert torch.equal(out[k * B:(k + 1) * B], x)


def test_repeat_batch_shares_storage_until_written():
    # expand-based repeat should not blow up memory: a (B, big) tensor tiled K
    # times must not allocate K*B*big fresh elements before any write.
    x = torch.zeros(2, 1024)
    out = repeat_batch(x, 8)
    assert out.shape == (16, 1024)
    assert torch.equal(out, torch.zeros(16, 1024))


def test_expand_kv_cache_tuple_format():
    B, n_heads, S, head_dim = 2, 4, 16, 8
    layer = (torch.randn(B, n_heads, S, head_dim), torch.randn(B, n_heads, S, head_dim))
    cache = (layer, layer)
    K = 4
    out = expand_kv_cache(cache, K)
    assert isinstance(out, tuple) and len(out) == 2
    for (k, v) in out:
        assert k.shape == (K * B, n_heads, S, head_dim)
        assert v.shape == (K * B, n_heads, S, head_dim)
    # First B rows of the expanded keys match the original.
    assert torch.equal(out[0][0][:B], layer[0])


def test_expand_kv_cache_dict_format():
    B, n_heads, S, head_dim = 2, 4, 16, 8
    cache = {
        0: {"key_states": torch.randn(B, n_heads, S, head_dim),
            "value_states": torch.randn(B, n_heads, S, head_dim)},
        1: {"key_states": torch.randn(B, n_heads, S, head_dim),
            "value_states": torch.randn(B, n_heads, S, head_dim)},
    }
    K = 3
    out = expand_kv_cache(cache, K)
    assert set(out.keys()) == {0, 1}
    for layer_kv in out.values():
        assert layer_kv["key_states"].shape == (K * B, n_heads, S, head_dim)
        assert layer_kv["value_states"].shape == (K * B, n_heads, S, head_dim)
