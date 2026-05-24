"""Integration pattern B: monkey-patch a served model's generate function.

This is how Keystone is wired into the vla-eval benchmark servers (GR00T N1.6,
X-VLA, StarVLA, pi0.5-JAX), where the model ships inside a Docker image and you
want to intercept only the function that maps an encoded observation to an action
chunk -- keeping the simulator, dependencies, and pre/post-processing identical
to the reproduced baseline.

The pattern: run the (expensive) backbone once, expand its outputs K-fold with
``repeat_interleave`` (interleaved layout), run the action expert's denoising
loop as one batched pass over K*B rows, reshape, and select. This snippet mirrors
the real X-VLA patch with the upstream calls left as stubs.
"""

import torch

from keystone import SelfConsistencyConfig, aggregate_actions


def install_keystone_patch(model, sc_cfg: SelfConsistencyConfig, chunk_size: int) -> None:
    """Replace ``model.generate_actions`` with a K-noise + prefix-reuse version.

    Args:
        model: the loaded policy exposing ``forward_vlm``, ``transformer``,
            ``action_space``, ``num_actions``.
        sc_cfg: Keystone configuration (``num_samples`` = K, aggregation, ...).
        chunk_size: number of timesteps actually executed before the next replan.
    """
    K = sc_cfg.num_samples

    @torch.no_grad()
    def patched(input_ids, image_input, image_mask, domain_id, proprio, steps: int = 10):
        model.eval()

        # 1. Run the backbone ONCE (the expensive vision-language pass).
        enc = model.forward_vlm(input_ids, image_input, image_mask)
        B = input_ids.shape[0]
        D = model.action_space.dim_action

        # 2. Expand prefix-side state K-fold, interleaved: row b*K + k is batch
        #    b's k-th replica. repeat_interleave shares storage where possible.
        enc_K = {
            k: (v.repeat_interleave(K, dim=0) if isinstance(v, torch.Tensor) else v)
            for k, v in enc.items()
        }
        proprio_K = proprio.repeat_interleave(K, dim=0)
        domain_id_K = domain_id.repeat_interleave(K, dim=0)

        # 3. K independent noises; batched flow-matching loop over K*B rows.
        x1 = torch.randn(K * B, model.num_actions, D, device=proprio.device, dtype=proprio.dtype)
        action = torch.zeros_like(x1)
        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((K * B,), i / steps, device=proprio.device, dtype=proprio.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = model.action_space.preprocess(proprio_K, x_t)
            action = model.transformer(
                domain_id=domain_id_K, action_with_noise=x_t_m, proprio=proprio_m, t=t, **enc_K
            )

        # 4. (K*B, T, D) -> (K, B, T, D) for the *interleaved* layout, then select.
        T_steps = action.shape[1]
        actions_KBTD = action.view(B, K, T_steps, D).permute(1, 0, 2, 3).contiguous()
        executed = min(T_steps, chunk_size) if chunk_size > 0 else T_steps
        agg = aggregate_actions(actions_KBTD.float(), sc_cfg, executed_steps=executed).to(action.dtype)

        # 5. Hand back in the original output format.
        return model.action_space.postprocess(agg)

    model.generate_actions = patched
