import torch

from tile_kernels.engram import fused_weight, engram_gate_fwd, engram_gate_bwd, grad_w_reduce


class EngramGateFn(torch.autograd.Function):
    """
    Fused engram gate with RMSNorm.

    Computes:
        gate = sigmoid(signed_sqrt(dot(RMSNorm(x, wh), RMSNorm(k, we)) * scalar))
        output = hidden_states + gate * v

    where ``signed_sqrt(x) = sign(x) * sqrt(|x|)`` and ``scalar = 1 / sqrt(hidden_size)``.
    A ``clamp_min`` is also applied.

    Args:
        hidden_states: [*, hc_mult, hidden_size], bf16.
        k:             [*, hc_mult, hidden_size], bf16.
        v:             [*, hidden_size], bf16.
        weight_hidden: [hc_mult, hidden_size], bf16. RMSNorm weight for hidden_states.
        weight_embed:  [hc_mult, hidden_size], bf16. RMSNorm weight for k.
        clamp_value:   float. Clamp range.
        eps:           float. RMSNorm epsilon.

    Returns:
        output: same shape and dtype as hidden_states.

    Note:
        If ``weight_hidden`` or ``weight_embed`` has a ``main_grad`` attribute, gradients are accumulated
        into ``main_grad`` in-place and ``None`` is returned for that parameter.
        Otherwise, gradients are returned.
    """

    @staticmethod
    def forward(ctx, hidden_states, k, v, weight_hidden, weight_embed, clamp_value, eps):
        origin_shape = hidden_states.shape
        *_, hc_mult, hidden_size = origin_shape

        x = hidden_states.view(-1, hc_mult, hidden_size)
        k = k.view(-1, hc_mult, hidden_size)
        v = v.view(-1, hidden_size)

        weight_fused = fused_weight(weight_hidden, weight_embed)
        output, dot, gate_score, rstd_x, rstd_k = engram_gate_fwd(
            x, k, v, weight_fused, eps, clamp_value,
        )

        ctx.save_for_backward(
            x, k, v, weight_hidden, weight_embed, weight_fused,
            dot, gate_score, rstd_x, rstd_k,
        )
        ctx.clamp_value = clamp_value
        ctx.origin_shape = origin_shape
        return output.view(origin_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (x, k, v, weight_hidden, weight_embed, weight_fused,
         dot, gate_score, rstd_x, rstd_k) = ctx.saved_tensors
        origin_shape = ctx.origin_shape
        clamp_value = ctx.clamp_value
        *_, hc_mult, hidden_size = origin_shape

        grad_out = grad_output.view(-1, hc_mult, hidden_size)

        grad_x, grad_k, grad_v, grad_w_partial = engram_gate_bwd(
            grad_out, x, k, v, weight_fused,
            dot, gate_score, rstd_x, rstd_k, clamp_value,
        )

        # Use main_grad (fp32 gradient buffer) if available, otherwise allocate fp32 grad tensors.
        # grad_w_reduce accumulates into grad_wh / grad_we in-place.
        main_grad_wh = getattr(weight_hidden, 'main_grad', None)
        main_grad_we = getattr(weight_embed, 'main_grad', None)
        grad_wh = main_grad_wh if main_grad_wh is not None else torch.zeros_like(weight_hidden, dtype=torch.float32)
        grad_we = main_grad_we if main_grad_we is not None else torch.zeros_like(weight_embed, dtype=torch.float32)
        grad_w_reduce(
            grad_w_partial, weight_hidden, weight_embed,
            grad_wh, grad_we,
        )

        v_origin_shape = origin_shape[:-2] + (hidden_size,)
        # Return None for weight grads when main_grad is used (already accumulated in-place).
        return (
            grad_x.view(origin_shape),
            grad_k.view(origin_shape),
            grad_v.view(v_origin_shape),
            None if main_grad_wh is not None else grad_wh,
            None if main_grad_we is not None else grad_we,
            None, None,
        )


engram_gate = EngramGateFn.apply
