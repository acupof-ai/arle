# Vendored from SGLang (Apache-2.0):
#   sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py
#   `tanh` (lines 933-935), `_apply_activation` (938-957), and
#   `act_and_mul_kernel` (960-1008). The SwiGLU epilogue between the fused MoE
#   GEMM1 (gate||up, [M*top_k, 2N]) and GEMM2 (down): out = silu(gate) * up,
#   reading gate from the first N columns and up from the second N columns of
#   each routed row, writing the [M*top_k, N] activation.
#
# Launch config resolved from `act_and_mul_triton` (the fused_experts_impl
# caller, unsorted routing): grid = (M*top_k,), expert_step = 1, BLOCK_SIZE =
# 512, ACTIVATION_TYPE = "silu", HAS_SWIGLU_LIMIT = False (Qwen3.6 has no SwiGLU
# clamp). `expert_ids_ptr` is the token-major topk_ids row buffer
# (`topk_ids.view(-1)`), used only to skip pad rows (`expert_id == -1`); on the
# AOT bf16 decode path every routed row is real (no -1), so the skip never
# fires, but we keep the arg + the branch byte-for-byte.
#
# DEVIATIONS (every intentional change vs. the upstream .py file):
#   1. The torch / sglang.srt.* module-level imports are DROPPED (see
#      arle_fused_moe.py DEVIATION 1 — AOT exec_module can't import torch).
#      Only `triton` + `triton.language` are imported.
#   2. Sibling launch wrappers and unrelated kernels are not vendored here.
# The live silu(gate)*up math is byte-equal to upstream.

import triton
import triton.language as tl


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _apply_activation(x, ACTIVATION_TYPE: tl.constexpr):
    """
    Apply activation function based on compile-time constant.

    Args:
        x: Input tensor (converted to float32 inside)
        ACTIVATION_TYPE: Compile-time constant string ("silu" or "gelu")

    Returns:
        Activated output in the same dtype as input
    """
    x = x.to(tl.float32)
    if ACTIVATION_TYPE == "silu":
        return x * tl.sigmoid(x)
    elif ACTIVATION_TYPE == "gelu":
        kAlpha = 0.7978845608028654
        return 0.5 * x * (1 + tanh(kAlpha * (x + 0.044715 * x * x * x)))
    else:
        raise ValueError(f"Unsupported activation: {ACTIVATION_TYPE}")


@triton.jit
def act_and_mul_kernel(
    gateup_output,
    down_input,
    hidden_size,
    expert_ids_ptr,
    expert_step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION_TYPE: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr = 0.0,
    HAS_SWIGLU_LIMIT: tl.constexpr = False,
):
    """
    Unified activation and multiply kernel that handles both sorted and unsorted routing,
    and both SiLU and GELU activations using compile-time constants.
    """
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2
    pid = tl.program_id(0)

    expert_id = tl.load(expert_ids_ptr + pid // expert_step)

    if expert_id == -1:
        return

    gateup_output_ptr = gateup_output + pid * hidden_size
    down_input_ptr = down_input + pid * half_hidden_size
    gate_output_ptr = gateup_output_ptr
    up_output_ptr = gateup_output_ptr + half_hidden_size

    for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < half_hidden_size

        gate_output = tl.load(gate_output_ptr + offset, mask=mask)
        up_output = tl.load(up_output_ptr + offset, mask=mask)

        if HAS_SWIGLU_LIMIT:
            gate_output = tl.minimum(gate_output, SWIGLU_LIMIT)
            up_output = tl.maximum(tl.minimum(up_output, SWIGLU_LIMIT), -SWIGLU_LIMIT)

        gate_output_activated = _apply_activation(gate_output, ACTIVATION_TYPE)
        gate_output_activated = gate_output_activated.to(InDtype)

        act_mul_output = gate_output_activated * up_output
        act_mul_output = act_mul_output.to(OutDtype)
        tl.store(down_input_ptr + offset, act_mul_output, mask=mask)
