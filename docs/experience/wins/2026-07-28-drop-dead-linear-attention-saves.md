# Drop 6 dead LinearAttentionCtx saves — ~10 GiB/LA-layer @256K

> Status: Code shipped; CPU tests + cuda-lane typecheck GREEN. Device gradient
> A/B `pending-remote` (H20, CUDA-only) — bundled with the MLP seq-chunk A/B.

## Context

An audit of every autograd op's `SavedContext` vs what its `*_backward` actually
reads found the linear-attention forward pins six tensors on the tape that no
backward path consumes:

| field | shape @256K, bf16/f32 | ~VRAM / LA layer |
|---|---|---|
| `q` `k` `v` `raw_output` | bf16 [1,seq,32,128] | ~2.0 GiB each |
| `a_inv` | bf16 [1,seq,32,64] | ~2.0 GiB |
| `g_cumsum` | f32 [1,seq,32] | ~32–48 MiB |

Proven dead on all three backward paths by source reading (not inference):
- **Device row-worker** (`backend_cuda.rs:4558-5015`, where batch==1 and, after
  per-row re-slice, batch>1 both land): neither kernel launch (`.arg(...)` lists)
  takes `q/k/v/a_inv/g_cumsum/raw_output` — only the 13 live tensors
  (`upstream,qkv,z,a_proj,conv1d_weight,dt_bias,a_log,norm_weight,preact,qkv_conv,beta,g,chunk_state`).
- **Host recompute** (`linear_attention.rs`, `initial_state.is_some()` path):
  rebuilds q/k/v/… from `qkv/z/proj/conv`; reads none of the saves.

The forward kernel still computes them internally as scratch — the waste was
retaining them on the tape from each LA layer's forward until backward, i.e. it
accumulated across every subsequent layer's forward.

## What changed

Stopped retaining the six. `LinearAttentionDeviceForwardResult` still returns
them (kernel unchanged), but `try_linear_attention_forward_device` no longer
`alloc_device_tensor`s them onto the tape — the returned handles drop at forward
end, freeing the VRAM immediately. Removed from `SavedContext::LinearAttentionCtx`,
`LinearAttentionDeviceBackwardArgs`, the backward guard/ensure/handle/args blocks,
and the batch>1 per-row re-slice. Live siblings (`preact,qkv_conv,g,beta,chunk_state`)
stay. Numerics-identical by construction — nothing read them.

## Verification

- `cargo test -p autograd` 37/37 pass incl. `test_linear_attention` +
  `linear_attention_prompt_boundary_carry_is_exact` (CPU exercises the host
  recompute path). clippy clean.
- cuda-lane Mac typecheck (`autograd`+`train`, `cuda,no-cuda`) green.
- **Device gradient A/B pending-remote**: on H20, run an LA-layer backward
  pre/post this commit and confirm `dqkv/dz/db/da/dconv/ddt/da_log/dnorm` are
  bit-identical (or within the MoE-nondet floor), + read the LA-layer writeback
  `pool_used_current` drop. Bundle with the MLP seq-chunk A/B (same GPU trip,
  same LA layers).

## Rule

Audit `SavedContext` vs the backward read-set before assuming a save is needed —
a forward intermediate the kernel computes internally does NOT need tape
retention if the backward recomputes or ignores it. Verify a "dead save" by
reading the kernel `.arg()` list, not by the field's mere presence in the ctx;
confirm on-device with a gradient A/B before shipping (host path passing only
proves the host path).
