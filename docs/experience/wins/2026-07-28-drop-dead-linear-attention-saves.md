# Drop 6 dead LinearAttentionCtx saves — ~10 GiB/LA-layer @256K

> Status: Shipped; CPU tests + cuda-lane typecheck GREEN; H20 gradient A/B
> measured (seq=40960, GREEN — loss bit-identical, VRAM drops).

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
- **Device gradient A/B — GREEN** (H20, agent-OPD synthetic writeback seq=40960,
  ThinkingCap-27B-FP8, LoRA r16 qv; parent `076d309cc` vs HEAD, single-variable;
  per-op `pool_used_current` at layer 62 (LA)):

  | metric | parent (6 saves kept) | HEAD (dropped) | Δ |
  |---|---|---|---|
  | mean_loss | 8.685793 | 8.685793 | **bit-identical** |
  | inner-backward peak | 74.0 GiB | 71.9 GiB | −2167 MiB |

  Loss bit-identical → the deleted saves do not change gradients (the
  numerics-identical claim holds). `pool_used` drops a constant 2167 MiB across
  every post-attention stage (the 6 freed tensors) — the seq=40960 slice of the
  ~10 GiB/LA-layer@256K claim; the drop grows O(seq) into the 256K regime.

## Rule

Audit `SavedContext` vs the backward read-set before assuming a save is needed —
a forward intermediate the kernel computes internally does NOT need tape
retention if the backward recomputes or ignores it. Verify a "dead save" by
reading the kernel `.arg()` list, not by the field's mere presence in the ctx;
confirm on-device with a gradient A/B before shipping (host path passing only
proves the host path).
