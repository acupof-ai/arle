# DSv4 incremental decode fixed — forward-level keepalive (cudarc event-tracking premature-free)

**Date:** 2026-06-04. **Backend:** CUDA (DSv4-Flash, TP=8/EP=8, 8×H20).
**Scope:** `crates/infer-cuda/src/{dsv4.rs,moe.rs,attention.rs}`.
**Status:** **VERIFIED 16/16** — full 16-token DSv4 incremental decode now exactly matches
the bf16 oracle (no dump, no diagnostic env). Unblocks the grouped-cache decode tok/s bench
(#21) + the SGLang-class perf roadmap (#24/#25).

## Context

DSv4 prefill was correct (token#1=11111 = oracle[0]) but **incremental decode (`start_pos>0`)
produced garbage** (`[16,11111,0,…]`), so the prior "3/3 16/16" was prefill-only. Wrongly
suspected (and cleared): wq_b TP-shard (fixed separately, prefill-verified), grouped-cache
prebuild (byte-identical), SW attention (prefill/decode-consistent), embedding/HC-expand
(both token-major consistent), executor state-lifecycle (resets only at `start_pos==0`).

## Root cause

`DeviceContext::on_device` (`crates/cuda-kernels/src/tensor.rs` ~:340) calls
`ctx.disable_event_tracking()` (intentional: avoids cudarc's hidden per-op stream waits in
CUDA-graph capture + runs a copy stream). The cost: a device buffer (`HiddenStates`/`CudaSlice`)
dropped at its Rust last-use is **freed immediately** — no wait for the in-flight async
same-stream kernels still reading it → the next alloc reuses that memory mid-flight →
corruption. DSv4 decode (per-layer fresh `HiddenStates::zeros` + drop, fast 1-token timing)
loses the reuse race; prefill (6 tokens, slower) survives.

**Why it took ~1.5h:** the garbage was identical across all 8 TP ranks (deterministic reuse,
read as "math bug") and first surfaced after `tp.all_reduce_sum` (the all-reduce read a
prematurely-freed input — innocent collective). Any buffer-clone / layer-dump / sync probe
**masked** it (the debug dump's `sources.push(stream.data.clone())` held buffers past sample)
— the probe-hides-the-race trap; the bug only reproduced *without* instrumentation.

## Fix

A **forward-level `Dsv4ForwardKeepalive`** in `dsv4.rs::forward_tokens` that holds clones of
EVERY per-call intermediate handle (all dtypes: `HiddenStates`/`DeviceVec`/`CudaSlice<f32/i32/i64/u8>`
— attention + MoE + DeepGEMM scratch + DeepEP), dropped only after `sample_cuda_token` (which
host-syncs). Unconditional (no env gate). The MoE/DeepGEMM helpers (`moe.rs`) + attention
(`attention.rs`) push their scratch into it. **Not** a global event-tracking re-enable (that
would re-add the hidden waits in the decode-graph path + risk Qwen-dense/Metal). Piecemeal
per-helper keepalive whack-a-moles (attention-only changed the wrong token 16→8760 but didn't
fix it; MoE/DeepGEMM scratch was the rest) — forward-level covers all loci at once.

## Verification

8×H20 TP=8/EP=8, `ARLE_DSV4_EXPERT_BACKEND=deepgemm`, `ARLE_DSV4_MOE_TRANSPORT=allreduce`,
prompt `[671,6102,294,8760,344]`, `max_new=16`, **no dump / no diagnostic env**, clean focused
diff (3 files), `cargo build --release -p infer-cuda --features cuda,nccl,deepep` green:
`clean_tokens = [11111,603,671,6102,294,8760,344,11111,603,671,6102,294,8760,344,11111,603]`
= **exact oracle match, 16/16**. (native-vs-deepgemm A/B both failed pre-fix → not backend-specific.)

## Rule

`DeviceContext` disables cudarc event-tracking, so eager GPU-forward paths must give explicit
lifetime discipline — a forward-level keepalive of all per-call device buffers until the
terminal host-sync. New forward paths inherit this. Symptom signature: correct-on-slow-shape /
garbage-on-fast-shape, identical across ranks, masked by any clone/dump/sync probe. See
`memory/reference_disabled_event_tracking_premature_buffer_free.md`.
