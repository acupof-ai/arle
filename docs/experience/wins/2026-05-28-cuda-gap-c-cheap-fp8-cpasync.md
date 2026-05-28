# GAP-C-cheap — `cp.async` pipeline for FP8 KV decode partial kernel

Related: [`docs/research/2026-05-28-arle-kernel-vs-sota-audit.md`](../../research/2026-05-28-arle-kernel-vs-sota-audit.md) §GAP-C.
INT8 sibling reference: [`docs/experience/wins/2026-05-27-int8-kv-kivi-per-channel-k-fix.md`](2026-05-27-int8-kv-kivi-per-channel-k-fix.md).

## Context

The 2026-05-28 SOTA audit flagged the FP8 per-channel-K decode partial
kernel (`decode_attention_fp8_per_channel_k_partial_kernel` in
`crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu`) as the
only quantized attention path still issuing blocking
`__nv_fp8x4_e4m3` global loads from inside the QK / PV compute loop.
The INT8 sibling (`decode_attention_int8_per_channel_k_partial_kernel`)
already used `__pipeline_memcpy_async` double-buffered tiles since the
2026-05-27 KIVI landing (`8afecffe`); FP8 had been left behind, with a
TODO-style comment in the INT8 header explicitly noting "cp.async
pipelining (which the INT8 sibling uses but the FP8 sibling does not)".

Audit named this gap **GAP-C-cheap**, ~80 LoC, ~2-3% Qwen3.5 quantized
decode wall-clock estimate.

## What worked

Direct mirror port of the INT8 cp.async template to the FP8 kernel:

1. **Shared-mem double buffer**: replace blocking
   `__nv_fp8x4_e4m3 packed = *reinterpret_cast<...>(K_data + base + d)`
   from inside the compute loop with
   ```c
   __shared__ __nv_fp8_e4m3 smem_k[2][TILE_TOKENS][HEAD_DIM];
   __shared__ __nv_fp8_e4m3 smem_v[2][TILE_TOKENS][HEAD_DIM];
   __shared__ float smem_v_scales[2][TILE_TOKENS];
   ```
   `TILE_TOKENS=16` (= `kQuantPageSize`, already module-scoped).
2. **`preload_page` lambda** issuing `__pipeline_memcpy_async` for K, V,
   and the V scale (K scale is per-channel and stays in registers from
   kernel entry), followed by `__pipeline_commit()`.
3. **Main loop** swaps stages on `page_local_idx & 1`, calls
   `__pipeline_wait_prior(0)` + `__syncthreads()`, fires the next tile's
   prefetch, then runs the existing online-softmax math reading from
   `smem_k[stage]` / `smem_v[stage]` / `smem_v_scales[stage]` instead of
   global memory.
4. **Math identical**: `__nv_fp8x4_e4m3` cast from the smem tile (instead
   of the previous global cast) drives the same vectorized dequant; KIVI
   asymmetric scheme (per-channel K + per-(row, head) V) preserved; the
   normalized-write contract from the 2026-05-26 fix is unchanged.
5. **Stale comments** at the INT8 header ("FP8 sibling does not [use
   cp.async]") and at the INT8 normalized-write block ("FP8 sibling at
   line ~766") were refreshed — both kernels now share the same
   pipelined structure and the cross-reference no longer needs the
   "yet" qualifier.

Diff size: 111 LoC vs the pre-change kernel (72 ins / 39 del), matching
the audit's ~80 LoC estimate.

## Local verification

- `crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu`
  compiles in the CSRC build chain (compile-side errors would surface
  during nvcc). Mac-host `cargo check -p infer --no-default-features
  --features cuda,no-cuda` is unrelated to this kernel (Rust-only
  typecheck) and the two errors that surface in `main.rs` /
  `scheduler/types.rs` are pre-existing at HEAD (separate NCCL
  scaffolding WIP, not introduced here).

## Pod validation — pending-remote

**Cannot run TPOT bench from this workspace right now.** The pod-exec
path is currently broken:

- `tn doctor` reports `ssh 127.0.0.1:12222: handshake failed — server
  might not be sshd (read: connection reset by peer)` for the `arle`
  host configured to the tunnel.
- Direct `/usr/bin/ssh -J jumpecs-hl.byted.org root@180.184.176.218`
  via `~/bin/pod-exec` is rejected at the jumpbox with `the server not
  allowed channel type: session`.
- `tn doctor` also reports `digest` host auth-failed.

This is an environment-side outage (jumpbox channel-type policy / SSH
tunnel reset), not a build problem with the change. Per CLAUDE.md
benchmark policy, this entry is `pending-remote` for the matched-A/B
TPOT numbers.

**Expected acceptance (when pod returns):**

| Workload | Acceptance |
|---|---|
| Qwen3.5-4B FP8 KV, c=1, 4k prompt | TPOT ≥ break-even (Δ ≤ +1%) |
| Qwen3.5-4B FP8 KV, c=16, 4k prompt | TPOT improvement ≥ 1% (audit estimate 2-3%) |

If c=16 wall-clock regresses > 1% on pod re-run, revert the FP8
kernel changes (commit on the same file) and write a paired errors
entry — that's the GAP-C-cheap kill condition.

## Provenance

- HEAD commit containing the FP8 cp.async kernel: `ab850f7a`. This
  commit was authored by a concurrent process in the workspace
  (probably a /loop or background agent) that bundled the FP8 kernel
  delta with a separate GAP-A planning doc and shipped it under a
  `docs(cuda):` message. The FP8 kernel content matches this entry's
  description and was confirmed via `git show ab850f7a:.../decode_attention_quantized.cu`.
- The INT4 WIP visible in `git status` (modified `K_dynamic_scales` /
  `k_dynamic` paths) is unrelated to this change — it remained
  unstaged through the FP8 work and is not part of GAP-C-cheap.
- Reflog: `ab850f7a HEAD@{1}` (concurrent agent) →
  `9ffaa622 HEAD@{0}` (this agent's intended FP8 commit, but the
  index/working-tree race left it carrying only the INT4 user-WIP
  delta; soft-reset and unstaged immediately to restore hygiene).

## Rule

When the INT8/FP8/INT4 quantized attention siblings diverge on a
structural optimization (cp.async pipelining, register pre-load, smem
double buffer), the audit asymmetry is the cheap win — direct mirror
port works because the algorithm (KIVI per-channel K + per-(row, head)
V, online-softmax, split-KV partials) is identical across precisions
and only the dequant cast changes.

## Next

- Re-run TPOT bench on pod when SSH path is restored; fill in the
  Δ% numbers and flip this entry from `pending-remote` to confirmed.
- GAP-C-medium (MMA QK / softmax via `mma.m16n8k16`) is the followup
  per audit — separate commit, separate review.

## Refs

- [`docs/research/2026-05-28-arle-kernel-vs-sota-audit.md`](../../research/2026-05-28-arle-kernel-vs-sota-audit.md) — GAP-C section
- [`docs/experience/wins/2026-05-27-int8-kv-kivi-per-channel-k-fix.md`](2026-05-27-int8-kv-kivi-per-channel-k-fix.md) — INT8 template this commit mirrors
- INT8 sibling: `decode_attention_int8_per_channel_k_partial_kernel`
- HEAD commit: `ab850f7a`
- KIVI: <https://arxiv.org/abs/2402.02750>
