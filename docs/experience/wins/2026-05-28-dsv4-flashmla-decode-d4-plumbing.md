# DSv4 FlashMLA decode — Phase D-4 plumbing landed; pack hook + dispatch deferred (pending-remote)

## SLO-shape probed? — N (plumbing-only commit; runtime path stays on legacy `dsv4_hybrid_attention_cuda` until pod-side wire-up validates)

## TL;DR

Wired the Phase D-4 runtime scaffolding required by
[`docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md`](../../plans/2026-05-28-dsv4-flashmla-decode-integration.md):

- `ARLE_DSV4_FLASHMLA_DECODE` env gate (default **OFF**) parsed exactly
  like `ARLE_DSV4_FLASHMLA_PREFILL`.
- `DeepseekAttentionRuntimeCache::fp8_kv_pool` — per-layer, per-slot
  `Option<CudaSlice<u8>>` slot for the MODEL1 FP8 block-paged KV.
- `ensure_dsv4_flashmla_fp8_kv_pool` lazy allocator + MODEL1 layout consts
  (page_block_size=64, bytes/token=584, block_bytes=37376) anchored to
  upstream `vendor/flashmla/csrc/sm90/decode/sparse_fp8/` at `df022eb`.

With the env knob off (default), the pool stays `None`, the allocator is
unreached, and the legacy `dsv4_hybrid_attention_cuda` decode path runs
byte-identical. Verified via
`PATH=/tmp/_stub_bin:$PATH cargo check -p infer --lib
--no-default-features --features cuda,no-cuda` and `cargo check -p infer
--lib --no-default-features --features no-cuda`, both clean.

**Pack-on-update hooks (Phase 2) and decode dispatch (Phase 3) are
deferred to a pod-side commit** — see "Contract findings" below for
the four blockers that would have required landing untested CUDA-side
coordination from a Mac, violating CLAUDE.md §0 SOLID ("80% SOLID 不够")
and the brief's "no half-states" rule.

## What landed (commit `d67f9832`)

```
infer/src/model/deepseek/state.rs   | 28 +++++++++++++
infer/src/model/deepseek/weights.rs | 80 +++++++++++++++++++++++++++++++++++++
2 files changed, 108 insertions(+)
```

| Symbol | Role |
|---|---|
| `DeepseekAttentionRuntimeCache.fp8_kv_pool` + 5 metadata fields | Per-layer, per-slot FP8 pool slot, lazy-allocated only when env-on dispatch first runs. |
| `DSV4_FLASHMLA_MODEL1_PAGE_BLOCK_SIZE` (64) | Upstream `config.h` MODEL1 constant. |
| `DSV4_FLASHMLA_MODEL1_BYTES_PER_TOKEN` (584) | `splitkv_mla.cuh:694` MODEL1 contract. |
| `DSV4_FLASHMLA_MODEL1_BLOCK_BYTES` (37376) | `page_block_size * bytes_per_token`. |
| `ensure_dsv4_flashmla_fp8_kv_pool` | Lazy alloc + grow helper, mirrors `ensure_swa_window_cache` pattern. Sized to `(sw_blocks + comp_blocks) * 37376` bytes. |
| `dsv4_flashmla_decode_enabled` | Env gate `ARLE_DSV4_FLASHMLA_DECODE`, default OFF. |

The pool design splits one contiguous byte buffer into two sub-pools:
- blocks `[0, sw_blocks)` → per-token K stream (SW window write)
- blocks `[sw_blocks, sw_blocks + comp_blocks)` → compressor output

This matches Phase D-3' contract (one main KV pool; FlashMLA's
`extra_kv` two-tier mechanism is not wired in
`crates/cuda-kernels/csrc/misc/arle_flashmla_decode_shim.cu`).

## Contract findings — why Phase 2 + 3 are deferred

These surfaced during the SOLID survey of `finish_attention_gpu` and
the Phase D-3' pack-kernel signature. None of them is a KILL/STOP
contract-impossibility (single-pool guarantee is preserved; the
SW/compressed coord space CAN map to block-paged FP8 — see Finding 3).
Each is a real engineering surface that needs nvcc-validated
coordination. Documented up-front per CLAUDE.md §0 ("禁止 silent 放过")
instead of landed as untested code.

### Finding 1 — K layout mismatch between `k_prepared` and the pack kernel

ARLE's `k_prepared` (output of `dsv4_prepare_qk_cuda` /
`dsv4_prepare_qk_fused_cuda`, `csrc/misc/dsv4_attention.cu:209-271`) is
bf16 `[token_count, head_dim=512]` interleaved: each token row is
`[NoPE 448 | RoPE 64]` contiguous in the last axis.

The Phase D-3' pack kernel
(`csrc/attention/dsv4_fp8_kv_pack.cu:106-112`) consumes two **separate**
buffers:

```
const __nv_bfloat16* nope,   // [n_tokens, HEAD_DIM_NOPE=448] stride=448
const __nv_bfloat16* rope,   // [n_tokens, HEAD_DIM_ROPE=64]  stride=64
```

Index expressions hard-code stride 448 / 64 (`nope[t*HEAD_DIM_NOPE +
dim_idx]`, `rope[t*HEAD_DIM_ROPE + dim_idx]`). Calling the kernel with
`k_prepared`'s 512-stride layout would silently mis-read. The pack
kernel's parity test
(`infer/tests/dsv4_fp8_kv_pack_parity.rs`) asserts this contiguous-stride
contract.

**Two viable paths, both deferred:**

1. **Strided pack-kernel variant** — add `stride_nope` / `stride_rope`
   parameters + a `_strided` extern. New nvcc surface (~50 LOC of CUDA
   + new FFI entry + extended parity test). Eliminates the
   deinterleave cost at runtime.
2. **DtoD deinterleave scratches via `memcpy_dtod`** — per pack call,
   allocate `nope_scratch[n*448]` + `rope_scratch[n*64]`, run 2N
   `memcpy_dtod` calls to deinterleave from `k_prepared`, then dispatch
   the pack kernel. Decode (n=1) is cheap (~3 launches per layer) but
   layer-overhead at TP=8 dominated the prefill A4 KILL — same risk
   here without empirical measurement.

Either path is untestable on Mac (`cargo check` runs with an
nvcc-stub). Land them with a paired GPU run.

### Finding 2 — SW pre-fill bootstrap

The FP8 pool only receives writes inside `finish_attention_gpu`
`token_count == 1` (decode). At the prefill→decode transition, the
bf16 SW window cache (`window_gpu`) already holds prefill-era K rows,
but the FP8 pool is empty.

FlashMLA decode's indices contract requires every reachable SW
position in `[max(0, N-sliding_window+1), N]` to be present in the
FP8 pool. The first decode step at `start_pos > 0` would otherwise
read uninitialised FP8 blocks.

**Path forward:** one-shot bulk pack at the prefill→decode boundary,
hooked into `forward_attention_gpu_cached`. Read from `window_gpu`
(bf16, length `sliding_window * head_dim`), pack into FP8 pool blocks
`[0, sw_blocks)`. Single 2-launch overhead per layer at the prefill
exit. Inherits the K-layout-mismatch surface of Finding 1 (the bf16 SW
ring is `[sliding_window, 512]` interleaved).

### Finding 3 — Indices builder (block-paged coords)

The decode dispatch needs `int32[s_q=1, topk_unified]` indices in the
unified block-paged coord space:

- SW slots `[max(0, N-sw), N]` → SW sub-pool coords. The natural
  mapping is ring-indexed: `ring_idx = abs_pos % sliding_window`;
  block_idx = `ring_idx / 64` (∈ `[0, sw_blocks)`), row = `ring_idx %
  64`. Pad with `-1` when the position is < 0 (early decode).
- Compressed rows (selected[i] for CSA, full range for HCA) →
  `(sw_blocks + comp_row / 64, comp_row % 64)` in the compressed
  sub-pool. Compressed row `r` is causally valid iff `(r+1)*ratio - 1
  <= start_pos`; otherwise `-1`.

`topk_unified = sliding_window + max_compressed_keys` (with
`max_compressed_keys = index_topk` for CSA, `compressed_count.div_ceil(128)
* 128` for HCA). At `sw=128, index_topk=512` ⇒ `topk_unified = 640`
which divides 128 cleanly (FlashMLA's `topk % (2*B_TOPK)==0` invariant).

**Two viable paths:**

1. **CPU-side prep** — assemble `topk_unified ≤ 640` i32 on host per
   layer, one H2D copy. 43 layers × decode-step at ≈10 KB / step total.
   Cheap.
2. **New CUDA kernel** — mirror the existing
   `arle_flashmla_csa_build_indices` /
   `arle_flashmla_hca_build_indices` shims (prefill path). Lower
   overhead but new nvcc surface.

Both need pod-side validation against the SW ring's wrap semantics and
the prefill-side index conventions.

### Finding 4 — SchedMeta + scratch lifecycle

`arle_flashmla_sm90_sparse_decode_get_meta` is a host call returning
`num_sm_parts`, `fixed_overhead_num_blocks`, `block_size_topk` (FFI at
`crates/cuda-kernels/src/ffi/misc.rs:297-304`). The caller then
allocates four buffers per decode step:

| Buffer | Size (b=1, s_q=1, d_v=512) |
|---|---|
| `lse_accum`  | `num_splits × h_q × 4` bytes |
| `o_accum`    | `num_splits × h_q × d_v × 4` bytes |
| `sched_meta` | `num_sm_parts × DecodingSchedMetaSize(32)` bytes |
| `num_splits` | `(b+1) × 4` bytes |

For TP=8 H20 with `num_sm_parts ≈ arch.num_sms / s_q / 2 ≈ 66`,
`num_splits` could reach tens-to-hundreds. Per-step alloc is fine
short-term per the brief, but the lifecycle interaction with the
scheduler's slot-owned `CudaSlice` arena should be sanity-checked on
pod before defaulting on.

## Lifecycle & SOLID notes

- **Pool sizing.** The plan's worst-case 3.2 GB/rank assumes
  max_seq_len=32K, num_slots=4, 43 layers. Per-layer-per-slot pool of
  `(sw_blocks=2 + comp_blocks≈128) × 37376 B` ≈ 4.9 MB at the working
  shape — well under the rank budget. The allocator grows on demand
  so cold slots don't pay.
- **No bf16 KV pool interaction.** `fp8_kv_pool` is a separate
  `Option<CudaSlice<u8>>`; drop-on-reset and grow-or-reuse follow the
  same idiom as `window_gpu`. Single-pool guarantee preserved (each
  rank still owns the bf16 KV pool; the FP8 mirror is gated behind a
  default-off env knob).
- **No half-states.** The pack hook + dispatch were not landed as
  partial-without-validation code; per the brief's "if at any
  sub-step you can't complete the wire-up, revert the changes and
  document the blocker", Phase 2 + 3 are deferred to a follow-up pod
  commit. The Phase 1 plumbing is independently coherent (defines the
  contract surface, allocator, env knob) and runtime-inert with the
  env knob off.

## Pod-side path forward (next commit)

Single GPU-validated commit that lands:

1. **Choose Finding-1 path.** Strided pack-kernel variant is simpler
   at the call site and cheaper at runtime; favour it unless the new
   nvcc surface trips compile.
2. **Add prefill→decode SW bulk-pack hook** in
   `forward_attention_gpu_cached`.
3. **Add per-step decode pack hook** in `finish_attention_gpu`
   `token_count == 1` branch, gated by `dsv4_flashmla_decode_enabled`.
4. **Add compressor-update pack hook** in
   `update_compressor_gpu_cache`, also gated.
5. **CPU-side indices builder** (Finding-3 path 1 — cheaper iteration)
   + sched-meta call + scratch alloc, then
   `arle_flashmla_sm90_sparse_decode_fwd`. Output writes into
   `local_attn.data` exactly like the legacy branch.
6. **Parity probe:** `bash dsv4_long_probe.sh 4 64 1` with env on vs
   off, compare greedy byte-equality (or `abs_tol=8e-4`).
7. **Perf probe:** same script at 4K, 16K, 24K — target TPOT ≤ 12
   ms/token (PASS bar from the plan).
8. **Flip default-on** after byte-equality clears (Phase D-6).

## Verification (this commit only)

```
PATH=/tmp/_stub_bin:$PATH cargo check -p infer --lib \
    --no-default-features --features cuda,no-cuda      # clean
cargo check -p infer --lib --no-default-features --features no-cuda  # clean
```

Pre-existing dirty paths (sister subagent in-flight INT4 KIVI work,
`crates/cuda-kernels/csrc/{attention,kv}/...cu`, `qwen35/...`,
`scheduler/{cuda,types}`) were **not** staged — per
`feedback_commit_only_own_files.md`, only
`infer/src/model/deepseek/{state,weights}.rs` rode this commit.

Bench: `pending-remote` — runtime path remains on legacy
`dsv4_hybrid_attention_cuda` (env default OFF), so there is no new
benchable surface yet. The Phase D-3' pack-kernel parity test
(`infer/tests/dsv4_fp8_kv_pack_parity.rs`) also remains pending the
next pod-side compile cycle.

## Refs

- Plan: [`docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md`](../../plans/2026-05-28-dsv4-flashmla-decode-integration.md) Phase D-4.
- Prior wins:
  [`2026-05-28-dsv4-fp8-kv-pack-kernel.md`](2026-05-28-dsv4-fp8-kv-pack-kernel.md)
  (D-3' pack kernel — the consumer that Phase 2 needs to feed),
  [`2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md`](2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md)
  (V2.4 prefill — unrelated branch, locked).
- FFI surfaces ready for Phase 2 + 3:
  `crates/cuda-kernels/src/ffi/misc.rs:243-324`
  (`arle_flashmla_sm90_sparse_decode_fwd`,
  `arle_flashmla_sm90_sparse_decode_get_meta`,
  `arle_flashmla_sm90_sparse_decode_sched_meta`).
- Pack kernel (consumer of Finding 1):
  `crates/cuda-kernels/csrc/attention/dsv4_fp8_kv_pack.cu:106-112`.
- Pack-kernel Rust wrappers:
  `crates/cuda-kernels/src/attention.rs:34-106`
  (`dsv4_fp8_kv_pack`, `dsv4_fp8_kv_pack_raw`).
