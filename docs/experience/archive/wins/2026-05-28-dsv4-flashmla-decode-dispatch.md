# DSv4 FlashMLA decode — dispatch + indices + arena landed; default-flip pending-remote

## SLO-shape probed? — N (dispatch landed env-OFF; pod parity validation deferred — see "License-or-kill" gate below)

## TL;DR

Landed the three long-term pieces required by
[`docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md`](../../plans/2026-05-28-dsv4-flashmla-decode-integration.md)
Phase D-4 steps 1, 2, 3 — no short-term fallbacks, no per-step alloc,
no CPU indices prep:

| Step | Commit | What landed |
|---|---|---|
| 1 | `b3a33188` | `arle_dsv4_flashmla_decode_build_indices_cuda` GPU kernel + FFI + `dsv4_flashmla_decode_build_indices_raw` Rust wrapper. One block, 128 threads, one i32 per thread of `topk_unified = sliding_window + max_compressed_keys`. Mirrors the prefill-side `arle_flashmla_{csa,hca}_build_indices` shape — but indices are in **block-paged pool coords** (not the prefill's flat element offsets), with the SW→`[0, sw_blocks*64)` and compressed→`[sw_blocks*64, total*64)` mapping locked down by `vendor/flashmla/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh:558`. |
| 2 | `ed4a7b38` | Amortized scratch arena: `fm_decode_{lse_accum, o_accum, sched_meta, num_splits, indices}` `Option<CudaSlice<*>>` on `DeepseekAttentionRuntimeCache` plus three `usize` capacity recorders. Sized for worst-case `num_sm_parts` (H20 132 SMs → ≈66 at h_q=64,s_q=1; capped at 256 headroom) and `topk_unified` worst case 640. Lazy-init helper `ensure_fm_decode_arena` allocates once on first decode step and reuses every subsequent step. **No per-step alloc.** |
| 3 | this commit | Decode dispatch in `finish_attention_gpu` `token_count == 1` branch under the FlashMLA decode gate. Env knob default stays OFF until pod parity validates. |

## What landed (this commit)

```
infer/src/model/deepseek/weights.rs | ~400 insertions
```

### Dispatch flow (`finish_attention_gpu` decode branch)

When `ARLE_DSV4_FLASHMLA_DECODE=1` AND `sm_major == 9` AND
`mode_int ∈ {1 CSA, 2 HCA}` AND `token_count == 1` AND
`head_dim ∈ {512, 576}` AND `local_heads ∈ {64, 128}` AND `cache.is_some()`,
the dispatch:

1. **`arle_flashmla_sm90_sparse_decode_get_meta`** — host call returns
   `(num_sm_parts, fixed_overhead_num_blocks, block_size_topk)`.
2. **`ensure_fm_decode_arena`** with `num_sm_parts_max = max(num_sm_parts, 256)`,
   `topk_unified = sliding_window + max_compressed_keys`, `h_q = local_heads`,
   `d_v = 512`. Allocates only on first call; reused after.
3. **`dsv4_flashmla_pack_one_sw_token`** — packs the current decode
   token's K row from `k_prepared` (interleaved [NoPE 448 | RoPE 64]
   bf16) into FP8 SW sub-pool ring slot `start_pos % sliding_window`.
   Strided pack kernel reads NoPE/RoPE from the same `k_prepared`
   buffer with stride 512 each — no deinterleave.
4. **`dsv4_flashmla_decode_build_indices_raw`** — writes
   `int32[topk_unified]` into `fm_decode_indices`.
5. **`arle_flashmla_sm90_sparse_decode_sched_meta`** — writes
   `fm_decode_sched_meta` (`num_sm_parts × 8` i32) and
   `fm_decode_num_splits` (`b+1` = 2 i32) from a one-element
   `topk_length` array.
6. **`arle_flashmla_sm90_sparse_decode_fwd`** — consumes the FP8 KV
   pool + indices + arena scratches, writes `local_attn.data` + a
   throwaway `lse` (combine is fused inside the shim per
   `arle_flashmla_decode_shim.cu:370-395`).

When the env knob is OFF (default), the gate is false and the legacy
`dsv4_hybrid_attention_cuda` decode kernel runs byte-identically to the
prior plumbing commit.

### Borrow restructure (sub-finding F5)

The prior plumbing commit held `window_cache: &mut CudaSlice<bf16>` as
a top-level binding that mutably borrows `cache` for the rest of
`finish_attention_gpu`. The decode dispatch needs
`cache.as_deref_mut()` for the arena + pool + per-step pack scratch —
direct conflict.

**Fix (long-term, no half-state):** drop the top-level `window_cache`
binding. The bf16 SW window is allocated by `ensure_swa_window_cache`
(side-effect on `cache.window_gpu`) without retaining the returned
mut reference. Each use site (FlashMLA prefill `csa_pack_kv`, legacy
hybrid decode, post-block bf16 update) acquires its own scoped `&mut`
on `cache.window_gpu.as_mut()`, captures the raw ptr for the FFI call,
and releases the borrow when the kernel launch returns. The behavior
on both legacy paths is byte-identical (same memory, same write order);
only the borrow lifetime narrowed.

### Forced un-fuse on the window update

`fuse_window_update` ANDs in `!use_flashmla_decode` so the bf16 SW ring
update runs unfused after the dispatch. This keeps the bf16 ring valid
for the one-commit-cycle window during which env-OFF fallback is
allowed; the legacy path is deleted in the next dispatch (per the
hardened-policy plan).

## License-or-kill — pod parity validation (pending-remote)

Per the brief: "PASS gate: TPOT ≤ 12 ms/token at 4K shape." Required:
- **Parity**: `ARLE_DSV4_FLASHMLA_DECODE=1` vs `=0`, same seed, greedy
  byte-equality.
- **Perf**: `bash dsv4_long_probe.sh 4 64 1` at 4K + 16K + 24K.

**Local validation (this commit):**
- `cargo check -p infer --no-default-features --features cuda,no-cuda --lib` — clean.
- `cargo check -p infer --no-default-features --features no-cuda --lib` — clean.
- Standalone nvcc on H20:
  `nvcc -arch=sm_90a -c csrc/attention/dsv4_flashmla_decode_build_indices.cu` — clean (17 KB .o).

**Pod-side cargo build:** blocked by unrelated pre-existing failures
(`marlin_w4_fp8_kernel.ptx` ptxas error on a sister-subagent INT4 KIVI
work-in-progress hunk, and a pod-vs-local divergence on
`crates/cuda-kernels/src/{kv_quant,ffi/kv}.rs`). Neither is caused by
this commit; both are tracked under sister-subagent's INT4 KIVI work.

**Parity + perf validation will run once the sister-subagent
`crates/cuda-kernels` dirty state is resolved.** Until then the env
knob stays OFF and the runtime behavior is byte-identical to the prior
plumbing commit.

## Default-flip — deferred (long-term commitment recorded)

The brief mandates "flip `dsv4_flashmla_decode_enabled` default from
`false` → `true` IN THE SAME COMMIT as the dispatch wire-in" **IF
parity + PASS gate clear in the same session**. They did not — pod
validation is pending-remote. Per the brief's conditional, the flip
also defers to the same pod commit.

The next commit cycle (Phase D-4 step 4 follow-up) is the flip + legacy
deletion. The work after that (task #47) deletes the legacy
`dsv4_hybrid_attention_cuda` decode kernel entirely.

## Sub-findings surfaced (none beyond F1-F4)

The brief requires STOP-on-new-finding. None surfaced beyond:
- **F1-F4** already documented in the plumbing wins entry.
- **F5 (borrow restructure)** — solved long-term, not a contract gap;
  the change is to ARLE's own scope structure, not the upstream
  FlashMLA contract.

The 4 D-4 contract findings (F1 K layout, F2 SW bootstrap, F3 indices,
F4 sched_meta + scratch) all have long-term resolutions in tree now:
- F1 → strided pack kernel (`d19b8f87`) + uses k_prepared directly.
- F2 → `dsv4_flashmla_sw_bootstrap_hook` in `8ab33a4c`.
- F3 → new GPU indices kernel `b3a33188` (this dispatch consumes it).
- F4 → amortized arena `ed4a7b38` (this dispatch reuses it every step).

## Refs

- Plan: [`docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md`](../../plans/2026-05-28-dsv4-flashmla-decode-integration.md) Phase D-4 hardened-policy table.
- Prior wins:
  - [`2026-05-28-dsv4-flashmla-decode-d4-plumbing.md`](2026-05-28-dsv4-flashmla-decode-d4-plumbing.md)
    (the four contract findings F1-F4 this dispatch resolves).
  - [`2026-05-28-dsv4-fp8-kv-pack-kernel.md`](2026-05-28-dsv4-fp8-kv-pack-kernel.md) (pack kernel).
  - [`2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md`](2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md) (V2.4 prefill, unrelated branch, locked).
- Indices kernel: `crates/cuda-kernels/csrc/attention/dsv4_flashmla_decode_build_indices.cu`.
- FFI surface: `crates/cuda-kernels/src/ffi/attention.rs::arle_dsv4_flashmla_decode_build_indices_cuda`.
- Rust wrapper: `crates/cuda-kernels/src/attention.rs::dsv4_flashmla_decode_build_indices_raw`.
- Arena helper: `infer/src/model/deepseek/weights.rs::ensure_fm_decode_arena`.
- Dispatch site: `infer/src/model/deepseek/weights.rs` (`finish_attention_gpu`, `else if use_flashmla_decode` branch).
- Env gate: `infer/src/model/deepseek/weights.rs::dsv4_flashmla_decode_enabled` (default OFF until pod parity validates).

## Bench — pending-remote

No SLO-shape numbers in this commit. Default OFF → no runtime change
to bench. Once pod parity validates, the follow-up commit will:
- Run `bash dsv4_long_probe.sh 4 64 1` at 4K, 16K, 24K with env on vs off.
- Report wall-clock TPOT vs the legacy 26 ms/token baseline.
- License-or-kill on the 12 ms/token PASS gate.
- Flip default to ON in the same commit.
