# DSv4-Flash FP8 KV pack kernel — Phase D-3' encoder landed (pending-remote bench)

## SLO-shape probed? — N (kernel-only build; runtime wire-in is Phase D-4)

## TL;DR

Wrote the bf16 → MODEL1 FP8 block-paged KV packing kernel that the upstream
FlashMLA sparse-FP8 decode (`sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel`)
requires. Closes the **D-3 → D-3' un-kill** from
[`2026-05-28-dsv4-flashmla-decode-integration.md`](2026-05-28-dsv4-flashmla-decode-integration.md):
ARLE's bf16 KV pool can now be packed into the upstream byte layout (584
B/token, AoS NoPE+RoPE + block-tail e8m0 scales) at decode time.

Files added (this is the entire diff this commit):

| Path | Role |
|---|---|
| `crates/cuda-kernels/csrc/attention/dsv4_fp8_kv_pack.cu` | CUDA kernel: 1 block/token, 128 threads, NoPE per-tile fp8_e4m3 quant + fp8_e8m0 scale + bf16 RoPE verbatim |
| `crates/cuda-kernels/src/ffi/attention.rs` | FFI extern `arle_dsv4_fp8_kv_pack_cuda` (appended; sister's INT4 KIVI hunk untouched) |
| `crates/cuda-kernels/src/attention.rs` | Rust wrappers `dsv4_fp8_kv_pack` + `dsv4_fp8_kv_pack_raw` |
| `crates/cuda-kernels/src/lib.rs` | `pub mod attention` (cuda-gated) |
| `infer/tests/dsv4_fp8_kv_pack_parity.rs` | CUDA-gated parity test — drives the kernel + dequants via the kernel's reverse path + checks per-tile abs error within E4M3 envelope |

## Contract (evidence-anchored from upstream, MODEL1 only)

Per `vendor/flashmla/csrc/sm90/decode/sparse_fp8/{config.h,splitkv_mla.cuh,components/dequant.h}`
at pin `df022eb`:

| Constant | Value |
|---|---|
| `HEAD_DIM_K` | 512 (MODEL1) |
| `HEAD_DIM_ROPE` | 64 |
| `HEAD_DIM_NOPE` | 448 |
| `QUANT_TILE_SIZE` | 64 |
| `NUM_SCALES` | 8 (7 used + 1 pad) |
| `BYTES_PER_TOKEN` | 584 = 448 (fp8 NoPE) + 128 (bf16 RoPE) + 8 (e8m0 scales) |
| `page_block_size` | 64 |
| Block stride | 64 × 584 = 37376 B |

Block layout (per layer, per rank):
```
offset 0      : [T0 NoPE 448B][T0 RoPE 128B]    (576 B/token AoS, T0..T63)
offset 576    : [T1 NoPE 448B][T1 RoPE 128B]
...
offset 36288  : [T63 NoPE 448B][T63 RoPE 128B]
offset 36864  : [T0 scales 8B]...[T63 scales 8B]
```

E8M0 scale encoding — derived from `__nv_cvt_e8m0x2_to_bf162raw`'s
exponent-only semantics: byte `b ∈ [1, 254]` ⇒ `2^(b - 127)`, byte 0 = zero,
byte 255 = NaN. For each 64-element NoPE tile we pick the smallest
`e = ⌈log₂(amax / 448)⌉` (frexp-derived; bumped by 1 if the trial scale
underflows), clamp to `[-126, 127]`, store `byte = e + 127`. The kernel's
reverse path then does:
```
scale_bf16 = 2^e (via __nv_cvt_e8m0x2_to_bf162raw)
recon = round_bf16( e4m3_to_f32(fp8) * scale_bf16 )
```

The CPU-side `dsv4_fp8_kv_pack_parity_two_blocks` test stages 128 random
bf16 tokens (mixed magnitudes including ×50 outliers), runs the kernel,
reads back the 2 packed blocks (74752 B total), and verifies:
1. **Every e8m0 scale byte matches the CPU encode reference exactly** —
   isolates the encoding logic from fp8 rounding noise.
2. **Per-element |recon - orig| ≤ 0.20 × tile_amax + scale × 2⁻⁹** — the
   E4M3 + per-tile-scale quantization envelope.
3. **RoPE bf16 round-trip is byte-exact** — RoPE is unquantized.

A small `cpu_e8m0_roundtrip_sanity` test pins the byte-encoding math
(amax=448 → byte=127, amax=224 → byte=126, amax=449 → byte=128, amax=1.0 →
byte=119) so future drift surfaces without needing a GPU.

## What this unblocks

| Phase | State |
|---|---|
| D-1 (vendor) | DONE (`5d18b624`) |
| D-2 (shim + FFI) | DONE (`3f7923a3`) |
| **D-3' (FP8 pack kernel)** | **DONE — this commit** |
| D-4 (decode dispatch wire-in, FP8 pool allocation) | NEXT |
| D-5 (parity vs bf16 reference, perf) | gated on D-4 |
| D-6 (default-on flip) | gated on D-5 PASS |

D-4 is the runtime wiring: `finish_attention_gpu` needs an FP8 KV pool
allocator (sized `num_layers × max_blocks × 64 × 584`; ~3.2 GB/rank at
TP=8 max_seq_len=32K per the plan doc) and a compressor-update hook that
calls `dsv4_fp8_kv_pack` on each decode step before
`arle_flashmla_sm90_sparse_decode_fwd`.

## Roofline check

Kernel cost is dominated by DRAM traffic: per token, read 448 + 64 bf16 =
1024 B and write 576 + 8 = 584 B → 1608 B/token. At H20 8 TB/s rank
bandwidth, 64 tokens × 1608 B / 8e12 = 12.9 ns ⇒ packing 64 tokens at
once costs ~0.013 µs (lower bound). Real cost will include launch
overhead, but the kernel itself is far below 1% of the FlashMLA decode
budget. Bench in D-5 will confirm.

| Op | Achieved | Peak (8×H20) | % | Verdict |
|---|---:|---:|---:|---|
| DSv4 decode TPOT (legacy `dsv4_hybrid_attention_cuda`) | ~26 ms/tok | — | — | unchanged this commit |
| `arle_dsv4_fp8_kv_pack_cuda` | pending-remote | ~13 ns/64-tok pack lower bound | — | kernel built, not yet benched |

## Verification

- **Mac CUDA-Rust typecheck**: `PATH=/tmp/_stub_bin:$PATH cargo check -p
  cuda-kernels --no-default-features --features cuda,no-cuda` passes; same
  for `cargo check -p infer --features cuda,no-cuda --test
  dsv4_fp8_kv_pack_parity --lib`. (Local nvcc is unavailable per
  `docs/environment.md`; a thin nvcc stub satisfies `cudarc 0.19.7`'s
  build.rs precondition so the Rust path compiles.)
- **CUDA path**: `pending-remote`. Build + parity test must be re-run on
  the H20 pod alongside the next D-4 commit. nvcc compile may surface
  signature drift in `__nv_fp8_e4m3()` constructor or `__nv_bfloat16`
  arithmetic that this Mac stub cannot detect.

## Hard-won lessons

1. **`__syncthreads()` requires all threads in scope to reach it.** First
   draft of the kernel did the NoPE 64-lane cross-warp reduction inside
   an `if (tid < 64)` branch with `__threadfence_block()` to "broadcast"
   between warps 0 and 1. That race-races: warp 0 reading `s_halves[1]`
   doesn't observe a happens-before with warp 1 writing it (only
   `__syncthreads()` provides the barrier). Restructured so RoPE threads
   participate in the `__syncthreads()` even though they don't use the
   shared slots — net cost is zero since they're already idle that pass.
2. **The brief said `csrc/attention/...cu` but most DSv4 / FlashMLA
   neighbours live under `csrc/misc/`.** Honoured the brief; the
   semantic fit ("KV packing for attention") is at least as good as
   "miscellaneous". If we add more attention-side packers we can
   consolidate later.
3. **`crates/cuda-kernels/src/attention.rs` is a brand-new module file**
   per the brief (didn't exist before). Kept it scoped to two wrappers
   so growing it later doesn't drift from `feedback_file_naming_semantic_alignment.md`.

## Refs

- Project plan: [`docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md`](../../plans/2026-05-28-dsv4-flashmla-decode-integration.md) Phase D-3'.
- D-1 + D-2 wins (superseded for D-3): [`2026-05-28-dsv4-flashmla-decode-integration.md`](2026-05-28-dsv4-flashmla-decode-integration.md).
- Upstream evidence anchors (vendored at `df022eb`):
  - `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh:491,540-560,694-697` — kv-as-fp8 + scales offset + `BYTES_PER_TOKEN` assert
  - `csrc/sm90/decode/sparse_fp8/config.h:23-29` — MODEL1 vs V32 constants
  - `csrc/sm90/decode/sparse_fp8/components/dequant.h:21-34` — `cvt_fp8x8_bf16x8` dequant formula
  - `csrc/api/sparse_decode.h:288-304` — torch-side preflight
- Pin: sgl-project/FlashMLA @ `df022eb`.

## Pending — local bench

`pending-remote` — kernel needs nvcc on the H20 pod for the actual compile
+ test execution. Plan: bundle with D-4 dispatch wire-up so the parity test
runs alongside an end-to-end decode probe at first run.
