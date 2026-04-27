# CUDA TileLang decode HD256 — guidellm sweep, cuda, 2026-04-27

> Tranche 3 stub — HD256 paged-decode swap to TileLang AOT cubins.
> Created at the same time as the implementation tranche per
> `docs/plans/tilelang-integration.md` §8 (full-integration / "全部接入"
> series). Replace this file with a completed entry (and a separate
> `…-off` baseline entry) once the L4 + H100 sweeps run.

## Goal

- Optimization (per `docs/bench-and-trace-spec.md` §goal taxonomy): measure
  end-to-end TTFT / ITL / saturation throughput delta between the
  FlashInfer HD256 paged-decode path (default build,
  `flashinfer_batch_decode_hd256_run`) and the TileLang HD256 decode path
  (`--features cuda,tilelang-attn`) on the Qwen3.5 full-attention layers.
  Coverage: Qwen3.5-0.8B (q8/kv2), Qwen3.5 medium / 14B / 32B-class
  (q16/kv4), Qwen3.6 MoE 30B-A3B (q16/kv2) on L4 floor and H100.

## Hypothesis

- The TileLang HD256 decode kernel mirrors the HD256 prefill kernel
  one-for-one on the FFI / dispatch side; the cubin internals differ only
  where the decode shape demands it: `BLOCK_M=1` (one Q row per request),
  `BLOCK_N=64`, no causal mask (qlen=1 means the single Q row legally
  attends to every KV position), grid `(1, num_q_heads, batch_size)`.
  FlashAttention-2 online softmax + paged-KV walk are line-for-line the
  upstream `tile-ai/tilelang/examples/flash_attention/example_gqa_decode*`
  template the L4 floor wins entry (2026-04-26) told us to mirror.
- Decision threshold (Phase-1 §5 of plan):
  **≥10% on TTFT/ITL or saturation tok/s** to advance the swap as the
  default; ±5% flat → ship-and-hold under flag; ≥5% worse → keep
  FlashInfer canonical (flag off-by-default — already the case).

## Command

```bash
# Off (default build, FlashInfer HD256 paged-decode)
scripts/bench_guidellm.sh tilelang-decode-hd256-off

# On (TileLang HD256 paged-decode)
scripts/bench_guidellm.sh tilelang-decode-hd256-on
```

Invoked via: pending remote L4 floor + H100 hosts (user-driven verification).

## Environment

- **Backend:** cuda
- **Models:** Qwen/Qwen3.5-0.8B, Qwen/Qwen3.6-30B-A3B (MoE), Qwen/Qwen3.5-medium
  / 14B / 32B-class (all bf16; full-attn head configs (8,2), (16,2), (16,4)
  — covers all three AOT specializations).
- **Hardware:** pending remote NVIDIA L4 + H100 hosts
- **Commit:** pending; will be filled in once the implementation tranche
  lands and the remote sweep runs.
- **Feature set (off):** `CUDA_HOME=/usr/local/cuda cargo build --release --features cuda` (workspace root)
- **Feature set (on):**  `CUDA_HOME=/usr/local/cuda cargo build --release --features cuda,tilelang-attn` (workspace root)
- **Non-default flags / env vars:** `INFER_TILELANG_PYTHON` if a non-default
  Python interpreter is needed for AOT, otherwise none.
- **Server launch (off):** `INFER_FEATURES="cuda" scripts/start_infer.sh models/Qwen3.5-0.8B 8000`
- **Server launch (on):**  `INFER_FEATURES="cuda,tilelang-attn" scripts/start_infer.sh models/Qwen3.5-0.8B 8000`
  Both invocations build and run the `infer` binary directly so the
  TileLang feature reaches the Qwen3.5 full-attention decode path
  (Tranche 3 — joins HD128 prefill from Phase 0, HD256 prefill from
  Tranche 2, and TC-decode HD128 alias from Tranche 4).

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260427`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh tilelang-decode-hd256-{on,off}`

## Results

- Status: `pending-remote`
- Local verification completed (macOS workspace):
  - `cargo check -p infer --no-default-features --features cuda,no-cuda`
  - `cargo check -p infer --no-default-features --features cuda,no-cuda,tilelang-attn`
  - Both `cfg` arms of `ops::flashinfer_run_layer_hd256` compile cleanly.
  - The single Qwen3.5 HD256 decode call site (BF16 path in
    `qwen35/batch_decode.rs`) passes the new `batch_size`, `max_qlen`,
    `total_pages` args and a `qo_indptr_gpu` slice; the FlashInfer
    `plan_hd256` call in `plan_attention` is cfg-gated behind
    `cfg(not(feature = "tilelang-attn"))`.

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| synchronous | pending | pending | pending | pending | pending | pending |
| saturation  | pending | pending | pending | pending | pending | pending |

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | pending |
| peak waiting | pending |
| peak prefill_queue | pending |
| peak kv_util | pending |
| `prefix_hit_rate` | pending |

## Problems

- No CUDA runtime host is available in this macOS workspace, so the
  canonical `guidellm` sweep is pending.
- Numerical-parity guard: TileLang and FlashInfer accumulate in different
  orders inside the tensor-core epilogue. The
  `infer/test_data/Qwen3.5-0.8B.json` substring match should pass, but if
  it fails on the remote run, the swap is reverted and an `errors/` entry
  is opened.
- Decode is single-token Q per request — kernel occupancy is bound by
  `(batch_size × num_q_heads)` blocks, not by Q-tile sweeps. Small batch
  on L4 (~ 1× 8 heads = 8 blocks) may underfill the SM grid; the H100
  sweep is the primary win-detection target for this tranche.

## Learnings

- Filled in after the remote run, per the bench-and-trace spec.

## Δ vs baseline

- **Baseline:** the matched `tilelang-decode-hd256-off` run on the same
  commit + same host, taken as the immediately-preceding sweep
  (per `feedback_matched_ab_for_small_bench_effects.md`).
- **Delta table:** pending remote run.

| metric | baseline (off) | now (on) | Δ% |
|---|---|---|---|
| TTFT p50 @ synchronous | pending | pending | pending |
| ITL p50 @ saturation   | pending | pending | pending |
| out tok/s @ saturation | pending | pending | pending |

## Artefacts

- Raw: pending
- CSV: pending
- HTML: pending
- Service trace: pending

## Notes

- What changed in code since baseline:
  - `crates/cuda-kernels/build.rs` — added
    `TILELANG_DECODE_HD256_HEAD_CONFIGS = [(8,2), (16,2), (16,4)]` next to
    the prefill HD256 list, and a parallel AOT loop in
    `compile_tilelang_aot_kernels` that emits one cubin per head config
    from `tools/tilelang/batch_decode_paged_hd256.py`. All generated C
    wrappers link into the same `libtilelang_kernels_aot.a`. The
    `cargo:rerun-if-changed=tools/tilelang` directory walk already covers
    the new `.py` file.
  - `crates/cuda-kernels/src/ffi/attention.rs` — added a new
    `tilelang_decode_hd256_decl!` macro (same FFI shape as the HD256
    prefill twin — only the cubin internals differ) and three
    `tilelang_batch_decode_paged_hd256_q{8_kv2,16_kv2,16_kv4}_run_cuda`
    declarations gated behind `cfg(feature = "tilelang-attn")`.
  - `infer/src/ops/attention.rs` — extended
    `ops::flashinfer_run_layer_hd256(...)` with three new TileLang-only
    params (`batch_size`, `max_qlen`, `total_pages`) and a new
    `qo_indptr_gpu` slice. Default builds keep the existing
    `flashinfer_batch_decode_hd256_run` call exactly as before; under
    `cfg(feature = "tilelang-attn")` it matches on
    `(num_qo_heads, num_kv_heads)` to pick the matching FFI symbol and
    dispatches with the standard varlen Q / paged-KV arguments + the
    TileLang-only runtime scalars (TileLang 0.1.9 auto-promotes
    `T.symbolic` shape vars into kernel arguments). Unsupported pairs
    return an `anyhow!` error pointing at the three lockstep lists to
    extend (`SUPPORTED_HEADS` in the Python kernel module,
    `TILELANG_DECODE_HD256_HEAD_CONFIGS` in `cuda-kernels/build.rs`, and
    the FFI macro arms).
  - `infer/src/model/qwen35/batch_decode.rs` — cfg-gated the
    `metadata.plan_hd256(...)` call inside
    `BatchDecodeBuffers35::plan_attention` behind
    `cfg(not(feature = "tilelang-attn"))` (TileLang is plan-less, so the
    FlashInfer plan_info isn't consumed). The metadata uploads
    (positions, kv_indptr, kv_indices, kv_last_page_len, qo_indptr) in
    `update_metadata` stay unconditional. The single BF16 HD256 decode
    call site now passes `batch_size`, `max_qlen`, `total_pages`,
    computed from the metadata host-side scratch (`qo_indptr_h`,
    `indptr_h`) — note the field names differ from the prefill side
    (`qo_indptr_host` / `kv_indptr_host`), so this tranche reads the
    decode-side equivalents (`qo_indptr_h` / `indptr_h`).
  - No `.cu` source changes. No
    `crates/cuda-kernels/tools/tilelang/batch_decode_paged_hd256.py`
    edits — the kernel was authored upstream of this tranche and is
    treated as read-only here. The three AOT specializations
    (q8_kv2 × q16_kv2 × q16_kv4 × HD256 × page_size=16) cover Qwen3.5
    full-attn 0.8B / MoE 30B-A3B / medium / 14B / 32B-class head
    configs — extend in lockstep when adding a new family member.
- Suspected cause of any regression: the HD256 decode kernel uses
  `BLOCK_M=1` (one Q row per tile) and grid
  `(1, num_q_heads, batch_size)`. On L4 with small `batch_size × num_q_heads`,
  the SM grid may underfill and FlashInfer's split-KV reduction pattern
  (which extracts more parallelism on small-batch decode) can dominate.
  H100 with higher SM count is the primary win-detection target.
- Follow-ups: replace this stub with completed off + on entries. If
  win ≥10%, retire `flashinfer_batch_decode_hd256_run` from the Qwen3.5
  decode hot path per Phase 1 plan.
