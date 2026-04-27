# CUDA TileLang prefill HD256 — guidellm sweep, cuda, 2026-04-27

> Tranche 2 stub — HD256 paged-prefill swap to TileLang AOT cubins.
> Created at the same time as the implementation tranche per
> `docs/plans/tilelang-integration.md` §8 (full-integration / "全部接入"
> series). Replace this file with a completed entry (and a separate
> `…-off` baseline entry) once the L4 + H100 sweeps run.

## Goal

- Optimization (per `docs/bench-and-trace-spec.md` §goal taxonomy): measure
  end-to-end TTFT / ITL / saturation throughput delta between the
  FlashInfer HD256 paged-prefill path (default build,
  `flashinfer_batch_prefill_paged_hd256_run`) and the TileLang HD256
  prefill path (`--features cuda,tilelang-attn`) on the Qwen3.5
  full-attention layers. Coverage: Qwen3.5-0.8B (q8/kv2), Qwen3.5 medium /
  14B / 32B-class (q16/kv4), Qwen3.6 MoE 30B-A3B (q16/kv2) on L4 floor and
  H100.

## Hypothesis

- The TileLang HD256 kernel mirrors the Phase 0 HD128 kernel one-for-one
  (identical varlen Q / paged-KV semantics, identical FFI shape) — the
  only deltas are `HEAD_DIM=256`, `BLOCK_N=32` (halved to fit sm_89's
  100 KB shared-memory cap), and `SUPPORTED_HEADS = {(8,2), (16,2), (16,4)}`.
  If Phase 0 HD128 wins on H100 / L4, HD256 should at minimum match the
  delta — the kernel pattern (FlashAttention-2 online softmax, tile-based
  paged-KV walk, GemmWarpPolicy.FullRow on both Q@K and P@V, bf16-narrow
  P rebuffer between softmax and P@V) is line-for-line the upstream
  `tile-ai/tilelang/examples/flash_attention/example_gqa_*` template that
  the L4 floor wins entry (2026-04-26) explicitly told us to mirror.
- Decision threshold (Phase-1 §5 of plan):
  **≥10% on TTFT/ITL or saturation tok/s** to advance the swap as the
  default; ±5% flat → ship-and-hold under flag; ≥5% worse → keep
  FlashInfer canonical (flag off-by-default — already the case).

## Command

```bash
# Off (default build, FlashInfer HD256 paged-prefill)
scripts/bench_guidellm.sh tilelang-prefill-hd256-off

# On (TileLang HD256 paged-prefill)
scripts/bench_guidellm.sh tilelang-prefill-hd256-on
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
  TileLang feature reaches the Qwen3.5 full-attention prefill path
  (Tranche 2 — joins HD128 prefill from Phase 0 and TC-decode alias from
  Tranche 4).

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260427`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh tilelang-prefill-hd256-{on,off}`

## Results

- Status: `pending-remote`
- Local verification completed (macOS workspace):
  - `cargo check -p infer --no-default-features --features cuda,no-cuda`
  - `cargo check -p infer --no-default-features --features cuda,no-cuda,tilelang-attn`
  - Both `cfg` arms of `ops::prefill_attention_paged_run_hd256` compile cleanly.
  - All four Qwen3.5 HD256 paged-prefill call sites (single + batched,
    each with plan + run) are converged on `ops::prefill_attention_paged_run_hd256`
    (run step) and the FlashInfer `plan_hd256` calls are cfg-gated behind
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

## Learnings

- Filled in after the remote run, per the bench-and-trace spec.

## Δ vs baseline

- **Baseline:** the matched `tilelang-prefill-hd256-off` run on the same
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
    `TILELANG_PREFILL_HD256_HEAD_CONFIGS = [(8,2), (16,2), (16,4)]` next to
    the HD128 list, and a parallel AOT loop in
    `compile_tilelang_aot_kernels` that emits one cubin per head config
    from `tools/tilelang/batch_prefill_paged_hd256.py`. All generated C
    wrappers link into the same `libtilelang_kernels_aot.a`. The
    `cargo:rerun-if-changed=tools/tilelang` directory walk already covers
    the new `.py` file.
  - `crates/cuda-kernels/src/ffi/attention.rs` — added a new
    `tilelang_prefill_hd256_decl!` macro (same FFI shape as the HD128
    twin — only the cubin's baked `head_dim` differs) and three
    `tilelang_batch_prefill_paged_hd256_q{8_kv2,16_kv2,16_kv4}_run_cuda`
    declarations gated behind `cfg(feature = "tilelang-attn")`.
  - `infer/src/ops/attention.rs` — added
    `ops::prefill_attention_paged_run_hd256(...)` as a sibling helper to
    `BatchPrefillPagedPlan::run_hd256`. Default builds delegate to
    `plan.run_hd256(...)`; under `cfg(feature = "tilelang-attn")` it
    matches on `(num_q_heads, num_kv_heads)` to pick the matching FFI
    symbol and dispatches with the standard varlen Q / paged-KV
    arguments + the TileLang-only `max_qlen` / `total_pages` /
    `num_pages` runtime scalars (TileLang 0.1.9 auto-promotes
    `T.symbolic` shape vars into kernel arguments). Unsupported pairs
    return an `anyhow!` error pointing at the three lockstep lists to
    extend (`SUPPORTED_HEADS` in the Python kernel module,
    `TILELANG_PREFILL_HD256_HEAD_CONFIGS` in `cuda-kernels/build.rs`,
    and the FFI macro arms).
  - `infer/src/model/qwen35/prefill.rs` — converged the four HD256
    prefill call sites (single + batched, each with plan + run) onto
    `ops::prefill_attention_paged_run_hd256` (run step) and cfg-gated
    the two `bufs.plan.plan_hd256(...)` calls behind
    `cfg(not(feature = "tilelang-attn"))` — TileLang is plan-less, so
    the FlashInfer plan_info isn't consumed. The host-side scratch
    (qo_indptr_host, kv_indptr_host, kv_last_page_len_host,
    start_pos_host) IS still uploaded for both paths; only the
    FlashInfer `plan_hd256` step is skipped. `max_qlen` and
    `total_pages` are computed at the call sites from
    `qo_indptr_host` / `kv_indptr_host` (mirrors the T4 pattern from
    qwen3/batch_decode.rs:1247-1255).
  - `infer/src/model/qwen35/prefill_buffers.rs` — annotated
    `PagedPrefillBuffers35::plan` with
    `#[cfg_attr(feature = "tilelang-attn", allow(dead_code))]` since
    TileLang is plan-less but the field stays under both cfg arms while
    FlashInfer remains the default path (Tranche 2: "不着急删除").
  - No `.cu` source changes. No
    `crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd256.py`
    edits — the kernel was authored upstream of this tranche and is
    treated as read-only here. The three AOT specializations
    (q8_kv2 × q16_kv2 × q16_kv4 × HD256 × page_size=16) cover Qwen3.5
    full-attn 0.8B / MoE 30B-A3B / medium / 14B / 32B-class head
    configs — extend in lockstep when adding a new family member.
- Suspected cause of any regression: the HD256 kernel halves `BLOCK_N`
  to 32 (vs HD128's 64) to fit shared memory on sm_89; that doubles the
  number of KV-tile iterations vs HD128. If TileLang's pipeline
  (NUM_STAGES=2) cannot hide the extra iteration overhead, the kernel
  will be latency-bound on sm_89 even if it wins on Hopper. The §6
  Hopper retune from the plan covers this.
- Follow-ups: replace this stub with completed off + on entries. If
  win ≥10%, retire `flashinfer_batch_prefill_paged_hd256_run` from the
  Qwen3.5 hot path per Phase 1 plan (decode HD128 / HD256 migration).
