# CUDA TileLang TC-decode HD128 alias — guidellm sweep, cuda, 2026-04-27

> Tranche 4 stub — TC decode aliased onto Phase 0 prefill HD128 cubins.
> Created at the same time as the implementation tranche per
> `docs/plans/tilelang-integration.md` §8 (full-integration / "全部接入"
> series). Replace this file with a completed entry (and a separate
> `…-off` baseline entry) once the H100 / L4 sweeps run.

## Goal

- Optimization (per `docs/bench-and-trace-spec.md` §goal taxonomy): measure
  end-to-end TTFT / ITL / saturation throughput delta between the
  FlashInfer TC decode path (default build, `flashinfer_tc_decode_run`) and
  the TileLang TC decode alias (`--features cuda,tilelang-attn`) on the
  Qwen3 BF16 batched-decode hot path. Coverage: Qwen3-0.6B / 1.7B / 4B /
  8B head configs (q16/q32/q40/q64, kv8) on L4 floor and H100.

## Hypothesis

- FlashInfer's `flashinfer_tc_decode_run` is "use the prefill kernel for
  decode-shaped inputs to get tensor-core utilization". Its kernel
  signature and semantics (varlen Q via `qo_indptr`, paged KV with
  `kv_indptr` / `kv_indices` / `kv_last_page_len`, causal mask within Q)
  are identical to the existing AOT cubin family
  `tilelang_batch_prefill_paged_hd128_q{16,32,40,64}_kv8_run_cuda` from
  Phase 0. The alias should produce numerically identical output (modulo
  fp accumulation order) and either match or improve TTFT/ITL at
  saturation. Decision threshold (Phase-1 §5 of plan):
  **≥10% on TTFT/ITL or saturation tok/s** to advance the alias as the
  default; ±5% flat → ship-and-hold under flag; ≥5% worse → keep
  FlashInfer canonical (flag off-by-default — already the case).

## Command

```bash
# Off (default build, FlashInfer TC decode)
scripts/bench_guidellm.sh tilelang-tc-decode-off

# On (TileLang TC decode alias)
scripts/bench_guidellm.sh tilelang-tc-decode-on
```

Invoked via: pending remote L4 floor + H100 hosts (user-driven verification).

## Environment

- **Backend:** cuda
- **Models:** Qwen/Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B
  (all BF16; q-head counts 16/32/40/64 — covers all four AOT specializations)
- **Hardware:** pending remote NVIDIA L4 + H100 hosts
- **Commit:** pending; will be filled in once the implementation tranche
  lands and the remote sweep runs.
- **Feature set (off):** `CUDA_HOME=/usr/local/cuda cargo build --release --features cuda` (workspace root)
- **Feature set (on):**  `CUDA_HOME=/usr/local/cuda cargo build --release --features cuda,tilelang-attn` (workspace root)
- **Non-default flags / env vars:** `INFER_TILELANG_PYTHON` if a non-default
  Python interpreter is needed for AOT, otherwise none.
- **Server launch (off):** `INFER_FEATURES="cuda" scripts/start_infer.sh models/Qwen3-4B 8000`
- **Server launch (on):**  `INFER_FEATURES="cuda,tilelang-attn" scripts/start_infer.sh models/Qwen3-4B 8000`
  Both invocations build and run the `infer` binary directly so the
  TileLang feature actually reaches both the prefill path (Phase 0) and
  the TC-decode alias (Tranche 4).

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260427`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh tilelang-tc-decode-{on,off}`

## Results

- Status: `pending-remote`
- Local verification completed (macOS workspace):
  - `cargo check -p infer --no-default-features --features cuda,no-cuda`
  - `cargo check -p infer --no-default-features --features cuda,no-cuda,tilelang-attn`
  - Both `cfg` arms of `flashinfer_tc_run_layer` compile cleanly.
  - All three Qwen3 BF16 TC-decode call sites (mixed batch + LoRA + graph)
    are converged on `ops::flashinfer_tc_run_layer`.

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
  orders inside the tensor-core epilogue. The `infer/test_data/Qwen3-4B.json`
  substring match should pass, but if it fails on the remote run, the
  alias is reverted and an `errors/` entry is opened.

## Learnings

- Filled in after the remote run, per the bench-and-trace spec.

## Δ vs baseline

- **Baseline:** the matched `tilelang-tc-decode-off` run on the same
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
  - `infer/src/ops/attention.rs::flashinfer_tc_run_layer` — extended the
    signature with `batch_size`, `max_qlen`, `total_pages`. Under
    `cfg(not(feature = "tilelang-attn"))` the body is unchanged
    (FlashInfer `flashinfer_tc_decode_run` call). Under
    `cfg(feature = "tilelang-attn")` it dispatches to the same
    `tilelang_batch_prefill_paged_hd128_q*_kv8_run_cuda` symbols as
    Phase 0 prefill — alias, not a new kernel.
  - `infer/src/model/qwen3/batch_decode.rs` — collapsed the three
    BF16-decode call sites (mixed batch + LoRA + graph) onto
    `ops::flashinfer_tc_run_layer`, replacing the inline `ffi::
    flashinfer_tc_decode_run` block in the mixed-batch path. Cfg-gated
    the two `tc_plan(...)` calls behind `cfg(not(feature = "tilelang-attn"))`
    — TileLang is plan-less, so the FlashInfer plan_info isn't consumed.
    The host-side scratch (positions, kv_indptr, kv_indices,
    kv_last_page_len, qo_indptr) IS still uploaded for both paths; only
    the FlashInfer `tc_plan` step is skipped.
  - `crates/cuda-kernels/src/flashinfer.rs` — exposed
    `FlashInferDecodeMetadata::indptr_h` so the dispatch sites can read
    `total_pages` (= last entry of host-side kv_indptr) without a
    device-side round trip.
  - No `.cu` source changes. No `crates/cuda-kernels/build.rs` changes.
    No new FFI declarations. The four AOT specializations
    (q16/q32/q40/q64 × kv8 × HD128 × page_size=16) cover Qwen3 0.6B /
    1.7B / 4B / 8B head configs — Qwen3-14B/32B inherit q40 / q64.
- Suspected cause of any regression: the prefill HD128 cubin is tuned
  for many-Q-rows-per-request workloads, while pure decode batches have
  exactly 1 Q row per request. The kernel may waste shared memory on a
  Q-tile that is mostly empty. If that is what we see, the next tranche
  adds a `tilelang_batch_decode_paged_hd128_*` specialization with
  Q-tile = 1.
- Follow-ups: replace this stub with completed off + on entries. If
  win ≥10%, retire `flashinfer_tc_decode_run` from the BF16 hot path
  per Phase 1 plan (decode HD128 / HD256 migration).
