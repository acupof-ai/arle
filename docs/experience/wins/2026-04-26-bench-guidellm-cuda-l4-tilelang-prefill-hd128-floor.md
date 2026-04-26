# TileLang prefill HD128 AOT — guidellm c=1..16, cuda L4 floor, 2026-04-26

> L4-floor entry per
> [`docs/plans/tilelang-integration-verification.md`](../../plans/tilelang-integration-verification.md)
> §0 — sm_89 numbers are recorded but **do not** drive the §5
> ship/revert decision; that gates on H100. The pending-remote H100
> stub stays in place.

## Goal

- Optimization (per `docs/bench-and-trace-spec.md` §goal taxonomy):
  matched A/B between the FlashInfer prefill HD128 path
  (`--features cuda`) and the new TileLang AOT path
  (`--features cuda,tilelang-attn`) on the same commit, same machine,
  same flags. Phase 0 was previously parked on a chain of
  TileLang-side blockers; this run closes it.

## Hypothesis

- TileLang AOT compiles end-to-end and produces numerically correct
  output on sm_89.
- Performance vs FlashInfer is roughly flat or slightly worse on
  Ada — TileLang's Hopper-only wins (TMA, WGMMA, warp-spec) don't
  fire here.

## Command

```bash
# off (FlashInfer)
cargo build --release -p infer --no-default-features --features cuda
/tmp/infer-off --model-path models/Qwen3-4B --port 8000 \
  --num-slots 16 --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 --max-prefill-tokens 16384

# on (TileLang AOT)
cargo build --release -p infer --no-default-features --features cuda,tilelang-attn
/tmp/infer-on --model-path models/Qwen3-4B --port 8000 \
  --num-slots 16 --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 --max-prefill-tokens 16384

scripts/bench_guidellm.sh cuda-l4-tilelang-{off,on} \
  --concurrencies 1,2,4,8,16 --max-seconds 60 --warmup 5 \
  --processor /content/workspace/agent-infer/models/Qwen3-4B
```

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4, 24 GB, sm_89, driver 580.82.07, CUDA 12.8.93
- **Commit (off + on):** `2a4ff6ce`
- **Toolchain:** rustc 1.95.0, nvcc 12.8.93, zig 0.14.0, tilelang 0.1.9
- **Non-default flags / env vars:** `INFER_CUDA_SM=89`,
  `CARGO_HOME=/tmp/cargo-home-local`, `INFER_TRITON_PYTHON=/usr/bin/python3`.
- **AOT pipeline (on build):**
  `tilelang.compile()` → pull `adapter.device_kernel_source` →
  nvcc to raw cubin against TileLang's bundled `tl_templates/cuda` +
  `cutlass/include` → embed cubin in C wrapper → wrapper sets
  `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 49152` and
  passes the same value as `cuLaunchKernel`'s `sharedMemBytes`.

## Numerical parity

Sanity probe on tilelang-on:

```
$ curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-4B","prompt":"The capital of France is","max_tokens":15,"temperature":0}'

→ " Paris. The capital of Germany is Berlin. The capital of Italy is"
```

Greedy answer matches the FlashInfer baseline at the same prompt.
Per-token logprobs are within bf16 noise of the off-build; full
e2e suite was not rerun (existing pre-existing greedy drift on
`Qwen3-4B.json` is unrelated to this kernel — see
`memory/project_remote_cuda_box.md`).

## Results — paired concurrency table (matched-flags A/B)

| conc | TileLang TTFT p50 ms | FlashInfer TTFT p50 ms | Δ TTFT | TileLang ITL p50 ms | FlashInfer ITL p50 ms | Δ ITL | TileLang tok/s | FlashInfer tok/s | Δ tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  1 |    838.7 |    719.3 | +16.6% | 35.20 | 35.24 | −0.1% | 26.31 | 26.56 | −0.9% |
|  2 |   1742.8 |   1518.6 | +14.8% | 38.41 | 38.77 | −0.9% | 45.37 | 45.21 | +0.4% |
|  4 |   2714.8 |   2354.4 | +15.3% | 41.20 | 41.62 | −1.0% | 52.34 | 53.31 | −1.8% |
|  8 |   4185.3 |   3838.0 |  +9.0% | 49.07 | 52.30 | −6.2% | 68.15 | 66.94 | +1.8% |
| 16 |  16361.9 |  16356.9 |   flat | 49.07 | 49.27 | −0.4% | 66.55 | 65.92 | +1.0% |

## Results — raw `tilelang-on` table

| conc | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---:|---:|---:|---:|---:|---:|
|  1 |    838.7 |    854.9 | 35.20 | 35.22 | 26.31 | 0.109 |
|  2 |   1742.8 |   1823.3 | 38.41 | 38.43 | 45.37 | 0.182 |
|  4 |   2714.8 |  15917.2 | 41.20 | 41.49 | 52.34 | 0.218 |
|  8 |   4185.3 |  19914.9 | 49.07 | 57.77 | 68.15 | 0.273 |
| 16 |  16361.9 |  31910.0 | 49.07 | 57.80 | 66.55 | 0.273 |

## Decision matrix vs plan §5

Per [`tilelang-integration.md`](../../plans/tilelang-integration.md) §5,
applies on H100 only — L4 numbers are floor:

| metric | Δ on L4 | §5 verdict (if H100) |
|---|---|---|
| TTFT p50 @ synchronous (c=1) | +16.6% | would trigger **revert** (≥5% regression) |
| out tok/s @ saturation (c=16) | +1.0% | flat (within noise) |

L4 floor verdict: **functional, ship-and-hold pending H100.** The +15%
TTFT regression at low concurrency on Ada (sm_89) is consistent with
TileLang's design — its wins fire on Hopper's TMA / WGMMA / warp-spec
intrinsics. On Ada the FlashInfer prefill is already well-tuned and
TileLang's generic CuTeDSL path lacks the specialized intrinsics that
would let it overtake. Decision is deferred to the H100 spike per
plan §0; do not retire the
[pending-remote H100 stub](./2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md).

## What landed in code

`2a4ff6ce feat(cuda): tilelang prefill HD128 AOT works end-to-end on TileLang 0.1.9` —
the unblock. Five files changed:

| file | purpose |
|---|---|
| `crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py` | Added `batch_size: T.int32, max_qlen: T.int32` runtime scalars; fixed P@V dtype; `policy=T.GemmWarpPolicy.FullRow`; hoisted alpha rescale to 2D parallel. |
| `crates/cuda-kernels/tools/tilelang/gen_tilelang_aot.py` | Rewrote AOT pipeline: pull device source from `adapter.device_kernel_source`, parse kernel signature, nvcc to cubin, lift dynamic shared-memory cap, pass it to `cuLaunchKernel`. |
| `crates/cuda-kernels/build.rs` | Probe TileLang install path for `tl_templates/` + `cutlass/include`; pass them along with `cuda-arch` to gen_tilelang_aot.py. |
| `crates/cuda-kernels/src/ffi/attention.rs` | FFI signature: added `num_pages, total_pages` between `max_qlen` and `num_q_heads`. |
| `infer/src/ops/attention.rs` | Compute and pass `num_pages = meta.pool.max_total_pages` and `total_pages = sum(s.num_pages)` at the call site. |

## Problems

- TTFT regression at c=1..4 (≥9%) — expected on Ada; would block §5
  ship-decision on H100. Not actionable on L4.
- The shared-memory cap lift (49152 bytes) is parsed from
  `host_kernel_source` regex. Brittle vs TileLang ABI changes; if a
  future version restructures the host source layout, the regex
  needs updating. The error message in `gen_tilelang_aot.py` calls
  this out so the next person knows where to look.
- `num_pages` and `total_pages` had to be added to the user-facing
  FFI signature because TileLang 0.1.9 auto-promotes every
  `T.symbolic` shape var into a kernel argument. Future kernel work
  should keep this invariant in mind: any new symbolic dimension
  needs a corresponding C-wrapper expression in
  `WRAPPER_FILL_RULES`.

## Learnings

- **TileLang 0.1.9 produces a TVM-FFI shared object, not a raw
  cubin.** Phase 0's prior `compiled.cubin_path` probe was looking
  for a file that doesn't exist on cold-cache compile. The right
  hook is `adapter.device_kernel_source` (in-memory CUDA source),
  which we then nvcc ourselves.
- **Symbolic shape promotion is automatic.** Every `T.symbolic("x")`
  the kernel references becomes an int32 kernel argument. The C
  wrapper has to fill all of them, including the duplicated
  `<name>_1` slots TileLang adds for re-use.
- **Dynamic shared memory needs the cap lift.** sm_89's default
  per-block cap is 48 KB; HD128 prefill needs ~48 KB for
  Q/K/V tiles. Without `cuFuncSetAttribute(...,
  MAX_DYNAMIC_SHARED_SIZE_BYTES, N)`, every launch fails with
  `illegal memory access`. The same lift is needed on H100, just
  with a higher target size.
- **Mirror upstream's flash_attention examples.** The three
  pre-fix kernel "improvements" we landed in `4d9c65f0` (FullRow
  policy, hoisted alpha, p_bf16 narrow) are line-for-line what
  `tile-ai/tilelang/examples/flash_attention/example_gqa_*`
  do. The lesson from the version bisect (every 0.1.5..0.1.9 rejected
  the pre-fix kernel) was: don't trust permissive auto-inference,
  copy the upstream pattern.

## Δ vs baseline

- **Baseline:** the `--features cuda` build at the same commit
  `2a4ff6ce` (this run's `tilelang-off` matched A/B side).
- **Delta table:** included above.

## Artefacts

- Off (FlashInfer): see
  [`2026-04-26-bench-guidellm-cuda-l4-mixed-batch-vs-f98ca92.md`](./2026-04-26-bench-guidellm-cuda-l4-mixed-batch-vs-f98ca92.md)
  for raw c=1..16 numbers (the `--features cuda` matched-flags run
  was the off side of this A/B too).
- On (TileLang): `bench-output/2026-04-26-cuda-l4-tilelang-on/{benchmarks.{json,csv,html},service_stats_*}`.
- AOT artefacts: `target/release/build/cuda-kernels-*/out/tilelang_aot/batch_prefill_paged_hd128_q{16,32,40,64}_kv8/{*.c,*.cubin,*_device_kernel.cu}`.

## Notes

- **What changed in code since the previous failed Phase 0 attempt
  (commit `9896d25` and earlier):** the `gen_tilelang_aot.py`
  rewrite, the kernel-side `T.int32` scalar args + FlashAttention-2
  alignment, and the FFI `num_pages` / `total_pages` plumbing.
- **Suspected cause of the c=1..4 TTFT regression:** generic
  CuTeDSL emit on sm_89; FlashInfer's `BatchPrefillPagedPlan` path
  is hand-tuned for Ada and uses prefetch + softmax fusion patterns
  TileLang's auto-codegen doesn't replicate. Confirming on Hopper
  is the §5 next step.
- **Follow-ups:**
  - H100 spike per `tilelang-integration-verification.md` §4 to
    drive the §5 ship/revert decision.
  - Once H100 confirms wins, propagate the `T.int32` scalar pattern
    to the decode HD128/HD256 kernels (Phase 1).
  - Pin `tilelang==0.1.9` in `pyproject.toml` to lock the ABI we
    just wrote against.
