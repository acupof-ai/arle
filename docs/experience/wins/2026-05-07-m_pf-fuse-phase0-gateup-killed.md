# 2026-05-07 · M_pf-fuse Phase 0 KILLED — gate-up fusion misses TTFT license

## Goal

Optimization license-or-kill for M_pf-fuse Phase 0: fuse Qwen3 prefill
`gate_proj + up_proj` into one `gate_up_proj` load-time concat plus one GEMM,
then split the halves in a fused SiLU-mul kernel.

Success threshold from `docs/plans/M_pf-fuse-prefill-gemm.md`: longctx
4k/c=4 TTFT improves by at least 8% versus P0' default Split.

## Hypothesis

Because vLLM/SGLang/TRT-LLM use merged gate-up MLP weights, removing one FFN
GEMM per layer should reduce Qwen3-4B longctx 4k/c=4 TTFT from `1976.4 ms` to
about `1818 ms` and license Phase 1 QKV fusion.

## Command

Server:

```bash
TMPDIR=/var/tmp \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
INFER_QWEN3_FUSED_GATE_UP=1 \
RUST_LOG=info \
cargo run --release -p infer --features cuda -- \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --num-slots 8 \
  --max-seq-len 12288 \
  --kv-cache-dtype auto
```

GuideLLM:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
scripts/bench_guidellm.sh m_pf-fuse-gateup-c4 \
  --concurrencies 4 \
  --max-seconds 60 \
  --warmup 10
```

## Environment

- GPU: RTX 4070 Ti SUPER 16 GiB
- CUDA: 13.2, target SM `8.9`
- Model: `infer/models/Qwen3-4B`, BF16 weights, FP8 paged KV pool
- Feature set: `infer --features cuda`
- Commit under test: dirty workspace after `22d9317`, before this killed-entry
  commit
- Runtime envelope: `--num-slots 8 --max-seq-len 12288 --kv-cache-dtype auto`
- Raw artifacts: `bench-output/2026-05-07-m_pf-fuse-gateup-c4/`

## Results

GuideLLM headline:

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc4 | 2009.6 | 11.4 | 2005.9 | 2024.0 | 27.20 | 19.42 | 0.03 | 19.42 | 19.53 | 19.54 | 6.96 | 6.98 | 4.0 | 152.49 | 2592.94 | 2688.42 | 131104 | 8192 | 0.56 |

Service trace:

- Samples: `74/74` OK; failed: `0`
- Peak active: `4`; peak running_batch: `4`; peak waiting: `0`
- Plan labels: `idle=23060`, `decode=2302`, `prefill=31`, `split=0`,
  `mixed=0`
- Peak prefill_queue: `3`
- Peak KV util: `84.5%`
- Prefix hit rate: `0.0%`

## Delta vs Baseline

Baseline: P0' default Split from
`docs/experience/wins/2026-05-07-m3.9-mixed-policy-budget-fix.md`, raw
`bench-output/2026-05-07-p0prime-default-split-c4/`.

| Metric | P0' default Split | M_pf Phase 0 fused gate-up | Delta |
|---|---:|---:|---:|
| TTFT p50 | 1976.4 ms | 2005.9 ms | **+1.5% worse** |
| ITL p50 | 19.27 ms | 19.42 ms | +0.8% worse |
| out tok/s | 153.83 | 152.49 | **-0.9%** |
| conc p50 | 4.0 | 4.0 | flat |

Against vLLM 4k/c=4 from
`docs/experience/wins/2026-05-07-m_b22-vllm-longctx-baseline.md`, fused
gate-up remains behind: TTFT p50 `2005.9 ms` vs vLLM `1174.9 ms`, and output
throughput `152.49` vs vLLM `159.17` tok/s.

## Problems

- **License miss:** Phase 0 needed at least 8% TTFT improvement; measured TTFT
  regressed by 1.5%.
- **Default path disposition:** the fused gate-up substrate is kept, but runtime
  dispatch is opt-in only through `INFER_QWEN3_FUSED_GATE_UP=1`. Default Qwen3
  serving stays on separate `gate_proj + up_proj`.
- **Scope:** the opt-in fused loader is single-rank only. Tensor-parallel loads
  stay on separate gate/up weights rather than reintroducing GPU-side concat.
- **Allocator footgun fixed during implementation:** the first load-time concat
  used GPU-side temporary matrices and reduced post-load free HBM from
  `7.03 GB` to `5.83 GB`, shrinking the KV pool. The final implementation does
  host-side safetensors concat followed by one H2D copy; post-load free HBM
  returned to `7.03 GB`, with `59264` max KV tokens.

## Learnings

- Industry-standard fusion is necessary substrate, but not automatically a
  macro-bench win on ARLE's current cuBLAS shapes. Fewer GEMM calls did not
  overcome the larger output matrix cost at longctx 4k/c=4.
- Operator fusion must be licensed by real TTFT, not by call-count math.
- Load-time weight concat should happen on host memory unless a GPU concat is
  proven to release allocator-visible HBM before KV pool sizing.

## Decision

M_pf-fuse Phase 0 gate-up fusion is **KILLED as a default path-cut**.

The implementation lands only as opt-in substrate for future experiments. Do
not proceed to Phase 1 QKV fusion on the basis of the gate-up hypothesis alone;
QKV needs a separate license experiment or stronger kernel-level evidence.
