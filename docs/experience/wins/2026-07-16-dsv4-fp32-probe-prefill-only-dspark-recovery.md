# DSv4 FP32 probe limited to prefill — unblocks DSpark (MTP) decode — CUDA, 2026-07-16

> Status: Shipped

## Goal

The all-boundaries FP32 compressor (see
[2026-07-16-dsv4-fp32-compressor-all-boundaries.md](2026-07-16-dsv4-fp32-compressor-all-boundaries.md))
fixed the #146/#150 prefill corruption but its guard let the FP32 probe run on
every decode token too — including the DSpark (MTP) draft phase, which calls
`compressor_forward` per single-row draft decode. Each draft token re-ran the
FP32 input GEMM + compressor update, making speculative decoding expensive.
Guard the probe to prefill only so MTP draft runs on the BF16 path.

## Changes

1. `attention.rs` (`2e5ef6503`, `60be54d9a`): `compressor_forward` FP32 probe
   guard limited to `start_pos_device.is_none()` (prefill). Decode (batched,
   full-flatten, graph, MTP draft) always has `start_pos_device = Some` → BF16
   path. Guard simplified from 5 conditions to 3: `precomputed.is_none()` and
   `defer_update.is_none()` are redundant — every prefill call site has all
   three `None`, every decode call site has `start_pos_device = Some`.
2. `scripts/bench_throughput.py` (`fc7fdd34e`): removed the strict
   `output_events == completion_tokens` check that failed under MTP
   (multiple accepted tokens per SSE event); usage-based token count is the
   ground truth.

## Correctness (needle gate, needle 738291)

**Depth 0.0 — ALL PASS** (9 lengths, 3/3 exact).

**Depth 0.5 — NO MISSES:**

| Length | exact | partial | miss |
|--------|-------|---------|------|
| 115 | 0 | 3 | 0 |
| 180 | 0 | 3 | 0 |
| 241 | 3 | 0 | 0 |
| 300 | 3 | 0 | 0 |
| 446 | 3 | 0 | 0 |
| 1000 | 0 | 3 | 0 |
| 2000 | 3 | 0 | 0 |
| 4000 | 3 | 0 | 0 |
| 8000 | 3 | 0 | 0 |

Partial results (len=115, 180, 1000) are mid-prompt retrieval behavior
("738" vs full "738291"), not failures. Prefill FP32 probe still runs.

## Performance (DSpark MTP, guidellm concurrent, 20 prompts, 60s max)

| Rate | ITL p50 | ITL p99 | TTFT p50 |
| ------ | --------- | --------- | ---------- |
| 1 | 40.8ms | 45.8ms | 126.7ms |
| 4 | 40.8ms | 53.4ms | 7587.4ms |
| 8 | 40.8ms | 53.8ms | 15644.6ms |
| 16 | 40.8ms | 53.7ms | 33897.1ms |

ITL p50 is flat at 40.8ms across concurrency — decode is compute-bound. Zero
`fp32_probe` log hits during the decode-heavy bench; the probe no longer runs
on the MTP draft path.

### Attribution note (SOLID)

The earlier "~2× vs previous fp32all" comparison was INVALID — it conflated two
independent changes. The `fp32all` baseline (19–25 tok/s output) ran **without
MTP** (eager decode, `--spec-type` unset) while the new numbers run **with
MTP** (`--spec-type mtp`). The 48–49 tok/s reflects MTP speculative decoding
working correctly now that the FP32 probe no longer bogs down each draft token;
it is NOT the isolated effect of removing the probe from decode.

To isolate the probe effect, the correct A/B is: same MTP config, old guard
(FP32 probe on decode draft tokens) vs new guard (FP32 probe prefill-only). We
do not have the old-guard + MTP number (it would be slow — each draft token
running the FP32 GEMM+probe — and was the motivation for this fix). The
all-boundaries bench's −17% to −36% total-tok/s cost (eager, no MTP) is the
best available isolated estimate of the probe-on-decode overhead.

## Environment

- Host / GPU: 8× NVIDIA H20 (97.9 GB each), driver 535.161.08
- CUDA: 12.9 (V12.9.86)
- Model / dtype: DeepSeek-V4-Flash-FP8
- TP / EP: 4 / 4 (GPUs 1–4; GPU 0 occupied)
- Server: `INFER_TP_SIZE=4 INFER_EP_SIZE=4 INFER_CUDA_DEVICES=1,2,3,4 arle serve --backend cuda --port 8000 --spec-type mtp`

## Learnings

The FP32 probe is a prefill-only correctness fix (#146, #150); decode never
needed it (single-token, BF16 path is sufficient). The all-boundaries
extension accidentally ran it on every decode token — `start_pos_device`
(`Some` in all decode paths, `None` in prefill) is the single discriminator.
For MTP this is especially costly: the draft phase emits many candidate tokens,
each of which would re-run the FP32 probe. Prefill-only guard keeps the
correctness fix while letting MTP draft run on the fast BF16 path.

Cross-workload A/B must hold the decode mode (eager vs MTP) constant — a
no-MTP baseline vs an MTP result measures MTP, not the variable under test.
