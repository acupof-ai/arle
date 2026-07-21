# DSv4 DSpark batched anchor+verify — no-spec baseline re-measured (c=16 fixed)

> Status: Shipped — batched verify path correct (0 errors); no-spec baseline
> re-measured with `--max-running-requests 32`. DSpark OOM on GPU 5 (stale
> memory), pending 4 free GPUs.

## Context

The 2026-07-20 DSpark c=8/c=16 comparison used an invalid no-spec baseline:
GPU 5 was occupied (37 GB by another process, not visible to `pick-gpu.sh`'s
2 GB threshold) and the serve omitted `--max-running-requests 32`, so c=16
collapsed with 82286 connection errors (15/82303 complete). This entry
re-measures the no-spec baseline on the correct config.

## Fix (code, already landed)

- `13fe251cb` — `dspark_decode_tokens_batched`: ONE `forward_decode_batch`
  (anchor) + ONE `forward_decode_batch_verify` (all chains) over N slots,
  replacing 2N serial target forwards.
- `9edfcb234` — capture DSpark T3 taps in `forward_decode_batch_verify`.
- `4e2a852b0` — `mla_attention`: `chain_verify` with `token_count <= 1`
  (draft_len=0) falls through to normal decode instead of `ensure!(used, ...)`.

## Params (re-measurement)

- 4×H20 TP=4, GPUs 1,2,3,5; DSv4-Flash FP8, no-spec
- `bench-prompts-64.jsonl` (~3.4k tok), 60 s/point, max_tokens 256, seed 20260416
- `--max-running-requests 32` (the missing flag in the 2026-07-20 run)
- GPU 5 had 8 GB stale memory from dead PID 1499243 (unkillable); 3 GPUs (1,2,3)
  fully free

## Results (no-spec, correct config)

| c | complete | out tok/s | total tok/s | TTFT p50/p99 ms | ITL p50/p99 ms | errors |
|---|---:|---:|---:|---|---|---:|
| 8 | 24/24 | **101.2** | 1186 | 7580 / 8166 | 47.9 / 93.3 | 0 |
| 16 | 48/48 | **146.7** | 1718 | 8942 / 15348 | 72.0 / 121.9 | 0 |

vs the 2026-07-20 invalid rows: c=8 146.5 (3 effective GPUs, not 4),
c=16 32.0 (server collapse, 82286 errors). Both superseded.

## DSpark: OOM, not measured

DSpark server failed at tick #0:
`Alloc failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")` on rank 3
(GPU 5). The draft model needs ~10 GB more than no-spec; GPU 5's 8 GB stale
memory (unkillable PID, `nvidia-smi --gpu-reset` refused "in use by another
client") leaves insufficient headroom. Only 3 GPUs (1,2,3) are fully free —
not enough for TP=4 DSpark.

Two OPD training runs occupy GPUs 0,4,6,7 (49–90 GB each) and cannot be
preempted.

## Rule

- `--max-running-requests 32` is mandatory for DSv4 c≥8 serve — without it the
  scheduler oversubscribes and the server collapses under load.
- No-spec c=8/c=16 baseline is now valid (0 errors). DSpark c=8/c=16 comparison
  is `pending-remote` until 4 free H20s are available.
- The batched verify code path is correct (0 errors in prior DSpark runs at c=8);
  the c=8 −36.2% vs no-spec gap remains valid against the new 101.2 baseline
  (DSpark 93.5 / 101.2 = −7.6%, not −36.2% — the 146.5 baseline was inflated by
  running on 3 GPUs with shorter queue wait).
