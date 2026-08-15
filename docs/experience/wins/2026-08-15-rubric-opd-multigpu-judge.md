# rubric-opd multi-GPU judge (TP=4) + single-GPU student — CUDA, 2026-08-15

> Status: Shipped

## Goal

Make `rubric-opd` training work with a multi-GPU judge (DeepSeek-V4-Flash, TP=4)
while the student stays single-GPU (Qwen3.6-27B-FP8, TP=1). End-to-end: student
rollout → judge verdict → CE writeback.

## Hypothesis

Two blockers:
1. GPU 0 memory contention: student (~27 GB) + judge rank-0 (~74 GB) > 96 GB H20.
2. `posix_fadvise(WILLNEED)` prefetch on a separate fd did not populate the mmap
   page cache — judge rank-0 deadlocked in page faults after the fadvise.

Fixes:
1. Judge child sets `CUDA_VISIBLE_DEVICES=1..=tp_size`; student keeps GPU 0.
2. Prefetch: `mmap` + `madvise(MADV_WILLNEED)` instead of `posix_fadvise`.

## Parameters

```bash
INFER_TP_SIZE=4 arle train rubric-opd \
  --student-model /data00/Qwen3.6-27B-FP8 \
  --teacher-model /data00/DeepSeek-V4-Flash-FP8 \
  --prompts-file examples/opd/sample-prompts.jsonl \
  --rubric-task math --rounds 1 --samples-per-prompt 2 \
  --max-new-tokens 256 --max-verdict-tokens 256 --writeback-cap 2
```

- Baseline: N/A (did not run — rank-0 deadlocked / OOM)
- Treatment: commit ed3864466, `--features cuda,nccl`

## Environment

- Host: 8×H20 (96 GB each), 1.9 TB RAM
- Model: student Qwen3.6-27B-FP8 (TP=1, GPU 0), judge DSv4-Flash-FP8 (TP=4, GPUs 1-4)

## Results

Round 0: prompts=20 accepted=14 distinct=10 parse_err=11 trained=2 mean_loss=0.2198.

Judge load: 4 workers engine-ready, ~74 GB/GPU on GPUs 1-4. GPU 0 free for student.

## Problems

- `posix_fadvise(WILLNEED)` on a separate fd returns in 0.0 s but does not populate
  the mmap page cache on this kernel (5.4.250). rank-0 then faults on every page
  and deadlocks. `mmap` + `madvise(MADV_WILLNEED)` on the same mapping fixes it.
- sccache build dropped the `nccl` feature; rebuilt with `RUSTC_WRAPPER=`.

## Learnings

PASS. Multi-GPU judge + single-GPU student works end-to-end. The judge's rank-0
must NOT share GPU 0 with the student (27+74 > 96 GB). Prefetch must use
`madvise` on the mmap'd region, not `posix_fadvise` on a separate fd.
