# DSv4 HCA Batch FlashMLA Decode

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44s, TPOT ~4.85ms, E2E ~7.7s, output throughput ~196 tok/s.

The current ARLE batched decode path already batches the DSv4 Q/K/V projection
stage, but the compressed attention core still looped over scheduler rows and
called FlashMLA decode as `b=1` for every row. That is structurally different
from the SGLang best-practice path, where decode metadata and FlashMLA run over
the active batch.

## What Worked

Implemented a narrow HCA-only batch FlashMLA decode tranche:

- added per-row `start_pos[]` CUDA entrypoints for fused Q/K RoPE prep and
  FlashMLA output inverse-RoPE;
- added a batch SW FP8 pack-slot filler for shared FP8 KV pool coordinates;
- wired ARLE DSv4 batch decode to prepare a `b=N, s_q=1` FlashMLA launch
  descriptor for HCA layers when `ARLE_DSV4_SHARED_KV_POOL=1` and the FlashMLA
  MODEL1 decode shape gate is satisfied;
- kept CSA/SWA layers on the old row loop;
- kept `local_attn -> wo_a -> wo_b -> all-reduce` on the old per-row path,
  because the prior batched output-projection attempt was killed by correctness
  validation.

Local verification before remote CUDA build:

- `cargo fmt --check` passed.
- `git diff --check` passed.
- `cargo check -p infer --no-default-features --features no-cuda` passed.
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda` passed.

Remote CUDA build / correctness / TPOT validation is pending.

## Rule

Do not claim DSv4 target performance from this tranche alone. It only removes
one `b=1` HCA FlashMLA decode structure; correctness and target-workload TPOT
must come from the pod after CUDA build and decode validation.
