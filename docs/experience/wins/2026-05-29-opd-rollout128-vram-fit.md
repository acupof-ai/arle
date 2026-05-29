# OPD rollout-128 VRAM attribution + lm_head-window lever — 16 GiB sm_89

> Memory-fit investigation (not a guidellm sweep). Target metric: peak VRAM
> on RTX 4070 Ti SUPER (16 GiB, sm_89). Goal: make OPD `--rollout-len 128`
> fit so the "longer rollout = more on-policy signal" hypothesis can be tested.
> **Verdict: PARTIAL — cheap levers shave the lm_head transient (−224 MiB at
> rollout-64, bit-identical loss) but rollout-128 still does NOT fit; the
> dominant bucket is the per-layer activation tape (~30 MiB/token), which needs
> gradient checkpointing, not a cheap lever.**

## Setup

- Teacher: `Qwen3___5-4B` (BF16 on disk, ~9.3 GB file)
- Student: `Qwen3___5-0___8B-Base` (BF16 on disk, 1.75 GB file; 452 BF16 + 36 tiny F32 norm tensors — hybrid linear-attn/mamba model)
- LoRA r=16 `attention-qv`, `--kl-chunk-size 16`, `--opd-kl-mask completion-only`, `--sft-anchor student-rollout --gkd-lambda 0`
- infer rollout = default (2nd in-process infer engine for the student)
- Build env: `CUDA_HOME=/opt/cuda NVCC_CCBIN=g++-14 INFER_TILELANG_PYTHON=.venv/bin/python TORCH_CUDA_ARCH_LIST=8.9 ARLE_CUDA_DISABLE_FLASHMLA=1`, `--release`
- Attribution method: driver `cuMemGetInfo` (added `CudaBackend::mem_get_info`) logged at every load/step boundary, cross-checked with a 0.1 s `nvidia-smi memory.used` sampler. Driver framing is ground truth (includes CUDA context + all pool allocations); nvidia-smi sampler captures the transient backward peak.

## VRAM attribution (driver `mem_get_info`, rollout-64, MiB)

| Phase | used | Δ | Bucket | Hypothesis verdict |
|---|---:|---:|---|---|
| backend init | 1277 | — | CUDA ctx + driver + msedge (~131) | — |
| + train student base (0.8B) | 2717 | **+1440** | train-crate frozen base, **BF16 on device** | **(a) REFUTED** — base is BF16 (~1.6 GB), not F32 (~3.2 GB). The `is_bf16_cuda_frozen_base_tensor` path fires (187 device-only tensors). The 3.08 GB seen in `host_tensor_bytes` is HOST RAM, evicted later — not VRAM. |
| + infer student engine (0.8B) | 4333 | **+1616** | 2nd student copy (infer BF16 weights + KV + graph) | **(b) CONFIRMED** — student held twice = 1440 + 1616 ≈ **3.06 GB** for one 0.8B model. |
| + teacher infer engine (4B) | 12499 | **+8166** | teacher BF16 weights (~8 GB) + small KV | **(c) CONFIRMED** — BF16 ~8 GB (F32 would be 16 GB, impossible). Already the infer-teacher BF16 bridge from 2026-05-21. |
| optimizer init | 12499 | +0 | AdamW state lazy (LoRA only, 639 K elems ≈ negligible) | — |
| before train step (eval ON) | 14247 | +1748 | **step-0 eval live residue** (teacher 4B forward over 5 prompts leaves ~1.7 GB live; `--eval-steps 999` drops step-start to 12503) | — |
| **backward peak** (nvidia-smi) | **15432–15880** | **+~2.9–3.4 GB** | **per-layer activation tape over the full prefix** (24 layers, dense `causal_sdpa` scores + MLP intermediates + grads), + lm_head logits + grads | **dominant, ~linear in rollout (~30 MiB/token)** |

### Transient scaling (matched, all `--eval-steps 999`, step_start = 12503 MiB)

| rollout | backward peak | transient | fits? |
|---:|---:|---:|---|
| 64 | 15432 | ~2929 MiB | YES (loss 9.644847887103e-5, bit-identical) |
| 80 | 15816 | >3313 MiB | NO (`add_into_device` OOM) |
| 96 | 15880 | >3377 MiB | NO |
| 128 | 15816 | est. ~4.85 GB | NO |

Transient grows ~30 MiB/token → rollout-128 needs ~4.85 GB backward on top of
the 12.5 GB resident floor = ~17.3 GB > 16 GB. **The per-token activation tape,
not lm_head logits or any fixed buffer, is the binding constraint** — exactly the
condition under which gradient checkpointing (and only it) is licensed.

## Lever chosen + why

Cheapest correctness-preserving transient reductions in `backward_chunked_kl_rollout`:

1. **Student lm_head over the KL window only** (not the full scored prefix).
   `forward_logits_window` runs the hidden forward over `[0..seq_end]` (causal
   attention needs the full prefix) but projects lm_head only over the
   completion window. Eliminates the full `[1, seq_end, vocab]` student logits
   (~142 MiB at 144×248320) **and** the `slice_bwd` grad buffer of the same size
   (the original rollout-128 OOM site). Numerically identical: causal logits at
   position p are independent of tokens after p.
2. **Teacher slice with the tape DISABLED + free the full teacher logits.** The
   teacher is a fixed target (no gradient), so a tape-enabled slice was needlessly
   registering a `slice_bwd` grad buffer the size of the full teacher logits.
   Slicing tape-off + `store.free` reclaims ~142 MiB.

These are the levers the task named ("compute lm_head + KL only over kl_range").
Applied; they are real but bounded — they shave the vocab-wide part of the
transient, which is small relative to the activation tape.

## Levers rejected (with evidence)

- **Student base F32 → BF16** (hypothesis a): already BF16 on device; no win.
- **`--no-cuda-graph`**: 14247 → 14241, no effect (graph isn't the bucket).
- **`--kl-chunk-size 4`**: still OOMs at `log_softmax_bwd`; chunking already bounds the per-chunk softmax — the binding allocations are the full sliced logits + activation tape, not the softmax intermediates.
- **`ARLE_OPD_INFER_ROLLOUT=0`** (drop infer-student, save ~1.6 GB resident): COUNTERPRODUCTIVE — the train-crate autograd rollout builds a larger tape and OOMs even with 3.2 GB more headroom. The infer-rollout path is the memory-efficient architecture.
- **`cuMemPoolTrimTo`** (reclaim async-pool residue): REVERTED. Freed only 96 MiB (the +1748 is LIVE infer-engine memory, not cached-free pool), and trimming the shared device pool while infer engines hold async state corrupted a later forward ("cuda synchronize failed"). Backend-isolation violation; killed.

## Before / after (peak VRAM, MiB)

| Config | before lever | after lever | Δ |
|---|---:|---:|---:|
| rollout-64 (production, eval on) | 15656 | **15432** | **−224** |
| rollout-128 (eval on) | 15784 (OOM `slice_bwd`) | 15496 (OOM `add_into_device`) | −288 peak, **still OOM** |

Correctness: rollout-64 step-1 loss `9.644847887103e-5` is **bit-identical**
before and after the lever; all 3 steps match magnitude (~1e-4). Step time
slightly faster (lm_head processes fewer positions): mean 12.3 s vs 12.8 s.

## rollout-128 fit verdict

**Does NOT fit on 16 GiB via cheap levers.** Closest approach: peak 15496 MiB
(eval on) / 15816 (eval off) before OOM — ~1.4 GB short. The gap is the per-layer
activation tape (~4.85 GB at rollout-128 vs ~2.9 GB at rollout-64). To fit
rollout-128 needs ONE of:

- **gradient/activation checkpointing on the 24 layers** (the licensed heavy
  change — the activation tape IS confirmed dominant and ~linear in tokens), or
- freeing one of the two 0.8B student copies (~1.6 GB; architecturally invasive —
  merges the infer-rollout and train-forward student roles), or
- a 24 GB card.

## Rule

- For OPD memory-fit, attribute with driver `cuMemGetInfo` per phase BEFORE
  optimizing; host `tensor_host_bytes` is RAM not VRAM and is misleading (the
  3.08 GB "student F32" was host-only, evicted).
- Transient backward at rollout-128 is **per-layer-activation-tape-bound**
  (~30 MiB/token), not lm_head/logits-bound. lm_head-window levers help only the
  small vocab-wide slice; the activation tape needs gradient checkpointing.
- Never trim the shared CUDA async pool while in-process infer engines hold live
  async state — it is both ineffective (live, not cached) and unsafe (corrupts
  pending infer work). Backend-isolation rule applies to the device pool too.
