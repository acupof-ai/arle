# Train-infer FP8 weight-share (`--share-frozen-base`) — verified correct, flipped default-ON

## Context

`2cff1465` added the train-infer FP8 weight-share opt-in (`--share-frozen-base`):
the autograd LoRA student's frozen FP8 base projections point **zero-copy** at the
co-resident rollout engine's resident FP8 base (same primary CUDA context), so an
OPD arm holds ONE base instead of two. The 2026-06-22 wins entry
([before](2026-06-22-train-infer-weight-share.md)) shipped it **default-OFF** with
the **correctness gate + bench pending-remote**, and pre-specified the plan: flip
default-ON with a `--no-share-frozen-base` opt-out once the verdict landed.

This entry is the **after**: the pending gate, run on H20.

## Measured (Qwen3.6-27B-FP8, rubric-opd `--self-consistency`, GPU 5, 1 s mem sampling)

Same binary, same prompts, share OFF vs ON, sequential on one GPU:

| | share OFF | share ON (default after this) |
|---|---|---|
| **peak GPU mem** | 61.1 GB | **44.0 GB** (−17 GB / −28%) |
| **mean_loss** | 0.9745 | **0.9749** (match, 4e-4) |
| accepted / trained / parse_err | 4 / 2 / 0 | **4 / 2 / 0** (identical) |
| share engaged | — | `borrowing 256 resident FP8 base projections (zero-copy)` |
| ran clean | ✅ | ✅ |

**Verdict: PASS.** The zero-copy aliased FP8 base produces functionally identical
training (loss + acceptance match to noise) while dropping ~17 GB. The ~17 GB
(vs the ~27 GB full base) is because only the layer projections are shared; lm_head
/ LoRA / optimizer / KV / the engine's own base stay separate.

## Default flipped ON (`--no-share-frozen-base` opt-out)

`share_frozen_base` CLI arg → inverted to `no_share_frozen_base` (default false) on
rubric-opd + agent-opd; production mapping `share = !args.no_share_frozen_base`.
Safe by construction (`frozen_base_fp8_pointers`, qwen35.rs:3231):
- **FP8 single-GPU** → shares (the measured win).
- **non-FP8 single-GPU** → returns an empty pointer table → graceful normal load
  (no break, no benefit).
- **TP > 1** → rejects with a clear error; OPD training is single-GPU, so this path
  doesn't occur. Pass `--no-share-frozen-base` to force the byte-identical two-copy
  load if ever needed.

## Devops side-finding (fixed: `679f49a7`)

Verification first OOM'd: rubric-opd landed on an occupied GPU 0 despite
`INFER_CUDA_DEVICE=5`, because the **autograd backend binds cudarc device 0 and does
not read `INFER_CUDA_DEVICE`** (only the infer engine does). `scripts/pod.sh run`
now pins via `CUDA_VISIBLE_DEVICES=<gpu>` + `INFER_CUDA_DEVICE=0`, so an OPD run's
*both* backends land on the chosen GPU — the correct per-agent GPU isolation.

## Rule

A pending-remote default-flip is a hypothesis until the same-binary A/B runs:
sharing was *plausibly* correct (zero-copy alias of the same bytes) but unverified
for 1 day — the A/B confirmed loss-identity + the ~17 GB drop AND surfaced the
autograd-ignores-`INFER_CUDA_DEVICE` isolation bug that only shows under a busy
GPU 0. Verify, then flip. See [[project_new_h20_sglang_box_devops]].
