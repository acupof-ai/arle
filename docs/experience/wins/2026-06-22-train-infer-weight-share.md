# Train-Infer FP8 Weight Sharing (训推一体) — opt-in, one shared frozen base

**Commit:** 2cff1465 (`feat(train): opt-in train-infer FP8 weight-share (--share-frozen-base)`).
**Status:** landed opt-in (default OFF, byte-identical); correctness parity gate = in progress.

## Default status & how to enable (训推一体)

- **Currently OPT-IN, NOT default.** Pass `--share-frozen-base` to
  `arle train rubric-opd` (`args.rs:1102`, `default_value_t = false`). Without it
  the loop loads two FP8 base copies (byte-identical default path, unchanged).
- **Preconditions** (the flag only helps, and is only sound, when these hold):
  FP8 student checkpoint (e.g. `Qwen3.6-27B-FP8`) **and** single GPU (same primary
  CUDA context — the zero-copy import compares context by ordinal). A bf16 student
  or multi-GPU TP cannot share; the flag is a no-op / unsupported there.
- **Why not default yet:** a default flip needs a *correctness license* — a
  controlled share-vs-no-share forward-parity check (the shared FP8 base must
  produce the same logits as a private copy). 2cff1465 showed the loop runs stable
  with CE loss in the two-copy range (~0.066), but that is suggestive, not a parity
  proof. The parity A/B (greedy, `rounds=1`, same prompts, share vs no-share,
  compare per-round `mean_loss`) is running now; **identical loss → flip default-on
  with a `--no-share-frozen-base` opt-out** (per the KV-features "default-on +
  opt-out + scheduled gate + one-line revert" precedent).
- **What it buys:** one base instead of two → **~27 GB freed** on the shared GPU,
  which is exactly the CE-backward headroom a larger-budget retrain needs.

## Context

The OPD rubric loop loaded the 27B **twice** — an autograd training copy + an
infer-cuda rollout/eval engine (~two FP8 base copies resident). ckl: collapse to
one shared frozen base ("训推一体 / 一份权重训推公用, 状态各存"), train also FP8.

## What worked

- **`--share-frozen-base` (opt-in, default off).** The autograd student's frozen
  FP8 base projections **import (zero-copy)** the infer engine's resident FP8
  `DeviceMatrix` pointers via `CudaStream::upgrade_device_ptr` (no D2D copy), same
  device **primary** context (the `backend_cuda.rs` context guard compares by value
  → same-ordinal handles pass). Only the trained suffix + LoRA + AdamW + KV stay
  per-subsystem.
- **Borrowed FP8 storage leaks-on-drop** (`mem::forget(self.weight.clone())`):
  the infer engine owns + frees the bytes once; the autograd view never frees.
  (Review caught a `ptr::read+forget` double-free in the first draft — that leaks a
  *copy* while drop-glue still frees the original → would `cuMemFree` the infer
  engine's live bytes.)
- **Phase-B offload skipped for the shared base** when sharing (`freed rollout=0`,
  base kept resident) — else freeing it crashes the CE forward reading the alias.
- **Load order conditional**: default = student-first (preserves the engine's
  `num_slots` clamp = post-student free VRAM, byte-identical); shared = engine-first
  (so the student can import). Review caught that always-engine-first over-reserves
  KV → default-path student OOM.

## Measured (GPU7, Qwen3.6-27B-FP8, --share-frozen-base, slots=2, rounds=1)

- `borrowing 256 resident FP8 base projections from the rollout engine (zero-copy)`.
- GPU7 peak **39.6 GB** with sharing = ONE base (~27 GB FP8) + small trained student
  + KV + activations. Without sharing the student would allocate its own ~27 GB FP8
  base → **~27 GB saved** (the 7 default-path capability seeds are the two-copy control).
- Phase-B `freed rollout=0` (base resident); CE micro-batch 1→2 ran with **no
  alloc_zeros** → the offload-skip + base-resident CE headroom holds at slots=2.
- Ran load → base eval (produced answers) → phase-A → phase-B → phase-C CE = stable.

## Rule

Sharing FP8 device buffers across the autograd↔infer-cuda boundary works on one GPU
(same primary context) via zero-copy `upgrade_device_ptr` + leak-on-drop; the sharp
edges are (1) the borrowed Drop must NOT free (clone+forget, not ptr::read+forget),
(2) skip the CE-phase offload of the shared base, (3) keep load-order conditional so
the default num_slots clamp is unchanged. Default opt-in/off until the needle gate
licenses a flip; tune `--rollout-num-slots` for CE headroom in shared mode (offload
no longer frees the base).

## Follow-ups

- **In progress:** share-vs-no-share forward-parity gate (greedy `rounds=1`, same
  16 prompts, compare per-round `mean_loss`); pass → flip default-on with a
  `--no-share-frozen-base` opt-out + update this entry with the verdict.
- num_slots vs CE-headroom sweep in shared mode (offload-skip keeps base resident).
- pending-remote: a guidellm/throughput bench (more KV slots from the freed ~27 GB).
