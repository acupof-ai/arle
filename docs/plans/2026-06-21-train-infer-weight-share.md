# Train-Infer Weight Sharing (训推一体) — one FP8 base, train+infer shared

**Status:** planned (one-step / Increment 1 acked by ckl 2026-06-21). Gated on a
real VRAM-residency measurement before implementation (the "54 GB bf16" baseline
was wrong — see §Findings).

## Goal

In the OPD rubric loop the 27B is loaded **twice** — an autograd training copy
(`train_cli.rs:1430`) and an infer-cuda rollout/eval engine
(`train_cli.rs:1501`). Collapse to **one shared FP8 base**: the frozen base
lives once; only the trained suffix/LoRA + optimizer + KV are per-subsystem.

## Findings (verified to file:line by Plan pass)

1. **FP8 frozen-base training already exists.** The autograd loader auto-loads a
   frozen tensor as FP8 block-scaled iff source dtype is `F8_E4M3`
   (`qwen35_loader.rs:1003-1047`, whitelist `is_fp8_cuda_frozen_base_tensor`
   `:1095-1116`); the forward already routes `CudaFp8BlockScaled` through
   `matmul_bt_device_f32_fp8_block_scaled` (`backend_cuda.rs:888`). Frozen-FP8 +
   trained-bf16 coexist (residency decided per-`TensorId` by `requires_grad`).
   The student dir **is** FP8 (`/data01/models/Qwen3.6-27B-FP8`), so the
   frozen-FP8 path is *likely already active* — the 54 GB figure was the
   checkpoint save-format (`Qwen35StudentWeights::FullMaterialized{bf16}`,
   `qwen35_checkpoint.rs:67`), NOT VRAM residency. **MEASURE before assuming.**

2. **The CUDA "context wall" is a non-issue on one GPU.** Both backends call
   `CudaContext::new(ordinal)` which in cudarc 0.19.7 **retains the device
   primary context** (`cudarc .../core.rs:74`). The guard at
   `backend_cuda.rs:362` compares context **by value** (`cu_device/cu_ctx/
   ordinal`), so two `CudaContext::new(0)` objects are `==` → cross-engine FP8
   handle sharing on the same ordinal **passes the guard as-is**. No
   context-bridging needed; just plumb the infer `DeviceMatrix` FP8 pointers
   into an autograd handle (zero-copy).

3. **The real blocker is the CE-phase offload.** Phase-B offloads the rollout
   engine during CE (`rubric_opd.rs:360-374`) to free VRAM. If autograd points
   at the infer engine's FP8 bytes, offload **frees the bytes autograd reads →
   crash.** One-step sharing therefore **requires redesigning Phase-B offload**
   to keep the shared FP8 base resident (offload only KV/scratch).

## VRAM math (to be confirmed by measurement)

- Shared: ~27 GB FP8 base (one copy) + trained-suffix bf16 + LoRA + AdamW + KV.
- Frees ~27–54 GB vs current → ~8–15 extra KV slots at ~3.6 GB/slot (1536 cap)
  → relaxes the `num_slots` clamp (`train_cli.rs:1519`) that throttles eval.

## Measurement gate (do FIRST, §0)

Sample GPU0 VRAM across phases on a live run (free — rides the capability run):
base-eval (both copies resident) vs CE (rollout offloaded). The drop at CE =
rollout-engine resident size; the residual = autograd student + KV. Confirms
(a) whether frozen-FP8 is active, (b) the real duplicate size / saving.

## Implementation (Increment 1, one-step) — >5 files, acked

1. **Infer borrow API** — expose each base projection's FP8 device ptrs
   `(qweight_u8, scale_f32, rows, cols, block_m, block_k)`:
   `infer-api/loaded.rs` (near `remerge_student_lora:605`) → `infer-cuda/
   executor.rs:3124` → `infer-cuda/qwen35.rs:2566-2707` (pristine-base cache).
2. **Autograd zero-copy FP8 import** — `import_fp8_block_scaled_device_ptr(...)`
   on the `Backend` trait + `CudaBackend` (next to `import_bf16_device_ptr_as_f32`
   `backend_cuda.rs:1116`), constructing a `CudaFp8BlockScaledStorage` **view**
   over the foreign ptr (no copy), same primary context.
3. **Loader wiring** — in `qwen35_loader.rs:1140-1162`, frozen base tensors
   *import* (not `upload_fp8_block_scaled` which copies) against the infer
   engine's exposed ptrs. Loader receives an engine borrow.
4. **Load ordering** — `train_cli.rs`: build `LoadedInferenceEngine` (`:1511`)
   **before** the autograd student (`:1433`); pass engine borrow into the loader.
5. **Offload redesign (the crux)** — `rubric_opd.rs:360-374`: exclude the shared
   FP8 base from Phase-B offload (offload KV/scratch only). Add a one-time
   handoff stream sync before the first autograd forward.

LoRA consistency is already in-memory + idempotent (`sync_lora_from_store` →
`remerge_student_lora` from pristine base, `infer_student.rs:280`,
`qwen35.rs:2524`) — unchanged by this.

## Risks

- **(a) FP8 frozen forward numerics** vs bf16/f32 — gate on needle /
  self-consistency (`scripts/needle_gate.py`) before/after; identical risk
  whether or not we share, so validate at the measurement step.
- **(b) offload/lifetime (sharp)** — see §Findings #3; the offload redesign is
  mandatory or it crashes mid-CE. Cross-stream reads need an event fence
  (`CudaPipelineFence`, `tensor.rs:217`).
- **(c) LoRA-apply consistency** — low; already in-memory idempotent.
- **(d) grad-checkpointing** still required for the CE activation set
  (independent of weight residency; `train_cli.rs:1441`).

## Correctness gate

Needle ladder ×3 same-config + self-consistency (NOT byte-identity) on the
shared-FP8-base forward vs the current path, per CLAUDE.md KV-precision gate.

## Review findings (post-impl, 2026-06-21) — agent diff in worktree-agent-ad159d818cedd561e

Implemented behind `--share-frozen-base` (default off), 13 files, typechecks pass.
Critical review (Claude; no codex credits) found:

1. **[FIXED in worktree] Borrowed-FP8 `Drop` double-free.** `backend.rs` used
   `ptr::read(&self.weight)` + `forget` to "leak" the foreign-owned `Arc`s — but
   `ptr::read` copies (refcount unchanged); the original field is still dropped by
   drop-glue → `Arc` 1→0 → `CudaSlice::Drop` `cuMemFree`s the **infer engine's**
   bytes → double-free / UAF at shutdown. Fixed to `mem::forget(self.weight.clone())`
   (bump strong count + leak the bump → drop-glue decrement lands at ≥1, never frees;
   infer engine frees once).

2. **[TODO] Load-order is NOT byte-identical for the default path.** The diff loads
   the rollout engine FIRST in BOTH paths. The engine's `num_slots` clamp is computed
   from *post-weights free VRAM*; engine-first sees ~96 GB free (student not yet
   loaded) → clamps high (~16 slots / ~58 GB KV) → the student then loads into the
   remainder → **default-path OOM** (original student-first order saw less free →
   ~10 slots → fit). FIX: make the order conditional — default = student-first
   (preserve clamp); `--share-frozen-base` = engine-first. (Single if/else around the
   two loads in `train_cli.rs:1486-1573`; duplicate the ~20-line engine-load block or
   extract a closure.)

3. **[TODO, GPU-measure] num_slots vs CE headroom in the shared path.** Sharing SKIPS
   the Phase-B rollout offload (base stays resident, aliased). So during CE the engine
   KV (num_slots × slot) competes with CE activations — unlike the default path which
   offloads the engine. If the engine over-reserves KV (engine-first seeing 96 GB
   free), the shared CE can OOM. The shared run needs a CONSERVATIVE `--rollout-num-slots`
   (or offload only KV during CE, keeping the base). Resolve empirically on GPU7:
   build the worktree, run `--share-frozen-base` with small `--rollout-num-slots`,
   measure (a) base VRAM ~27 GB lower than flag-off, (b) CE does not OOM, (c) needle/
   self-consistency gate. Then tune slots / decide default.

Status: scaffolding + Drop fix done; items 2-3 are the focused finish (load-order
conditional + GPU-verify the shared path + tune num_slots), then needle gate + wins
entry before any default consideration. Default stays opt-in/off throughout.
