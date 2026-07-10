# OPD engine weight time-share (offload/reload) lands rollout-256 step-1 on 16 GB

## Context

rollout-256 long-CoT OPD OOMs in `tape.backward` on the 16 GB RTX 4070 Ti
SUPER: ~9.3 GB resident (teacher W4 3.1 GB + infer-student 1.6 GB + train-student
1.4 GB + ctx 1.3 GB + scratch 1.7 GB) leaves ~6.6 GB, but the rollout-256
backward needs ~8 GB
([`errors/2026-05-29-opd-rollout128-train-crash.md`](../errors/2026-05-29-opd-rollout128-train-crash.md)).
The OPD step has memory-disjoint phases — during the student backward BOTH the
infer-student (rollout-only) and the infer-teacher (already scored, logits
cached) are idle — so the vLLM-sleep / verl-HybridFlow time-share applies
([`research/2026-05-29-opd-memory-best-practice.md`](../../research/2026-05-29-opd-memory-best-practice.md)
Tier 2). This is the user-approved unblock to test full (non-truncated) CoT
([`errors/2026-05-29-opd-gsm8k-rollout64-regression.md`](../errors/2026-05-29-opd-gsm8k-rollout64-regression.md):
rollout-64 truncates CoT → −10 pp GSM8K).

Build: `CUDA_HOME=/opt/cuda NVCC_CCBIN=g++-14 INFER_TILELANG_PYTHON=.venv
TORCH_CUDA_ARCH_LIST=8.9 ARLE_CUDA_DISABLE_FLASHMLA=1`, `--release`, sm89.

## What landed

**Infer-engine weight offload/reload API (format-agnostic).**
- `DeviceMatrix::offload_to_host` / `reload_from_host` (cuda-kernels
  `tensor.rs`): snapshots EVERY device buffer (dense bf16 + INT8/INT4 qweight +
  scales + Marlin packed/scales/channel-scales + hybrid W4A8/W4-FP8 sidecars +
  TurboQuant packed) to host RAM, frees the VRAM (1-elem placeholder), returns
  bytes freed. `DeviceVec::offload_to_host`/`reload_from_host` and
  `offload_raw_slice`/`reload_raw_slice` for the bare `CudaSlice<f32>` SSM
  fields. `DeviceContext::{mem_info_bytes, trim_memory_pool}` helpers.
- `Qwen35Model::{offload_weights_to_host, reload_weights_to_device,
  is_offloaded}` (qwen35 `weights.rs`): walks embed/lm_head/norm/cos·sin +
  every block (layernorms, MLP gate/up/down, full or linear attn) in a fixed
  order; LoRA-merge-safe (snapshots the live merged weights; the pristine
  `lora_base_cache` host copy is untouched). Full `cuCtxSynchronize` device
  barrier around the D2H/H2D round-trip.
- Wired through `ModelForward` trait → `EngineOffloadRequest` scheduler control
  message (single-writer thread that owns the model) → `SchedulerHandle` →
  `RequestHandleInferenceEngine` → `LoadedInferenceEngine::{offload,reload}_engine_weights`.
- Train: `InferStudent`/`InferTeacher`/`MultiTeacher` offload+reload;
  `TeacherForward` trait hooks; OPD `opd.rs` offloads BOTH idle engines together
  inside `backward_chunked_kl_rollout` after the teacher scores (on a quiesced
  device) and reloads before the next step's rollout/scoring. Gated by
  `ARLE_OPD_ENGINE_OFFLOAD=1`; default path unchanged.

**Pre-existing regression fixed (bundled, attributed):** the oplib dispatch
relocation (a33ed6fd/3d171bf2) made `(batch=1, DenseBf16)` resolve to
`Bf16GraphsafeGemm` in BOTH phases, but the single-token `gemv` launcher only
serviced `Bf16Gemv` → `unreachable!("no matching weight storage")` on any dense
single-token decode (the tied-lm_head last-row projection in
`forward_token_logits`). This broke ALL `forward_token_logits` (existing
`forward_token_logits` test red on HEAD), i.e. OPD teacher/student scoring.
Added a `Bf16GraphsafeGemm` arm to the gemv launcher (`ops/linear.rs`) routing
the dense single-token through the same handwritten BF16 GEMV (numerically
identical for N=1).

## Results

**Parity (asserting test `infer/tests/engine_offload_parity.rs`, 296-token
prefill = rollout-256 teacher shape, 2 offload→reload cycles each):**
- dense Qwen3.5-0.8B student: freed **1435.1 MiB**, argmax+logits-checksum
  **bit-identical** before/after (and after a 2nd cycle).
- W4A8-Marlin Qwen3.5-4B teacher: freed **2979.3 MiB**, **bit-identical** —
  the packed/quantized side buffers round-trip correctly.

**Goal — rollout-256 step-1 TRAINS (no OOM):** W4 teacher + 0.8B student LoRA
r16 α32, gsm8k-train.jsonl, kl-chunk-size 16, `ARLE_OPD_ENGINE_OFFLOAD=1`,
`CUDA_LAUNCH_BLOCKING=1`:
- both engines offloaded inside the step: student 1435 MiB + teacher 2979 MiB
  = **~4.4 GB freed before the backward**.
- VRAM (nvidia-smi): **8623 MiB → 4753 MiB** during the backward — the
  backward fits with ~11 GB free vs. the ~7.3 GB that OOM'd without offload.
- `loss = 1.0175e-4` (finite, ~1e-4 as expected); rollout_len=297 (full CoT);
  step ≈ 51 s (rollout 14 s, teacher 0.1 s, student fwd 5.7 s, backward 29 s).
- **Control:** identical config WITHOUT offload OOMs at step 1 with the
  documented masked "cuda synchronize failed" (free 7288 MiB < ~8 GB needed) —
  confirms the offload is what makes rollout-256 fit.

## Multi-step blocker — RESOLVED (2026-05-30, follow-up)

**All 10 rollout-256 OPD steps now train cleanly** (`ARLE_OPD_ENGINE_OFFLOAD=1`,
both engines time-shared). The earlier "step-2 `CUDA_ERROR_ILLEGAL_ADDRESS` at
the teacher reload" diagnosis was **wrong on two counts** — the real story (all
evidence-backed, two via `compute-sanitizer` / a `marlin_packed`-presence
counter, not inference):

1. **The illegal address is a D2D use-after-free in the teacher logits bridge,
   not a teacher-reload race.** `InferTeacher::forward_logits_device` D2D-copies
   the infer engine's logits buffer into the train backend via
   `import_bf16_device_ptr_as_f32` → `cuMemcpyDtoD_v2`. That copy issues on the
   per-thread default stream, which is **not** host-blocking under
   `disable_event_tracking()`; the infer engine then frees its logits buffer
   (`cuMemFreeAsync`) as the bridge returns, racing the still-running copy.
   `compute-sanitizer` named it exactly: *"Use-after-free … accessed after it is
   free'd"* at `cuMemcpyDtoD_v2` vs `cuMemFreeAsync`, sizes = `seq_len × vocab ×
   2` (the BF16 teacher logits). It vanishes under `CUDA_LAUNCH_BLOCKING=1`
   (ordering bug, not corruption). **Fix:** a single `cuCtxSynchronize` after the
   D2D copy in `crates/autograd/src/backend_cuda.rs::copy_bf16_device_ptr_to_local`,
   before the foreign source can be freed.

2. **The "missing W4A8 Marlin-packed side buffer" was the inter-step KL eval
   forwarding the *offloaded* teacher, not a reload bug.** The engines were
   offloaded inside the backward and reloaded only at the *next* step's start;
   between steps the example's `maybe_eval` (and the checkpoint save) ran a
   teacher/student forward against a 1-elem placeholder → the Marlin GEMM
   tripped on `marlin_packed.is_none()`. A `DBG_RELOAD` `marlin_packed`-count
   confirmed the reload restores every layer; the offload/reload **primitive is
   correct** (the W4 parity test passes). **Fix:** reload both engines at the
   **end of the backward** (`backward_chunked_kl_rollout`), so the offload
   window is strictly inside the heavy student backward and the engines are
   resident for whatever the caller does next (eval / checkpoint / next rollout).
   Reload is idempotent, so the next step's pre-rollout/pre-scoring reloads
   become no-ops.

Also landed: `ARLE_OPD_ENGINE_OFFLOAD` is now a **mode** — `1`/`all` (both),
`student` (student only, frees ~1.4 GB — OOMs on the longest CoTs), `teacher`
(teacher only, frees ~3 GB), `off`. `all` is the recommended setting and the one
verified below.

### Verified — rollout-256, 10 steps, `ARLE_OPD_ENGINE_OFFLOAD=1`

W4A8-Marlin-4B teacher + 0.8B-Base student LoRA r16, gsm8k-train.jsonl,
kl-chunk-size 16, default eval (steps 0/2/5/10), `--save-every 10`:

- **All 10 steps + all 4 inter-step evals + checkpoint save: clean, EXIT=0.**
- loss `1.0172e-4 → 8.998e-5` (all finite ~1e-4, −11.5% over 10 steps);
  rollout lengths 287–**370** tokens (full CoT) all fit.
- Per step both engines offload (student 1435 MiB + teacher 2979 MiB = **4.4 GB
  freed**) before the backward, reload after.
- VRAM (nvidia-smi): during-backward peak **used ≈ 9213 MiB / free ≈ 6730 MiB**;
  the 370-token backward — which OOM'd in `student`-only mode (only 1.4 GB freed)
  — fits comfortably with the full 4.4 GB freed.
- step ≈ 69 s mean (rollout 14–46 s, teacher ≈ 0.1 s, student fwd ≈ 7 s,
  backward ≈ 30–37 s); 722 s total.
- **Checkpoint loads**: `final/adapter_model.safetensors` = 24 tensors /
  638 976 elems (q/v LoRA A·B), zero non-finite, valid PEFT r16 config.

## Rule

- **A use-after-free across two allocators surfaces at the *next* sync, not the
  faulting op — `compute-sanitizer` names the real culprit in minutes.** Three
  weeks of "it's a teacher-reload race" dissolved when memcheck pointed at
  `cuMemcpyDtoD_v2` vs `cuMemFreeAsync` on the BF16-logits bridge. A
  cross-allocator D2D read on the per-thread default stream needs an explicit
  fence before the *source's* owner frees it; `disable_event_tracking()` removes
  the implicit ordering you'd otherwise get.
- **"Primitive is correct" must be proven by the symptom's own counter, not a
  green isolation test.** The W4 parity test passed precisely because it never
  forwarded an *offloaded* model; the OPD failure was the eval doing exactly
  that. A 2-line `marlin_packed`-presence `eprintln` settled offload-vs-reload-
  vs-caller in one run — far faster than the multi-allocator-race theory it
  replaced.
- **Scope the offload window to the operation that needs the VRAM.** Offloading
  for the whole inter-step gap (reload only at next step) breaks every caller
  that touches the engine between steps (eval, checkpoint). Reload at the end of
  the backward; the headroom you need is during the backward, nowhere else.
- **A masked "cuda synchronize failed" is OOM until a blocking run says
  otherwise** (re-confirms the rollout128 errors-entry rule): the no-offload
  control reproduced the exact documented OOM, anchoring the freed-VRAM win.

### Known pre-existing (out of scope here)

- `engine_offload_parity.rs` **dense-student** case is **red on HEAD** (argmax
  preserved, logits checksum drifts) — independent of this fix (the diff touches
  no offload/reload primitive). Benign for the greedy OPD rollout (argmax-
  stable); flagged for a separate dense-bf16 reload audit.
