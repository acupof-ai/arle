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

## Problems / open blocker (multi-step)

**rollout-256 step-2 `CUDA_ERROR_ILLEGAL_ADDRESS` at the teacher reload** —
even with `CUDA_LAUNCH_BLOCKING=1`. Step-1 trains cleanly; step-2's
infer-student reload + rollout succeed, but the infer-teacher reload+forward
faults.

Diagnosis (SOLID-isolated):
- The offload/reload **primitive is correct**: the 296-token parity test does
  offload→reload→forward TWICE for both the dense student and the packed W4
  teacher, single-engine, non-blocking, **bit-exact** — so it is neither weight
  corruption nor a long-sequence forward bug.
- It is a **multi-allocator concurrency/ordering issue**: the OPD step has three
  co-resident CUDA contexts sharing the device's async memory pool
  (infer-student, infer-teacher, train autograd) with cudarc
  `disable_event_tracking()` (no auto cross-stream waits). The fault only
  appears when the engines run concurrently with the train backend across
  steps; a full `cuCtxSynchronize` at offload/reload was not sufficient. NOT
  the CUDA graph (`--no-cuda-graph` reproduces) and NOT the trim (removed).
- Hit the 2-strike limit on the cross-stream race; the primitive + step-1 goal
  are proven, the multi-step reload ordering is deferred.

Next step (not yet landed): order the teacher reload's allocations against the
train backend's outstanding pool ops explicitly (event/stream fence across the
shared primary context), or stage the reload through a private (non-pool)
allocation; then re-run 3-step rollout-256 + rollout-512.

## Rule

- **Prove the primitive in isolation before blaming the integration.** A
  296-token, 2-cycle, both-format parity test cleanly separated "offload/reload
  is correct" from "the 3-allocator training context races" — without it the
  illegal-address would have looked like weight corruption.
- **A masked "cuda synchronize failed" is OOM until a blocking run says
  otherwise** (re-confirms the rollout128 errors-entry rule): the no-offload
  control reproduced the exact documented OOM, anchoring the freed-VRAM win.
- **`disable_event_tracking()` means co-resident allocators do not auto-order.**
  Sharing one device pool across infer + train CUDA contexts needs explicit
  cross-stream fences; a per-context `cuCtxSynchronize` does not by itself
  serialize another context's async pool frees/allocs.
