# BF16 Frozen-Base Implementation Path

Related: `docs/projects/2026-05-24-opd-mainline-task-backlog.md` T5 and
`docs/experience/errors/2026-05-24-gkd-real-corpus-tape-oom-kill.md`.

## 1. Where is the f32 frozen-base resident memory today?

Current source already has a partial CUDA BF16 frozen-base path; the May 22 "all frozen base widens to f32" diagnosis is stale for rank-2
matrix weights (`docs/research/2026-05-22-opd-frozen-bf16-lora-student-step0-audit.md:8-11`,
`crates/train/src/qwen35_loader.rs:822-881`).  The OPD example creates a CUDA `TensorStore`,
then loads the student via `load_qwen35_lora_from_hf_dir` with LoRA rank 16, alpha 32, and
`LoraTargetSet::AttentionQv` (`crates/train/examples/opd_step_cuda_infer_teacher_train.rs:41-50`,
`crates/train/examples/opd_step_cuda_infer_teacher_train.rs:180-194`).  That
train loader documents the mode as "frozen base plus trainable LoRA adapters"
(`crates/train/src/qwen35_loader.rs:547-550`).  `LoadMode::LoraStudent` constructs
`Qwen35Model::new_with_lora_targets` (`crates/train/src/qwen35_loader.rs:551-558`,
`crates/train/src/qwen35_loader.rs:621-626`).  In `Qwen35Model::new_internal`,
`base_requires_grad` is true only for scratch-train without LoRA; LoRA-student base tensors are
frozen (`crates/train/src/qwen35.rs:1182-1205`).  The example target set is only full-attention
q/v projections (`crates/train/src/lora.rs:21-42`).  LoRA A/B adapter tensors are f32 trainable
`TensorId`s (`crates/train/src/lora.rs:118-127`).  The base linear weight is a `TensorId`;
its forward path is `matmul_bt(flat_x, self.weight)` before optional adapter-delta add
(`crates/train/src/lora.rs:85-90`, `crates/train/src/lora.rs:147-193`).  The loader records
`bf16_cuda_frozen_base` in each planned load (`crates/train/src/qwen35_loader.rs:675-683`).
That flag is true only for LoRA-student mode, frozen slots, BF16 checkpoint dtype,
CUDA backend, and an allowed tensor name (`crates/train/src/qwen35_loader.rs:822-835`).
Allowed direct-BF16 frozen tensors are rank-2 only: embedding, lm_head, full-attention
q/k/v/o projections, linear-attention in/out projections, and MLP gate/up/down projections
(`crates/train/src/qwen35_loader.rs:837-855`).  For those tensors, the loader reads
BF16 bits, uploads through `upload_bf16_bits`, and replaces the tensor device handle
(`crates/train/src/qwen35_loader.rs:864-881`).  `replace_device_handle` clears the host `Vec<f32>`
and makes the device handle authoritative (`crates/autograd/src/tensor.rs:291-307`).  The CUDA
handle type is `CudaBf16Storage(CudaSlice<u16>)` (`crates/autograd/src/backend.rs:110-140`).
CUDA overrides `upload_bf16_bits` to return `DeviceHandle::CudaBf16`
(`crates/autograd/src/backend_cuda.rs:560-570`).  The fallback path still widens
F32/BF16/F16 checkpoint data into `Vec<f32>` and stores `Tensor::new(data, ...)`
(`crates/train/src/qwen35_loader.rs:467-484`, `crates/train/src/qwen35_loader.rs:884-895`).
Therefore the large rank-2 frozen matrices are not f32 resident today when the checkpoint
tensor is BF16 and the backend is CUDA (`crates/train/src/qwen35_loader.rs:822-881`).
The f32 frozen train-side tensors that remain are the tensors excluded by the rank-2/name
gate.  Per-layer input and post-attention RMSNorm weights are rank-1 frozen tensors
(`crates/train/src/qwen35.rs:1237-1278`, `crates/train/src/qwen35_loader.rs:837-840`).
The final RMSNorm weight is rank-1 and frozen (`crates/train/src/qwen35.rs:1527-1534`,
`crates/train/src/qwen35_loader.rs:837-840`).  Full-attention q_norm/k_norm
weights are rank-1 frozen tensors (`crates/train/src/qwen35.rs:1311-1349`,
`crates/train/src/qwen35_loader.rs:837-840`).  Linear-attention conv1d weight is rank-2 in the train
model, but its suffix is not in the BF16 allow-list (`crates/train/src/qwen35.rs:1392-1436`,
`crates/train/src/qwen35_loader.rs:837-855`).  Linear-attention `dt_bias`, `a_log`,
and output `norm` are rank-1 frozen tensors (`crates/train/src/qwen35.rs:1393-1456`,
`crates/train/src/qwen35_loader.rs:837-840`).  RoPE cos/sin caches are f32 non-checkpoint
frozen tensors and are included in the model id set (`crates/train/src/qwen35.rs:1536-1542`,
`crates/train/src/qwen35.rs:2747-2773`).  `qwen35_checkpoint.rs` is save-side: it saves full
materialized weights or adapter-only weights (`crates/train/src/qwen35_checkpoint.rs:67-78`,
`crates/train/src/qwen35_checkpoint.rs:116-135`).  Adapter-only save builds the registry
from `student.adapter_name_map()` and does not change frozen-base residency during training
(`crates/train/src/qwen35_checkpoint.rs:268-310`).  The infer/runtime Qwen3.5 path is
already BF16-first: `DeviceVec` is BF16, `DeviceMatrix` is BF16 unless packed/quantized,
and `load_tensor_1d/2d` load those types (`crates/cuda-kernels/src/tensor.rs:596-642`,
`crates/cuda-kernels/src/tensor.rs:1197-1207`, `infer/src/weight_loader.rs:122-176`).
Runtime Qwen3.5 loads embedding, attention projections, MLP projections, and norms through
those device types, except linear-attention `A_log` and `norm_weight`, which are f32
(`infer/src/model/qwen35/weights.rs:300-455`, `infer/src/model/qwen35/weights.rs:455-467`).
Runtime serve LoRA merges q/v adapters into dense BF16 base matrices at load
time; that is not the train-side LoRA path (`infer/src/model/qwen35/lora.rs:1-7`,
`infer/src/model/qwen35/lora.rs:165-205`).

## 2. What is the autograd tape allocator's sizing rule?

I did not find a current code path that allocates directly by `prompt_max_tokens * rollout_len`.  The OPD rollout validator uses
an additive bound: `total_len = prompt_len + rollout_len` (`crates/train/src/opd.rs:462-469`).
The step validates that additive shape before rollout (`crates/train/src/opd.rs:901-919`).
The rollout vector starts as `prompt_ids` and appends exactly `cfg.rollout_len` generated tokens
(`crates/train/src/opd.rs:953-958`, `crates/train/src/opd.rs:968-1018`).  The teacher/student
KL logits are shaped `[1, rollout.len(), vocab]` (`crates/train/src/opd.rs:1061-1077`,
`crates/train/src/opd.rs:1080-1085`).  `kl_distill_loss` validates `num_positions == logits.numel()
/ vocab` (`crates/train/src/loss.rs:28-32`, `crates/train/src/loss.rs:89-115`).  Corpus-truth GKD
adds a second student forward over `prompt_ids + corpus_tokens` (`crates/train/src/opd.rs:587-650`).
That SFT path computes `total_len = prompt_ids.len() + corpus_tokens.len()`, then forwards
the student over the combined sequence (`crates/train/src/opd.rs:610-650`).  The example runs
`maybe_eval(0, ...)` before the train loop, so "before first train step" can be eval-only prompt
memory, not rollout memory (`crates/train/examples/opd_step_cuda_infer_teacher_train.rs:570-579`,
`crates/train/examples/opd_step_cuda_infer_teacher_train.rs:929-965`).  Eval computes
teacher and student logits over each prompt and passes `prompt.len()` as `num_positions`
(`crates/train/examples/opd_step_cuda_infer_teacher_train.rs:967-995`).
The prompt file loader is capped by `args.prompt_max_tokens`
(`crates/train/examples/opd_step_cuda_infer_teacher_train.rs:446-453`).  The infer
teacher runtime is loaded with capacity `prompt_max_tokens + rollout_len
+ 32`, but that is teacher runtime capacity, not the autograd allocation
rule (`crates/train/examples/opd_step_cuda_infer_teacher_train.rs:256-267`,
`crates/train/examples/opd_step_cuda_infer_teacher_train.rs:509-534`).  The CUDA
allocator primitive is direct: `size = shape_size(shape)` and `alloc_zeros::<f32>(size)`
(`crates/autograd/src/backend_cuda.rs:602-617`).  The largest documented train allocation
shape is logits/loss over sequence and vocab: `[B, S, V] = 2 x 512 x 248070 x 4 B ~=
1 GB` (`crates/autograd/src/backend_cuda.rs:1140-1148`).  Matmul outputs allocate
f32 `m * n` or `batch * m * n` (`crates/autograd/src/backend_cuda.rs:301-392`).
BF16 RHS matmul allocates temporary BF16 activation/output buffers, then widens the
result back to f32 (`crates/autograd/src/backend_cuda.rs:394-467`).  The 2026-05-24
error entry says the allocator is keyed on `prompt_max_tokens x rollout_len`
(`docs/experience/errors/2026-05-24-gkd-real-corpus-tape-oom-kill.md:64-68`).  The
source-backed correction is: OPD KL is bounded by `prompt_max_tokens + rollout_len`,
corpus SFT by `prompt_max_tokens + completion_tokens`, and the dominant tensor bytes
multiply sequence length by `vocab_size * sizeof(f32)` (`crates/train/src/opd.rs:462-469`,
`crates/train/src/opd.rs:610-650`, `crates/autograd/src/backend_cuda.rs:1140-1148`).  The failed
run used `--prompt-max-tokens 512`, `--rollout-len 8`, `--gkd-lambda 0.3`, and `--sft-anchor
corpus-truth` (`docs/experience/errors/2026-05-24-gkd-real-corpus-tape-oom-kill.md:24-34`).
The same note reports MMLU prompt p95 501 and max 503, so eval step 0 can already reach near-512
prompt-logits shape (`docs/experience/errors/2026-05-24-gkd-real-corpus-tape-oom-kill.md:11-16`,
`crates/train/examples/opd_step_cuda_infer_teacher_train.rs:967-995`).

## 3. What is the conversion cost for BF16 frozen base?

Projection and lm_head matrices are trivial because the current train path already supports BF16 RHS for the
allowed rank-2 tensors.  Train `linear_forward` flattens input and calls `matmul_bt`
(`crates/train/src/qwen35.rs:2284-2326`).  LoRA base projection also calls `matmul_bt(flat_x,
self.weight)` (`crates/train/src/lora.rs:147-193`).  Autograd `matmul_bt` ensures device handles and
dispatches to the backend (`crates/autograd/src/ops/matmul.rs:54-93`).  CUDA `matmul_bt` detects
BF16 RHS and calls `matmul_bt_device_f32_bf16` (`crates/autograd/src/backend_cuda.rs:707-733`).
That path rounds f32 activations to BF16, uses cuBLAS GEMM_EX with BF16 operands and FP32
compute, then widens BF16 output to f32 (`crates/autograd/src/backend_cuda.rs:394-467`).
Backward through a BF16 frozen RHS has a BF16-aware lhs-gradient path
(`crates/autograd/src/backend_cuda.rs:3178-3195`).  Plain `matmul` is medium if ever needed for
frozen BF16 RHS.  CUDA `matmul` still calls `cuda_slice` for both operands, and `cuda_slice`
rejects `CudaBf16` on the f32-only path (`crates/autograd/src/backend_cuda.rs:685-704`,
`crates/autograd/src/backend_cuda.rs:263-274`).  Current frozen base weights flow through
`matmul_bt`, not through SDPA activation `matmul` (`crates/train/src/qwen35.rs:2284-2326`,
`crates/autograd/src/ops/attention.rs:52-80`).  Embedding is trivial because it is already wired.
Autograd embedding uses the lazy CUDA path and passes the table handle into `Backend::embedding`
(`crates/autograd/src/ops/embed.rs:13-84`).  CUDA embedding supports `DeviceHandle::CudaBf16`
through `embedding_bf16_to_f32` (`crates/autograd/src/backend_cuda.rs:3782-3884`).
The BF16 embedding kernel widens BF16 table rows into f32 output
(`crates/autograd/src/backend_cuda/kernels/embedding.cu:38-63`).  Device-token rollout embedding
also has a BF16 table path (`crates/autograd/src/backend_cuda.rs:3887-3988`).  RMSNorm weights
are medium.  Train `qwen35_rmsnorm` offsets norm weights by ones, then calls autograd `rmsnorm`
(`crates/train/src/qwen35.rs:250-272`).  Autograd RMSNorm makes the weight host-resident and
passes `&[f32]` into the backend (`crates/autograd/src/ops/norm.rs:9-77`).  CUDA RMSNorm takes
`weight: &[f32]` and launches `rms_norm_f32` (`crates/autograd/src/backend_cuda.rs:1373-1388`,
`crates/autograd/src/backend_cuda/kernels/rms_norm.cu:9-45`).  A BF16 norm-weight tranche therefore
needs a backend API sibling or a device-weight RMSNorm path.  Full attention after projections
is trivial/no-op for frozen-base BF16.  Train Qwen35 full attention consumes q/k/v projection
outputs, then runs q/k norm, RoPE, SDPA, gate, and o_proj (`crates/train/src/qwen35.rs:492-594`).
Autograd SDPA itself is f32 activation math built from `matmul`, softmax, and `matmul`
(`crates/autograd/src/ops/attention.rs:52-80`).  The autograd decode SDPA fast path is
f32-only and rollout-only (`crates/autograd/src/backend_cuda/kernels/attention.cu:1-87`,
`crates/autograd/src/ops/attention.rs:123-150`).  Changing SDPA precision is
a separate activation-memory project, not required for frozen-base residency.
Linear attention core is hard if converted fully.  Train linear attention calls four
projection linears, then `linear_attention_core` with conv1d, dt_bias, A_log, and norm
(`crates/train/src/qwen35.rs:912-949`).  `linear_attention_core` forces every input to host
and reads f32 data (`crates/autograd/src/ops/linear_attention.rs:30-89`).  Its backward
is host f32 `Vec<f32>` math (`crates/autograd/src/ops/linear_attention.rs:300-420`,
`crates/autograd/src/ops/linear_attention.rs:851-878`).  Leaving those small
non-projection tensors f32 is medium/low risk; moving the linear-attention core to
real CUDA BF16/f32 is hard.  Runtime/infer kernels are BF16 references, not the taped
train path.  Runtime embedding dispatch reads BF16 `DeviceMatrix` and writes BF16
hidden states (`infer/src/ops/embedding.rs:8-33`, `infer/src/ops/embedding.rs:110-134`).
Runtime RMSNorm reads BF16 weights and BF16 activations (`infer/src/ops/norm.rs:7-43`,
`crates/cuda-kernels/csrc/misc/norm.cu:636-642`).  Runtime linear dispatch calls dense BF16
GEMM kernels (`infer/src/ops/linear.rs:2088-2126`, `infer/src/ops/linear.rs:2167-2278`).
Runtime paged attention selects TileLang BF16 kernels (`infer/src/ops/attention.rs:1099-1210`).
Runtime GDR uses BF16 qkv/b/a/dt and f32 A_log/state (`infer/src/ops/recurrent.rs:8-12`,
`infer/src/ops/recurrent.rs:124-169`).  The taped OPD student uses autograd CUDA/NVRTC kernels,
not the runtime `crates/cuda-kernels/csrc` path (`crates/autograd/src/backend_cuda/kernels.rs:7-58`,
`crates/autograd/src/backend_cuda/kernels.rs:61-117`).

## 4. Is there an existing BF16 path?

Yes: teacher-logit bridge, autograd BF16 storage, and Qwen35 loader support.  The May 21 win imported infer-owned BF16
device logits into autograd as f32 CUDA handles without host materialization
(`docs/experience/wins/2026-05-21-arle-autograd-bf16-d2d-bridge.md:3-12`).
That test preserved BF16 bytes and matched exact BF16-to-f32 widening
(`docs/experience/wins/2026-05-21-arle-autograd-bf16-d2d-bridge.md:22-28`).
It deferred cross-runtime OPD correctness and wall-clock validation
(`docs/experience/wins/2026-05-21-arle-autograd-bf16-d2d-bridge.md:30-38`).  That bridge
solves teacher-logit import, not frozen student base residency.  Autograd now has
`DeviceHandle::CudaBf16` (`crates/autograd/src/backend.rs:131-140`).  CUDA upload,
readback, eval validation, BF16 embedding, and BF16 RHS `matmul_bt` all exist
(`crates/autograd/src/backend_cuda.rs:560-570`, `crates/autograd/src/backend_cuda.rs:620-658`,
`crates/autograd/src/backend_cuda.rs:667-682`, `crates/autograd/src/backend_cuda.rs:3782-3988`,
`crates/autograd/src/backend_cuda.rs:707-733`).  The train loader already stores selected rank-2
frozen BF16 base tensors as device-only BF16 (`crates/train/src/qwen35_loader.rs:822-881`).
What remains is not "add BF16 frozen base" generically; it is to measure whether this path
is active on the failing run, quantify remaining f32 resident weights, and verify whether
the real OOM is `[S,V]` tape memory (`crates/autograd/src/backend_cuda.rs:1140-1148`,
`docs/experience/errors/2026-05-24-gkd-real-corpus-tape-oom-kill.md:52-58`).

## 5. Recommended tranche shape

Commit 1: residency accounting only, no math change.
Add a CUDA loader/accounting diagnostic that counts LoRA-student frozen element counts by
handle kind: `CudaBf16`, `Cuda`, and host-only f32 (`crates/autograd/src/backend.rs:131-140`,
`crates/train/src/qwen35_loader.rs:822-881`).  Test in isolation with student load; no full OPD
smoke required.  Risk: low.  Commit 2: allocation-shape diagnostic around first eval/step.
Instrument the central allocator or a scoped wrapper around `CudaBackend::zeros`, which
is the common f32 allocation site (`crates/autograd/src/backend_cuda.rs:602-617`).
Run with `--eval-steps 0`, because the example evaluates before the train
loop (`crates/train/examples/opd_step_cuda_infer_teacher_train.rs:570-579`,
`crates/train/examples/opd_step_cuda_infer_teacher_train.rs:929-995`).  Risk: medium;
keep logging gated so timing is not confused with training.  Commit 3a: BF16 device-weight
RMSNorm only if Commit 1 proves norm memory matters (`crates/train/src/qwen35.rs:1237-1278`,
`crates/train/src/qwen35.rs:1311-1349`, `crates/autograd/src/ops/norm.rs:37-77`).  Risk:
medium; test RMSNorm parity in isolation before OPD.  Commit 3b: linear-attention frozen tensor
work only if separately licensed.  The current train core is host f32, so do not mix it with
RMSNorm (`crates/autograd/src/ops/linear_attention.rs:30-89`).  Risk: hard.  Commit 4: full
real-corpus OPD smoke.  Use the failing shape from the error note: 4B teacher, 0.8B student,
real corpus, `--prompt-max-tokens 512`, `--rollout-len 8`, lambda 0.3, and corpus-truth
SFT (`docs/experience/errors/2026-05-24-gkd-real-corpus-tape-oom-kill.md:24-39`).
Acceptance is reaching `eval_summary step=0` and at least one `train_step`
line, because the reported failure happened before the first train step
(`docs/experience/errors/2026-05-24-gkd-real-corpus-tape-oom-kill.md:41-54`).  Do not combine
BF16 residency expansion with prompt filtering, activation checkpointing, or loss/tape changes
in the same commit.

## Summary table

| tranche | lines | risk | depends on |
| --- | ---: | --- | --- |
| Residency accounting for LoRA student handles | 80-140 | low | current loader |
| Allocation-shape diagnostic at eval/step | 60-120 | medium | allocator path |
| BF16 device-weight RMSNorm | 120-220 | medium | residency proof |
| Linear-attention frozen tensor BF16 | 200-400 | hard | separate license |
| Full real-corpus OPD smoke | 0-40 | high | fresh GPU and prior gates |

## What I did NOT verify

I did not run the failing OPD command, so there is no fresh peak-memory counter.
I did not inspect the actual safetensors dtype in the local 0.8B model; the direct BF16 path requires `Dtype::BF16` (`crates/train/src/qwen35_loader.rs:822-835`).
I did not prove the 2026-05-24 OOM is model-base memory; code evidence points more strongly to `[S,V]` logits/loss tape memory for the real-corpus shape (`crates/autograd/src/backend_cuda.rs:1140-1148`, `docs/experience/errors/2026-05-24-gkd-real-corpus-tape-oom-kill.md:52-58`).
I did not benchmark BF16 RHS matmul drift against the f32-base LoRA path.
I did not audit optimizer-state memory beyond the example's use of `student_trainable_params` (`crates/train/examples/opd_step_cuda_infer_teacher_train.rs:196-198`, `crates/train/examples/opd_step_cuda_infer_teacher_train.rs:557-568`).
