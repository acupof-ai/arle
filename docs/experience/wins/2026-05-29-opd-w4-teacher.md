# OPD W4 native-resident teacher frees ~5 GB, unblocks rollout-128/256

## Context

Goal: shrink the OPD teacher (Qwen3.5-4B dense BF16) from ~8 GB resident to a
native-resident W4 (~2-3 GB) so the 16 GB RTX 4070 Ti SUPER (sm89) has headroom
for rollout-128 *and* rollout-256 alongside the student + infer rollout engine.
Teacher precision does not matter (it only emits KL targets), so the simplest
naive symmetric RTN W4 is fine.

Prior facts (re-confirmed): FP8 does not save VRAM (loader dequantizes
FP8→BF16-resident, `weight_loader.rs:1001`); Marlin W4A8 keeps weights packed
4-bit resident. ARLE supports symmetric Marlin W4A8 / sym-GPTQ only.

Build env: `CUDA_HOME=/opt/cuda NVCC_CCBIN=g++-14
INFER_TILELANG_PYTHON=…/.venv/bin/python TORCH_CUDA_ARCH_LIST=8.9
ARLE_CUDA_DISABLE_FLASHMLA=1`, `--release`, mem_fraction_static=0.05, one GPU job.

## What Worked

**Quant source: self-quantized (no loadable W4 existed on ModelScope).** Earlier
search found only zero-point AWQ (unsupported), AX650 hardware GPTQ, and a 9B
GPTQ. Used the repo's existing `scripts/quantize_qwen3_w4a8.py` (naive
max-scale symmetric RTN, `gptq_scales=None` default path) targeting ARLE's
MarlinW4A8 side-tensor convention. One run, no thrash:

```
.venv/bin/python scripts/quantize_qwen3_w4a8.py \
  --src /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --dst /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B-W4A8-marlin
# converted 307 linear tensors; on-disk 8.8G → 3.2G
```

The script quantizes every shape-qualifying 2D linear (`in%128==0 &&
out%256==0`): full-attn q/k/v/o + MLP gate/up/down + the linear-attn proj
matrices (in_proj_qkv/z, out_proj). Narrow projections (linear_attn in_proj_a/b,
out=32), conv1d, norms, and the tied embed_tokens stay dense.

**One loader fix required** (`infer/src/weight_loader.rs`, the `marlin_w4a8`
branch in `load_tensor_2d_maybe_quantized_with_config`): the W4A8 branch
unconditionally demanded a `.marlin_w4a8_qweight` sibling for *every* tensor it
loaded, but the qwen35 single-GPU loader routes the dense-only linear_attn
in_proj_a/b through this same path. Added a per-tensor fall-through to dense
bf16 when the packed sibling is absent. **Critical detail:** the guard probes
the shards via `find_tensor(...).is_err()`, NOT `weight_map.contains_key()` —
`load_shard_info` returns an *empty* weight_map for single-file checkpoints
(the script writes one `model.safetensors`, no index), so a `weight_map` check
falsely reports every quantized tensor missing and silently dropped the whole
model to the dense path (the original failure: `in_proj_qkv.weight not found`).

Quant auto-detection (`QuantLoadConfig::from_model_path` →
`MarlinW4A8`) works unchanged from the script's `config.json`
`quantization_config = {quant_type: marlin_w4a8, group_size: 128}`.

## Results (sm89, 16 GiB, mem_fraction_static=0.05, infer rollout ON)

Example: `opd_step_cuda_infer_teacher_train`, student =
Qwen3.5-0.8B-Base (LoRA r16), teacher via `--teacher-model <dir>` (infer engine).

VRAM (MiB), label `03_after_teacher_infer_load` minus `02_after_infer_student_load`:

| teacher        | used@03 | teacher delta | free@step-start |
|----------------|---------|---------------|-----------------|
| BF16 4B (dense)| 12524   | **8166**      | 3281            |
| **W4A8 4B**    | 7500    | **3142**      | **8347**        |

**Teacher footprint 8166 → 3142 MiB (−5024 MiB, −62%).** The BF16 8166 MiB delta
reproduces the user's prior measurement exactly. Free VRAM at step start
+5066 MiB.

Fit:
- **rollout-128**: W4 teacher + student + rollout load and eval cleanly, 8347 MiB
  free (vs 3281 MiB BF16). eval KL = `2.561e-5` (finite, sane ~1e-5).
- **rollout-256 (bonus)**: with W4, the **full train step completes** —
  `06_after_train_step_1 used_mib=9216 free_mib=6727`, KL `2.561e-5`. This is
  infeasible under BF16 (only 3.3 GB free before the step even starts).

KL sanity: W4 teacher eval KL `2.561e-5` vs BF16 `1.689e-5` — same order of
magnitude, finite. The W4 logits differ from BF16 (expected; precision-insensitive
KL target).

## Known pre-existing issue (NOT caused by this change)

At **rollout-128** the train step (not eval) hits
`OPD student chunk KL Qwen3.5 forward autograd error: cuda synchronize failed`.
Confirmed **identical with the BF16 teacher** and across kl_chunk_size ∈ {32,128},
so it is a pre-existing rollout-128-shape student-forward autograd bug, independent
of teacher precision. rollout-256 completes the step. Eval (full teacher+student
forward + KL) succeeds at both sizes. Tracked separately; this change is purely
the teacher VRAM reduction.

## Rule

ARLE single-file (no-index) safetensors checkpoints carry an **empty** weight_map
— per-tensor presence checks in the loader MUST probe the shards via
`find_tensor`, never `weight_map.contains_key`. A naive RTN symmetric W4A8
self-quant via `scripts/quantize_qwen3_w4a8.py` is the canonical, low-effort
path for a precision-insensitive OPD teacher; the dense fall-through keeps narrow
linear-attn projections (out<256) and norms intact while still cutting the 4B
teacher to ~3 GB resident.

Preserved: dense BF16 teacher remains the default/fallback (no W4A8 config →
unchanged path). 594 infer lib tests green (no-cuda); no-cuda typecheck +
clippy clean on the touched file.
