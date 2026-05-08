# W4A8 GPTQ end-to-end gate FAIL — kernel ≠ script,layout divergence in GPTQ-aware pack

> Phase 1b script-level LICENSED at 0.02% drift across 4 layers(`e753af7`)
> but end-to-end `test_w4a8_vs_bf16_token_diff` produces **token id 0
> garbage**(`"!!!!!!!!!!!!!!!!!!!!"`)。
>
> Loader detection issues resolved this tick(metadata mismatch fixes
> committed)。The W4A8 path NOW correctly engages,but the CUDA kernel
> produces NaN/garbage output despite Python `manual_unpack_w4a8` round-trip
> verifying near-zero drift。**Kernel-vs-Python-diag divergence is the new
> debugging surface**。

## What I did this tick

1. **Discovered loader metadata mismatch**:`load_quant_meta` priority
   order is GGUF → TurboQuant → GPTQ-via-quantize_config.json → AWQ →
   config.json `quantization_config` fallback。Script wrote
   `quantize_config.json` which forced GPTQ branch → `marlin_w4a8: false`。
2. **Fixed by deleting `quantize_config.json` + patching config.json
   inline `quantization_config: {quant_type: marlin_w4a8, group_size: 128}`**。
   Loader correctly detects MarlinW4A8 → engages W4A8 path。
3. **Updated `convert_gptq_w4a16_to_w4a8_marlin.py`**:
   - Removed `quantize_config.json` write
   - Added `config.json` patch with inline `quantization_config`
4. **Re-ran `test_w4a8_vs_bf16_token_diff`**:loader OK,kernel produces
   garbage。

## Empirical evidence — kernel garbage vs script PASS

```
BF16:  " Paris. The capital of Germany is Berlin. The capital of Italy is Rome..."
W4A8:  "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
       ↑ all token id 0 (<pad> or <unk>)

W4A8 vs BF16: matched first 0/32 tokens, diff 100.0%
First divergence: idx=0 bf16=Some(12095) w4a8=Some(0)
```

But Python script verification produces:
- `scripts/verify_gptq_w4a8_repack_quality.py` on 4 layers: **0.02-0.03% rel diff,PASS**
- `scripts/diag_w4a8_pack_roundtrip.py` on multi-shape: **PASS**

**Gap**:Python `manual_unpack_w4a8`(in diag)recovers weights to
0.02% accuracy,but CUDA kernel `marlin_w4a8_kernel.cu` reading the same
packed tensors produces complete NaN/garbage。

## Hypotheses

1. **`pack_w4a8(gptq_scales=...)` produces `s_group` / `s_channel` in a
   layout subtly different from naive max-scale path**(which kernel
   was tested against per `01ace86` audit)。The Python `manual_unpack`
   also was written against the naive path,so it inherits the same
   "wrong" assumption — both diverge from kernel in the SAME way →
   they agree but kernel disagrees。

2. **Kernel expects `s_channel` derived as `s_channel = max(|w|)/127.0`
   from data**(naive path),but GPTQ-aware path computes
   `s_channel = max_g(s_gptq)`(different math)。If kernel hardcodes a
   normalization factor,GPTQ-aware values may overflow or zero-out。

3. **Bit-pack convention mismatch**:GPTQ qweight U8 has weights at
   integer levels 0-15(unsigned),decoded to int and shifted by 8
   produces signed -8..7。Kernel may expect a different sign convention。
   Naive path quantizes BF16 directly → integer 0..15 with implicit -8
   shift in kernel。GPTQ-derived path may double-apply the shift。

## Verification path forward(codex own,kernel-side)

1. **Single-layer 1 prompt token kernel-vs-Python comparison**:run
   forward(BF16 hidden_state, W4A8-GPTQ packed weights)through both
   the CUDA kernel AND the Python `manual_unpack` followed by manual
   matmul。Compare resulting hidden states element-wise。If diverge,
   bug is in kernel(or in the kernel's reading of GPTQ-aware packed
   tensors)。

2. **Diff `pack_w4a8(naive)` vs `pack_w4a8(gptq_scales)` outputs at
   same input weight tensor**:if naive path produces output that
   the kernel reads correctly(token diff was high but tokens were
   non-zero,not garbage),and GPTQ-aware produces token id 0 garbage,
   the difference between the two pack modes' output tensors is the
   bug surface。

3. **Inspect `s_channel` magnitudes**:naive `s_channel = max(|w|)/127`
   produces values ~`max(|w|)/127`。GPTQ-aware computes
   `s_channel = max_g(s_gptq)` which is `max_g(max_per_group/7) =
   max(|w|_per_group)/7` ≈ `max(|w|_total)/7`(if max element falls
   in any group)≈ **18× larger than naive**。If kernel uses
   `s_channel × s_group × dequant_int` and naive path expects
   `s_channel ~ max/127`,passing `~max/7` overflows by 18×。

Hypothesis 3 is strongest:**`s_channel` magnitude convention mismatch
between naive and GPTQ-aware paths**。

## Action

Codex action(`12a54da` follow-up):
- Verify `pack_w4a8(gptq_scales=...)` produces `s_channel` magnitude
  compatible with `marlin_w4a8_kernel.cu` expectations
- If not,either:
  - Re-derive `s_channel = max(|w|_total)/127` from data even in
    GPTQ-aware mode(keeping kernel-compat magnitude)
  - Or modify kernel/loader to accept GPTQ-aware s_channel layout

Claude action(this tick):
- ✅ Fixed loader metadata detection(this entry)
- 🔧 Document divergence,hand back to codex
- ⏳ Defer further work until codex investigates kernel-side

## Cross-references

- Script-level LICENSE:`e753af7`(0.02% drift across 4 layers)
- Conversion script post-fix:`scripts/convert_gptq_w4a16_to_w4a8_marlin.py`(this commit)
- pack_w4a8 GPTQ-aware:`12a54da`(`scripts/quantize_qwen3_w4a8.py:94-130`)
- Kernel:`crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`
- Loader path detection:`infer/src/quant.rs:328-365` `load_quant_meta`
- Loader QuantLoadConfig:`infer/src/weight_loader.rs:493-498` `MarlinW4A8`
- Test:`infer/tests/greedy_consistency.rs:294` `test_w4a8_vs_bf16_token_diff`
- Codex audit kernel CLEAN(naive path):`01ace86`

## Status

- ✅ Phase 1b script-level LICENSED(`e753af7`)
- ✅ Loader metadata detection FIXED(this commit:script + checkpoint patch)
- ❌ End-to-end greedy_consistency FAIL — kernel produces token 0 garbage
- 🔧 Hypothesis:`s_channel` magnitude convention mismatch(`max/127` naive
  vs `max/7` GPTQ-aware)— kernel may overflow by 18×
- ⏳ Codex investigation:kernel-vs-Python single-layer forward comparison

## Rule

When script-level diag PASSES but end-to-end kernel FAILS,**Python
manual unpack is mirroring the same bug as the kernel from a different
direction** — both may diverge from kernel "correctness"(however
defined)。The fix is NOT to make Python diag stricter;it's to compare
kernel output against an independent reference(e.g., naive-path kernel
output on same weights,or BF16 reference matmul)。

Skill v1.3.0 anti-pattern:script-level PASS without end-to-end CUDA
kernel verification is NOT sufficient evidence of correctness。Always
gate on actual kernel inference through `greedy_consistency`(or
similar e2e test)before declaring "calibration preserved"。

## 2026-05-08 Update - Superseded Root Cause

Codex kernel-side isolation refuted the kernel-divergence hypothesis.
Direct single-layer `marlin.w4a8_mul` checks against the PR #31 extension
matched an independent PyTorch reference at ~0.4-0.55% relative error
across q/k/v/o/gate/up/down projections and M=1/5/16.

The actual source of the end-to-end garbage was upstream of W4A8:
`scripts/convert_gptq.py` decoded GPTQ `qzeros` without the AutoGPTQ
`+1` offset. The public checkpoint's layer-0 `q_proj.qzeros` unpacked to
stored value 7, which represents real zero-point 8. The old converter
used 7 directly, shifting every converted W4A16 weight by one scale unit.

After adding `+1` to `zeros_unpacked`:

```bash
INFER_TEST_MODEL_PATH=/home/ckl/projects/arle/infer/models/Qwen3-4B-GPTQ-Int4-converted-zpfix \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --features cuda --test greedy_consistency \
  test_greedy_solo_vs_concurrent -- --nocapture

INFER_TEST_W4A8_MODEL_PATH=/home/ckl/projects/arle/infer/models/Qwen3-4B-GPTQ-W4A8-zpfix \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --features cuda --test greedy_consistency \
  test_w4a8_vs_bf16_token_diff -- --nocapture
```

Results:
- Corrected W4A16 GPTQ source generated coherent English and passed
  solo-vs-concurrent token equality.
- Corrected W4A8 generated exactly the BF16 32-token continuation for
  "The capital of France is" (`matched first 32/32 tokens, diff 0.0%`).

Conclusion: this entry's "kernel != script" diagnosis was a useful
debugging route but is superseded by the GPTQ `qzeros` off-by-one root
cause. See
`docs/experience/errors/2026-05-08-gptq-qzeros-off-by-one-broke-w4a8-source.md`.
