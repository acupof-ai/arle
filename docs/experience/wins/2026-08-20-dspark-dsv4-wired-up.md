# DSpark speculative decoding wired up for DSv4-Flash — CUDA, 2026-08-20

> Status: Shipped (correctness verified, speedup marginal at default SPS params)

## Context

DSpark is DeepSeek V4's block-draft speculative decoding: a 3-stage draft model
(mtp.0 entry, mtp.1 middle, mtp.2 exit with Markov + confidence heads) drafts
`dspark_block_size=5` tokens per step, tapping base model layers 40-42. The
scaffolding (`load_dspark_draft`, `Dsv4DsparkDraft`, `Dsv4DsparkExec`,
`dspark_decode_tokens`) existed but was not wired up: the native MTP loader
ran for DSpark serves and crashed on `mtp.0.enorm.weight` (the base model's
mtp.0/1/2 are DSpark stages, not native MTP layers).

## What Worked

1. **Skip native MTP when DSpark is active** (`cd0a14ca3`): `!dspark_on &&`
   added to the MTP load condition in `dsv4/load.rs:448`. The DSpark executor
   dispatches on `self.dspark.is_some()` and never calls `mtp_forward_level`,
   so `model.mtp = None` is correct for DSpark.

2. **NVFP4 vs FP8 E4M3 detection** (`4b0a87742`): `is_nvfp4` checked only the
   scale dtype (F8_E8M0), but standard DSv4 FP8 (E4M3 weight + E8M0 128×128
   block scales) also uses E8M0 scales. DSpark draft experts are FP8 E4M3, not
   NVFP4 — the misdetection doubled the hidden dim (4096→8192) and crashed the
   first forward. Fix: require weight dtype I8/U8 (packed E2M1) for NVFP4.

3. **W4AFP8 finalize kernel + SwiGLU cap re-applied** (`ee8640904`): the
   original fix commit `7cde1ce84` was replaced by `6f1b416e4` which kept only
   3 of 5 fixes. The missing two: (a) `w4a8_amax_finalize_kernel` converts raw
   amax → amax/448 between the amax and quantize kernels — without it the
   CUTLASS epilogue reads raw amax as the dequant factor (448× amplification);
   (b) SwiGLU 256-block cap zeroed output past 32 routed rows at i_dim=2048.
   Base model produced garbage until both were re-applied.

4. **Loader shape validation** (`a1b6d3e6a`): rank check (`weight.shape.len() != 2`)
   + cross-expert w1/w3/w2 shape consistency in the NVFP4→W4AFP8 conversion.
   Prevents panics on non-2D weights and silent corruption from mixed-shape
   experts in the fused w13 buffer.

5. **Log message** (`7bd63d589` + `2a090ec13`): the spec-decode log now
   distinguishes DSpark ("loading base layers plus DSpark draft (native MTP
   skipped)") from native MTP.

## Result

DSpark serve on DSv4-Flash-0731 (NVFP4 base, TP=2, 128K context):

| Test | DSpark | Base (no spec) | Delta |
|------|--------|----------------|-------|
| Math correctness (4 questions) | 4/4 PASS | 4/4 PASS | — |
| 200-token decode | 34.7 tok/s | 33.7 tok/s | +3% |
| 500-token decode | 34.7 tok/s | 35.3 tok/s | -1.7% |

DSpark is correctly wired up and produces correct output, but the speedup is
marginal at the default SPS parameters (`bias_ms=211`, `row_ms=0.53` — tuned
for Qwen3.5, not DSv4). The DSpark draft adds 12.7GB VRAM and 3 MoE forward
passes per step; at DSv4's ~30ms/token base decode, the draft overhead offsets
the spec decode gain. SPS parameter tuning for DSv4 is a separate optimization
task.

## Environment

- Host / GPU: H20 pod, GPUs 2,3 (96GB each)
- Driver / CUDA: sm_90, CUDA 12.x
- Model: DeepSeek-V4-Flash-0731 (NVFP4, 43 layers, 256 experts)
- Draft: DeepSeek-V4-Flash-DSpark-draft-fp8 (3 stages, block=5, target=[40,41,42])
- TP: 2, slots: 6, max_seq_len: 131072
- Binary: `w4afp8-fix` (commit `ee8640904`)

## Rule

When a fix commit is replaced or rebased, verify every fix in the original
commit is present in the replacement — `7cde1ce84` had 5 fixes, `6f1b416e4`
kept only 3, and the missing 2 (finalize kernel, SwiGLU cap) shipped broken
W4AFP8 for a day. `git diff <original>..<replacement> -- <file>` catches this.
