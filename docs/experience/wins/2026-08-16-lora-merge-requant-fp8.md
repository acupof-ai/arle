# FP8 merge-requant: 27B all-linear LoRA merge fits one GPU

**Context.** The BF16 merge lane keeps the retired FP8 base (23.2 GB) plus the
merged dense BF16 (46.4 GB) resident — about 3× base bytes — so a 27B
`all-linear` LoRA sync OOMed on one 96 GB H20 (previous failure: layer 59
`mlp.gate_proj` BF16 promotion alloc).

**What worked.** `--lora-merge-fp8` (default off) quantizes each merged matrix
back into the FP8 serving slots and drops the dense copy:
- New kernel `quantize_bf16_to_fp8_block_scaled_cuda`, dual of the existing
  dequant: one CUDA block per weight block, shared-memory amax reduction,
  scale = amax/448.
- `DeviceMatrix.pristine_fp8` holds the base pair after the first requant so the
  re-merge stays idempotent; device addresses do not move, so share-frozen-base
  aliases stay valid. `merge_base_fp8()` is the single accessor used by promote,
  merge, restore, and the frozen-base export.
- Requant runs **per layer**, not per update. Requanting after the whole update
  left every promoted matrix resident and the peak was unchanged — the 27B
  all-linear run still OOMed at layer 59. Per-layer keeps the dense peak one
  layer wide.

**Result.** Single GPU, 27B FP8 student, `all-linear`, `--self-consistency`
(judge-free), 2 rounds: round 0 trained 6 accepted rollouts (loss 0.0542),
round 1 trained 6 (loss 0.0653), merge + requant over all 48 layers each round,
exit 0. Store residency 23 200 MiB FP8 + 4 895 MiB BF16 per round.

**Limit.** Residency after requant is pristine FP8 + merged FP8 ≈ 2× base
(≈46 GB). An in-process 27B judge (29.5 GB) still cannot reload onto the same
GPU after training; a judge reload OOM leaves the engine half-loaded and the
next forward reports `fp8_block_scaled missing qweight_u8`. Use a second GPU for
the judge, or `--self-consistency`.

**Rule.** A peak-residency fix must run at the granularity of the peak. Batching
the release step after the whole traversal leaves every intermediate alive and
measures as no change at all.
