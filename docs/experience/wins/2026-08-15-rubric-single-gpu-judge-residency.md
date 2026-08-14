# Single-GPU rubric-opd: judge residency + FP8 merge verified

**Context.** Verifying the frozen-base FP8 non-zero-delta merge lane
(`--share-frozen-base`) end-to-end on one H20 GPU: 27B FP8 student and a 27B
FP8 in-process judge (judge is model-agnostic text I2-wire; Flash is not
required). Three failures blocked the first attempts, all on d7d2366fe..89b891905.

**What worked.**
1. `all-linear` LoRA targets on 27B cannot merge on one GPU by design: retired
   FP8 (23.2 GB) + merged BF16 (46.4 GB) ≈ 3× base bytes stay resident.
   `attention-qv` fits and exercises the same merge path.
2. Phase-D reloaded the judge (29.5 GB) before the per-round LoRA sync, evicting
   the merge headroom → judge reload is now lazy (`FlashJudge::ensure_resident`),
   owned by the judge object because the CLI calls `run_rubric_rounds` once per
   round and any loop-local flag resets across calls (e14a4caf5). `judge_batch`
   bypassed the first fix and had to be routed through the same gate (89b891905).
3. The engine-ready channel sent `err.to_string()` (outermost context only), so
   OOM roots printed as "row fuse + <tensor>". `format!("{err:#}")` across the
   channel restored the chain (d872cc37c); first reproduction after the fix
   printed the full `H2D copy failed: CUDA_ERROR_OUT_OF_MEMORY` root.

**Result.** 2 rounds on GPU 4: round 0 accepted=2 trained=2 loss 0.0780, LoRA
sync into the FP8 rollout engine clean; round 1 lazy judge reload fired,
accepted=2 trained=2 loss 0.0709; exit 0.

**Rule.** Residency state belongs to the object that owns the engine, not to the
loop that happens to drive it — every public entry point that submits work must
pass through the same residency gate. Errors that cross a `String` boundary must
be flattened with `{err:#}`, never `to_string()`.
