# CUDA MoE/FP8 serving fixes + scheduler/perf + DSv4 EP — 8×H20 pod-verified

## Context

Follow-on to the kernel-pipeline registry redesign
([2026-06-23 entry](2026-06-23-cuda-registry-ffi-codegen-pending-remote.md)):
end-to-end runtime verification of "all 3.6 / v4 / Qwen3.5-MoE at TP>1" on the
8×H20 box (sglang-test container, GPUs 4-7, allreduce-MoE + DeepGEMM-native).
Surfaced + fixed several loader/serving gaps and one latent EP bug. Build/serve
on the persistent `/host/arle-c2` clone via `scripts/pod.sh` discipline
(detached, BUILD_EXIT-gated, killed by exact PGID). Models fetched via `oniond`
(`BUCKET=ai-infra`); the pod's external net is throttled so cargo fetches route
through ckl's reverse SOCKS5 (`all_proxy=socks5h://127.0.0.1:1080`, the
canonical `pod-build-env.sh`).

## What Worked (pod-verified, needle = RAW + per-model template, ×3 DET)

- **C1 — Qwen3.5/3.6 MoE FP8 loads at TP** (commit `4d6caf65`). `load_linear_qkv_sharded`
  was BF16-only, aborting `model_type=qwen3_5` MoE FP8 checkpoints at TP>1 on
  `linear_attn.in_proj_qkv` (F8_E4M3). Made it quant-aware (reuse
  `shard_head_blocks_column_parallel` for the FP8 weight + its BF16 block-scale,
  head_dim 128 == block_m 128 → 1:1). **Qwen3.6-27B-FP8 (a `qwen3_5` shared-expert
  MoE) at TP=4: needle exact ×3 DET all 9 lengths 115→8000.** c=1 prefill 485 ms,
  decode 48.2 tok/s. This is the Qwen3.5/3.6-MoE-on-CUDA proof.
- **C2 — DSv4 FP8 loader accepts F32 power-of-2 block-scales** (commit `a8ca816d`).
  The DSv4-native loader hard-required `F8_E8M0` `.scale`; the common
  vLLM/DeepSeek-V3-style FP8 stores block-scales as F32. Decoded evidence: those
  F32 values are exact powers of two, so `byte = (f32_bits>>23)&0xff` is the
  lossless E8M0 encode → feed the unchanged E8M0 path. Pod: the F32-scale
  `/host/DeepSeek-V4-Flash-FP8` now **loads past all FP8 projections** (wq_a/wq_b/
  wkv/wo_b) that previously aborted; oniond UE8M0 byte-identical passthrough (no
  regression, needle still exact). Remaining: that ckpt also stores `wo_a` as
  dense BF16 (a separate dtype-routing follow-up).
- **C3 — vanilla Qwen3-MoE → actionable error** (commit `70380db8`). `qwen3_moe`
  (not `qwen3_5`) is a Metal target; the qwen35 CUDA loader is hardwired for
  gated-attn + shared-expert + `language_model` prefix. Replaced the opaque serde
  `RawConfig` error with `Qwen3MoeUnsupported` → `"use --backend metal"`. Verified.
- **C4 — DSv4 over-length graceful + default raise** (commit `70380db8`). Ingress
  cap was `.max()`'d 8× above the slot capacity → a >4096-tok prompt passed
  ingress then crashed all TP workers at the forward assert. Bound the cap DOWN
  via `.min()` (respecting a stricter user flag); raised `DSV4_DEFAULT_MAX_SEQ_LEN`
  4096→32768. Pod: **DSv4 needle at len=8000 now passes** (previously killed the
  engine).
- **DeepSeek-V4-Flash (oniond UE8M0) TP=4**: needle exact (MoE non-det at 1-2
  lens, expected); c=1 decode 38.9 tok/s; allreduce-MoE `c跑` 24.7→31.2→42.3→52.6
  tok/s (c=1→16, scales ~2.1×).
- **Cleanup (−~20k lines tracked)**: dropped committed device-kernel `.cu`
  (`c936a826`), 62 redundant binary cubins (`255ead11`), dead flashinfer vendor +
  toml cruft (`c34b2486`), 9 dead FFI externs + their `.cu` (`c7eb88cd`, pod nvcc
  compile confirmed the deletions). Kept the consumed per-SM `.c` (the
  prebuilt-release tranche will drop those).

## Scheduler / perf finding (measured A/B, not inferred)

All three batching techniques are **supported + default-ON** (the rewrite has no
policy gate, unlike the deleted monolith): continuous/in-flight batching
(`admit_waiting` every tick, cap = `num_slots` default 4), chunked prefill
(2048 floor, inert at short prompts), mixed prefill+decode (planner emits
`ForwardMode::Mixed` unconditionally). The KILL docs (`mixed-default-kill`,
`chunked-prefill-size-kill`) are **monolith-era perf-default verdicts**, not
"off/broken".

**Qwen3.6-27B-FP8 MoE TP=4 saturates ~68 tok/s by c=4** (c=1=41.5, c=8/16/32 flat
≈67). Matched A/B (`--num-slots 4` vs `16`, same `--total-pages`): num_slots=4
already reaches the ceiling at c=8 (continuous batching pipelines the queue);
16 slots add only +10% (→75). **Verdict: operator-bound, not slot-bound** — the
ceiling is the MoE expert grouped-GEMM at small batch + TP=4 per-layer allreduce,
not the slot cap. The lever is an MoE-decode batching kernel (or EP), not slots.

## DSv4 EP (DeepEP) — 2 latent bugs fixed; c=1 correct, concurrency NON-FUNCTIONAL (commit `5a2b8273`)

"Use GPUs 4-7" surfaced a latent device-binding bug:
- **Toolchain**: `dsv4_toolchain.sh build_infer` hardcoded `--features cuda,nccl`,
  never adding `deepep` for `--moe-backend native-deepep` → couldn't produce an EP
  binary. Made the feature list MOE_BACKEND-aware.
- **Device-by-rank**: `deepep-sys/csrc/deepep_buffer.cpp` did
  `cudaSetDevice(self->rank)` (TP rank 0..3), but workers run on the real ordinal
  from `INFER_CUDA_DEVICES` (CUDA_VISIBLE_DEVICES is never set). On a non-0-based
  set (4,5,6,7) that selected the WRONG GPUs → `Buffer::sync` barrier failed with
  `cudaErrorInvalidResourceHandle`. Worked historically only because production
  EP=8 uses GPUs 0-7 (rank==ordinal). Fix: thread the real `ctx.ordinal` →
  `cudaSetDevice(device_id)` + re-pin in sync/dispatch/combine. 0-based:
  byte-identical.
- `native-deepep` is the intranode **replicated-token path (killed 2026-06-01)**;
  the validated EP path is `deepep_ll` (per-rank slice, NVSHMEM). Built NVSHMEM in
  (`ARLE_DEEPEP_NVSHMEM_DIR` = the pip `nvidia-nvshmem` pkg, `nvshmem=true`);
  DeepEP @ `d4f41e4` cloned via the proxy (the sglang `/sgl-workspace/DeepEP` is a
  newer non-`legacy/` layout deepep-sys rejects).
- **`deepep_ll` (NVSHMEM EP) on 4,5,6,7 — c=1 single-stream ONLY**: needle exact
  ×3 DET at the lengths run (115→1000, `num_max_dispatch_tokens_per_rank=256`), so
  the EP forward path is numerically correct at batch 1 (transcript only — not
  re-captured in a commit/log artifact; treat as evidence-grade c=1, not archived).
  **Concurrency is NON-FUNCTIONAL, not "tuned later": every concurrent `c跑` (c=1/4/8/16)
  returned 0 successful requests (0 tok/s, ok=0/N), and `=512` made even the needle
  NONDET exact=0.** Root tension: `num_max_dispatch_tokens_per_rank` must be ≥ the
  concurrent batch (≥512/rank) but that OOMs alongside DSv4's ~40 GB/rank weights on
  96 GB cards; 256 overflows under any concurrency. **EP's entire point is throughput,
  so EP throughput is UNVERIFIED — a hard blocker (no batch size fits on 96 GB),
  not a perf follow-up.** Next: shrink the KV arena / lower `INFER_DSV4_MAX_SEQ_LEN`
  to fit ≥512 LL buffer, then re-run the c-sweep and capture the log.

## Rule

- **A negative/loader gap is "which model + which serialization", not a backend
  verdict.** Decode the actual config AND the safetensors tensor dtypes per
  component before generalizing: two checkpoints with byte-identical configs
  differed only in scale serialization (F32 vs E8M0); the F32 values were
  powers-of-two → a 1-line lossless encode, not the risky dequant route the first
  design assumed. The indexer/`hc_*` F32 tensors were never the problem (correct
  in both) — only the FP8 MLA projections' scale dtype.
- **A latent multi-GPU bug hides whenever the test set is 0-based-contiguous.**
  `cudaSetDevice(rank)` worked on GPUs 0-7 (rank==ordinal) for the entire prior
  life of the DeepEP path; only `INFER_CUDA_DEVICES=4,5,6,7` exposed it. Verify
  device-affinity code on a non-0-based GPU set.
- **Saturation is slot-bound vs operator-bound only by a measured num_slots A/B.**
  The c-knee numerically matching `num_slots` is circumstantial; the A/B (4 vs 16)
  showing +10% (not ~4×) is what proves the operator ceiling.
- **Kill ≠ useless; prefer the validated path.** `native-deepep` intranode was
  KILLed as a replicated-token trace path; `deepep_ll` is the validated per-rank
  one — don't re-animate a killed path, switch transports.
