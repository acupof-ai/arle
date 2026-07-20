# hd256/FP8 temp>0 sampling corruption — root-cause + complete fix

> Status: Active — `9851ced6b` CONFIRMED a regression and REVERTED
> (`485eefe0d`, 2026-07-20). The norm handling was a DETOUR — the CUDA loader
> was already correct. The ORIGINAL bug (temp>0 salad, greedy coherent, on
> hd256/**FP8**) is still OPEN; temp=0.3 workaround stays. Next: bf16 E1
> isolation (FP8-noise vs hd256-compute) for the real bug.

## The norm detour — RESOLVED (9851ced6b reverted)

`9851ced6b` loaded the per-layer `input_layernorm`/`post_attention_layernorm` as
`w−1`, which REGRESSED greedy to salad on both 27B FP8 models (decoded, two clean
builds). The CUDA code was already correct. Reverted.

**Why the CUDA norm handling is correct — the Metal reference settles it.**
`crates/infer-metal/src/qwen35.rs`: `qwen35_norm_needs_offset_correction` detects
offset weights by `mean(|w|) < 0.75`, then `qwen35_normalize_direct_norm_weight`
converts them with `w + 1`, then applies standard `x·inv_rms·(w+1)`. That is
IDENTICAL to the CUDA `rms_norm_batched_offset_kernel`'s hand-written
`x·inv_rms·(1 + weight)`. So the `(1+w)` kernel is not a bug — it is the offset
convention these HF checkpoints ship, matched.

Per-tensor convention (measured, local Qwen3.5-0.8B; matches Metal's threshold):

| tensor | mean\|w\| | convention | correct CUDA load |
|---|---|---|---|
| input/post_layernorm | 0.24 / 0.085 (<0.75) | offset | **raw** → kernel `(1+w)` |
| final norm | 3.3 (>0.75) | direct | `w−1` → kernel `(1+w)` = 3.3 |

`9851ced6b` wrongly applied the final-norm `w−1` to the offset input/post norms
→ `(1+(w−1)) = w ≈ 0.24` → 1/5 scale per layer, compounding to salad. The
"latent final-norm bug" hypothesis is void: final `mean|w| 3.3 > 0.75` is direct,
`w−1` is correct.

**Robustness follow-up (optional):** the CUDA loader hard-codes the per-tensor
convention (input/post raw, final `w−1`) instead of detecting it like Metal
(`mean|w| < 0.75`). Data-driven detection would have prevented `9851ced6b`
entirely. Small, mirror Metal — do it if another checkpoint breaks the hard-code.

## The REAL bug (still open): temp>0 salad on hd256/FP8

Independent of the norm detour. Pre-`9851ced6b` (`13426a8de`): greedy + temp=0.3
coherent, temp=1.0 salad, on both FP8 models. Leading hypothesis: FP8 logit-tail
noise (both affected models are FP8; greedy argmax survives, temp>0 samples the
mis-scaled tail). Decisive isolation = bf16 E1 below (download is staged,
62.8 GB).

## Crux confirm + E1 (in flight / next)

- **Crux (running):** reverted-HEAD vs `00224faa0`, 27B FP8, greedy → expect
  coherent vs salad. Confirms the revert.
- **E1 (next, for the REAL bug):** hd256 **bf16** @ temp=1.0. Coherent → FP8
  noise → fix in FP8 quant/dequant or raise hot path to bf16. Salad → hd256
  compute residual (a path b4b293f0c/norms didn't touch).

## Verdict up front

Every agent-OPD rollout on the hd256/FP8 student (`Qwen3.6-27B-FP8`) at the
default `--rollout-temperature 1.0` samples from a **corrupted distribution** —
silent quality degradation across the whole lane. Decoded cases (same binary
`13426a8de`, top_k=20 top_p=0.95):

- current student @ temp=1.0 → corrupt (Swedish token-bleed, broken docstrings)
- current student @ temp=0.3 → clean 3/3
- ThinkingCap-27B-FP8 (hd256) @ temp=1.0 → full multilingual salad; @0.3 clean

Temperature-graded. `b4b293f0c` (hd256 q/k RMSNorm OFFSET→STANDARD) fixed the
**argmax** (greedy coherent) but left a **residual distribution-shape error**:
low temp sharpens it away, high temp samples the mis-scaled/noisy tail. The
sampler plumbing is NOT the cause — validated on hd128 (Qwen3-4B coherent at
temp=1.0+nucleus, same binary). Nucleus over a corrupted distribution can't
rescue it.

## Open confounder (why Phase 0 exists)

The only control so far — hd128 **bf16** vs hd256 **FP8** — varies head_dim AND
quantization at once. Two live hypotheses, not yet separated:

- **(A) FP8 noise** — 64-layer FP8 accumulation jitters the logit tail; greedy
  survives, temp>0 samples wrong ranks. Leading hypothesis: code read confirms
  the hd256 decode q/k norm is already STANDARD (`decode_prep_paged_hd256`,
  qwen35.rs:6191/7905), turboquant's `k_norm` is the FP8-KV dequant scale (not
  the RMSNorm), and `lm_head`/`embed_tokens` are in `modules_to_not_convert`
  (logit projection is bf16) — so the obvious RMSNorm/logit paths are clean.
- **(B) hd256 compute residual** — a remaining wrong-convention / scale error in
  a hd256 path `b4b293f0c` didn't touch (prefill / RoPE / attention softmax
  scale), independent of FP8.

Inference points at (A), but inference is hypothesis — Phase 0 measures.

## Phase 0/1/2 — DONE (root cause found + patched, see header)

Superseded by `9851ced6b`. The bf16 isolation below is no longer needed; kept for
the record.

## Phase 0 — decisive attribution (gates the fix branch; ~10 min, no rebuild)

- **E1 — hd256 + bf16 @ temp=1.0.** Resume the bf16 ThinkingCap download (killed
  at 94%), serve on the same binary, temp=1.0 top_k=20 top_p=0.95.
  - bf16 coherent → **(A) FP8 noise** → Phase 2-A.
  - bf16 salad → **(B) hd256 compute residual** → Phase 2-B.
- **E2 — layerwise divergence probe.** Same prompt, bf16 vs FP8, per-layer
  hidden/logit rank correlation → the layer/op where the distribution first
  diverges. Confirms E1 and localizes the fix.

## Phase 1 — root-cause to file:line (branch on Phase 0)

- **A (FP8 noise):** E2 localizes the FP8 op. Check for a missing dequant /
  wrong scale / an op that should accumulate in bf16. If it is genuine whole-
  chain accumulation with no single point, the fix is raising the hot path
  (attention or the last N layers) to bf16 accumulation.
- **B (RMSNorm residual):** decode already excluded — audit prefill
  (`prefill_attention_hd256.cu`, `prefill_attention_paged_prep.cu`), RoPE, and
  attention softmax scale for hd256 vs the hd128 convention.

## Phase 2 — implement (branch-specific, kernel level)

Concrete kernel/GEMM dtype-or-convention change; branch A/B per Phase 0.
Line-level spec written only after Phase 1 (no filing a hypothesis as a fix).

## Phase 3 — acceptance

temp=1.0 coherent on hd256 **FP8 and bf16** + needle gate ×3 + long-generation
nonascii=0 end-to-end + no perf regression → restore `--rollout-temperature 1.0`.

## Phase 4 — interim + progress spine

- **Now:** `--rollout-temperature` default 1.0 → 0.3 (stopgap; >0 keeps F.6
  logprobs), gated on the long thinking-on coherence spot-check. Lane stops
  degrading immediately. Optionally tighten `--rollout-top-p 1.0→0.95` /
  `--rollout-top-k 0→20` (nucleus hygiene).
- errors/ entry: the whole silent-degradation finding.
- Revert the workaround + restore temp=1.0 once Phase 2 lands.

## Coordination

The tree carries **uncommitted hd256 WIP** (`cuda-kernels/build.rs`,
`kernels.toml`, `qwen35-spec/src/lib.rs`). Reconcile with those before any hd256
kernel edit — the earlier concurrent-commit collision (my sampling commit folded
into another agent's) must not repeat. Confirm the WIP's intent first.

## Links

- Sampling-defaults fix (validated, landed): `b9207defa` content, schema.rs
  `SamplingDefaults`.
- hd256 RMSNorm greedy fix: `b4b293f0c`,
  [wins/2026-07-17-hd256-rmsnorm-convention-fix.md](../experience/wins/2026-07-17-hd256-rmsnorm-convention-fix.md).
- Relay/root task: #48.
