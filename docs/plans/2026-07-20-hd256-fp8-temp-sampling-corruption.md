# hd256/FP8 temp>0 sampling corruption — root-cause + complete fix

> Status: Active — root cause FOUND & FIXED in source (`9851ced6b` +
> `bf66a3854`, 2026-07-20). Confounder resolved: **Branch B (hd256 compute
> residual, FP8-independent)**. Remaining = Phase 3 empirical verify on a
> rebuilt binary, then revert the temp=0.3 workaround. Phase 0 bf16 isolation is
> now unnecessary (the mechanism is identified and patched).

## Root cause (RESOLVED)

`b4b293f0c` fixed the hd256 q/k RMSNorm convention but left the **per-layer
`input_layernorm` / `post_attention_layernorm`** loaded raw. All Qwen3.5/3.6
norms ship in STANDARD format (~1-centered); the `rms_norm_offset` trunk kernel
applies `(1 + weight)`, so raw-loaded norms carried a ~2× multiplier per layer,
compounding across 64 layers. Fix (`9851ced6b`): load them via
`load_final_norm_offset` → `(w − 1)`, so `(1 + (w−1)) = w`. q/k_norm stay raw
(hd256 prep kernels apply `weight` directly — the STANDARD convention
`b4b293f0c` set). This is FP8-independent — the bf16 path had the same 2× bug.

**Open tension to close in Phase 3:** a 2×-per-layer compounding error should
have broken greedy too, yet greedy read as coherent pre-fix. Either greedy was
never rigorously gated on the 27B at this binary, or the mechanism magnitude is
overstated. Do not trust the commit message — MEASURE temp=1.0 on the rebuilt
binary before declaring done.

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
