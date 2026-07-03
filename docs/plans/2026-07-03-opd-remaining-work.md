# OPD remaining work — master execution plan

**State at handoff (2026-07-03):** toy agent-OPD round ≈ **10s**
(rollout ~3s + writeback 6.8s = forward 2.2 + backward 4.2 + tails 0.4),
44× vs the 438s origin. Correctness chain closed
([campaign](../experience/wins/2026-07-02-correctness-gate-campaign.md)).
Every item below has a measured motivation, implementation-level steps, a
license gate, and a kill condition. Order = ROI ÷ effort.

Shared verification harness (all items): pod launcher family
`/host/arle-build/run-*-toy1r.sh` (sed label/GPU/rounds; pick a free GPU via
nvidia-smi first — GPU 1 is tenant-shared), phase lines
`[masked-writeback] phase=*` + `[agent-opd] phase=*`, backward op table via
`ARLE_OPD_BACKWARD_PROFILE=1`, loss band 0.24–0.33 + `passed=1`, and for any
engine-path change: greedy turn-output md5 vs the previous build + (for
routing/kernel swaps) `GATE_PROFILE=generic BIN=$TREE/target/release/arle
MODEL=/host/Qwen3.6-27B-FP8 TEMPLATE=qwen3_nonthink NEEDLE_MAX_TOKENS=256
scripts/lever_gate.sh <label>`.

---

## P1.1 — Decode-graph license A/B (effort S, expected rollout −0.5–1s)

Rollout decode measured 27–36 tok/s with **~1074 kernel launches per token**
(host-bound; docs/plans/rollout-optimization.md lever ③). The whole-step
decode CUDA graph exists behind an env flip and is dropped/recaptured on every
LoRA remerge (executor.rs `decode_graph = None` in `remerge_student_lora`) —
correct by construction, never perf-licensed on the agent-OPD lane.

1. Find the flip: grep `decode_graph` env gating in
   `crates/infer-cuda/src/executor.rs` (capture entry + the env const near it).
2. Same-binary env-flip A/B, 3-round toy: decode segment wall (live-trace the
   turn window) + tok/s ± graph. Recapture cost per round must be amortized:
   count captures in the log (expect 1/round after the round's first decode).
3. Gates: loss band, passed=1, greedy md5 identical, WARNs 0.
4. License → flip the default for the agent-OPD lane (train_cli), keep serve
   default unchanged unless separately licensed. Kill: <15% decode gain or
   per-round recapture eats it.

## P1.2 — LA backward, full chunkwise GEMM form (effort L, backward 4.2→~1.5s)

The transfer-operator rewrite hit its pre-declared kill threshold (29.7%);
decoded floor = per-token `__syncthreads` chain
([outcome](linear-attention-chunked-backward.md)). The remaining 3.19s (76% of
backward) falls only to the fla-style formulation with NO intra-chunk token
loop.

1. Derive the chunkwise backward for THIS kernel's exact forward semantics
   (k-normalization → exp-gate decay → beta delta rule; forward recompute spec
   = `linear_attention.cu:560-690`). Reference: flash-linear-attention
   `fla/ops/gated_delta_rule` chunked backward (dq/dk/dv/dbeta/dg as per-chunk
   GEMMs against saved chunk states + one reverse chunk-carry recurrence).
   The k-norm and conv/preact chain stay in the existing epilogue kernels.
2. Emit as TileLang stages beside the seven forward stages in
   `crates/cuda-kernels/tools/tilelang/gated_delta_rule.py` (preferred: same
   codegen + release-bundle pipeline, kernels-publish auto-ships them) OR
   native CUDA if TileLang fights the reverse scan. Register in `kernels.toml`.
3. Wire behind `cuda_linear_attention_backward_device`
   (`crates/autograd/src/backend_cuda.rs:4186`) as tier 1; keep BOTH existing
   kernels (chunk-parallel default, mono via `ARLE_LA_BACKWARD_MONO=1`) as
   fallbacks during licensing.
4. Gates: extend `cuda_linear_attention_qwen{35,36_27b}_chunked_grad_matches_cpu`
   (crates/autograd/tests/test_linear_attention.rs — multi-chunk shapes,
   max_abs ≤ existing tolerance); same-binary A/B on the op row. License ≥2×
   op-level (71→≤35ms); kill <2× — then this lane is closed for good and the
   remaining backward work moves to P1.3's traffic reduction.
   Note the parity-test naming trap: `--list` the test binary before filtering.

## P1.3 — bf16-native pipeline, T0 attribution gate (effort S; T1+ only if licensed)

Full tranche plan: [bf16-native-autograd.md](bf16-native-autograd.md). The
plan's own arithmetic says conversions ≈ 4–10% of forward — the 10–30% upside
is a launch/alloc-overhead HYPOTHESIS. Run T0 ONLY (conversion counters +
nsys one round + long-seq probe), then the doc's three kill conditions decide
whether T1–T4 run at all. Do not start T1 before T0 numbers exist.
Arithmetic-solid regardless of kill: tape 24→13GB moves the
`should_checkpoint` cliff (qwen35.rs) ~2× in seq — re-evaluate after P2.1 if
production trajectories hit the cliff.

## P2.1 — Frozen-prompt-KV lane: device carry (effort M-L, unlocks 15k+ writebacks)

The lane's linear-attention carry path is host-only by documented deferral;
at measured production shape (gen_len≈13.6k) it moves 48 layers × ~2.5 passes
onto a single-thread scalar recurrence — the lane cannot meet its purpose
(audit finding #1; [audit](../experience/wins/2026-07-03-engine-dispatch-audit-fixes.md)).

1. Extend `LinearAttentionDeviceForwardArgs/Result`
   (`crates/autograd/src/backend.rs:337-363`) with
   `initial_state`/`initial_conv_window` in and `final_state`/`conv_tail` out.
2. `cuda_linear_attention_forward_device` (`backend_cuda.rs:~3720`): replace
   the hardcoded `alloc_zeros` initial_state (:3836-3839) with the optional
   carry upload; the recurrent chunk kernel already threads state — surface
   `final_state` (already computed, :3785) + the last `conv_kernel-1` rows of
   qkv as the conv tail.
3. Route `linear_attention_core_with_carry{,_taped}`
   (`crates/autograd/src/ops/linear_attention.rs:352/:548`) through a
   `try_*_device` first, mirroring `linear_attention_core:226`; delete the
   "Always host" branches once licensed (deletion-refactor).
4. Backward: remove the `has_carry` device carve-out (:1207) — the device
   backward needs the carry only as the chunk-0 incoming state, which is the
   same `chunk_state` slot mechanism the chunked kernels already consume.
5. cat_seq/cat_heads (audit #2): route rank-4 seq-concat through the existing
   `backend().concat_axis2` device path (`ops/attention.rs:690` idiom); host
   loops (:933, :759) become the CPU fallback arm.
6. Gates: the lane's existing Gate-A-exact test
   (`test_frozen_prompt_kv_writeback.rs`) + a NEW at-shape wall-clock A/B
   (lane on vs baseline writeback at seq≈14.5k — the lane must WIN, that is
   its purpose). Kill: if after device carry the lane still loses to baseline
   at 15k, close the lane and pursue long-seq via P1.3's tape shrink instead.

## P2.2 — Production multi-task batching validation (effort S, config only)

Rollout is continuous-batching-capable; the toy runs 1 task serially. No code.
1. Build a 4–8 task `tasks.jsonl` (SWE-shape, staged sandboxes), run
   `--task-limit 8 --samples-per-prompt 2` one round on a free GPU.
2. Measure: aggregate rollout wall vs 1-task × N (target ≥3× at 8 tasks),
   VRAM peak (KV pool sizing), sandbox CPU contention (pytest walls).
3. Gate: per-task pass parity with serial. Kill: engine serializes rollouts →
   file the scheduler finding, do NOT hand-roll app-level batching.

## P3 — Spec decode / MTP for rollout (effort L, only after P1.1)

Only if decode is still the rollout binder after the graph license: DSv4 has
EAGLE infra in-repo; Qwen3.6 hybrid needs the frozen-KV MTP notes
(memory: frozen-kv MTP for sparse attention). Requires its own plan doc +
the multi-prompt spec gate (≥2 prompts, memory rule). Do not start from this
document.

---

## Sequencing & dependencies

```
P1.1 (S)  ──┐                    P1.2 and P1.3-T0 are independent of P1.1;
P1.2 (L)  ──┼─→ re-measure round ─→ P2.1 (M-L) ─→ P2.2 (S) ─→ P3 (L, conditional)
P1.3-T0(S)──┘        (~7s?)
```

Standing rules for every item: license-or-kill with the pre-declared
threshold (the LA transfer-operator kill proved the discipline pays);
same-binary env-flip A/Bs; decode the failing case before overturning; wins/
entry per landed change; pod traps (tenant GPUs, exec-channel `&` hang,
pkill self-match) are documented in the launcher scripts.
