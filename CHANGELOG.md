# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This changelog should record more than feature additions. It should also record:

- breaking changes
- deprecated surfaces
- support-matrix changes
- migration notes when user action is required

Related governance docs:

- [docs/stability-policy.md](docs/stability-policy.md)
- [docs/support-matrix.md](docs/support-matrix.md)

## [Unreleased]

Progress spine. Entry classes recorded here the day they land: phase exits,
default flips, license-or-kill verdicts (AGENTS.md §Docs lifecycle & progress
spine).

- **2026-07-14 — DSv4 DSpark TP=4 concurrency licensed.** Loading the already
  EP4/TP4-sharded draft before KV planning replaced a false 19.9 GB/rank reserve
  with the measured 4,960 MB resident footprint, raising slots **1→33**. The
  TP-unsafe sequential B>1 draft path was deleted; batches use the target decoder
  until a TP-safe batched verify lane exists. GuideLLM c=1/4/8/16 throughput is
  **45.04/80.06/120.66/141.46 tok/s**: -1.0%/+71.6%/+159.1%/+203.9%, zero errors.
  [bench](docs/experience/wins/2026-07-14-dspark-resident-budget-tp4.md).

- **2026-07-14 — DSv4 DSpark prompt router licensed for H20 TP=4.** DSpark
  output throughput moved from +6.3% at 32 prompt tokens to -12.4% at 128 and
  -18.6% at 8K. The opt-in `--dspark-max-prompt-tokens 64` router preserves the
  short-prompt path and restores 128/8K to within 1% of no-spec; defaults remain
  unchanged. [bench](docs/experience/wins/2026-07-14-dspark-prompt-router-tp4.md).

- **2026-07-14 — V100 (sm_70) prefill `cudaErrorNotSupported` fixed.** Two
  fixes, both gated exclusively on compute-major ≤ 7 so the sm_80+ hot path is
  byte-identical: (1) BF16 GEMM on Volta (no BF16 tensor cores — only FP16/FP32)
  now casts BF16→FP16, runs an FP16 tensor-core GEMM, casts back, skipping
  cublasLt's bad-algo heuristic; compute-major cached per device to avoid the
  uncached-per-step −77% decode regression. (2) `allow_sm70 = true` for the
  HD256 q8_kv2 paged-attention prefill/decode kernels (the 0.8B dense config),
  so the sm_70 cubin is compiled instead of the runtime `cudaErrorNotSupported`
  stub. [gemm](docs/experience/wins/2026-07-14-v100-sm70-bf16-gemm-fp16cast.md),
  [paged-attn](docs/experience/wins/2026-07-14-v100-sm70-paged-attention-allow-sm70.md).

- **2026-07-14 — DSv4 DSpark correctness PASS, opt-in unchanged.** Restored the
  official HC-lane mean, native BF16 Markov weights, and accepted-prefix recurrent
  fold: coherent 128-token output with **61/170 accepted (35.9%)** on H20 TP=4.
  Also bounded checkpoint prefetch to rank zero plus page-cache capacity, removing
  the observed 4-rank full-checkpoint read amplification. [correctness](docs/experience/wins/2026-07-14-dspark-dsv4-accept-and-correctness.md),
  [load](docs/experience/wins/2026-07-14-loader-tp-rank0-prefetch.md).

## [0.3.0] - 2026-07-12

Headline: **DSpark speculative decoding** for DSv4/Qwen3.6, a **CUDA kernel
`csrc/` reorg + content-addressed prebuilt-kernel release**, and a
**strategy-driven agent-OPD harness**. ~1 month of runtime + training work since
v0.2.1.

### Added

- **DSpark block-draft speculative decoding** (`--spec-type dspark`) — DSv4
  dual-stream draft + 3-stage backbone orchestrator + draft→verify→accept loop
  (T1–T4.4). P1 LICENSED: **2.39× decode short-ctx / 3.14× at ~3K** vs no-spec
  (Qwen3.6-27B, H20 TP=1). See §Verdicts.
- **Unified kernel set** — one full-build binary serves Qwen AND DSv4; the
  model-family kernel partition was deleted, releases key on SM tier only
  (`89ea8e7c4`). [win](docs/experience/wins/2026-07-11-unified-kernel-set-one-binary-qwen-and-dsv4.md).
- **Content-addressed prebuilt kernel bundle** — immutable source-addressed
  TileLang cubin bundle on the `kernel-artifacts` release; the zero-Python T1
  release lane fetches it instead of regenerating AOT (`6cb2c0054`).
- **Strategy-driven agent-OPD harness** — pluggable update strategy
  (`--update-strategy rejection-ce | sao-dis`) + dense partial-credit reward
  (fraction of fail-to-pass passing), off-policy DIS diagnostics.

### Changed

- **2026-07-12 — CUDA kernel `csrc/` reorg** (`a07a48d90`, `9fc53e7e4`,
  `051edb29b`). Exploded the 19-file `misc/` junk drawer into domain dirs (new
  `sampling/`·`norm/`·`recurrent/`·`elementwise/`; DSv4 MLA/DSA/MHC + FlashMLA/FA3
  shims → `attention/`; `kvcacheio/` → `kv/`) — every family now aligns 1:1 with
  its `src/ffi/*.rs` split. Deleted dead code (−6545 LOC): 3 Marlin W4/W4A8 GEMM
  `.cu` + `marlin_pf8/`, `kv/{paged_kv_append,scatter_kv}.cu`, and 5 `src/ffi/`
  extern decls (all 0-caller). `csrc/` now = 56 `.cu` in 10 kernel dirs.
  Byte-identical, bench-exempt (no runtime path changed).
  [win](docs/experience/wins/2026-07-12-kernel-csrc-reorg.md).

- **2026-07-11 — DSv4 decode-region KV reuse default ON** (`6230d9d3d`).
  Multi-turn concurrent throughput **+25%**; default flip after multi-shape
  verification.

### Verdicts

- **2026-07-11 — DSpark draft-KV: cap full-layer at per-request ceiling** (Qwen3.6-27B,
  CUDA, `1ee72d809`). The DFlash draft full-attention layer sized per-slot KV from the
  128K KV-pool floor (`max_seq_len`), not `max_total_tokens` — 512 MB/slot, clamping
  slots and blocking >4K prompts. Cap at `min(max_seq_len, max_total_tokens)`: lossless
  (scheduler admits nothing longer). Pod-verified: draft/slot **544→64 MB** at
  `--max-total-tokens 8192`, slots **32→256**, dspark tok-s/accept unchanged (2.49×/3.76×,
  above the P1 anchors), 13K prompt now fits one slot. P2.5 prefix-restore partial-ctx
  drafting was ALSO found already-implemented + verified holding (accept 0.18–0.22 on
  prefix-hit turns, 100% partial-ctx chains, no plain-decode fallback).
  [win](docs/experience/wins/2026-07-11-dspark-draft-kv-cap-per-request-ceiling.md).

- **2026-07-11 — DSpark/DFlash block-draft spec-decode: P1 LICENSED** (Qwen3.6-27B,
  CUDA). z-lab DFlash backbone drafter (`--spec-type dspark`) nets **2.39× decode
  tok/s short-ctx / 3.14× at ~3K** vs no-spec on H20 TP=1, B=1 greedy, no-prefix-hit
  — clears the 1.15× kill by a wide margin, above the 1.03× native-MTP ceiling that
  motivated adoption. block-16 verify overhead does NOT eat the win; accept-rate
  *rises* with ctx (0.199→0.228). Correctness PASS (needle + self-consistent).
  NOT a default flip: OPD rollout regime (91% prefix hit, 20–45K ctx) still gated on
  P2.5 (prefix-restore) + the 544 MB/slot draft-KV memory clamp.
  [win](docs/experience/wins/2026-07-11-dspark-p1-license-qwen36-27b.md).

- **2026-07-11 — DSv4 decode-region reuse: DEFAULT FLIPPED ON** (`--dsv4-decode-reuse`,
  was opt-in). Multi-turn concurrent A/B (token-preserving harness, the shape
  guidellm can't express) on H20 TP=4: aggregate throughput **+25.3% at c=16**,
  TTFT p50 halved (−52%), TPOT −18.7% — the win scales monotonically with
  concurrency. No single-shot regression (guidellm independent-prompt A/B is a
  byte-wash; finish-capture D2H ~free when reuse doesn't fire). ON-path
  correctness pod-verified across the campaign (crash-repro 24/24, needle-exact).
  Two binding shapes cleared → flip. The throughput lever was the reuse feature
  itself; the pinned-DRAM (#5) and admission-watermark (#6) knobs were KILLED
  (bad ROI / unsafe-no-cascade → #160).
  [wins](docs/experience/wins/2026-07-11-dsv4-decode-reuse-multiturn-concurrent-throughput.md)

- **2026-07-11 — Agent-OPD round −30.1% wall (H20 GPU1 3-arm A/B), quality-neutral
  (`894be29fa`)**: DSpark serial-B=1 decode LICENSED (rollout −29% / eval −30%,
  1.41×; engagement proven by net speedup + 78 `[dspark-draft]` lines; already
  default-on). Writeback grad-checkpoint offload now **seq-adaptive**
  (`writeback_offload_for_seq` = flag && seq_len≥4096) — short trajectories skip
  the host round-trip (backward −36%, writeback −33% at seq≈1276), long ones
  self-protect from the seq≥~9600 allocator OOM (errors/2026-06-28). Wins:
  [dspark-decode-and-seq-adaptive-offload](docs/experience/wins/2026-07-11-agent-opd-dspark-decode-and-seq-adaptive-offload.md).

- **2026-07-10 — DSv4 finish-write-through decode-region reuse: crash-fix gate
  PASS (opt-in `--dsv4-decode-reuse`), default flip pending perf**: v1
  (`79b5dbb17`) engaged (multi-turn match 640→704, +1 page into the decode
  region) but crashed the TP serve (`pool seq_len 494 != append_pos 485`) —
  the sub-page tail beyond `matched_len` has no radix content identity. v2
  (`28b8cd7bb`) added a continuation guard (reuse the tail only when
  `prompt[matched_len..finish_len] == entry.tail_tokens`). Pod re-verify TP=8:
  OFF 15/15 DET byte-identical; crash-repro 24/24 exact, zero
  `seq_len != append_pos`; multi-turn published 10 pages into the decode region,
  no over-restrict. OFF default byte-identical; flip needs a
  token-id-preserving perf harness.
  [wins](docs/experience/wins/2026-07-10-dsv4-finish-writethrough-decode-reuse.md)
  · [errors](docs/experience/errors/2026-07-10-dsv4-finish-writethrough-tail-content-identity.md)
- **2026-07-10 — DSpark-on-OPD default flip: quality-neutral LICENSED
  (opt-in), concurrency ≥4 DEFERRED**: final gate — pass-rate quality-neutral
  (n=16 dspark 9/16 ≥ plain 7/16, zero systematic per-task loss, CIs overlap;
  lossless-spec expectation confirmed). c=1 aggregate 1.9×; c≥4 unattributable
  under shared-box KV clamp (DFlash draft reserves 2560 MB/slot → co-tenant
  46 GB squeezes slots 256→6, OOM — not a dspark structural failure). No code
  default changed: dspark stays the licensed opt-in (`--dspark-draft-model`)
  until a clean-GPU c-sweep clears the concurrency leg.
  [wins](docs/experience/wins/2026-07-10-dspark-opd-default-flip-gate.md)
- **2026-07-10 — DSv4 Route A prefix reuse "identity formula fix": REVERTED
  (`4ad32362e`)**. The original claim below was wrong: the change never executed
  on DSv4-Flash (demand-paged skips it; its pod numbers licensed the
  copy-restore path) and broke V32/GLM band contiguity. Kept for the record:
  ~~`prepare_kv_batch` and `mirror_full_band` hardcoded `slot*lsp + i` instead of
  using engine-provided `slot_pages[i]`; 89.7% hit rate, 3.3× cold→hot~~.
  [errors](docs/experience/errors/2026-07-10-dsv4-prefix-reuse-identity-fix-was-noop-and-v32-hazard.md)
- **2026-07-10 — Qwen FP8 small-M dense GEMM: DeepGEMM from M=2 LICENSED;
  M=1 GEMV variants KILLED**: measured crossover (DeepGEMM flat 47.5–57.8 µs
  in M vs ~linear GEMV) moves `QWEN_FP8_DEEPGEMM_DENSE_MIN_M` 16→2 — matched
  same-tree A/B +5–9% dspark greedy csv / +2–5% rust, needle ×3 exact DET on
  both lanes. M=1 stays on the GEMV: smem-x and x-in-registers variants both
  measured slower (attributed via the new `fp8_wread_probe`: the per-row x
  tail is the whole 1.78-vs-2.9 TB/s gap; achievable read BW is 3.5 TB/s,
  not the 4.0 spec).
  [wins](docs/experience/wins/2026-07-10-qwen-fp8-smallm-deepgemm-crossover.md)
- **2026-07-10 — DSv4 KV-reuse Phases 2b+3b SHIPPED** (#154): whole-slot
  park deleted (−869 LOC; preemption rides the 2a prefix-state pool;
  `--kv-oversubscription` on DSv4 now fails loud) and FlashMLA bands are
  demand-paged — the 16K slot cliff dissolves (**3 → 117 slots** at
  `--max-total-tokens 16384`, same-day paired A/B). Correctness lanes
  green (E1 15/15 ×2 arms, E2 10/10 @4.16× warm TTFT, restore→batched
  kill-test 25/30→30/30 post codex-R3 fix); E6 c=4 wall **+3.8%** miss
  documented with attribution (slots 0.9pp; zeroing/growth-storm ruled out
  by ablation; residual needs nsys). Wins:
  `docs/experience/wins/2026-07-10-dsv4-park-deletion-phase2b.md`,
  `docs/experience/wins/2026-07-10-dsv4-band-demand-paging-phase3b.md`.

- **2026-07-10 — DSpark on the OPD rollout serve: wall-clock POSITIVE**
  (first e2e A/B, CC-as-harness, 16 real swe_smith tasks): matched-task
  rollout wall **−25.1%**, 4.11 tok/step, partial-ctx engaged on 90% of
  chains, deep-ctx accept 3.46 > cold 2.08; pass-rate movement within
  single-sample noise (9/16 vs 6/16, ~1.1σ). Default flip still gated on:
  multi-sample pass-rate, `/v1/stats` accept export, and wiring
  `dspark_draft_model` into the in-process `train agent-opd` engine
  (train_cli.rs:2434 — serve-only today).
  [wins](docs/experience/wins/2026-07-10-opd-e2e-dspark-rollout-ab.md)
- **2026-07-10 — DSv4 prefix reuse RELICENSED (Phase 2a, content-keyed
  host-resident state pool)**: cross-request reuse relanded on the
  post-Route-A-deletion baseline — entries keyed by radix host-page identity
  (D1 unrepresentable), pool = L2 (zero HBM), L3 mmap spill unbudgeted. Pod
  evidence gate green: warm TTFT **4.19×** (0.768→0.184 s), resend 10/10
  after the derived-state fix (`0b5bd3d55` — the FP8 band is decode-lane
  DERIVED state, never captured/restored; rebuilt from restored bf16
  staging), L3 read-back exact, publish overhead −0.35% (free).
  [wins](docs/experience/wins/2026-07-10-dsv4-prefix-state-pool-phase2a.md) ·
  plan [Phase 2](docs/plans/2026-07-09-dsv4-kv-reuse-seam-refactor.md).

- **2026-07-10 — DSpark sampled (temp>0) spec decode LICENSED**: device-side
  filter/chain-rejection kernels (`e22a41637`/`9f2dd5b3b`) take sampled spec
  from 34.8 (−7.5% vs plain) to **64–106 tok/s = 1.8–3.0× plain sampling**;
  determinism (cache-off same-seed byte-identical), needle 3/3, greedy lane
  regression-free. OPD 3-turn rollout shape: 62–77 tok/s sampled vs ~36 plain.
  Next walls: 16 per-step draft syncs (~36 ms sampled draft), greedy
  prefix-hit accept drop (3.11→1.92).
  [wins](docs/experience/wins/2026-07-10-dspark-sampled-device-path.md)
- **2026-07-10 — DSpark partial-ctx drafting (P2.5) LICENSED; sampling RNG
  cleared**: prefix-hit requests re-seed speculation (`8edde59c7`); multi-turn
  accept −11/−22% within band, 101–112 tok/s vs 42–44 plain; whole-restore
  −67% accept but 95 tok/s ≥ anchor, greedy byte-identical, needle 3/3 —
  sidecar fallback not needed. Same-seed-twice PASSES with prefix cache
  disabled → the 07-10 "determinism bug" was the lane/ctx confound, not RNG;
  determinism gates must control cache state. Env-sweep smoke Δ≈0%.
  [wins](docs/experience/wins/2026-07-10-dspark-partial-ctx-drafting.md)
- **2026-07-10 — DSpark trained heads NO-LICENSE (z-lab backbone stays);
  P2 sampling verify KILLED as-is**: FR Markov head +0.3–0.9 accept but
  draft 8.1→16.6 ms (per-row host loop) → ≤ z-lab tok/s; confidence
  truncation strictly harmful (conf=0 dominates); AEON block=11 −9% (12-row
  verify misses the B≥16 GEMM lane). Sampling lane: same-seed-twice FAILS
  (spec-path bug, plain lane passes) and host-side sampling lands −7.5% vs
  plain — fix determinism + device-side sampling before OPD rollout use.
  [wins](docs/experience/wins/2026-07-10-qwen36-dspark-dual-head-and-sampling-verdicts.md)
- **2026-07-10 — DSv4 Route A prefix reuse KILLED pending content-keyed
  redesign; warm-cache needle regression FIXED**: the Route A machinery
  (state pools, per-namespace tiers, restore path, host→FlashMLA page
  translation) deleted entirely (`bbaaea93b`, +67/−1553) after #154's
  bisection (origin `0198c3ba7`, amplifier page-sharing series) and a
  9-defect restore-path review; device page tables now refresh via a
  dirty-bit contract on every host-band change. Pod acceptance 6/6 (solo
  15/15 both cache states, concurrent 120/120, +193 MB pool budget
  reclaimed, park intact):
  [wins](docs/experience/wins/2026-07-10-dsv4-route-a-deletion-regression-fix-acceptance.md) ·
  [errors](docs/experience/errors/2026-07-09-dsv4-route-a-flashmla-needle-regression-bisected.md) ·
  reland plan [Phase 2](docs/plans/2026-07-09-dsv4-kv-reuse-seam-refactor.md).

- **2026-07-10 — Qwen3.6 DSpark block draft LICENSED (short-ctx greedy)**:
  36.2 ms/step, 104–108 tok/s = 2.4× plain decode on H20 after quant-lane
  routing (row-serial GEMV → DeepGEMM/cuBLASLt at B≥16); needle ×3 +
  self-consistency PASS, plain-decode control unregressed. OPD-rollout claim
  still gated on long-ctx A/B + the prefix-restore draft-ctx gap.
  [wins](docs/experience/wins/2026-07-10-qwen36-dspark-block-draft-licensed-2p4x.md)

- **DSv4 decode-kernel levers #141/#142/#143 LICENSED (2026-07-04).** uint4-vectorized
  FP8 GEMVs + TILE-templated batch accumulator + warp-parallel mhc_params tail with a
  fused params|pre_rms_norm decode-graph pair. Matched binary-pair A/B (TP=4/EP=4,
  8×H20, same shell): decode TPOT 39.57→24.90 ms (−37.1%, MTP-off c=1) and
  31.27→20.94 ms/committed-tok (−33.0%, MTP-on 2015-in); needle 3/3 + count gates
  clean, MTP drift shown pre-existing via paired baseline control.
  [wins/gemv](docs/experience/wins/2026-07-03-dsv4-gemv-uint4-tile-template.md) ·
  [wins/mhc](docs/experience/wins/2026-07-03-dsv4-mhc-tail-parallel-fused.md)

- **Agent-OPD toy-corpus capability lane KILLED; harness + 12-round loop SHIPPED (2026-07-03).**
  Five measured escalations (surface cues, gold-module scenery, turn budgets)
  all left the untrained 27B at ceiling on synthetic small-repo bug-fix tasks
  (8/8 → 24/24 → 22/24→0/24 cliff) — classic single-line bugs are
  pattern-matched, and read→edit completes in 2 turns. What shipped: the full
  curve harness (corpus gen + self-check, `scripts/agent_opd_curve.sh`,
  plotter, held-out eval channel), the tape-footprint 3× margin fix (OOM at
  seq≈1350), sandbox `__pycache__` staleness fix, and a 12-round 27B run
  (loss 0.376→0.155, pass-rate ≥ baseline, zero OOM). Phase 2 =
  teacher-rescue on real SWE-Pro.
  ([kill](docs/experience/errors/2026-07-03-agent-opd-toy-corpus-saturation-kill.md) ·
  [run](docs/experience/wins/2026-07-03-agent-opd-27b-loop-stability-12rounds.md) ·
  [plan](docs/plans/2026-07-03-agentic-opd-27b-capability-curve.md))

- **Phase 2 re-scoped; whole-step decode CUDA graph RE-KILLED (2026-06-21).**
  The B=1 chain-map/roofline shows the wall is foundation-bound (per-step
  `ctx.sync` + cross-process barrier; HBM ~2.8% util, 36× below roofline) —
  the graph lever measured −41%. MTP stays acceptance-gated opt-in
  (break-even ~57% accept; typical 50–53% is a wash); no universal spec-decode
  default. #70 closed.
  ([chain-map](docs/plans/2026-06-20-dsv4-b1-decode-chain-map.md))

### Train / OPD

- **OPD stack review-driven hardening (2026-07-06).** **Landed:** KL scale
  centralized behind `kl_batchmean_scale` + gradient regression test (guards the
  2026-06-16 LR-collapse); rollout arm → `--rollout-engine {infer,train}`
  (`ARLE_OPD_INFER_ROLLOUT` deleted); 490-line `gkd_anchor` split into phase
  helpers (490→251, zero behavior change); dead `rubric_writeback_ce_step`
  deleted; OPD-vs-RFT naming de-drifted (`agent-opd`/`rubric-opd` are RFT, not
  distillation). **Planned (pod-gated):** Metal OPD backend, real-SWE
  teacher-in-loop curve, overload-chain collapse.
  ([kl-guard](docs/experience/wins/2026-07-06-opd-kl-batchmean-scale-guard.md) ·
  [flags](docs/experience/wins/2026-07-06-opd-engine-knobs-cli-flags-pending-remote.md) ·
  [split](docs/experience/wins/2026-07-06-opd-gkd-anchor-phase-helpers-pending-remote.md) ·
  [dedrift](docs/experience/wins/2026-07-06-opd-rft-naming-dedrift-dead-code.md) ·
  [metal-plan](docs/plans/2026-07-06-opd-metal-training-backend.md) ·
  [swe-plan](docs/plans/2026-07-06-opd-real-swe-eval-teacher-in-loop.md) ·
  [chain-plan](docs/plans/2026-07-06-opd-step-overload-chain-collapse.md))

### CUDA

- **Qwen3.6 serves on CUDA (2026-06-29):** FP8 MoE via DeepGEMM; batched paged
  decode scales c=1→8 (Qwen3.6-27B-FP8, 1×H20: 21 → 26 tok/s aggregate).
  ([wins](docs/experience/wins/2026-06-29-cuda-qwen36-paged-batched-decode.md))
- **Qwen3.5-122B-A10B serves at TP4** via GQA KV-head replication;
  numerical-completion gate pending a clean re-run.
  ([wins](docs/experience/wins/2026-06-29-cuda-gqa-replication-122b-tp4.md))
- **GLM-5.2 (`glm_moe_dsa`, DSv4-DSA family) wired on the DSv4 path** —
  forward tranches landed, verification pending-remote; not
  production-verified. (wins `2026-06-19-glm52-*`)

### Metal

- **Qwen3.6 NextN/MTP spec decode shipped (2026-06-21)** on the canonical
  Metal model.
  ([wins](docs/experience/wins/2026-06-21-metal-qwen36-mtp-spec-decode.md))
- **VLM bring-up:** Gemma4 forward + image smoke landed (2026-06-15);
  DeepSeek-OCR wired (2026-06-24/25, vision numerics not yet faithful).
  Quality/throughput validation pending for both.

### Server

- **`/v1/chat/completions` now supports `stream=true`** (SSE
  `chat.completion.chunk` frames with `reasoning_content`/`content` deltas;
  closes the R5 tranche-2 deferral, #79). Multimodal chat streaming still
  fails closed with 400.
  ([wins](docs/experience/wins/2026-07-02-http-chat-sse-streaming.md))

### Repo

- **Renamed `agent-infer` → `arle`** across source, config, and docs
  (2026-06-29).

## [0.2.1] — 2026-06-15

> Consolidated section: tags `v0.1.5` (2026-05-02), `v0.2.0` and `v0.2.1`
> (both 2026-06-15) were cut without changelog sections. Everything below
> spans v0.1.4 → v0.2.1; per-tag artifacts live on GitHub Releases.

### Runtime rewrite — `infer-*` stack becomes the serving truth (2026-06-04)

- **Breaking:** the monolithic `infer` crate is deleted (`e81b98fb`,
  ~167k LOC). Serving stack: `infer-plan` → `infer-seam` → `infer-core` →
  `infer-cuda`/`infer-metal` → `infer-server`/`infer-api`; `infer-api`
  (`LoadedInferenceEngine`) is the single programmatic front door. Any command
  referencing `-p infer` is stale. Consolidated verification + performance
  verdict:
  [final report](docs/projects/2026-06-04-qwen35-dsv4-final-report.md).

### Training surface — OPD-only (2026-05-18)

- **Breaking:** scratch pretrain / SFT / GRPO / multi-turn RL surfaces are
  deleted; OPD is the only training axis.
  ([pivot](docs/projects/2026-05-18-opd-only-pivot.md))

### DSv4 perf campaign — adopt official kernels (2026-06-06 → 06-15)

- Official DSA indexer default-on: decode 124 ms → 26 ms flat @4096.
  ([wins](docs/experience/wins/2026-06-07-dsv4-official-dsa-default-on.md))
- FlashMLA `sparse_fwd` + FP8 DeepGEMM prefill default-on: 7.2 s → 3.48 s.
  ([wins](docs/experience/wins/2026-06-07-dsv4-prefill-official-kernels-default-on.md))
- Phase 0 debt closed 2026-06-10 (#56–#59). KV precision parity gate re-ported
  as correct-inference (needle ladder, not byte-identity); FlashMLA decode +
  fused-wqkv correctness LICENSED; pooled/contig-MoE default flip KILLED
  (−24%).
  ([lever verdicts](docs/experience/wins/2026-06-10-dsv4-lever-gate-license-or-kill.md))
- Seam-level KV-dtype dispatch `--kv-cache-dtype` (default bf16 unchanged);
  INT8/FP8 correctness LICENSED, opt-in pending a perf license (2026-06-12).
  ([wins](docs/experience/wins/2026-06-12-cuda-quant-kv-dispatch-int8-fp8.md))
- Phase 1 batched-lane keystone closed (#61 2026-06-11, #60 2026-06-15): DSv4
  B>1 decode takes the batched serving lane by default; residual c>1
  throughput lever is DP-attn (#89).

### OPD train (CUDA) — new beta surface

- **OPD mainline queue moved from experiment-only to operator-facing workflow.**
  `arle train opd --student-model <dir>` now runs the real HF-dir OPD path
  instead of the old pending stub, using the Qwen3.5 loader and `opd_step`
  directly. The 2026-05-24/25 queue also landed code-only chunked-logits KL
  parity, KV-tier observability counters, the default-off T2 coordinator
  wireframe, SFT-anchor corpus attribution, and a CPU-only capability-eval
  preflight for the P5 pure-OPD 5k adapter. Live task ordering, deferred GPU
  gates, and artifact links are tracked in
  [`docs/projects/2026-05-24-opd-mainline-task-backlog.md`](docs/projects/2026-05-24-opd-mainline-task-backlog.md).
- **End-to-end OPD CUDA training stack landed on Qwen3-0.6B.** Single-session
  32-commit arc through kill-or-license-gated wins brings the OPD step at the
  moderate Qwen3.5-like shape to **48.5 ms** on RTX 4070 Ti SUPER —
  **1.71× faster than the like-for-like PyTorch CUDA reference (83 ms)** —
  and the real Qwen3-0.6B checkpoint OPD step to **0.164 s/step** (~170×
  over a naive scratch CPU baseline). CPU/CUDA loss bit-equivalent to
  relerr 1.276e-6. Convergence verified at lr=1e-7 with held-out
  exact-overlap **50 → 82.8 %** by step 5000 (KL/NLL still monotonically
  falling). Five parallel axes killed cleanly via SOLID gates with
  recorded errors entries (forward_last_logits, merge_grad sharing, SDPA
  mask-softmax fusion, high-level CUDA Graph rollout capture, SwiGLU
  silu+multiply fusion). New CUDA op surfaces: `matmul_bt` forward +
  backward, in-place AdamW, KV cache for OPD rollout, device-resident
  RoPE / argmax, fused causal-SDPA decode, fused attention-prepare
  layout, fused grad clip. Usage manual:
  [`docs/projects/2026-05-21-arle-opd-cuda-usage-manual.md`](docs/projects/2026-05-21-arle-opd-cuda-usage-manual.md).
  Cycle wrap:
  [`docs/projects/2026-05-21-opd-cuda-cycle-wrap.md`](docs/projects/2026-05-21-opd-cuda-cycle-wrap.md).
  Industry positioning:
  [`docs/projects/2026-05-21-opd-industry-positioning-best-framework.md`](docs/projects/2026-05-21-opd-industry-positioning-best-framework.md).

### Observability

- Added low-overhead HTTP `request_trace` JSON summaries for streaming and
  buffered requests, including TTFT, total latency, token throughput,
  KV/prefix-cache state, scheduler phase EMA, pipeline, and preprocess
  snapshots. Added `scripts/bench_dsv4_trace_http.py` to run DSv4 HTTP smoke
  cases and collect matching `request_trace` entries from server logs without
  enabling CUDA-synchronizing per-layer tracing.
- Fixed DSv4 distributed HTTP submissions so concurrent client requests keep
  the same logical queue order on every rank. `DistributedSchedulerGroup` now
  serializes cross-rank fanout submission, preventing rank 0 and follower ranks
  from entering different per-request token coordinators under concurrent
  traffic.
- Allowed DSv4 decode to run scheduler batches larger than one via the existing
  per-slot decode path. This keeps multi-slot distributed HTTP fanout alive
  while the vectorized DSv4 B>1 decode kernel work remains pending.
- Added DSv4 HTTP TP/EP axis overrides through the existing `INFER_TP_SIZE`
  / `ARLE_TP_SIZE` and `INFER_EP_SIZE` / `ARLE_EP_SIZE` env vars. The default
  remains the legacy overlapping TP=world, EP=world layout. The first 8xH20
  profiling pass confirms the current runnable DSv4 layout is decode
  communication-bound: default TP=8/EP=8 performs 86 all-reduces per generated
  token per rank, and nsys observed 22016 NCCL all-reduce kernels for a
  32-token decode window. Evidence and industry comparison are recorded in
  [`docs/experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md`](docs/experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md).
- Added committed DSv4 trace artifacts under
  `docs/trace-artifacts/2026-05-14-dsv4-decode/`,
  including the compressed raw nsys report/database, `nsys stats`, client JSON,
  server log, and SHA256 manifest. The trace record no longer depends on remote
  `/tmp` files.
- Added DSv4 DeepEP MoE trace artifacts under
  `docs/trace-artifacts/2026-05-14-dsv4-deepep/`,
  including compressed BF16 and FP8 combine trace logs, parsed summaries, remote
  build evidence, default trace-off post-checks, and the current bottleneck
  callout for return-side combine exchange plus local expert GEMMs.
- Added a current 8xH20 DSv4 single-token Nsight trace under
  `docs/trace-artifacts/2026-05-14-dsv4-deepep/nsys-one-token-current/`.
  The `max_tokens=2` streaming request returned `霓灯` and produced exactly one
  `step_decode_kernel_launch` wave across 8 ranks. The isolated token takes
  266.020 ms wall; decode-only nsys shows `cuStreamSynchronize`,
  async allocation/free, launch/memset churn, and NCCL send/recv ahead of the
  actual attention and GEMV kernels.
- Added a refreshed 2026-05-15 DSv4 single-token Nsight trace under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-one-token-current/`.
  With send/recv route and route-logits scratch reuse in place, the same
  one-token decode shape is now 158.439 ms wall. The remaining ranked costs are
  async allocation/free, launch/memset churn, D2H route readbacks, NCCL
  SendRecv/AllReduce, and local expert FP8/FP4 GEMV.
- Added 2026-05-15 DSv4 padded-dispatch Nsight records under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/`.
  The negative first trace (`nsys-single-token-padded-dispatch`) shows that
  padding without removing the dead send-count kernel regresses to 136.908 ms;
  the fixed trace (`nsys-single-token-padded-dispatch-skip-count`) validates the
  shipped B=1 decode path at 123.955 ms and records the remaining ranked costs.
- Added the DSv4 padded peer-combine Nsight trace under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-token-padded-peer-combine/`.
  The real 8xH20 run keeps the `霓彩` output and shows the single-token decode
  wave at 112.133 ms after pre-summing padded return rows per origin peer.
- Added the DSv4 fused dispatch payload Nsight trace and matching HTTP smoke
  under `docs/trace-artifacts/2026-05-15-dsv4-deepep/`.
  The real 8xH20 run keeps the `霓彩` output, cuts decode-window SendRecv
  launches from 1,032 to 688 by exchanging hidden rows and route metadata in
  one BF16 payload, and records a fresh isolated single-token decode wave at
  118.985 ms. The trace-off `decode64` smoke returns normal English content at
  12.22 post-first tok/s and the arithmetic case returns `410`; the nsys run
  makes clear that NCCL exchange/reduction, launch overhead, allocator churn,
  D2H, and local expert GEMV still dominate.
- Added the DSv4 route-grouped pair GEMV Nsight trace under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-route-pair-gemv/`.
  The opt-in `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1` run keeps the `霓彩` output
  and measures a 117.894 ms single-token decode wave. The decode-window top
  costs are now explicit: `ncclDevKernel_SendRecv` at 50.338 ms per rank
  range, FP4 route pair GEMV at 19.616 ms, FP4 route `w2` GEMV at 10.487 ms,
  FP8 GEMV at 9.408 ms, plus allocator/free and launch overhead.
- Added a fresh user-requested single-token `nsys` rerun under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-current-user/`.
  The real 8xH20 `/root/DeepSeek-V4-Flash` run returns exact arithmetic `406`
  and measures a 94.841 ms decode wave. The slow stack is reduce-scatter
  combine, local FP8/FP4 expert GEMV, residual all-reduce/send-recv,
  attention/MHC/route kernels, and high per-token launch/alloc/free/D2H
  runtime overhead, not sampler time.
- Added the matching DSv4 single-token `NCCL_PROTO=LL128` negative trace under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-nccl-ll128/`.
  The arithmetic request still returns `406`, but the isolated decode wave is
  94.936 ms versus 94.841 ms on the current default reference, and
  reduce-scatter combine is slightly worse at 21.371 ms per rank-range.
  Protocol selection alone is therefore not the next default decode fix.
- Added an opt-in DSv4 return-combine overlap experiment behind
  `ARLE_DSV4_COMBINE_OVERLAP=1`. The path creates a second EP NCCL
  communicator on a dedicated communication stream and delays routed-output
  consumption with an explicit CUDA fence so shared expert compute can overlap
  the reduce-scatter. The real 8xH20 run returns exact arithmetic `406`, but
  the trace regresses from 94.841 ms to 104.359 ms because all-reduce timing
  and cross-stream event overhead outweigh the reduce-scatter improvement.
  The matching default-off HTTP smoke still reaches 12.05 post-first tok/s,
  so the overlap experiment remains disabled by default.
- Added a fused DSv4 B=1 padded DeepEP local expert prepare kernel and matching
  trace records under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-small-local-pack-prepare/`
  and
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/bench-small-local-pack-prepare-smoke/`.
  The real 8xH20 run returns exact arithmetic `406`, cuts H2D runtime calls
  from 1,040 to 696, cuts `cuMemsetD8Async` calls from 1,232 to 544, and keeps
  trace-off `decode64` at 12.05 post-first tok/s. The single captured nsys wave
  is 92.602 ms due to noisier D2H/AllReduce timing, so this is recorded as
  small-call cleanup rather than a wall-time win.
- Reused per-layer DSv4 incremental attention projection buffers for `c_q`,
  `c_q_normed`, `q_raw`, `kv_raw`, and `kv_normed`. The real 8xH20
  single-token `nsys` run returns exact arithmetic `406` and moves the decode
  wave from 94.841 ms to 90.946 ms, while `cuMemAllocAsync` calls drop from
  6,760 to 5,040 and `cuMemFreeAsync` calls drop from 3,048 to 1,328 inside
  the decode range. The matching HTTP smoke keeps normal Chinese/English
  streaming output and exact math, with `decode64` at 11.89 post-first tok/s.
- Added a direct current-path single decode-token Nsight breakdown under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-current-breakdown/`.
  The real 8xH20 `/root/DeepSeek-V4-Flash` run returns exact arithmetic `406`
  and measures a 105.205 ms isolated second-token decode wave. The top stack is
  now explicit: 16,177 CUDA launches, reduce-scatter combine, local FP8/FP4
  expert GEMV, all-reduce, attention/MHC/route kernels, and 347 D2H calls for
  per-layer synchronization. The actual D2H activity payload is only 44,044
  bytes, confirming the current issue is MoE communication/compute plus
  launch/runtime synchronization granularity, not sampler time or copy bandwidth.
- Switched additional full-write DSv4 runtime scratch buffers from zeroed
  allocation to uninitialized allocation: expert/shared/grouped hidden scratch,
  route logits, per-layer hidden scratch, and MHC parameter scratch. The real
  8xH20 single-token `nsys` trace under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-expanded-uninit/`
  returns exact arithmetic `406`, moves the isolated decode wave from
  105.205 ms to 88.554 ms, and cuts `cuMemsetD8Async` from 3,640 calls /
  6.932 ms per rank range to 1,920 calls / 2.839 ms. The trace still points at
  reduce-scatter combine and local FP8/FP4 expert GEMV as the main bottlenecks.
  The matching HTTP smoke under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/bench-expanded-uninit-smoke/`
  keeps normal Chinese/English multi-token output, exact math `410`, and
  `decode64` at 11.94 post-first tok/s.
- Extended the DSv4 uninitialized scratch cleanup to MoE dispatch, payload,
  recv/local-route, active grouped, and combine buffers. The real 8xH20
  single-token `nsys` artifact under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-moe-scratch-uninit-rerun/`
  returns exact arithmetic `406`, moves the isolated decode wave from
  88.554 ms to 87.667 ms after a rerun, and cuts `cuMemsetD8Async` from
  1,920 calls / 2.839 ms per rank range to 1,232 calls / 1.558 ms. The
  matching HTTP smoke under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/bench-moe-scratch-uninit-smoke/`
  keeps normal Chinese/English multi-token output, exact math `410`, and
  `decode64` at 12.06 post-first tok/s.
- Moved DSv4 grouped expert weight/scale pointer tables into
  `DeepseekV4MoeBlock` load-time caches for the opt-in grouped/route-grouped
  expert paths and future raw-pointer DeepGEMM integration. On the real 8xH20
  `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1` trace, exact arithmetic remains `406`,
  H2D activity drops from 1,918 calls / 374,752 bytes to 440 calls / 7,808
  bytes, H2D runtime drops from 5.490 ms to 1.380 ms, and the route-grouped
  single-token wave moves from 105.808 ms to 94.828 ms. The path remains
  default-off because reduce-scatter combine and route-wise FP4/FP8 GEMV still
  dominate; the default DeepEP smoke still returns math `410`, normal Chinese
  writing, and normal English decode text.
- Added the DSv4 default-path warm decode Nsight trace under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-default-warm-decode/`.
  The run warms a real decode first, then profiles a second single decode token
  on 8xH20. The output remains `霓彩`, the decode wave is 128.130 ms, and the
  trace confirms allocator/free overhead is steady-state rather than only
  first-decode initialization: the decode window still records 8,453
  `cuMemAllocAsync` calls and 6,048 `cuMemFreeAsync` calls. The slow stack is
  NCCL SendRecv/AllReduce, local FP8/FP4 expert GEMV, launch/runtime overhead,
  allocator/free churn, and route-count D2H synchronization.
- Added the DSv4 expert-wise grouped GEMV negative Nsight trace under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-expert-grouped/`.
  With `ARLE_DSV4_GROUPED_EXPERTS=1`, the real 8xH20 run keeps the `霓彩`
  output but regresses the warmed single-token decode wave to 145.693 ms.
  The trace shows `ncclDevKernel_SendRecv` at 58.049 ms per rank range, FP4
  grouped gate/up GEMV at 23.162 ms, FP4 grouped `w2` GEMV at 11.428 ms, and
  elevated route-count D2H synchronization. This confirms the opt-in grouped
  GEMV path remains default-off and that the target remains true grouped
  GEMM/DeepGEMM with DeepEP overlap.
- Added the DSv4 route-grouped pair trace-off HTTP comparison under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/bench-route-grouped-pair-vs-default/`.
  Default fused-dispatch decode keeps `decode64` at 11.47 completion tok/s and
  arithmetic at `410`; `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1` returns normal text
  and the same arithmetic answer but regresses `decode64` to 6.54 completion
  tok/s. Route-wise grouped GEMV remains default-off.
- Added DSv4 incremental stream scratch recycling and captured both the HTTP
  smoke and Nsight follow-up under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/bench-stream-recycle/`
  and
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-stream-recycle/`.
  The real 8xH20 run keeps normal text and arithmetic `410`; the isolated
  warmed decode wave improves from 128.130 ms to 111.798 ms, with
  `cuMemAllocAsync` dropping from 8,453 calls / 16.802 ms to 7,757 calls /
  12.574 ms and `cuMemFreeAsync` from 6,048 calls / 13.801 ms to 5,352 calls /
  11.096 ms. HTTP `decode64` stays effectively flat at 11.48 tok/s, so the
  main target remains NCCL plus local expert GEMV.
- Added DSv4 GPU compressor projection scratch reuse for `kv_raw` and
  `score_raw`, with trace artifacts under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/bench-compressor-projection-scratch/`
  and
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-compressor-projection-scratch/`.
  The real-output checks still pass (`decode64` normal text, arithmetic `410`)
  and alloc/free calls fall again (`cuMemAllocAsync` 7,757 -> 6,765,
  `cuMemFreeAsync` 5,352 -> 4,360), but HTTP `decode64` remains flat at
  11.47 tok/s and the single nsys wave is not a wall-time win because D2H/NCCL
  timing dominates this capture.
- Added DSv4 incremental attention scratch Nsight and HTTP artifacts under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/bench-attention-scratch/`
  and
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-attention-scratch/`.
  The real 8xH20 run returns normal multi-token output and arithmetic `410`;
  the isolated single-token decode wave is 97.042 ms after B=1 attention
  scratch cuts decode-window free calls from 4,360 to 3,048 without retaining
  prompt-sized prefill buffers. The trace directly answers the current
  bottleneck question: sampler is not in the top stack; NCCL SendRecv/AllReduce,
  D2H route-count synchronization, launch/runtime overhead, local expert
  FP8/FP4 GEMV, and attention/MHC kernels dominate.

### CUDA

- Added a default DSv4 B=1 padded BF16 combine reduce-scatter path behind
  `ARLE_DSV4_COMBINE_REDUCE_SCATTER` (default `1`). Expert ranks now sum padded
  route outputs into one row per origin peer and call NCCL `ReduceScatter`
  directly into the owner-rank output hidden row, with `0` preserving the prior
  grouped SendRecv combine. Real 8xH20 DSv4 validation against
  `/root/DeepSeek-V4-Flash` keeps normal Chinese/English streaming output and
  exact arithmetic `410`; `decode64` measures 12.05 post-first tok/s. The
  matching single-token nsys wave moves from 97.071 ms to 94.923 ms, replacing
  the old 23.163 ms SendRecv combine bucket with a 20.443 ms ReduceScatter
  bucket plus 3.259 ms residual SendRecv. This is a modest communication-shape
  cleanup; local expert grouped GEMM/DeepGEMM, DeepEP overlap, launch reduction,
  scratch reuse, and D2H readback removal remain the main performance targets.
- Reused per-layer DSv4 DeepEP dispatch scratch for route setup, rank count
  exchange buffers, packed send hidden rows/metadata, and local expert
  count/offset/cursor buffers. On the 8xH20 default path, trace-off math smoke
  reached 7.7-7.8 tok/s for 12 generated tokens, traced
  `ffn_deepep_dispatch_combine` p50 dropped to 1.552 ms, and the profiled
  `cuMemAllocAsync`/`cuMemFreeAsync` call count fell from 136,825 to 111,531 in
  the 8-token Nsight window. Remaining bottlenecks are still stream sync,
  return-side NCCL send/recv, and local expert GEMV/GEMM.
- Reused DSv4 DeepEP send-route token/slot buffers across decode steps and
  removed the unused `expert_token` output from `dsv4_pack_received_experts`.
  The 8xH20 trace-off math/writing smoke remained normal at 7.94-8.09
  completion tok/s, while the single-token nsys window reduced decode-only
  `cuMemAllocAsync` calls from 11,980 to 11,097 and `cuMemFreeAsync` calls from
  11,988 to 11,105. Remaining allocator pressure now sits in recv/local route
  buffers plus combine scratch and still needs a broader lifetime/graph pass.
- Reused DSv4 DeepEP B=1 decode recv/local route scratch for received hidden
  rows and metadata, local expert packed rows/weights/route slots, and
  route-output rows. Prefill preallocates only a small `ep_world * topk` decode
  capacity so long prompts do not retain prompt-sized route buffers. The real
  8xH20 DSv4 smoke stayed correct at 8.24-8.79 completion tok/s, and the
  single-token nsys window improved from 191.152 ms to 148.253 ms while
  reducing decode-only `cuMemAllocAsync`/`cuMemFreeAsync` calls to
  9,480/9,488 and `cuMemsetD8Async` calls to 10,554.
- Reused the DSv4 B=1 decode MoE route-logits buffer and preallocated its
  one-token scratch during prefill. This is an allocator-count cleanup rather
  than a confirmed wall-time win: the single-token nsys window reduced
  decode-only `cuMemAllocAsync`/`cuMemFreeAsync` calls again to 9,136/9,144 and
  `cuMemsetD8Async` calls to 10,210, while the captured wall time was noisy
  at 162.062 ms versus the prior 148.253 ms.
- Reran the post-scratch DSv4 single-token Nsight capture on 2026-05-15. The
  fresh one-token decode wave measured 158.439 ms wall and confirms the
  remaining cost center is not sampler or KV-cache lookup: runtime allocation,
  launch, memset, D2H routing readbacks, NCCL exchange/reduction, and per-expert
  GEMV still dominate before attention.
- Reused per-layer DSv4 shared expert scratch during DeepEP decode and added an
  in-place BF16 add kernel for accumulating shared expert output into the routed
  MoE output. Real 8xH20 smoke stayed correct at 9.07-9.50 completion tok/s,
  while the single-token nsys wave improved from 158.439 ms to 140.111 ms and
  decode-only `cuMemAllocAsync`/`cuMemFreeAsync` calls fell from 9,136/9,144 to
  7,416/7,424. The same step restored the CUDA `argmax_batch_readback_into`
  re-export required by Qwen3.5 CUDA builds. The scratch is gated to B=1 decode
  so long prefill does not retain prompt-sized shared expert buffers.
- Optimized the gated DSv4 grouped expert prototype behind
  `ARLE_DSV4_GROUPED_EXPERTS=1` by caching per-layer local expert weight
  pointer arrays and launching indexed active experts instead of rebuilding
  active pointer tables every step. The route remains opt-in: 8xH20 trace-off
  smoke improved grouped math latency to 2.37-2.40 s and short writing latency
  to 2.69 s, but traced `ffn_deepep_local_experts` p50 is still 1.196 ms versus
  roughly 0.46 ms on the default scratch-reuse path. The harness is ready for
  the next replacement with real grouped GEMM/DeepGEMM.
- Added a gated DSv4 grouped gate/up pair GEMV launch for the same
  `ARLE_DSV4_GROUPED_EXPERTS=1` harness. The FP8/FP4 pair kernels compute
  `w1` and `w3` in one grouped launch when format, shape, and block-scale
  layout match, otherwise the path falls back to separate grouped GEMV
  launches. 8xH20 nsys with `ARLE_DSV4_MOE_BACKEND=deepep` confirms
  `dsv4_fp4_grouped_gemv_pair_batch_kernel` runs in decode, but the grouped
  harness remains default-off: the decode window is still dominated by NCCL
  send/recv plus allocation/free and launch churn, not by the missing gate/up
  fusion alone.
- Added a gated DSv4 MoE combine exchange experiment via
  `ARLE_DSV4_COMBINE_DTYPE=fp8`. The path quantizes return-route BF16 rows to
  FP8 E4M3 with per-row FP32 scales, exchanges the FP8 payload through NCCL
  `Uint8` send/recv plus scale exchange, and dequantizes back to BF16 before
  the existing route-slot combine kernel. It is validated on 8xH20 but remains
  opt-in because the measured 1,039-token prefill trace is not faster than the
  BF16 combine default.
- Reused per-layer DSv4 HyperConnection/MHC temporary buffers in the
  incremental attention and FFN paths. The 8xH20 trace-off smoke set improved
  from roughly 5.5/5.6/6.0 tok/s to 6.3/6.2/7.3 tok/s for two math cases and
  one short writing case, while traced decode `attn_mhc` and `ffn_mhc` p50
  dropped to 0.088 ms and 0.085 ms respectively.
- Reused DSv4 incremental attention scratch for prepared Q/K, local attention
  output, and the `wo_a` latent projection, gated to B=1 decode so prefill does
  not retain prompt-sized buffers. The real 8xH20 HTTP smoke remains correct
  (`decode64` normal text, writing normal Chinese, arithmetic `410`), while the
  paired single-token Nsight capture reduces warmed decode `cuMemFreeAsync`
  calls from 4,360 to 3,048.
- Added a default DSv4 local expert segment-input path for DeepEP decode. When
  `w1` and `w3` are DSv4 block-scaled FP8/FP4 matrices, the per-expert fallback
  now runs their GEMV directly from the packed `expert_hidden` segment and
  skips the old D2D copy into `scratch.input`; unsupported formats still use
  the original copy fallback. Real 8xH20 nsys against `/root/DeepSeek-V4-Flash`
  kept the `霓虹` streaming output, reduced decode-only `cuMemcpyDtoDAsync_v2`
  from 871 calls / 1.795 ms per rank range to 613 calls / 1.240 ms, and moved
  the isolated single-token wave from 146.448 ms to 145.104 ms. The trace
  confirms this is a small cleanup: allocator/runtime churn, D2H route
  readback, NCCL SendRecv/AllReduce, and per-expert FP8/FP4 GEMV remain the
  dominant costs.
- Reused per-layer DSv4 incremental hidden scratch for attention/FFN
  HyperConnection pre-projection and RMSNorm temporaries. Real 8xH20 nsys
  against `/root/DeepSeek-V4-Flash` kept the streaming `霓虹` output and moved
  the isolated decode wave from 145.104 ms to 135.390 ms. Decode-only
  `cuMemAllocAsync`, `cuMemFreeAsync`, and `cuMemsetD8Async` calls each dropped
  by 1,376, matching four one-token temporary buffers across 43 layers and 8
  ranks. The remaining ranked costs are launch/runtime overhead, D2H route
  readback, NCCL SendRecv/AllReduce, and local expert FP8/FP4 GEMV.
- Removed the default DSv4 DeepEP AllGather route's redundant 32-byte
  `send_rank_counts` host readback. The AllGather count matrix is now
  collected before route packing and reused to derive both send and receive
  counts; the `ARLE_DSV4_COUNT_EXCHANGE=sendrecv` fallback keeps the previous
  readback. Real 8xH20 nsys kept the `霓虹` output, moved the single-token
  decode wave from 135.390 ms to 129.768 ms, and reduced decode-only D2H calls
  from 887 to 543. The remaining D2H cost is the 256-byte all-rank count matrix
  readback, ahead of deeper device-side count-prefix or countless dispatch
  work.
- Added the default DSv4 B=1 padded dispatch fast path for DeepEP decode. When
  the count exchange mode is the default AllGather route, decode now uses fixed
  `ep_world * topk` route slots, initializes unused slots as invalid, skips the
  unused send-rank zero/count kernel, and avoids the count AllGather plus its
  256-byte all-rank D2H readback. Set `ARLE_DSV4_PADDED_DISPATCH=0` to force
  exact-count dispatch. Real 8xH20 nsys kept the `霓彩` streaming output, moved
  the single-token decode wave from 129.768 ms to 123.955 ms, removed
  `ncclDevKernel_AllGather` from the decode window, and reduced decode-only D2H
  calls from 543 to 344. The remaining slow stack is NCCL SendRecv/AllReduce,
  launch/runtime and allocator/memset/free churn, local-count D2H, and local
  expert FP8/FP4 GEMV.
- Optimized the B=1 padded return-side combine exchange by summing valid padded
  route outputs into one BF16 row per origin peer on the expert rank before the
  return send/recv. This keeps the same `霓彩` streaming output, reduces
  returned combine rows by 8x, moves the real 8xH20 single-token decode wave
  from 123.955 ms to 112.133 ms, and drops `ncclDevKernel_SendRecv` time from
  25.211 ms to 23.329 ms per rank range. The local expert FP8/FP4 GEMV timings
  are unchanged, so true grouped GEMM/DeepGEMM remains the next compute target.
- Added and gated the default-path DSv4 single-expert `w1`/`w3` pair GEMV
  experiment behind `ARLE_DSV4_PAIR_EXPERT_GEMV=1`. The real 8xH20 trace kept
  the `霓彩` output and proved the new `dsv4_fp4_gemv_pair_batch_kernel` runs,
  but it regressed the local expert work on the current B=1 decode shape
  (`23.207 ms` per rank range for the pair kernel, 127.412 ms decode wave), so
  the shipped default remains the split GEMV path while the next compute target
  stays true grouped GEMM/DeepGEMM.
- Added and gated a DSv4 route-wise grouped expert experiment behind
  `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1`. It runs local experts directly from
  padded received route slots and removes the local-count D2H readback from the
  top decode runtime list, but the real 8xH20 nsys trace regressed to a
  145.669 ms single-token wave because `dsv4_fp4_route_gemv_batch_kernel`
  costs 35.895 ms per rank range. The path remains default-off and documents
  why the next compute step needs DeepGEMM-style grouped GEMM rather than
  route-wise GEMV.
- Added a clean 8xH20 decode-only HTTP comparison for the gated
  `ARLE_DSV4_PAIR_EXPERT_GEMV=1` path. The default split expert GEMV path
  reaches 11.79 post-first tok/s on `decode64`, while pair GEMV reaches
  7.70 tok/s; both return normal sequence text and the arithmetic check returns
  `410`. This keeps pair GEMV default-off and confirms the next compute target
  is real grouped GEMM/DeepGEMM rather than single-expert gate/up fusion.
- Added `HiddenStates::uninit` for CUDA call sites that immediately overwrite
  every element and switched DSv4 decode temporaries plus generic GEMM/add/SwiGLU
  outputs to use it where safe. Real 8xH20 DSv4 HTTP smoke remains correct
  (`decode64` reaches 11.99 post-first tok/s and the arithmetic check returns
  `410`), and single-token nsys shows `cuMemsetD8Async` dropping from 8,789
  calls / 11.855 ms per rank range to 2,957 calls / 4.180 ms. The isolated
  decode wave moves from 125.497 ms to 112.724 ms; NCCL exchange, launch
  overhead, async allocation/free, and local expert FP8/FP4 GEMV remain the top
  targets.
- Added the DSv4 B=1 fused dispatch payload experiment. Padded DeepEP decode
  appended route metadata as raw BF16 words behind each hidden row and exchanged
  hidden+metadata through one BF16 grouped send/recv instead of separate BF16
  hidden and I32 metadata exchanges. Real 8xH20 nsys kept the output correct,
  reduced SendRecv launches from 1,032 to 688, and recorded the isolated decode
  wave at 118.985 ms; NCCL SendRecv/AllReduce, launch/runtime overhead,
  allocator churn, D2H, and local expert FP8/FP4 GEMV remained the next targets.
- Optimized the gated route-wise grouped expert experiment by pairing its
  route-local `w1` and `w3` GEMV launches for matching DSv4 block-scaled FP8 or
  FP4 weights, falling back to split route GEMV when format or shape differs.
  The real 8xH20 nsys run lowers the prior route-grouped regression from
  145.669 ms to 117.894 ms, but it remains default-off because single-token
  decode is still dominated by NCCL SendRecv, route GEMV work, launch overhead,
  and async allocation/free. The main target remains true grouped
  GEMM/DeepGEMM with DeepEP overlap.
- **🎉 W4-hybrid prefill graph capture closes 4k/c=4 gap — Tier 1 STRONG
  PROCEED** (`a56b7a9`/`c44788f` 2026-05-10). Path B.2 bucketed prefill
  graph allocation key reduces capture key churn from 388 unique → **7
  unique** (98% reduction) with **98.5% LRU dominant key reuse rate**.
  Engine-side TTFT p50 **2000ms → 150ms = -92.5%** improvement on
  4k/c=4 prefill-dominant workload (server-side ground truth via
  `/v1/stats engine_ttft_us`; client-side guidellm 0.6.0 TTFT
  measurement separately broken per `e8d82b0` — bench tool bug, not
  substrate). Throughput **+632%** in matched-control 60s window
  (53 → 388 requests). Codex's "second-order bucketing" insight
  (captured scalar launch parameters use bucket capacity, not exact
  dim from first capture) was load-bearing for the win and added to
  skill v1.7.0 anti-pattern catalog. Followup: n=3 σ-tight re-bench +
  guidellm streaming fix. Evidence:
  [`docs/experience/wins/2026-05-10-bench-40-pathB2-tier1-strong-proceed.md`](docs/experience/wins/2026-05-10-bench-40-pathB2-tier1-strong-proceed.md).
- W4-hybrid Qwen3 paged-prefill **CUDA Graph capture** lands as opt-in
  via `INFER_PREFILL_GRAPH=1` + `INFER_HYBRID_W4A8_PREFILL=1` (`35fc3cf`).
  Phase 1 functional gate: prefill-lifetime `MarlinPrefillScratch`
  lifecycle + multi-key 8-d graph cache (token / page layout / start_pos)
  + W4 graphsafe weight gating for dense BF16, W4A16 Marlin, W4A8 Marlin,
  and W4-hybrid. Default behavior unchanged when env vars unset.
  Throughput license deferred: scout bench A vs B (graph OFF baseline
  TTFT p50 1628.9 ms vs graph ON 1627.8 ms = Δ -0.07%) detected
  capture-key churn — Path A multi-key direction KILLED, Path B
  device-memory `start_pos` re-licensed P0 (`e462c53`). Evidence:
  [`docs/experience/wins/2026-05-10-bench-p24-w4a8-prefill-graph-hoist.md`](docs/experience/wins/2026-05-10-bench-p24-w4a8-prefill-graph-hoist.md),
  [`docs/experience/errors/2026-05-10-37-throughput-bench-killed-pathA-multikey-churn.md`](docs/experience/errors/2026-05-10-37-throughput-bench-killed-pathA-multikey-churn.md).

### Long-context (cross-backend)

- **RoPE scaling support** (YARN / Linear / NtkAware) wired through
  `Qwen3Config::rope_scaling` and `Qwen35Config::rope_scaling` (Phase
  1+2 closed via 7 atomic commits + 51 unit tests). Helpers
  `compute_scaled_inv_freq` and `compute_attention_factor` ship in both
  spec crates. CUDA backend integration via
  `weight_loader::precompute_rope_with_scaling` (qwen3 path) +
  `precompute_rope_with_qwen35_scaling` thin shim. Vanilla path
  (`rope_scaling = None`) is bit-equivalent to the legacy
  `precompute_rope` formula (verified by
  `vanilla_inv_freq_matches_legacy_formula` test). Long-ctx bench
  validation (Qwen3-4B 64k YARN×2 / 128k YARN×4 + FP8 KV) deferred to
  Phase 3; CUDA-side viable on RTX 4070 Ti SUPER 16 GB per
  [`docs/plans/2026-05-10-rope-yarn-phase3-cuda-bench-plan.md`](docs/plans/2026-05-10-rope-yarn-phase3-cuda-bench-plan.md).
  Apply to a model dir via [`scripts/setup_qwen3_yarn_config.py`](scripts/setup_qwen3_yarn_config.py).
  Consolidation:
  [`docs/experience/wins/2026-05-10-m-rope-yarn-scaling-phase1-phase2-landed.md`](docs/experience/wins/2026-05-10-m-rope-yarn-scaling-phase1-phase2-landed.md).

### Structured-output (xgrammar)

- `crates/xgrammar-sys` Rust safe wrapper over upstream
  `mlc-ai/xgrammar` v0.1.34 lands as Phase 1 FFI scaffold (codex's #26).
  Default build is a stub that compiles without native sources or
  network; `--features real` builds a C++ shim against a pinned
  upstream checkout via `cc`. Wrapper surface:
  `GrammarCompiler` / `CompiledGrammar` / `GrammarMatcher` /
  `bitmask_size` / per-step bitmask fill APIs. No HTTP, scheduler,
  sampler, or GPU sampling integration yet — that is follow-up
  tranche work. Plan:
  [`docs/plans/M_xgrammar-ffi-scaffold.md`](docs/plans/M_xgrammar-ffi-scaffold.md).

### Metal

- Qwen3.5-0.8B MLX 4bit single-request step-driver reaches 305.5 tok/s mean
  / 304.7 p50 on M4 Pro 20c for `1024/256`. The matched GGUF Q4_K_M
  exact default remains 202.1 tok/s direct for correctness, while the
  opt-in native-q4 load path reaches 236.7 tok/s direct / 239.8 tok/s
  step-driver, so current status surfaces no longer present the historical
  211.7 tok/s GGUF-only profile as the Metal SOTA headline. Evidence:
  `docs/experience/wins/2026-04-28-bench-metal-qwen35-0p8b-mlx4bit-qknorm-default.md`.
  Native-q4 GGUF evidence:
  `docs/experience/wins/2026-04-28-bench-metal-qwen35-0p8b-gguf-native-q4.md`.


> Older releases (0.1.x — pre-rewrite): see [CHANGELOG-history.md](CHANGELOG-history.md)
