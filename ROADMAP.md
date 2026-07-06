# ARLE Roadmap

Updated 2026-07-02. Derived planning surface. On any conflict the canonical
doc wins:

- Strategic master: [`docs/projects/2026-06-10-arle-master-strategy-v2.md`](docs/projects/2026-06-10-arle-master-strategy-v2.md)
- Support status: [`docs/support-matrix.md`](docs/support-matrix.md)
- Workspace topology: [`docs/codebase-map.md`](docs/codebase-map.md)
- Architecture boundaries: [`docs/architecture.md`](docs/architecture.md)
- Benchmark process: [`docs/bench-and-trace-spec.md`](docs/bench-and-trace-spec.md)
- Contributor contract: [`AGENTS.md`](AGENTS.md)

## Positioning

ARLE is a Rust-native, device-neutral inference runtime with integrated
local agent and **On-Policy Distillation (OPD)** workflows. The serving
truth is the `infer-*` rewrite stack (`infer-plan` → `infer-seam` →
`infer-core` → `infer-cuda`/`infer-metal` → `infer-server`/`infer-api`);
the monolithic `infer` crate was deleted 2026-06-04. `arle` is the unified
front door. Product mainline = coding-agent runtime (local Metal
single-user + self-hosted CUDA multi-tenant). DSv4-Flash on 8×H20 is the
technical wedge and engine forge, not a separate product line. Training
is OPD-only (2026-05-18 pivot — see
[`docs/projects/2026-05-18-opd-only-pivot.md`](docs/projects/2026-05-18-opd-only-pivot.md)).

## Active Priorities (strict serial — master strategy v2 §3)

Progress tracker: [umbrella #55](https://github.com/cklxx/arle/issues/55).
Issues carry `phase-N` labels; off-path infra carries `infra`.

| Phase | Status | Goal | Exit condition | Anchor |
| --- | --- | --- | --- | --- |
| **0 — Debt** | ✅ **Closed 2026-06-10** (#56–#59). Open residue: [#68](https://github.com/cklxx/arle/issues/68) model-generic KV-quant parity gate (Qwen 4-precision matrix) — does not re-block Phase 1. | Long-ctx correctness closeout, 256K admission real fix, KV-precision-parity re-port to `infer-cuda`, truth-surface resync. | All four items closed; parity harness unlocks the gated FlashMLA/fused-wqkv/contig-MoE default flips. | v2 §3 Phase 0 |
| **1 — Batched serving lane (keystone)** | ✅ **Keystone closed** ([#61](https://github.com/cklxx/arle/issues/61) 2026-06-11 · [#60](https://github.com/cklxx/arle/issues/60) 2026-06-15): DSv4 B>1 decode takes the batched lane by default. Residual c>1 throughput lever: DP-attn ([#89](https://github.com/cklxx/arle/issues/89)). | True batched lowering per [`unified-batched-kvpool-abstraction`](docs/plans/2026-06-07-unified-batched-kvpool-abstraction.md) (`KvBatchDescriptor` + `ModelKvAdapter`, DSv4 first). `cd421794` (sequential plan-split, c≥2 no longer crashes) is the starting point, not the goal. | c-sweep clears TTFT+ITL+tok/s per bench spec; then deepep_ll-vs-allreduce A/B at its real lane, license-or-kill. | v2 §3 Phase 1 |
| **2 — Spec decode default-good** | **Re-scoped 2026-06-21.** [#70](https://github.com/cklxx/arle/issues/70) **CLOSED — whole-step graph KILLED** by the [B=1 chain-map/roofline](docs/plans/2026-06-20-dsv4-b1-decode-chain-map.md) (−41%, foundation-bound: per-step `ctx.sync` + cross-process barrier — the skew-anatomy re-license is overturned by measurement, launches aren't the wall). [#62](https://github.com/cklxx/arle/issues/62) closed. | The B=1 wall is latency/foundation-bound (HBM ~2.8% util, **36× below roofline**); MTP d2 is **acceptance-gated** (break-even ~57%; typical 50–53% → wash, +14% only on high-accept ShareGPT) → **stays opt-in, not default-flipped**. The throughput headroom is in **batching** (Phase 1), not single-stream. | No universal spec-default; MTP opt-in for high-acceptance workloads. | [chain-map](docs/plans/2026-06-20-dsv4-b1-decode-chain-map.md), v2 §3 Phase 2 |
| **3 — Product re-aim** | **ACTIVE** — [#64](https://github.com/cklxx/arle/issues/64)/[#65](https://github.com/cklxx/arle/issues/65) closed-completed (2026-06-23 / 06-29: OPD GPU resume, Qwen3.6 CUDA serving); live: [#71](https://github.com/cklxx/arle/issues/71) AIPC, [#90](https://github.com/cklxx/arle/issues/90) SOPD, [#102](https://github.com/cklxx/arle/issues/102)/[#103](https://github.com/cklxx/arle/issues/103) train-side | long-ctx mission restart on the new substrate, OPD GPU experiments resume, **AIPC route** (Metal single-user convergence + HIP/ROCm third backend on the seam — local unified-memory hardware, no pod contention), **SOPD self-training axis** (umbrella [#90](https://github.com/cklxx/arle/issues/90) → children [#91](https://github.com/cklxx/arle/issues/91)–[#98](https://github.com/cklxx/arle/issues/98), incl. [#98](https://github.com/cklxx/arle/issues/98) rubric-graded A5 — the open-ended bridge; teacher-free LoRA self-distill at rollout time; OPD-only, never GRPO; gated on its own Phase-0 keystone, off the serial critical path — see below). | Per-item; mission threshold ≥1.30 stands. | v2 §3 Phase 3, [SOPD plan](docs/plans/2026-06-14-self-training-lora-opd-sopd.md) |

Off-path / opportunistic (no serial-phase contention):
[#69](https://github.com/cklxx/arle/issues/69) DSv4 serve cold-boot ~6 min
(rank-0 serialization + 8× read amplification);
[#91](https://github.com/cklxx/arle/issues/91) SOPD Phase-0 keystone —
**closed-completed 2026-06-14 (PASS)**; children
[#93](https://github.com/cklxx/arle/issues/93)–[#98](https://github.com/cklxx/arle/issues/98)
unlocked (prerequisite [#92](https://github.com/cklxx/arle/issues/92)
prefix-cache epoch-invalidation still open);
[#123](https://github.com/cklxx/arle/issues/123) DSpark umbrella —
semi-autoregressive draft + confidence-scheduled verify (children #124–#131),
the spec-decode acceptance-wall attack after the Phase-2 re-scope; off-path
until a child earns a license.

OPD substrate maintenance (off-path, 2026-07-06 review): hardening landed (KL
scale guard, `--rollout-engine` flag, `gkd_anchor` split, dead-code + naming
de-drift); three pod-gated follow-ups planned —
[Metal OPD backend](docs/plans/2026-07-06-opd-metal-training-backend.md),
[real-SWE teacher-in-loop curve](docs/plans/2026-07-06-opd-real-swe-eval-teacher-in-loop.md),
[overload-chain collapse](docs/plans/2026-07-06-opd-step-overload-chain-collapse.md).

Killed/deferred work (B=1 per-kernel micro-levers, deepep_ll default-on,
classical spec, 5–6 ms-on-H20 framing, FlashInfer migration, ROCm, …) is
enumerated in master strategy v2 §5 — re-doing a KILLED item requires
overturning its evidence first. Precedent: the whole-step decode graph
lever was re-licensed 2026-06-10 by the
[nsys skew anatomy](docs/experience/wins/2026-06-10-dsv4-nsys-skew-anatomy-rewrites-lever-board.md),
then **RE-KILLED 2026-06-21** by the
[B=1 chain-map/roofline](docs/plans/2026-06-20-dsv4-b1-decode-chain-map.md)
(−41%, foundation-bound — launches aren't the wall; [#70](https://github.com/cklxx/arle/issues/70)
closed). The per-kernel/alloc/host-overhead micro-lever KILL stands.

## Next-Model Priority Order

Currently shipped: Qwen3-dense + Qwen3.5/3.6 hybrid·MoE (CUDA + Metal —
Qwen3.6 now serves on CUDA via FP8 MoE/DeepGEMM, no longer Metal-only),
DSv4-Flash (CUDA 8×H20 TP=8/EP=8). Metal also runs VLMs (Gemma4, DeepSeek-OCR
bring-up) + DiffusionGemma. The model-coverage queue is ranked, not parallel:

1. **DeepSeek V4 (DSv4-Flash)** — active substrate (Phase 0–2 above).
2. **Qwen 3.6** — **CUDA serving landed** (no longer "next"): FP8 MoE via
   DeepGEMM, batched paged decode scales c=1→8 (Qwen3.6-27B-FP8 1×H20 21→26
   tok/s; wins `2026-06-29-cuda-qwen36-paged-batched-decode`). Metal canonical
   model with NextN/MTP spec-decode shipped (wins
   `2026-06-21-metal-qwen36-mtp-spec-decode`).

Active in-flight model items (no longer "Qwen3.6 next"):

- **GLM-5.2** (`glm_moe_dsa`, DSv4-DSA family, 256 experts) — wired on the DSv4
  CUDA path; forward tranches landed but **verification pending-remote** (wins
  `2026-06-19` glm52-* all pending-remote). Not production-verified.
- **Gemma4 / DeepSeek-OCR Metal VLMs** — Metal forward + image smoke landed;
  **quality/throughput validation pending** (Gemma4 wins `2026-06-15` gemma4-*;
  DeepSeek-OCR wired/bring-up, vision numerics not yet faithful, wins
  `2026-06-24/25` deepseek-ocr-*).
- **Qwen3.5-122B-A10B at TP4** — serves at TP4 via GQA KV-head replication (all
  4 worker engines ready); **numerical-completion gate pending** a clean re-run
  (wins `2026-06-29-cuda-gqa-replication-122b-tp4`).

Other families in the support matrix sit behind these and are not actively
scheduled.

Backend queue: CUDA + Metal are shipped. **HIP/ROCm + Vulkan** are the AIPC
lane ([#71](https://github.com/cklxx/arle/issues/71), #76/#77): code began
landing 2026-06-10/11 ahead of the Phase 3 ordering (`infer-hip`,
`infer-vulkan` + `hip-/vulkan-{sys,kernels}`; plans
[`2026-06-10-hip-backend-mvp.md`](docs/plans/2026-06-10-hip-backend-mvp.md),
[`2026-06-11-hip-onbox-runbook.md`](docs/plans/2026-06-11-hip-onbox-runbook.md)).
**Phase-ordering ratification pending** — see
[refactor roadmap §6](docs/plans/2026-06-12-architecture-refactor-roadmap.md).
Gemma4 already has a working **Metal VLM forward** (`gemma-spec`; SWA + full
attn, image-capable — smoke/bench landed, quality/throughput pending, see the
in-flight items above); the in-tree Vulkan Gemma4 order pin remains **unranked**
on the AIPC backend queue (same pending ratification).

## History

Released tags + bench evidence live in:

- [`CHANGELOG.md`](CHANGELOG.md) — per-version notes (latest: v0.1.5)
- [`docs/experience/wins/`](docs/experience/wins/), [`docs/experience/errors/`](docs/experience/errors/) — curated evidence log
- [GitHub Releases](https://github.com/cklxx/arle/releases) — tagged binaries
- `git log` — full history

Use [`docs/index.md`](docs/index.md) to find current documents. Anything not
listed there is not a source of truth.
