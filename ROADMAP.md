# ARLE Roadmap

Updated 2026-06-14. Derived planning surface. On any conflict the canonical
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
| **1 — Batched serving lane (keystone)** | **ACTIVE** ([#60](https://github.com/cklxx/arle/issues/60), [#61](https://github.com/cklxx/arle/issues/61)) | True batched lowering per [`unified-batched-kvpool-abstraction`](docs/plans/2026-06-07-unified-batched-kvpool-abstraction.md) (`KvBatchDescriptor` + `ModelKvAdapter`, DSv4 first). `cd421794` (sequential plan-split, c≥2 no longer crashes) is the starting point, not the goal. | c-sweep clears TTFT+ITL+tok/s per bench spec; then deepep_ll-vs-allreduce A/B at its real lane, license-or-kill. | v2 §3 Phase 1 |
| **2 — Spec decode default-good** | Queued ([#70](https://github.com/cklxx/arle/issues/70) kernel-base convergence first, then [#62](https://github.com/cklxx/arle/issues/62)). #70's graph/FlashMLA capture-safety work is in flight (`4b835fa4`…`e95e11b6`, pending-remote). | Decode kernel-base convergence (whole-step graph + lockstep step-start levers, re-licensed by the [skew anatomy](docs/experience/wins/2026-06-10-dsv4-nsys-skew-anatomy-rewrites-lever-board.md)), then frozen-KV MTP on DSv4 (checkpoint-native draft head, no training; cheap acceptance measurement first — the biggest unverified hypothesis). | Spec-on as default; wall-clock net win at B=1 + long-ctx; H20 target ~8–10 ms/token. | [frozen-KV design](docs/plans/2026-06-06-dsv4-frozen-kv-mtp-redesign.md), v2 §3 Phase 2 |
| **3 — Product re-aim** | Queued ([#63](https://github.com/cklxx/arle/issues/63), [#64](https://github.com/cklxx/arle/issues/64), [#65](https://github.com/cklxx/arle/issues/65), [#71](https://github.com/cklxx/arle/issues/71), [#90](https://github.com/cklxx/arle/issues/90)) | W3/W4 cross-engine baseline (owed since 2026-05-02), long-ctx mission restart on the new substrate, OPD GPU experiments resume, Qwen3.6 CUDA via the second `ModelKvAdapter`, **AIPC route** (Metal single-user convergence + HIP/ROCm third backend on the seam — local unified-memory hardware, no pod contention), **SOPD self-training axis** ([#90](https://github.com/cklxx/arle/issues/90) — teacher-free LoRA self-distill at rollout time; OPD-only, never GRPO; gated on its own Phase-0 premise-test, off the serial critical path — see below). | Per-item; mission threshold ≥1.30 stands. | v2 §3 Phase 3, [SOPD plan](docs/plans/2026-06-14-self-training-lora-opd-sopd.md) |

Off-path / opportunistic (no serial-phase contention):
[#69](https://github.com/cklxx/arle/issues/69) DSv4 serve cold-boot ~6 min
(rank-0 serialization + 8× read amplification);
[#90](https://github.com/cklxx/arle/issues/90) SOPD Phase-0 premise-test —
a cheap CUDA+Qwen3.5 probe of the inline self-update loop, runnable on the pod
off the Phase-1 critical path (zero new kernels; 4 of 5 architecture cruxes
proven dormant at this scope). Everything after its Phase-0 is gated on a PASS
and does not pre-empt the batched-lane keystone ([#60](https://github.com/cklxx/arle/issues/60)).

Killed/deferred work (B=1 per-kernel micro-levers, deepep_ll default-on,
classical spec, 5–6 ms-on-H20 framing, FlashInfer migration, ROCm, …) is
enumerated in master strategy v2 §5 — re-doing a KILLED item requires
overturning its evidence first. Precedent: the whole-step decode graph
lever was re-licensed 2026-06-10 by the
[nsys skew anatomy](docs/experience/wins/2026-06-10-dsv4-nsys-skew-anatomy-rewrites-lever-board.md)
(launch-gap drizzle measured at 29% of wall) and is tracked in
[#70](https://github.com/cklxx/arle/issues/70); the per-kernel/alloc/
host-overhead micro-lever KILL stands.

## Next-Model Priority Order

Currently shipped: Qwen3.5-family (CUDA + Metal), DSv4-Flash (CUDA 8×H20).
The model-coverage queue is ranked, not parallel:

1. **DeepSeek V4 (DSv4-Flash)** — active substrate (Phase 0–2 above).
2. **Qwen 3.6** — second priority. Metal canonical model today; CUDA
   serving lands as the second `ModelKvAdapter` (Phase 3).

Other families in the support matrix sit behind these two and are not
actively scheduled.

Backend queue: CUDA + Metal are shipped. **HIP/ROCm + Vulkan** are the AIPC
lane ([#71](https://github.com/cklxx/arle/issues/71), #76/#77): code began
landing 2026-06-10/11 ahead of the Phase 3 ordering (`infer-hip`,
`infer-vulkan` + `hip-/vulkan-{sys,kernels}`; plans
[`2026-06-10-hip-backend-mvp.md`](docs/plans/2026-06-10-hip-backend-mvp.md),
[`2026-06-11-hip-onbox-runbook.md`](docs/plans/2026-06-11-hip-onbox-runbook.md)).
**Phase-ordering ratification pending** — see
[refactor roadmap §6](docs/plans/2026-06-12-architecture-refactor-roadmap.md).
A `gemma-spec` + Vulkan Gemma4 order pin is also in-tree, **unranked** in the
model queue above (same pending ratification).

## History

Released tags + bench evidence live in:

- [`CHANGELOG.md`](CHANGELOG.md) — per-version notes (latest: v0.1.5)
- [`docs/experience/wins/`](docs/experience/wins/), [`docs/experience/errors/`](docs/experience/errors/) — curated evidence log
- [GitHub Releases](https://github.com/cklxx/arle/releases) — tagged binaries
- `git log` — full history

Use [`docs/index.md`](docs/index.md) to find current documents. Anything not
listed there is not a source of truth.
