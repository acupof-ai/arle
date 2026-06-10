# ARLE Roadmap

Updated 2026-06-10. Derived planning surface. On any conflict the canonical
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

| Phase | Goal | Exit condition | Anchor |
| --- | --- | --- | --- |
| **0 — Debt** | Long-ctx correctness closeout (seq≥241 residual, same-config-twice control), 256K admission band-aid → real fix, KV-precision-parity audit re-port to `infer-cuda`, truth-surface resync (this doc series). | All four items closed; parity harness unlocks the gated FlashMLA/fused-wqkv/contig-MoE default flips. | v2 §3 Phase 0 |
| **1 — Batched serving lane (keystone)** | True batched lowering per [`unified-batched-kvpool-abstraction`](docs/plans/2026-06-07-unified-batched-kvpool-abstraction.md) (`KvBatchDescriptor` + `ModelKvAdapter`, DSv4 first). `cd421794` (sequential plan-split, c≥2 no longer crashes) is the starting point, not the goal. | c-sweep clears TTFT+ITL+tok/s per bench spec; then deepep_ll-vs-allreduce A/B at its real lane, license-or-kill. | v2 §3 Phase 1 |
| **2 — Spec decode default-good** | Frozen-KV MTP on DSv4 (checkpoint-native draft head, no training). First step: cheap acceptance measurement on coherent workloads (GSM8K/ShareGPT) — the biggest unverified hypothesis. | Spec-on as default; wall-clock net win at B=1 + long-ctx; H20 target ~8–10 ms/token. | [frozen-KV design](docs/plans/2026-06-06-dsv4-frozen-kv-mtp-redesign.md), v2 §3 Phase 2 |
| **3 — Product re-aim** | W3/W4 cross-engine baseline (owed since 2026-05-02), long-ctx mission restart on the new substrate, OPD GPU experiments resume, Qwen3.6 CUDA via the second `ModelKvAdapter`. | Per-item; mission threshold ≥1.30 stands. | v2 §3 Phase 3 |

Killed/deferred work (B=1 per-kernel levers, deepep_ll default-on,
classical spec, 5–6 ms-on-H20 framing, FlashInfer migration, ROCm, …) is
enumerated in master strategy v2 §5 — re-doing a KILLED item requires
overturning its evidence first.

## Next-Model Priority Order

Currently shipped: Qwen3.5-family (CUDA + Metal), DSv4-Flash (CUDA 8×H20).
The model-coverage queue is ranked, not parallel:

1. **DeepSeek V4 (DSv4-Flash)** — active substrate (Phase 0–2 above).
2. **Qwen 3.6** — second priority. Metal canonical model today; CUDA
   serving lands as the second `ModelKvAdapter` (Phase 3).

Other families in the support matrix sit behind these two and are not
actively scheduled.

## History

Released tags + bench evidence live in:

- [`CHANGELOG.md`](CHANGELOG.md) — per-version notes (latest: v0.1.5)
- [`docs/experience/wins/`](docs/experience/wins/), [`docs/experience/errors/`](docs/experience/errors/) — curated evidence log
- [GitHub Releases](https://github.com/cklxx/arle/releases) — tagged binaries
- `git log` — full history

Use [`docs/index.md`](docs/index.md) to find current documents. Anything not
listed there is not a source of truth.
