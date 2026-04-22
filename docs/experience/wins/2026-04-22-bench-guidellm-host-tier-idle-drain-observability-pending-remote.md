# Prefix-skip and tier-recall observability pending canonical bench

## Goal

- Regression guard: ship request-level prefix reuse and slower-tier recall
  counters so the next canonical CUDA bench can explain KV skip rate, staged
  tier mix, promotion recall, and staged-prefix fallback instead of inferring
  them indirectly from logs.

## Hypothesis

- Extending `ServerMetrics` and wiring the scheduler's staged-prefix path into
  those counters should not intentionally change scheduling behavior, but it
  should make the next `Qwen3` / `Qwen3.5` trace decisive about whether the
  remaining throughput and OOM issues come from real tier activity or cold
  prefill.

## Command

```bash
scripts/bench_guidellm.sh prefix-tier-observability
```

Invoked via: `scripts/bench_guidellm.sh prefix-tier-observability`

## Environment

- **Backend:** CUDA
- **Model:** pending local rerun after this observability tranche lands
- **Hardware:** NVIDIA L4
- **Commit:** local workspace on `2026-04-22`
- **Feature set:** `cargo build --release -p infer --bin infer`
- **Non-default flags / env vars:** `ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig`
- **Server launch:** pending next canonical rerun

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| pending-local-rerun | pending-local-rerun | pending-local-rerun | pending-local-rerun | pending-local-rerun | pending-local-rerun | pending-local-rerun |

## Problems

- This tranche is observability-only, but the repo contract still requires a
  bench record. The canonical local `Qwen3 c16` rerun was started and then
  aborted to honor a user-requested `pull --rebase` + push cycle, so this entry
  remains `pending-local-rerun`.
- The new counters therefore ship today with tests + release build coverage,
  and the first post-rebase `Qwen3 c16` / `Qwen3.5` trace should update this
  stub with real numbers.

## Learnings

- Prefix-hit rate alone was too weak: it could tell us whether any request hit
  the radix cache, but not how many prompt tokens were skipped.
- Tier queue depth alone was also too weak: it could show whether the
  coordinator was busy, but not whether staged-prefix plans were coming from T1
  vs T2/T3, how much of that staged data was actually promoted, or how often we
  fell back to cold prefill.
- The new metrics surface closes those gaps without creating a parallel stats
  path.

## Δ vs baseline

- Baseline: prior `ServerMetrics` surface with only `prefix_hit_rate` and
  coordinator queue counters.
- Delta table: pending-local-rerun

## Artefacts

- Raw: `pending-local-rerun`
- CSV: `pending-local-rerun`
- HTML: `pending-local-rerun`
- Service trace: `pending-local-rerun`
- Server log: `pending-local-rerun`

## Notes

- Local verification completed:
  - `cargo fmt --all`
  - `cargo test --release -p infer --lib metrics:: -- --nocapture`
  - `cargo test --release -p infer --lib scheduler::cuda:: -- --nocapture`
  - `cargo build --release -p infer --bin infer`
- Related diagnosis:
  - `docs/experience/errors/2026-04-22-host-tier-idle-drain-local-trace-no-store-submit.md`
- The next canonical rerun should read:
  - `prefix_skip_rate`
  - `tier_src=h:/d:/r:`
  - `tier_recall`
  - `tier_promoted`
  - `tier_fallback`
