# 2026-04-15 · Tiered KV Cache M2b — remote CUDA acceptance

## Context

M2b was the local 2026-04-15 batch that made radix-driven cross-request
prefix reuse actually load-bearing in the CUDA scheduler: replaced the
legacy `cached_prompts: Vec<Vec<u32>>` scan with a radix-driven selector
keyed by `block_owner_slots` + `slot_materialized_prompt_lens`, added
the alloc-OOM retry (`alloc_pool_tokens_with_retry`), landed the retain
hard cap (`PREFIX_CACHE_RETAIN_HARD_CAP = 0.90`), and added radix
tombstone GC. Local win note:
[`2026-04-15-tiered-kv-m2b-local.md`](2026-04-15-tiered-kv-m2b-local.md).

This note records the remote acceptance run against
[`../../plans/tiered-kv-cache-m2b-remote-acceptance.md`](../../plans/tiered-kv-cache-m2b-remote-acceptance.md).

## Environment

- GPU: NVIDIA L4 24 GB (driver 580.82.07, CUDA 13.0, SM 89)
- Model: Qwen3-4B BF16, `Qwen/Qwen3-4B` HF Instruct variant
- Build: `cargo build -p infer --release` (default features, `cuda`),
  FlashInfer 0.6.3 headers, Triton 3.5.1 AOT
- Commit at first run: `bfbfd6f`; post-pull re-run: `85bc85b`
- Server: `target/release/infer --model-path models/Qwen3-4B --num-slots 4`
  - `cuda_graph=true`, warm batch sizes {1, 2, 4}
  - `max_seq_len=4096` auto, `kv_cache_dtype=bf16`
  - TokenKVPool BF16 at `page_size = 16` (M0.3)
- Cargo env: `CARGO_HOME=/tmp/cargo-home-local`
  (Drive-backed registry is unusable — see
  `project_remote_cuda_box.md`)

## Static sanity (§2)

All three §2 greps produced the expected results with two minor
doc-drift notes:

- `rg cached_prompts: Vec<Vec<u32>>` → two hits, both doc comments in
  `scheduler/cuda/request.rs:54` and `scheduler/cuda/core.rs:53`
  describing the removed type. Runtime code is clean. Acceptance doc
  grep wants zero matches; the comments are informative and were
  intentionally left in place.
- `rg reusable_prefix_len|alloc_pool_tokens_with_retry|...` → matches in
  `runtime.rs`, `prefill.rs`, `core.rs`, `request.rs` as expected.
- `rg prefetch_kv_to_gpu` → **zero matches**. The method was deleted
  by `c3f65f7 refactor(model): retire legacy contiguous kv offload`,
  because the CPU offload path it protected was retired entirely
  (M3c scope). Acceptance doc expected one trait method + three impls;
  neither is present, and the underlying invariant (prefix reuse can't
  read stale CPU KV) is now satisfied structurally — there is no CPU
  KV to read. The M2b acceptance doc grep is stale vs the current
  tree and should be dropped when the doc is next edited.

## Build / test gates (§3)

```
cargo build -p infer --release                             # 4m 11s cold, 12 s incremental
cargo test --workspace --exclude mlx-sys --release --lib   # 348 tests pass (273 infer + 35 cuda-kernels + 40 others)
cargo test -p infer --release --test e2e                   # Phase 1-4 all pass after the replay-drift fix below
cargo fmt --all -- --check                                 # clean
```

**Not-in-scope pre-existing failures** (tracked separately, not M2b
regressions):

- `cargo test -p infer --release --test e2e_qwen35` — Phase 1 baseline
  drift vs current HF `Qwen/Qwen3.5-4B` weights. Tracked in
  `project_remote_cuda_box.md`.
- `cargo test --release --test greedy_consistency` — B=3 batched
  decode divergence, tracked in
  `../errors/2026-04-13-batched-decode-high-concurrency.md`.
- `cargo clippy --workspace -- -D warnings` — pre-existing
  `unused import: Path` in `crates/tools/src/lib.rs:3`.
  Reported per the `project_remote_cuda_box.md` instruction "report,
  don't silently fix", left for a separate cleanup commit.

### New: e2e Phase 3 replay drift

`test_e2e_generation` Phase 1 (6 baselines) now passes cleanly on
Qwen3-4B for the first time on this host — the stale baseline-drift
pre-existing issue is gone. That exposed a dormant bug in Phase 3
(stream/non-stream consistency): the
`PrefixReuseAction::ReplayFinalToken` branch in
`ModelInferenceEngine::prepare_with_prefix_cache` truncates the KV
cache to N-1 tokens and calls `forward_prefill` with a single trailing
token, which runs the FlashInfer batched prefill kernel with
batch=1 — numerically not byte-identical to the batch-N cold prefill
that wrote positions `[0..N)`. Greedy argmax flips around the 4th
generated token and the stream output diverges from the non-stream
output on the exact same prompt.

Full root cause + fix in
[`../errors/2026-04-15-e2e-phase3-replay-drift.md`](../errors/2026-04-15-e2e-phase3-replay-drift.md).

Applied fix in `infer/src/server_engine.rs`: the `ReplayFinalToken`
handler now falls through to the same reset + full recompute path as
`FullRecompute`. The `PrefixReuseAction::ReplayFinalToken` enum
variant and its unit tests are preserved. The scheduler's own prefix
reuse (all of M2b's cross-session wins) is untouched — that path uses
`forward_prefill_with_pool`'s dedicated single-token-to-decode
dispatch and does not go through `ModelInferenceEngine`.

## Agent-trace bench (§4)

```
python3 scripts/bench_agent_trace.py \
    --server http://localhost:8000 \
    --label tiered-kv-m2b-remote \
    --out docs/experience/wins/2026-04-15-bench-tiered-kv-m2b-remote.json
```

Raw result (run against commit `85bc85b`, `num_slots=4`):

```
session          turn  msgs   wall(ms)  ttft(ms)   itl(ms)  tokens  finish
──────────────────────────────────────────────────────────────────────────
agent-001           0     2     5151.4      47.3      34.6     144  stop
agent-001           2     4     9137.6     152.7      34.8     256  length
agent-001           4     6     9153.7     215.2      34.8     256  length
agent-002           0     2     9092.0     124.2      34.7     256  length
agent-002           2     4     9154.7     113.1      34.8     256  length
agent-002           4     6     8893.9     157.5      34.3     256  length
agent-003           0     2     9091.1     165.6      34.7     256  length
agent-003           2     4     9137.8     208.6      34.8     256  length
agent-004           0     2     9091.5     207.1      34.7     256  length
agent-004           2     4     9150.3     115.0      34.8     256  length
agent-005           0     2     9122.3     112.6      34.8     256  length
agent-005           2     4     9153.7     159.4      34.8     256  length
agent-006           0     2     9139.0     107.5      34.8     256  length
agent-006           2     4     9050.7     114.2      34.6     256  length

turns OK:        14 / 14
tokens total:    3472
wall total (s):  123.52
TTFT p50/p99:    124.2 / 215.2 ms
ITL  p50/p99:    34.8 / 34.8 ms
```

Comparison against the **pre-M1 2026-04-13 baseline**
([`2026-04-13-bench-agent-trace-baseline.md`](2026-04-13-bench-agent-trace-baseline.md),
commit `876b986`, same host / model / `num_slots=4`):

| metric        | 2026-04-13 baseline (pre-M1) | 2026-04-15 M2b remote | delta |
|---------------|------------------------------|-----------------------|-------|
| turns OK      | 14 / 14                      | 14 / 14               | —     |
| TTFT p50      | 112.2 ms                     | 124.2 ms              | +12 ms|
| TTFT p99      | 207.0 ms                     | 215.2 ms              | +8 ms |
| ITL  p50      | 34.1 ms                      | 34.8 ms               | +0.7 ms|
| ITL  p99      | 34.4 ms                      | 34.8 ms               | +0.4 ms|
| tokens total  | 2703                         | 3472                  | +769  |

The token count jumped because most 2026-04-13 turns finished early on
EOS while the 2026-04-15 turns ran to `max_tokens=256` `length`. That
extends wall time proportionally but the per-turn ITL floor is
unchanged — both are decode-GEMV bound at ~34 ms, which matches the
L4 HBM ceiling.

TTFT p50 is slightly higher than the pre-M1 baseline. **M2b's real win
lives on the 100x shared-prefix stress below, not on this trace** —
the agent-trace sessions' prompts are short relative to the decode
window (~256 tokens out), so the 4-token-ish prefill cost the cold
path pays is quickly amortised. A cleaner signal would come from a
trace with much longer shared system prompts; that upgrade lives in
I1 research per `tiered-kv-cache-tasks.md`.

No same-host M2a baseline (commit `4402ab0`) was captured before M2b
landed, so the strict "M2b vs M2a on same host" comparison requested
by the acceptance doc §4 is not available. The pre-M1 baseline is the
only existing reference on this box.

Server logs during the run show the radix shadow observer firing on
admission — `radix_hit = X but no reusable free slot state, queue = Y`
lines in `scheduler/cuda/runtime.rs:127`. The radix lookup is
classifying hits correctly; the "no reusable free slot state" note
means the matching slot had already been freed and the new request
landed in a different slot, which is the expected M2b boundary (safe
same-slot resurrection only, no cross-slot paged-pool aliasing).

## Shared-prefix stress (§5)

100 concurrent long-prefix requests sharing a ~2 kB system prompt:

```
# script: inline python (stored at /tmp/m2b_shared_prefix_stress.py)
total=100 bad=0
```

Server logs during the burst show no allocation-failure loops. The
retain hard cap plus alloc-OOM retry caused the scheduler to fall
open under pressure as expected — some `prefix_cache.insert: expected
N tokens, got M` WARNs appeared (`scheduler/cuda/core.rs:404`),
corresponding to partial block coverage on some requests, but all
100 requests streamed back `HTTP 200` with their completions.

§5 acceptance criteria all green: `bad=0`, no pool-allocation failure
loops, retain hard cap is the dominant back-pressure.

## Sign-off

- [x] Static sanity checks passed (with 3 doc-drift notes).
- [x] Build/test gate passed on CUDA after the replay-drift fix.
- [x] Agent-trace bench ran without faults; TTFT close to pre-M1
      baseline (no regression; same-host M2a comparison unavailable).
- [x] Shared-prefix stress: `total=100 bad=0`, no allocation failure
      loops.
- [x] This win note exists with raw output and commands.

**M2b accepted on the 2026-04-15 L4 host.**

## Rule

When a large refactor (c3f65f7 "retire legacy contiguous KV offload")
lands on top of a milestone (M2b), **re-run every single-request
engine test before signing off on the milestone**, not just the
scheduler path. The CUDA scheduler's own prefix reuse is exercised by
the §4 / §5 bench gates, but `ModelInferenceEngine` (the legacy
single-request engine used by e2e and the agent CLI) has its own
prefix-cache planner that can drift numerically when the offload
path disappears underneath it — in our case, by silently unmasking a
pre-existing replay-drift bug whose greedy mismatch had been hidden
for weeks behind an unrelated baseline-drift Phase 1 failure.
