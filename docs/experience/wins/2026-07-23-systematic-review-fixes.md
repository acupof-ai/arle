# Systematic defect-review sweep — 26 findings fixed

> Status: Active

## Context

Full-runtime defect audit (26 scoped module reviews, each finding adversarially
re-verified against the source). 42 raw findings → 26 confirmed after a skeptic
pass refuted 16 (metal both targets, train-opd main loop, cuda-model,
shared-leaves, server-tokenizer came back clean). All 26 fixed here.

Severity of confirmed: 2 high, 9 medium, 15 low. No critical.

## What Worked

Fixes grouped by crate (each verified — see Verification):

**infer-core** (scheduler/cache correctness + memory):
- H1 `radix.rs` — evicted radix nodes never reclaimed → unbounded `nodes` growth
  + O(uptime) eviction scans. Added a `free: Vec<usize>` free-list: `sever_subtree`
  clears + pushes the slot, `insert` pops before appending. New test
  `evict_then_insert_reuses_freed_slot_and_matches`.
- M1 `lib.rs` — `cancel_request` / admission-reject dropped a parked request's
  `swap_key` without releasing the backend tier-store image (leak). Release via
  `executor.drop_kv_slot_entries(&[key])` at every waiting-drop site.
- M2 `lib.rs` — `completed` BTreeMap grew unbounded. `record_completed` helper
  caps it at 1<<16 (monotonic handles ⇒ pop_first drops the coldest); all four
  insert sites routed through it.
- LOW `lib.rs`/`prefix.rs`/`planner.rs` — deleted the dead reuse-priority ordering
  (`WaitingRequestHint` was always default at enqueue); admission is priority+FIFO.

**infer-server** (serving):
- H2 `lib.rs` — relay worker MutexGuard held across `relay_stream` collapsed
  streaming to serial. Scoped `recv()` so the guard drops first.
- LOW `coordinator.rs` — OpenAI streaming keepalive/disconnect (5s SSE-comment
  ping, mirrors /v1/messages); stats/metrics awaiter unregistered on send-fail;
  `std::fs::write` dumps moved to `spawn_blocking`.
- LOW `multiproc_relay.rs` — `accept_n` read-timeout on the WorkerHello read.
- LOW `schema.rs` — `stop` accepts the OpenAI single-string form.

**infer-api / seam / cli**:
- M3 `loaded.rs` — DSv4 ingress prompt cap re-bound to the MLA arena
  (`max_total_tokens`) instead of the default `total_pages`, so long prompts
  above the default-derived cap are no longer wrongly rejected.
- LOW `serve_engine.rs` — streaming deltas now report held-back / non-boundary
  token ids (cursor into `acc_ids`), so the streamed union equals
  `generated_tokens`.
- LOW `resource.rs`/`args.rs`/`loaded.rs` — mem_fraction_static doc band corrected
  to `[0.05, 0.97]`.

**train** (OPD):
- M4 `qwen35_checkpoint.rs` — latest-symlink publish failure no longer deletes the
  fully-written checkpoint (moved out of the cleanup-on-failure scope).
- LOW `update_strategy.rs` — `update_ce` averages over trained trajectories (not
  survivors incl. VRAM-skipped); GlobalTokenMean denominator excludes skipped.
- LOW `metrics.rs` — MlflowSink HTTP agent bounded to 5s so flush can't hang.

**autograd** (OPD carry gradients):
- M7/M8 `linear_attention.rs` — carry `initial_state` now enters the position-0
  decay grad, and the carried conv window contributes to `conv1d_weight`'s grad.
  New finite-diff gradcheck `linear_attention_carry_grad_matches_numeric`
  (a_proj/conv1d_weight/dt_bias/a_log analytic == numeric < 8e-3).

**kv-native-sys**:
- M9 `kv_tier.rs`/`lib.rs` — durable recall tier msyncs data pages then fsyncs the
  manifest (ordering barrier); volatile tier unchanged.

**infer-cuda** (typecheck-only locally — pending-remote for perf/runtime):
- M5 `attention/flashmla.rs` — `ensure(compress_ratio>0)` moved into the
  ratio-dependent arm so GLM SparseIndexed (ratio 0) loads.
- M6 `loader.rs` — BF16 grouped DeepGEMM gated off sm_120 (`&& !sm120`), mirroring
  the FP8 gate, so the `m_indices.expect` panic can't fire.
- LOW `spec_decode.rs` (per-step eprintln gated), `dsv4/dspark.rs` (overlapping
  cuMemcpyDtoDAsync → scratch bounce), `dsv4.rs` (decode-graph GLM exclusion),
  `quant_format.rs` (div-by-zero guard before the divide).

## Verification

- `cargo test -p infer-core` — 107 passed / 0 failed (incl. H1 free-list + swap_key).
- `cargo test -p autograd --test test_linear_attention` — 5 passed (incl. new
  carry gradcheck; numerically confirms M7/M8).
- `cargo test -p kv-native-sys --profile release-fast` — 31 passed (incl. M9 flush).
- Device-neutral compile + `clippy -D warnings` clean (arle/infer-core/seam/
  server/api/train/cli, cpu,no-cuda).
- infer-cuda: `CUDARC_CUDA_VERSION=12080 cargo check -p arle --features cuda,no-cuda`
  + clippy clean (typecheck only; nvcc/perf pending-remote H20).

## Bench / pod verification (2026-07-24)

CUDA half verified on **H20 sm_90** (DSv4-Flash-FP8 TP=4/EP=4) and built on
**Colab G4 sm_120** (Blackwell):
- **Build:** BUILD_EXIT=0 on both arches (sm_90 DeepEP + native DeepGEMM incl. the
  cli/train cuda lanes with no Mac typecheck; sm_120 226 AOT objects) — the CUDA
  fixes compile with real nvcc, not just Mac typecheck.
- **Needle gate:** 15/15 exact, 0 miss across 5 rungs — **no correctness
  regression** (matches champion).
- **Throughput c1/c4:** +0.5% / −0.8% vs the 2026-07-19 champion — **perf-neutral**
  on the per-request hot path (attention/MoE/quant/decode-graph, where the CUDA
  fixes live).
- **Regression found + fixed:** the LOW#18 accept_n hello-read timeout leaked into
  the steady-state relay reader → TP=4 c8+ serve teardown (framing desync). Fixed
  in `837b89d39`; pod-confirmed c8 48/48 + c16 64/64, no teardown. See
  errors/2026-07-24-relay-hello-timeout-leak-tp4-teardown.md.
- **Open (not attributed to this sweep):** a reproducible c16 throughput deficit
  (−40% vs 07-19 champion, batching/scaling — c8→c16 1.16× vs champion 1.58×),
  measured on a binary that also carries concurrent non-sweep changes
  (infer-cuda/executor.rs edits). This sweep's scheduler changes are O(1) /
  behavior-equivalent and don't touch batch formation, and c1/c4 were
  perf-neutral, so the mechanism points elsewhere — needs a champion-binary A/B +
  a sweep-isolated build to attribute.

## Rule

Full-runtime review = scoped module reviews + a per-finding adversarial refute
pass; 16/42 raw findings were plausible-but-wrong and only the skeptic caught
them. A silent-gradient fix ships with a finite-diff gradcheck that fails before
and passes after — reasoning alone is not the gate. And a timing-sensitive
multiproc change is invisible to `cargo test` + Mac typecheck + CI (TP=1-only
coverage) — only a pod **TP=N c8/c16 c-sweep** caught the relay teardown.
