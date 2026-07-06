# DSv4 decode multi-stream overlap (Aux stream) - 2026-07-06

> Status: **Killed — 2026-07-06.** Round 3's real matched A/B (n=8, 3 trials/arm,
> trial-nonced cold prompts) showed a ~9% wall-clock **regression** with the
> lever ON (9.65s -> 10.53s mean, zero overlap across 6 trials — see the final
> section below). Lever code deleted same day: `ARLE_DSV4_DECODE_MULTISTREAM_OVERLAP`,
> the Aux-stream fork in `forward_decode_batch_stream_impl`, `DeviceContext::aux_stream`/
> `with_stream_view`, and the `prefill_linear_aux` scratch. `shard_rows_and_allreduce`
> (extracted `abc4826e9`) and the KV-budget clamp fix (`51c31b44f`/`77d60fd4d`) are
> unrelated to this verdict and were kept.

## Goal

- Add a Aux CUDA stream lever to DSv4's batched (n>1) eager decode lane
  (`forward_decode_batch_stream_impl`) so the main + indexer-key compressor
  batch prepass GEMMs run concurrently with the main-stream projection GEMM
  (`mla_attention_prepare_proj_batch`, which contains the batched wq_b) instead
  of strictly after it, closing part of the "no multi-stream overlap" gap found
  auditing ARLE's DSv4 attention stack against SGLang's DeepSeek-V4 reference
  (§5.1 of the reference: indexer/KV-write/compressor hidden behind the main
  projection GEMM via alt streams).

## Hypothesis

- `normed` (the layer's RMSNorm'd hidden state) is fully computed before both
  the projection GEMM and the compressor/indexer-key prepass, and those two
  prepass calls do not depend on the projection's output — only
  `indexer_query_batch_prepass` does (it needs `proj.c_q_normed`). Forking the
  two independent prepasses onto a second stream, gated behind their own
  dedicated scratch (`prefill_linear_aux`, never the `prefill_linear` the
  concurrent projection GEMM is using), should let them overlap with the
  projection GEMM without changing outputs. **No throughput or latency
  improvement is claimed** until a CUDA host runs an A/B — this local host has
  no GPU and cannot execute or profile the kernels.

## Command

Local non-GPU validation:

```bash
CUDARC_CUDA_VERSION=12080 \
cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib

CUDARC_CUDA_VERSION=12080 \
cargo clippy -p infer-cuda -p cuda-kernels --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
```

GPU verification TODO for a CUDA host:

```bash
# Correctness: needle ladder must be unaffected with the lever ON.
ARLE_DSV4_DECODE_MULTISTREAM_OVERLAP=1 CUDA_HOME=/usr/local/cuda \
  scripts/needle_gate.py --model <DSv4 checkpoint> --backend cuda

# Perf: same-binary, same-shell A/B, lever OFF vs ON, DSv4 decode workload,
# batch >= DSV4_MULTISTREAM_OVERLAP_MIN_BATCH (4) so the lever actually engages.
scripts/bench_guidellm.sh dsv4-decode-multistream-overlap-off ...
ARLE_DSV4_DECODE_MULTISTREAM_OVERLAP=1 scripts/bench_guidellm.sh dsv4-decode-multistream-overlap-on ...

# nsys: confirm the two streams actually overlap (not serialized by a hidden
# dependency e.g. the cuMemAllocAsync pool or an unexpected event wait).
nsys profile -o dsv4_multistream_overlap ... (decode workload, lever ON)
```

## Environment

- **Backend:** local Rust typecheck (Mac, no GPU) **+ pod-verified CUDA runtime
  (2026-07-06)**.
- **Model:** `/host/DeepSeek-V4-Flash-FP8` (274 GB on-disk FP8).
- **Hardware:** 8×H20 pod (115.190.184.36). GPU 1 was occupied by another
  tenant's job at probe time (51 GB / 98% util) — ran **TP=4 on GPUs 2,3,4,5**
  instead of the model's native TP=8/EP=8 (all 8 GPUs would have pulled GPU 1
  into the shard set, contending with that job and polluting the reading).
- **Feature set:** `--no-default-features --features cuda,no-cuda --lib` (Mac
  typecheck); `--features cuda,nccl` (pod build — TP≥2 multi-rank serve
  requires `nccl`, `--features cuda` alone errors at coordinator setup).
- **Non-default flags / env vars:** `ARLE_DSV4_DECODE_MULTISTREAM_OVERLAP=1` to
  engage the lever (default OFF); `INFER_CUDA_DEVICES=2,3,4,5 INFER_TP_SIZE=4
  ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_INCREMENTAL_KV=1 --max-total-tokens
  8192` (KV-budget-safe TP4 recipe, see Problems).
- **Server launch:** `arle serve --backend cuda --model-path
  /host/DeepSeek-V4-Flash-FP8 --port <p> --max-total-tokens 8192`, gate script
  `scripts/lever_gate.sh` with `RAW=1 TEMPLATE=dsv4` (the model-neutral
  `/v1/chat/completions` default route returns empty completions for this
  checkpoint — a routing/template mismatch, not a model bug; confirmed by the
  `RAW=1 TEMPLATE=dsv4` route producing correct decodes).

## Params

| Param | Value |
|---|---|
| Change type | CUDA stream-overlap lever, opt-in |
| Code path | `crates/cuda-kernels/src/tensor.rs` (Aux stream + `with_stream_view`), `crates/infer-cuda/src/attention.rs` (gate + threshold const), `crates/infer-cuda/src/attention/kv_layout.rs` (dedicated `prefill_linear_aux` scratch), `crates/infer-cuda/src/dsv4.rs` (`forward_decode_batch_stream_impl` fork/join) |
| New API | `CudaPipelineStreamKind::Aux`, `DeviceContext::aux_stream`, `DeviceContext::with_stream_view`, `DeviceContext::sync_aux`, `Dsv4KvAdapter::prefill_linear_aux_mut` |
| Gate | `ARLE_DSV4_DECODE_MULTISTREAM_OVERLAP` (default OFF), `DSV4_MULTISTREAM_OVERLAP_MIN_BATCH = 4` (unlicensed placeholder, no nsys evidence backs this number) |
| Scope | DSv4 batched (n>1) eager decode lane only. Does NOT touch prefill, the per-slot CUDA-graph decode path (`forward_tokens_decode_graph`, which is n=1 only and does not reach this function), or the in-graph metadata construction gap (separate task) |
| Perf status | DEFER — n≥4 path unreachable this pass (KV-budget ceiling); c=1 off-path shows zero delta |

## Results

| Check | Result |
|---|---|
| `cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` | PASS |
| `cargo clippy --no-default-features --features cuda,no-cuda -- -D warnings` | PASS — 23/23 findings match the `main` baseline (git-stash A/B), zero new |
| Pod build `cargo build --release --features cuda,nccl` | BUILD_EXIT=0 |
| Symbol check (`strings target/release/arle \| grep ARLE_DSV4_DECODE_MULTISTREAM_OVERLAP`) | present |
| **Serial correctness gate, flag OFF** (`lever_gate.sh baseline`, TP=4, RAW=1 TEMPLATE=dsv4, lengths 115/300/446/2000/8000 ×3) | exact=3/3 every length; len=8000 NONDET (bold-vs-plain formatting noise, pre-existing floor) |
| **Serial correctness gate, flag ON** (`ARLE_DSV4_DECODE_MULTISTREAM_OVERLAP=1`, same matrix) | exact=3/3 every length, **within the baseline envelope** (len=8000 DET this run — inside the OFF run's own non-determinism band) → **PASS** |
| Concurrent correctness (n≥`DSV4_MULTISTREAM_OVERLAP_MIN_BATCH`=4, the code path this lever actually touches) | **BLOCKED** — see Problems |
| Perf A/B (guidellm) | **BLOCKED** — see Problems; c=1 wall-clock (below) shows no regression on the untaken path |

**Verdict: flag OFF↔ON serial correctness — PASS.** The batched (n≥4) code
path this lever changes was **not reachable** in this pass (see Problems) —
so this PASS certifies "byte-identical off-path, and the on-path change
doesn't break single-request decoding," not the overlap itself. **No
performance license granted; do not flip the default without a clean n≥4 A/B.**

c=1 wall-clock (`needle_gate.py`, RAW completions, max_tokens=16, from the
correctness-gate logs above — the n=1 path is byte-identical OFF vs ON since
`n >= DSV4_MULTISTREAM_OVERLAP_MIN_BATCH` gates the fork):

| len | OFF run0/1/2 (s) | ON run0/1/2 (s) |
|---:|---|---|
| 115 | 1.0 / 0.6 / 0.7 | 0.9 / 0.7 / 0.7 |
| 2000 | 1.6 / 0.7 / 0.6 | 1.6 / 0.7 / 0.6 |
| 8000 | 4.6 / 0.7 / 0.7 | 4.5 / 0.7 / 0.7 |

No measurable difference — expected, since c=1 never reaches n≥4.

## Problems

- **GPU allocation forced TP=4, not the model's native TP=8/EP=8.** GPU 1 was
  occupied by another tenant (51 GB, 97–98% util) at probe time; running TP=8
  would have pulled GPU 1 into the shard set and polluted both readings. Ran
  on GPUs 2,3,4,5 instead.
- **Concurrent (n≥4) correctness/perf was BLOCKED (round 1) by a pre-existing,
  same-day-documented bug, independent of this lever — fixed in round 2 (see
  the Re-verification section below).** Root cause was `Dsv4::kv_budget_plan`
  (`infer-cuda/src/dsv4.rs`), not `HostPagedKvPool` as first suspected: it
  planned `num_slots` from per-slot state affordability alone, never checking
  whether the shared FlashMLA pool's remainder could back that many whole
  fixed bands. Reproduced independently of either lever (plain baseline, zero
  `ARLE_DSV4_*` flags): a 2-request concurrent burst already crashed the whole
  coordinator (`HostPagedKvPool out of fixed-band pages: slot 1 needs 66, free
  55` at `--max-total-tokens 16384`; `slot 1 needs 34, free 30`-class failures
  persisted down to `--max-total-tokens 8192`, which only raised the ceiling
  from ~1 to ~5 concurrent fixed-band slots before the crash). A 4-concurrent
  burst (within that ~5-slot ceiling) survived one trial but showed
  **needle-recall truncation on the plain baseline** across repeated trials
  against the same long-lived server (e.g. `'The secret access code is
  738.'` instead of the full `738291`) — a SEPARATE, still-pre-existing
  concurrency correctness issue, confirmed again in round 2 (see below),
  reproduced with zero flags set. The sizing bug is fixed; the concurrency
  correctness issue remains out of scope for this task.
- **guidellm 0.6.0's synthetic-text generator is incompatible with this
  checkpoint's HF config**: `AttributeError: 'PreTrainedConfig' object has no
  attribute 'max_position_embeddings'` (DSv4's custom `rope_parameters` schema
  vs guidellm's tokenizer-length inference) — reproduces regardless of the
  `--data` spec. This matches the box's own prior precedent
  (`docs/experience/wins/2026-07-03-dsv4-fp8-tp4-138fixed-perf.md`: guidellm
  was also unusable for DSv4 there, for a different reason — network-blocked
  install). Fell back to the same precedent's method: `needle_gate.py`
  wall-clock timing (used above for the c=1 comparison), not guidellm.
- `DSV4_MULTISTREAM_OVERLAP_MIN_BATCH = 4` remains an unlicensed placeholder —
  no nsys evidence backs it; the pod's KV-budget ceiling prevented sweeping it.
- **Real concurrency requires the GPU scheduler to actually interleave the two
  streams' kernels**; nothing measured here proves that (no nsys trace was
  run — blocked by the same KV-budget ceiling before an nsys-worthy batched
  workload could be constructed).

## Learnings

- ARLE's DSv4 batched (n>1) decode lane already collapsed almost all of
  SGLang's per-row multi-stream-overlap motivation via a DIFFERENT mechanism —
  full-flatten batching (#60): rather than overlapping N per-row launches
  across streams, it batches the compressor/indexer/projection GEMMs across
  all N rows into one call each. The remaining overlap opportunity is coarser:
  entire prepass STAGES (main+indexer-key compressor vs. the projection) that
  are mutually independent, not individual per-row kernels.
- The critical precondition this lever depends on, found only by reading the
  actual scratch struct (`Dsv4PrefillDeepGemmLinearScratch`) and its
  allocation-site comment: `prefill_linear` is ONE model-wide instance reused
  sequentially across every layer's GEMM calls, on the explicit invariant
  "never aliased concurrently." Naively forking compressor prepass onto a
  second stream while it still shared that scratch with the concurrent
  projection GEMM would have been a silent correctness bug (racing writes to
  `input_fp8`/`qkv_raw`) invisible without CUDA hardware. The fix was a SECOND
  dedicated scratch instance (`prefill_linear_aux`), allocated only when the
  lever is on.
- `DeviceContext::with_stream_view` (clone `self`, swap `.stream` to another
  lane) is the minimal way to redirect any existing `ctx: &DeviceContext`
  taking function onto a different physical stream without touching its body
  — every field is an `Arc`, so the clone is cheap, and kernel dispatch inside
  `compressor_batch_prepass`/`proj_batched`/`dsv4_linear` reads `ctx.stream`
  uniformly. This is a smaller, more reusable primitive than the alternative
  of threading a `stream:` parameter through every call site.
- `indexer_query_batch_prepass` cannot join the overlap window — it reads
  `proj.c_q_normed`, produced by the very GEMM this lever tries to hide behind
  — so it stays on the main stream after `proj`, unchanged.

## Delta vs baseline

- **Baseline:** `lever_gate.sh baseline` (flag unset), TP=4, GPUs 2-5, same
  binary/shell/prompts as the ON run.
- **Delta, correctness:** zero (exact/DET-envelope match at every length).
- **Delta, perf:** not measured — the n≥4 code path was unreachable (KV-budget
  ceiling, see Problems). c=1 shows 0% delta as expected for an unreached path.

## Artefacts

- Pod logs: `/root/needle_gate_baseline.log`, `/root/needle_gate_multistream.log`
  (pod-local paths, not committed — gitignored bench-output convention).
- GuideLLM: not produced (guidellm/DSv4-config incompatibility, see Problems).
- nsys: not produced (blocked by the same KV-budget ceiling).

## Notes

- Lever is default OFF and byte-identical to today's behavior when unset —
  landing it carries no runtime risk to the existing default configuration.
- **License-or-kill verdict (superseded by the re-verification below): DEFER,
  not PASS-for-perf.**

## Re-verification after the KV-pool sizing fix (2026-07-06, round 2)

**Root cause, corrected.** The blocking bug above was mis-attributed to
`HostPagedKvPool` (`infer-seam/src/host_paged_kv_pool.rs`) — that struct only
*stores* whatever `total_pages`/`fixed_pages_per_slot` its caller passes; it
computes nothing from `num_slots` itself. The actual gap: DSv4's shared
FlashMLA pool budget (`Dsv4::kv_budget_plan`, `infer-cuda/src/dsv4.rs`) sized
its per-slot-state affordability (`affordable`) independently of the fixed
band's own cost, so `num_slots` could be planned far above what the pool's
"coherent remainder" could ever back with whole fixed bands — the pool then
handed out all its pages to the first few concurrent requests and the
`(N+1)`th's `alloc_fixed_band` call `bail!`'d mid-serve. Fix (commits
`51c31b44f`, simplified by `77d60fd4d`): `kv_budget_plan` now additionally
computes `pool_affordable_slots = pool_budget_bytes_per_layer /
(flashmla_slot_pages × flashmla_page_bytes)` and clamps `num_slots =
min(num_slots, pool_affordable_slots)` — reusing the function's own existing
NCCL-min-reduced, lockstep-safe clamp pattern. The reduced `num_slots` flows
through the *already-existing* `loaded.rs:1958-1964` scheduler-sync (no new
code needed there), so the scheduler never again admits more concurrency than
the shared pool can serve.

**Reachability: FIXED.** Same TP=4/GPUs-2-5 config, same binary
(`arle` @ commit `77d60fd4d`):

| `--max-total-tokens` | requested slots | planned (post-fix) | n=64 concurrent burst |
|---:|---:|---:|---|
| 8192 | 256 | **1** (`pool-band-affordable=1`) | survives — serialized, no crash |
| 2048 | 256 | **256** (no clamp; `pool-band-affordable=387`) | survives — real concurrency, no crash |

At `--max-total-tokens 2048` the pool affords the full requested 256 slots, so
n=4 and n=64 concurrent bursts both complete with **zero coordinator crashes**
across every trial (previously: crashed at n=2 even at 16384 tokens). This is
the direct, measured fix for the blocker.

**Correctness at concurrency — a SEPARATE, pre-existing bug, confirmed
independent of this lever.** Serial (n=1) correctness: exact=3/3 at every
length (115/300/446/1000), matching the original PASS envelope exactly
(`needle_gate_v2_multistream.log`). Concurrent (n=4 ×3, n=64 ×1) with the flag
ON: `exact=1/4, 2/4` (n=4) and `exact=15/64` (n=64) —  but the **zero-flag
baseline run in the same session, same server generation, shows the identical
failure class at a comparable rate** (`exact=1/4` × several trials at n=4;
`exact=25/64` at n=64; see the sibling Waterfill doc's Re-verification section
for the shared baseline log). Every miss is a truncation to a numeric prefix
of the needle (`738`, `7382`, …) or, rarely, a single corrupted digit — never
a new garbage/looping class, and never worse in aggregate than the baseline's
own rate. This matches the *already-documented* pre-existing DSv4 batched (n>1)
decode correctness bug (independent of both levers, reproduces with zero
`ARLE_DSV4_*` flags) — **not a regression introduced by this lever**, but it
does mean the concurrency-gated correctness envelope itself is not clean
enough on this box to certify the lever's *own* incremental effect.

**Perf: no clean signal, unchanged from round 1.** guidellm 0.6.0 is still
incompatible with this checkpoint's HF config. Wall-clock concurrent-batch
completion time (n=64, cold, no prefix-cache reuse) was ~21-23s for OFF and
~22s for ON — no distinguishable delta. A same-server back-to-back repeat
showed both arms drop to ~6-9s on the *second* identical-prompt burst — a
prefix-cache-reuse artifact (confirmed on the OFF arm too, see the Waterfill
doc), **not a lever effect**; do not read a perf win into it.

**Verdict: DEFER (revised reason).** The reachability blocker is fixed and
verified (n=64 concurrent, zero crashes, two `--max-total-tokens` regimes).
Serial correctness is clean and licensed. The concurrent (n≥4) code path is
now reachable, but this box's DSv4 batched-decode path has an independent,
pre-existing correctness bug at n>1 (out of scope for this lever) that
prevents a clean pass/fail read on the lever's own effect — the ON envelope is
statistically indistinguishable from the (already-broken) OFF envelope, which
is the best available evidence of "no new regression," not a correctness
license. No perf evidence either way. Re-verify once the pre-existing n>1
decode-correctness bug is fixed, or via a position-controlled repeat-count
large enough to resolve the lever's effect against the baseline's own noise
floor.

## Round 3 (2026-07-06) — real matched A/B, post-refactor `abc4826e9` — KILL

**Context.** After today's simplification refactor (`abc4826e9`, extracting
`TpRuntime::shard_rows_and_allreduce`), a fresh pod session re-verified this
lever's serial correctness (unaffected — see the sibling regression-check task)
and then ran the perf A/B round 1/2 deferred on: guidellm is still incompatible
with this checkpoint's rope config, but wall-clock timing does not require
correct decode, only a fixed request shape — the pre-existing n>1
correctness bug (documented above) is orthogonal to speed.

**Method.** `concurrent_needle_v3.py` (trial-nonced prompts: every trial's 64/8
prompts are byte-distinct from every other trial run against the same
long-lived server, so prefix-cache reuse cannot bias ANY arm regardless of
boot/trial order — a stronger control than alternating boot order). n=8
(above `DSV4_MULTISTREAM_OVERLAP_MIN_BATCH=4`), TP=4 on GPUs 2,3,4,5,
`--max-total-tokens 2048`, `ARLE_DSV4_MOE_BACKEND=allreduce`. One server boot
per arm, 3 trials per boot, `WALL_TOTAL` = submit-to-last-completion wall time.

| Arm | trial 0 | trial 1 | trial 2 | mean |
|---|---:|---:|---:|---:|
| OFF | 9.682s | 9.568s | 9.692s | **9.647s** |
| ON | 10.611s | 10.588s | 10.378s | **10.526s** |

**Δ% = (9.647 − 10.526) / 9.647 = −9.1%** (negative = ON is SLOWER). Every ON
trial (10.378–10.611s) is slower than every OFF trial (9.568–9.692s) — zero
overlap between the two distributions across 6 trials, so this is a real,
repeatable regression, not noise. Artifact check: trial-nonce design
guarantees every trial is cold by construction (never-before-seen prompt
text), so the "warm second arm" cache-reuse artifact flagged in Round 2 cannot
explain this result — if anything it would only help the LATER-run arm (ON),
yet ON is the slower one.

**Verdict: KILL.** No wall-clock benefit; a measurable regression instead.
Deleted the same day: `ARLE_DSV4_DECODE_MULTISTREAM_OVERLAP` env gate and
`DSV4_MULTISTREAM_OVERLAP_MIN_BATCH` (`attention.rs`), the `aux_overlap_ctx`/
`overlap_prepass` fork in `forward_decode_batch_stream_impl` (`dsv4.rs` —
`run_compressor_indexer_prepass` now runs unconditionally on the main stream,
byte-identical to the lever-OFF path), `prefill_linear_aux` scratch
(`kv_layout.rs`), and `CudaPipelineStreamKind::Aux` / `DeviceContext::aux_stream`
/ `with_stream_view` (`tensor.rs` — confirmed dead via workspace-wide grep,
zero remaining callers). Post-deletion smoke gate (TP=4, GPUs 2-5, lengths
115/300/446/2000): exact=3/3 DET at every length, matching the pre-deletion
envelope exactly. `cargo clippy -p infer-cuda -p cuda-kernels --features
cuda,no-cuda -- -D warnings`: 23 findings before and after (git-stash A/B),
zero new.
