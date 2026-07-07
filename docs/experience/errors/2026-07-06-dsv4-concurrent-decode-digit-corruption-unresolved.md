# DSv4 concurrent-decode digit corruption — FlashMLA-lane AND KV-reuse hypotheses KILLED

## Context

Discovered incidentally while pod-verifying two unrelated DSv4 decode levers
(2026-07-06): concurrent decode (n>1) on DSv4 TP=4 shows needle-gate exact
match ~40-47% failure vs clean at n=1 serial. Reproduces identically with
every `ARLE_DSV4_*` lever flag off, on a freshly booted server — predates and
is unrelated to that day's changes.

## Root Cause (localized, not confirmed to file:line)

Decoded ~15 trials of actual failing completions rather than trusting the
aggregate number:

- ~70% of misses: pure truncation, correct prefix, missing only the last 2
  digits of the needle.
- ~30%: digit corruption — first 3 digits always correct, divergence starts
  at digit 4 onward, in every observed failure with zero exceptions.
- Zero garbage/looping output, zero cross-request contamination (no case of
  one row's output containing another row's distinguishable content, even
  under byte-identical prompts across reruns).

Ruled out: MTP/spec-decode (off by default, confirmed via engine log),
cross-request KV/scratch aliasing (no distinguishable-content evidence),
server-state degradation (a fresh boot's first request already fails), pure
concurrency alone (n=8 at a short ~80-token prompt is 40/40 exact clean),
pure length alone (not monotonic — len=1300 clean while len=250/500/900
failed in the same run).

Established: requires **both** n>=3 concurrent decode rows in the same
batched kernel launch **and** prompt length past ~100-250 tokens.

Localized to the DSv4 batched FlashMLA/CSA decode lane — the only path
activating specifically at this joint (n, length) dependency:
`crates/infer-cuda/src/dsv4.rs:2421` (`forward_decode_batch_stream_impl`) →
`crates/infer-cuda/src/attention/flashmla.rs:815-1001`
(`build_layer_batch_meta` → `sched_meta_for_batch` split-KV scheduler
metadata → `sparse_decode_fwd_batched`), backed by the shared
`[num_sm_parts + max_batch]`-sized `lse_accum`/`o_accum` scratch
(`Dsv4FlashMlaDecodeBatchScratch`, `kv_layout.rs:165`) and the vendored
`arle_flashmla_decode_shim.cu` kernel. Same subsystem as an already-fixed
prior bug
([errors/2026-06-14-dsv4-batched-flashmla-decode-phaseB-correctness-kill.md](2026-06-14-dsv4-batched-flashmla-decode-phaseB-correctness-kill.md)).

Working hypothesis (unverified): the fixed-capacity split-accumulator can be
oversubscribed when live-row count × per-row split-count (which grows with
context length) is large enough, aliasing one row's in-flight partial
accumulator onto another's slot. This fits the observed signature — an
attention-output-level defect (not a stop-token bug), and a timing-dependent
race rather than a deterministic index bug (matches the random per-rerun
positioning even with identical prompts).

## Fix

None yet. `ARLE_DSV4_FLASHMLA_DECODE=0` (the existing lever to route around
this lane entirely) could not be used for a clean A/B at first: that
fallback path's KV-pool sizing was keyed to the FlashMLA banded layout and
admission-rejected almost every request at typical `--max-total-tokens`
ceilings — fixed same day in `4e44b0209` (decoupled
`dsv4_flashmla_decode_alloc_enabled`, a compile-time `HAS_FLASHMLA` question,
from the runtime `ARLE_DSV4_FLASHMLA_DECODE` kernel-choice flag). Pod-verified
the unblock: booting at the exact previously-rejecting config (TP=4,
`--max-total-tokens 2048`, GPUs 2/3/4/5, `ARLE_DSV4_FLASHMLA_DECODE=0`) and
re-sending the same 1706-prompt-token/1722-KV-page request that used to hit
`admission reject: request needs 1722 KV pages, pool has 1 free` — now 24/24
admit cleanly (3 trials × n=8, zero rejections in `serve_job1verify2.log`).

**A/B result (same day, same binary, same harness, same n/length sweep):
corruption REPRODUCES on the scalar eager kernel — hypothesis KILLED.**
Ran `concurrent_needle_v3.py` n∈{3,4,6,8} × prompt_len∈{100,250,500,900} × 3
reps, TP=4 GPUs 2/3/4/5, `--max-total-tokens 2048`:

| Arm | Requests | Exact | Miss | Miss rate |
|---|---|---|---|---|
| `ARLE_DSV4_FLASHMLA_DECODE=0` (scalar) | 252 | 200 | 52 | 20.6% |
| default (batched FlashMLA/CSA) | 196 (41/48 trials before an external SIGTERM tore the server down) | 146 | 50 | 25.5% |

Same failure signature on **both** arms — truncation (`'The secret access
code is 738.'`, correct prefix + missing tail) and digit corruption from
digit 4 onward (`738292`, `7382391`, `738123` vs needle `738291`) — at
comparable rates. If the batched FlashMLA/CSA lane (`flashmla.rs:815-1001`,
split-KV scheduler metadata + shared `lse_accum`/`o_accum` scratch) were the
root cause, routing around it entirely should have produced a clean 100%
match on the scalar arm. It didn't.

**Conclusion: not FlashMLA-specific.** The bug is in something shared between
both kernel paths — scheduler batch construction, tokenizer/sampler
concurrency handling, or the KV pool itself (allocation/addressing, not the
attention math) — not `flashmla.rs`. Next localization step: same n/length
sweep with `compute-sanitizer racecheck` (unblocked now — no lever flag
needed), targeting the scheduler's batch-row assembly and KV-pool slot
addressing rather than the FlashMLA scratch buffers.

## KV cache/page-reuse hypothesis — KILLED (2026-07-07)

Next candidate after FlashMLA-lane: a host/GPU synchronization gap around
freeing and reassigning physical KV pages — either via `RadixCache`
(`crates/infer-core/src/radix.rs`) matching a shared prompt prefix across
concurrent requests, or via raw same-slot page reuse
(`crates/infer-seam/src/host_paged_kv_pool.rs::free_slot`/`alloc`, pure
host bookkeeping, no sync/fence calls by design) racing a still-in-flight
GPU read from the prior occupant.

**Step 1 — prefix-sharing check (structural).** Read
`/host/arle-build/concurrent_needle_v3.py`: every concurrent request's
prompt is `"Trial{TRIAL}-Req{i}: " + TOPIC*n + PRE + CUE` — the per-request
salt sits at the very front of the filler, so every concurrent row's tokens
diverge within the first few tokens. The only shared prefix across ALL
requests, all trials, all runs is the fixed `wrap()` system preamble
(`<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>`), a handful of
tokens present identically even in the n=1/short-prompt cases that are
clean. A shared prefix that constant can't explain an (n≥3, length>250)-gated
defect — **cross-request `RadixCache` prefix reuse is structurally
implausible** as the sole mechanism before running anything.

**Step 2 — prefix-cache-off A/B (empirical).** Added a temporary,
diagnostic-only escape hatch (not a shipped feature) —
`crates/infer-api/src/loaded.rs`, `EngineLoadConfig::scheduler_config`: if
`ARLE_DISABLE_PREFIX_CACHE` is set, force `config.enable_prefix_cache =
false`. Rebuilt (`BUILD_EXIT=0`), reran the identical `job2_ab.sh` sweep
(n∈{3,4,6,8} × len∈{100,250,500,900} × 3 reps, TP=4 GPUs 2/3/4/5,
`--max-total-tokens 2048`) with `ARLE_DISABLE_PREFIX_CACHE=1`:

| Arm | Requests | Exact | Miss | Miss rate |
|---|---|---|---|---|
| default (prefix cache on) | 196 | 146 | 50 | 25.5% |
| `ARLE_DSV4_FLASHMLA_DECODE=0` (scalar) | 252 | 200 | 52 | 20.6% |
| `ARLE_DISABLE_PREFIX_CACHE=1` | 252 | 155 | 97 | 38.5% |

Same failure signature (truncation + digit-4-onward corruption, e.g.
`738231`/`7382`/`738.`  vs needle `738291`), zero admission errors/rejections
in either the client or server log. Disabling prefix cache entirely did
**not** reduce corruption — if anything the rate is comparably high (noise
at n=48 trials, not a real reduction). **Kills cross-request RadixCache
reuse as the mechanism**, confirming step 1's structural read.

**Step 3 — decisive fresh-boot-first-request test.** The narrower version
(same-slot/page reuse across *sequential* requests, independent of radix
matching — no fencing between a finished request's `free_slot` and a new
occupant's write to the same physical page) predicts corruption should be
*absent* on a server's very first-ever forward, since no page has ever been
freed/reused yet. Booted a fresh DSv4 server (TP=4, GPUs 2/3/4/5) and fired
n=8/len=500 as the literal first request batch (no warmup: DSv4's
`warmup()` is a no-op, `crates/infer-cuda/src/executor.rs`) — 8 concurrent
rows landing on 8 slots/physical KV bands never touched before:

```
CONCURRENT_SUMMARY N=8 trial=j3-fresh-1 exact=3 miss=[1, 3, 4, 6, 7]
```

5/8 miss **on the very first request this server ever processed** — same
truncation/digit-4 signature (`738.`, `7382.`, `**738292**`). Four more reps
against the same now-warm slots: exact 5/8, 5/8, 6/8, 2/8 — comparable to
the steady-state rate, no first-use vs. steady-state gap.

**This kills the entire KV-page-reuse framing, both variants.** There is no
reuse event on iteration 1 — every physical KV page these first 8 rows
write is being touched for the first time in the process's life, and the
bug already fires at the established rate. A host/GPU fencing gap around
page free-then-reuse cannot be the mechanism when the defect needs no reuse
to manifest.

**Signature re-read.** First-3-digits-always-right /
divergence-from-digit-4-onward, and truncation dropping only the *last* 2
digits, is a within-row late-decode-step pattern (something drifts across
successive decode steps of the SAME multi-token generation), not a
cross-row aliasing-at-a-point-in-time pattern — a sharper lead for the next
pass than "shared scratch buffer" or "stale page," but not chased further
in this pass (out of scope: verify-or-kill one hypothesis per pass).

The diagnostic-only `ARLE_DISABLE_PREFIX_CACHE` toggle in
`crates/infer-api/src/loaded.rs` is intentionally left in place (harmless,
env-gated, off by default) as a reusable A/B knob for future prefix-cache
investigations — it is not a user-facing feature and carries no default
behavior change.

## DSA topk row-ordinal-vs-slot-identity hypothesis — KILLED (2026-07-07)

Next candidate: the batched CSA/DSA indexer top-k SELECT step (never isolated
by the FlashMLA-lane A/B, which only swapped the final attention KERNEL, not
the selection feeding it) conflates a row's ordinal position `r` in this
step's batch (`0..n`) with its stable physical `slot_ids[r]`, so a row's
top-k selection silently reads/writes the wrong row once slot composition
churns mid-generation.

**Code read (write vs read addressing).** Traced the full write→read chain
in `csa_select_official_batched`
(`crates/infer-cuda/src/attention.rs:9339-9683`, called from
`dsv4.rs:3468`): the per-row prepare loop (`dsv4.rs:3031-3329`) gathers each
row's `q_i`/`weights`/`key_count` into the batch staging buffers at ordinal
offset `r` (`attention.rs:8778-8813`, `Dsv4DsaBatchedGather::row`), and
`slot_ids[r]` is used ONLY to build the block-table/page-table translation
(`attention.rs:9518-9540`, `csa_select_official_batched`'s `(b1)` block,
and the on-device twin `dsv4_dsa_build_select_meta_kernel`,
`dsv4_dsa_official.cu:665-690` — `block_table[r*num_pages+b] =
slot_ids[r]*num_pages+b`). The topk kernel itself
(`deepseek_v4_topk_transform_kernel`, `dsv4_dsa_official.cu:634-659`) reads
block `bid = blockIdx.x` (== the launch's row index, i.e. `r`, since the
grid is sized `batch_size = n`) and writes `out_selected`/`raw_indices` at
`bid * output_stride` — the SAME `r`. The consumer
(`Dsv4FlashMlaDecodeBatch::build_layer_batch_meta` →
`build_indices_batched`, `flashmla.rs:815-856`) reads `selected_batched` at
row stride `csa_topk`, populated by the SAME per-row loop via
`gather_selected_row`/`selected_batched_mut` at ordinal `r`
(`flashmla.rs:760-805`). **Write and read sides agree on `r` addressing
throughout — `slot_ids[r]` never leaks into the scratch-buffer offset
computation.** No static mismatch found by reading.

**Empirical confirmation (not just code reading).** Added a temporary
env-gated trace (`ARLE_DSV4_DSA_TRACE=1`, gated `n>1` only so it never fires
inside the n=1 CUDA-graph-capture decode lane) printing, per batched-select
call per row: `r`, `slot_ids[r]`, `context_lens[r]`, `positions[r]`, and a
selected-indices fingerprint. Pod-built (`cuda,nccl`), booted DSv4 TP=4
(GPUs 2/3/4/5), ran 5×`concurrent_needle_v3.py` n=4/len=500 trials
(3 reproduced corruption). Across every trace line in every trial —
including steps where the batch shrank (n=4→3, a row finished) and steps
where a slot was reused by a brand-new request within the same run —
**`r == slot_ids[r]` held in 100% of ~1500 trace lines.** The scheduler in
this workload always presents `slot_ids` in ordinal-sorted order, so the
conflation this hypothesis needs never gets an opportunity to manifest here.
**Hypothesis killed: r-vs-slot addressing is not the mechanism** (or at
least not reachable from this harness's admission pattern).

**Side observations, not chased further (out of scope for one pass):**
- `sel_first4` (first 4 selected block indices) was `[0,1,2,3]` in every
  single trace line regardless of row/step/context length — plausibly an
  attention-sink effect (the indexer scoring the first few compressed blocks
  highest every step), not inspected further.
- One trial showed a slot reused twice within the same short run (a fast
  request finished and freed slot 0; a new request was admitted into slot 0
  seconds later) — the freed-then-reused occupant happened to be the one
  request that corrupted that trial. This looked promising as a "stale
  per-slot ring-buffer bookkeeping on reuse" lead, but did NOT replicate: in
  a later trial the request that raced ahead and reused a slot finished
  CORRECTLY while two same-batch peers with no reuse history corrupted, and
  in two other trials multiple rows corrupted with no reuse event visible at
  all. Slot reuse is not a consistent predictor; not chased further this
  pass (would need a dedicated reuse-vs-no-reuse controlled A/B, not
  inference from 5 uncontrolled trials).

## Custom one-shot allreduce (`ARLE_COMM_BACKEND=auto`) — ruled out, no pod run needed (2026-07-07)

Considered whether the vendored one-shot `CustomAllreduce` kernel
(`crates/cuda-kernels/csrc/comm/custom_all_reduce.cu`, gated by
`TpRuntime::init_oneshot_comm`, `tp.rs:311`) could be a private-stream/flag
race (the codebase's own prior failure mode, see
`feedback_private_stream_needs_stream_wait` — DeepEP's dispatch/combine
missing a `stream_wait`), especially since it's licensed only by an isolated
comm bench, not the real serving workload. **Ruled out without a pod run**:
`crates/cli/src/args.rs:684` defaults `--comm-backend` to
`ServeCommBackendArg::Nccl`, not `Auto` — every prior A/B in this
investigation (and this session's own boots) logged `[comm-oneshot] disabled
via ARLE_COMM_BACKEND=nccl` (confirmed via `grep` across
`/host/arle-build/serve_*.log`). The one-shot kernel was never active in any
run that reproduced corruption; plain NCCL `all_reduce` is the only comm
path exercised, and it launches on `ctx.stream` (not a private stream) per
`tp.rs:462-490`. No further investigation needed on this lead.

## CUDA_LAUNCH_BLOCKING=1 A/B — GPU-async-race hypothesis KILLED (2026-07-07)

Cheapest remaining decisive test: does forcing every kernel launch to run
synchronously (no host-side launch queueing, no GPU-side overlap between
kernels/streams) change the corruption rate? If it disappears, the mechanism
is a missing-fence/async-ordering hazard exposed only by real overlap. If it
reproduces identically, the bug is not timing-dependent at the GPU-launch
level at all.

**Setup.** Smallest reliable repro from the DSA-hypothesis pass: n=4,
len=500, TP=4 GPUs 2/3/4/5, same env as `job2_ab.sh`
(`ARLE_DSV4_MOE_BACKEND=allreduce`, `ARLE_DSV4_INCREMENTAL_KV=1`,
`ARLE_DSV4_EXPERT_BACKEND=deepgemm`). Two server boots, control vs
`CUDA_LAUNCH_BLOCKING=1` exported before `arle serve`, otherwise identical.

**Pass 1 (5 reps/arm, 20 requests/arm)** — ambiguous on its own (Fisher exact
p=0.20), so extended to a second, bigger pass before concluding anything:

| Arm | Requests | Exact | Miss | Miss rate |
|---|---|---|---|---|
| control | 20 | 8 | 12 | 60% |
| `CUDA_LAUNCH_BLOCKING=1` | 20 | 13 | 7 | 35% |

**Pass 2 (15 reps/arm, 60 requests/arm)** — the decisive sample:

| Arm | Requests | Exact | Miss | Miss rate |
|---|---|---|---|---|
| control | 60 | 43 | 17 | 28.3% |
| `CUDA_LAUNCH_BLOCKING=1` | 60 | 43 | 17 | 28.3% |

Identical counts. Fisher exact on pass 2 alone: p=1.0. Combined pass1+pass2
(80 requests/arm): control 29/80 miss (36.25%), LB 24/80 miss (30%), p=0.50 —
no significant difference. Pass 1's apparent gap was noise from a small n.

Failure signature under `CUDA_LAUNCH_BLOCKING=1` is byte-for-byte the same
class as every prior pass — truncation (`'The secret access code is 738.'`)
and digit-4-onward corruption (`738292`, `738123`, `73829` vs needle
`738291`) — no new signature, no reduction in severity.

**Conclusion: not a GPU-kernel-launch-ordering race.** `CUDA_LAUNCH_BLOCKING=1`
removes all host-side launch queueing and inter-kernel/inter-stream overlap;
a hazard that depends on that overlap (a missing `stream_wait`, an async
memcpy racing a kernel read) would have been exposed as a rate change. It
was not — the mechanism is either a deterministic logic/numerics bug that
depends on (N, routing/content) rather than on timing, or a hazard outside
GPU-launch overlap entirely (see scope caveat below).

**Scope caveat, not fully closed by this test.** `CUDA_LAUNCH_BLOCKING=1`
only serializes GPU kernel launches; it says nothing about a *host-side*
(CPU-thread) data race. Closed separately, for free, by the architecture:
`ServeHandle`'s continuous-batching scheduler + backend executor live on one
dedicated engine thread (`crates/infer-api/src/serve_engine.rs`,
`crates/infer-server`'s `ServeHandle`) — HTTP handler threads only submit
tickets and collect results over a channel; the per-step batch construction,
`Dsv4DecodeBatch::from_rows`, and the forward call all execute on that single
thread, never concurrently with themselves. No host-thread race is possible
in the scheduler/decode-step-construction path regardless of how many HTTP
requests arrive simultaneously. Combined with the LB result, this narrows the
mechanism to a deterministic bug inside the single engine-thread's per-step
logic that only misfires when N>=3 real rows are coalesced into the same
batched forward call (and the prompt is long enough) — not a race of any
kind.

## Source-level review of the batched-decode chain — no static bug found (2026-07-07)

Per the non-race branch: reviewed every N-batched, content-shared (i.e., not
excluded by the earlier FlashMLA-vs-scalar A/B, which only swapped the
*attention* kernel) subsystem on the path from scheduler to sampled token,
looking for something simply WRONG and tied to (N, length) rather than a
race. No fix applied — flagging as reviewed-clean, not as a license to stop
looking.

- **`Dsv4DecodeBatch::from_rows`** (`crates/infer-cuda/src/executor.rs:1942`):
  builds `slot_ids`/`tokens`/`start_positions`/`positions` from `&[DecodeRow]`
  purely per-row (`row.slot`, `row.kv_seq_len`, `row.last_token`) — no
  batch-size-dependent arithmetic, no shared index. Asserts
  `slots[row.slot].seq_len() == row.kv_seq_len` per row.
- **`forward_decode_batch`'s row-selection** (`crates/infer-cuda/src/dsv4.rs:2369-2419`):
  samples row `r`'s token via `forward_stream_last_token(&stream, r + 1, ...)`,
  i.e. reads stream row `r`. This assumes the batched stream's row order
  matches `slot_ids` order throughout the whole layer stack, not just the
  DSA-select step already traced (`r == slot_ids[r]` 100% in ~1500 prior trace
  lines) — read `forward_stream_last_token`
  (`crates/infer-cuda/src/dsv4.rs:4516`) itself: a plain `seq_len - 1` row
  index into `copy_row_to_vec`/`head_hidden_from_stream`, no batch-size term.
- **Full MoE compact-decode pipeline** (allreduce transport, the path this
  config's `ARLE_DSV4_MOE_BACKEND=allreduce` actually takes — traced end to
  end since the FlashMLA A/B never excluded MoE, only the attention kernel):
  `dsv4_moe_forward` → `dsv4_moe_forward_masked_tail` → (`total_routes <= 128`
  for every n∈{3,4,6,8} tested here, so always) `dsv4_moe_forward_decode_fp8`
  (`crates/infer-cuda/src/moe.rs:2728-3083`) — router gemm → device routing
  (`dsv4_route_kernel`) → count/scan (`dsv4_count_local_experts_kernel`,
  `dsv4_exclusive_scan_i32_kernel`) → pack
  (`dsv4_pack_local_experts_with_slots_kernel`) → fused FP8 grouped
  gate/up/SwiGLU + down GEMM (`dsv4_fp8_grouped_swiglu_decode_kernel`/
  `dsv4_fp8_grouped_down_decode_kernel`,
  `crates/cuda-kernels/csrc/gemm/dsv4_fp8_decode_moe.cu`) → scatter
  (`dsv4_scatter_all_route_slots_kernel`) → combine
  (`dsv4_combine_route_slot_outputs_kernel`), all in
  `crates/cuda-kernels/csrc/moe/dsv4_route.cu`. Verified: `route =
  token*topk+k` layout is consistent between the router kernel that writes
  `indices[token*topk+k]` and the pack kernel that reads `token = route/topk`;
  the pack kernel's `packed_route_slot[slot] = route` and the scatter
  kernel's `route_out[route_slot*hidden_dim+col]` correctly round-trip
  packed-slot ↔ original-route-index; the two FP8 grouped-GEMM kernels' grid
  (`num_experts` in `blockIdx.z`, `max_count`-derived `blockIdx.y`) and their
  Rust FFI bindings (`crates/cuda-kernels/src/moe.rs:1567-1647`) pass
  `num_experts`/`max_count`/`n`/`k` in the same order the C signature expects
  — no positional-argument swap.
- **Also checked**: no `n == 2`/`n >= 3`-style batch-size special-casing
  anywhere in `dsv4.rs`; `DSV4_DECODE_GEMV_MAX_ROUTES = 128` and
  `DSV4_DECODE_CONTIG_MAX_ROUTES = 128` are both far above this repro's
  total-route counts (≤ 64), so no route-capacity ceiling is in play; the
  `ARLE_DSV4_MOE_CONTIG_DECODE` lever does not apply here — it only gates
  `dsv4_moe_forward_decode_pooled` (`moe.rs:3293`), a function not on this
  call path (`dsv4_moe_forward_masked_tail` branches straight to the FP8
  compact lane before that lever is ever consulted), so it can't be reused as
  an A/B knob for this specific repro without new code.

**Conclusion: still unresolved.** Every N-batched subsystem reachable from
source reading in this pass round-trips its indices correctly on paper. This
is consistent with either (a) a genuine numerical edge case in the FP8
dot-product / clamped-SwiGLU kernels that is data/routing-dependent rather
than an indexing bug (not exercised — would need per-step token-id-level
tracing of a corrupted row against its serial-n=1 reference, comparable
effort to the DSA trace but for the MoE path), or (b) a bug in a part of the
chain not yet reviewed this pass (the shared KV-batch-descriptor prep,
`prepare_kv_batch`, and the attention layers' non-kernel-choice-dependent
scaffolding around FlashMLA/scalar). No fix applied — inference from source
reading is not evidence per this doc's own prior lesson (the FlashMLA lane
"looked clean from source reading alone" too, and needed the lever A/B to
kill). Next step needs the same discipline: an env-gated per-row trace of
sampled token IDs (not just `slot_ids[r]`) at each decode step, diffed
against a serial n=1 run of the identical prompt, to localize which decode
step and which subsystem output first diverges — not further source reading.

## Token-ID-level diff trace — fixed-step corruption, NOT join/finish-correlated (2026-07-07)

Repro reconfirmed first (Step 0): `concurrent_needle_v3.py` n=4/len=500 TP=4
GPUs 2/3/4/5, 8 reps (32 requests) — 4 misses (12.5%), same signature
(truncation + digit-4-onward corruption). Sample too small to pin the exact
rate but not a surprise-clean result; proceeded.

**Instrumentation.** Added `ARLE_DSV4_DECODE_TRACE=1` (env-gated, rank==0
only, single pre-formatted `write_all` — the first cut used `eprintln!` with
multiple `{}` placeholders, which is *not* one `write()` syscall; 4 TP-rank
processes writing to the shared serve-log fd produced byte-level
field-interleaved garbage until fixed). One line per batched decode call:
`DSV4_TRACE call=<n> n=<batch_size> rows=[slot:start_pos:token,...]` —
`crate::dsv4::dsv4_decode_trace`, called from both `Dsv4Model::forward_decode_batch`
(B>1) and (new) `RealCudaExecutor::forward_decode_batch`'s B=1 branch (the
CUDA-graph decode-graph lane bypasses the model-level call entirely, so B=1
needed its own trace call site to get a token-level reference at all).

**Harness.** `trace_probe.py` (new, alongside `concurrent_needle_v3.py`): one
byte-identical fixed "TRACKED" prompt (prompt_tokens=456, deterministic
across every solo/concurrent invocation) + 3 filler rows offset by only
+1 TOPIC-repeat each (prompt_tokens 467/477/487 — small, unambiguous-but-not-
length-heterogeneous gaps, to avoid confounding the established near-
homogeneous-length repro regime while still letting the tracked row be
identified server-side purely from `start_position` == its own prompt-token
count).

**Reference (solo, n=1).** TRACKED prompt alone, greedy: tokens
`[8613,3278,4181,344,223,30143,17979,16,1]` at KV positions 456–464 →
`"The secret access code is 738291."` (17979 = "291", 16 = ".", 1 = EOS).

**4 corrupted trials, exact token-level diff** (of 14 concurrent n=4 reps,
4 corrupted — 28.6%, consistent with the established baseline):

| Trial | Join event (other rows enter batch) | TRACKED's own token stream (positions 456→) |
|---|---|---|
| rep3  | step index 6 (position 462, batch 1→4) | `[671,8613,3278,4181,344,223,30143,`**`18307`**`,16,1]` |
| rep4  | step index 6 (position 462, batch 1→4) | `[671,8613,3278,4181,344,223,30143,`**`18307`**`,16,1]` |
| rep10 | step index 6 (position 462, batch 1→4) | `[671,8613,3278,4181,344,223,30143,`**`18307`**`,16,1]` |
| rep11 | step index 2 (position 458, batch 1→3) | `[671,8613,3278,4181,344,223,30143,`**`18307`**`,16,1]` |

Every corrupted trial substitutes the **exact same wrong token** (18307 for
17979 — decodes to the wrong last digit, `738292` vs `738291`) at the
**exact same position** (KV depth 463 — the row's own 8th generated token),
regardless of when the join event happened. Rep11 is the decisive case: its
batch grew from 1→3 rows at TRACKED's step 2 (position 458), five steps
*before* the corruption; the batch composition was then **completely
stable** (same 3 slots, no join, no finish) for the 5 ticks bracketing the
corrupted step (calls 167–174 in the trace, positions 458–465) — the
corruption fires in the middle of an unchanging batch. Cross-checked rep3/
rep4/rep10 too: no slot's start_position shows an EOS (token `1`) landing on
the tick immediately before or at the corrupted tick — no finish event
correlates either.

**Verdict: (a), not (b).** The corrupted step is a fixed position, not
correlated with any observable batch-composition churn (neither a row
joining nor a row finishing at or near the corrupted tick). This directly
kills the "batch-composition-churn" framing as the trigger.

**Scope caveat — content-position vs. absolute-KV-depth not yet
separated.** Every trial here uses the *same* fixed TRACKED prompt, so
"KV depth 463" and "the row's 8th generated token" and "the token
representing the needle's last two digits" are the same number in every
trial — this dataset cannot distinguish "a bug tied to an absolute KV depth/
buffer boundary" from "a bug tied to a content-relative position within the
answer" (both would look identical here). Deciding between them needs a
second TRACKED prompt of a different length reproducing the same
substitution at a *different* absolute depth but the *same* relative
content-position (or vice versa) — not run this pass (information budget).

**Cheap, left in place** (same precedent as `ARLE_DISABLE_PREFIX_CACHE`):
`ARLE_DSV4_DECODE_TRACE=1` in `crates/infer-cuda/src/dsv4.rs`
(`dsv4_decode_trace`) and its executor.rs B=1 call site — off by default, one
`env::var_os` check per decode tick when unset, zero behavior change.

## compute-sanitizer racecheck/synccheck/memcheck — intra-kernel race KILLED, cross-kernel-fence still open (2026-07-07)

Per the token-trace round's own scope caveat ("the mechanism is invisible to
`CUDA_LAUNCH_BLOCKING`... exactly what `compute-sanitizer --tool racecheck`
is designed to catch"), ran the three genuinely batch-launched candidates
under `compute-sanitizer` against the live n=4 repro.

**Setup.** Pod ships `compute-sanitizer` 2025.2.1.0 at `/usr/local/cuda/bin/`.
`--kernel-name kernel_substring=<name>` (additive, matches inside the mangled
name) targeted three kernels, chosen from the source (`crates/cuda-kernels/csrc/misc/`)
by which ones are **actually launched `<<<n, BLOCK>>>` with `blockIdx.x` =
row/token ordinal in ONE call spanning the whole batch** (a prerequisite for
an *intra-kernel* cross-block race — a kernel launched once-per-row in a
host loop can't have one, since CUDA serializes same-stream launches):
- `dsv4_compressor_update_batched_kernel` (`dsv4_attention.cu:1136`, `<<<n,BLOCK>>>`,
  `rowi = blockIdx.x`) — genuinely batched, always (`full_flatten`, MODEL1
  B>1 canonical path per `dsv4.rs:2675` comment).
- `deepseek_v4_topk_transform_kernel` (`dsv4_dsa_official.cu:634`, `bid =
  blockIdx.x`) — genuinely batched, always.
- `dsv4_hybrid_attention_kernel` (`dsv4_attention.cu:1534`) — **traced this
  round and it is NOT the batched-race candidate the priority list assumed**:
  its `start_pos`/`start_pos_ptr` arg is a single SCALAR (`dsv4_graph_start_pos`
  derefs `*start_pos_ptr` unconditionally, no per-row array), and
  `forward_decode_batch_stream_impl` (`dsv4.rs:2519-2531`) memcpys each row's
  own position into its OWN `slot.start_pos_device` before a **per-row** call
  — so unless `batched_attn_lane` (FlashMLA's own batched decode kernels,
  already A/B-killed as a hypothesis in an earlier round) is active, this
  kernel is launched once per row, serialized on `ctx.stream`, never
  concurrently with itself. Included anyway (cheap, additive filter) but
  structurally can't have an intra-launch race under the non-FlashMLA fallback.

**Repro under instrumentation.** TP=4 hit a wall first: compute-sanitizer's
own device-memory overhead OOM'd model load (`DSv4 grouped weight alloc
failed`) — the **unsanitized** boot already runs at ~98% VRAM (`used
95561MB free 1947MB` after all slots, per a prior round's log), so ANY
added overhead breaks it. `--force-synchronization-limit 1` (trades
perf for lower tool memory — "reduces tool's device memory at the cost of
performance" per `--help`) fixed it; booted clean at `used 95595MB` across
GPUs 2/3/4/5. (Aside, addressing a mid-task steer: TP=1 on a single GPU was
tried first as a simpler alternative since TP/NCCL is orthogonal to an
intra-kernel batched-row race — but this checkpoint is 274GB on disk vs.
97GB/card, confirmed OOM at the identical `grouped weight alloc` step even
unsanitized; TP=4 is the real minimum shard for this model on H20.)

Confirmed the sanitizer library (`libsanitizer-public.so`) was mapped into
all 4 forked rank-worker PIDs (not just the coordinator) via `/proc/<pid>/maps`,
and that all 4 shared one inherited log fd (`/proc/<pid>/fd/3` →
`racecheck_<label>_<coordinator-pid>.out`) — the single-file-per-run output
is expected (fork before exec inherits the fd), not evidence of missed
coverage.

**Results** (`concurrent_needle_v3.py`, n=4, len=2000, `max_tokens=16`,
GPUs 2/3/4/5):

| Tool | Trials | Corrupted / total | Output |
|---|---|---|---|
| `racecheck` | 6 | 9/24 (37.5%) | Clean — only the `========= COMPUTE-SANITIZER` header, zero hazards, across all 6 trials |
| `synccheck` | 3 | 3/12 (25%) | Clean — same header-only result |
| `memcheck` (bonus) | 3 | 7/12 (58%, likely timing-shifted by ~3x heavier instrumentation — 105-126s/trial vs. 33-43s for racecheck/synccheck) | Clean w.r.t. the 3 target kernels — only 4x benign `CUDA_ERROR_INVALID_VALUE` on `cuCtxGetLimit` inside `cublasLtCreate` (a known harmless cuBLASLt/sanitizer init interaction, unrelated to DSv4) |

The corruption signature reproduced under every tool pass (the digit-4-onward
truncation/substitution bucket), so these are not "didn't trigger" clean
results — the bug fired repeatedly while instrumented, and none of the three
tools flagged anything in the targeted kernels.

**memcheck scope caveat (why this bonus check is weaker evidence than it
looks):** `radix_topk`'s un-bounded-checked write (`dsv4_dsa_official.cu:613`,
`output[pos] = idx` with `pos` from `atomicAdd(&s_counter,...)`, no `pos <
topk` guard in the `bin > threshold_bin` branch) was the motivating hypothesis
for adding memcheck — if a row's radix select ever counts more than `topk`
items above threshold, `pos` could exceed the row's own slice and spill into
the **next row's** region of `page_indices`. Memcheck did NOT flag this, but
memcheck only detects accesses **outside the whole `cudaMalloc`'d allocation**
— if `page_indices` is one contiguous `[n, output_stride]` buffer (likely,
matching the pattern of every other batched buffer in this file), a row
spilling into its NEIGHBOR's slice is still "in bounds" of the parent
allocation and structurally invisible to memcheck. This hypothesis is
**not ruled out** by this round; it needs either a source-level bound check
(read `radix_topk`'s round-3 exact-count invariant) or a deliberately
undersized/padded per-row allocation to make memcheck's boundary meaningful.

**Verdict: intra-kernel (cross-thread-block) race KILLED for the two
genuinely-batched kernels** (`dsv4_compressor_update_batched_kernel`,
`deepseek_v4_topk_transform_kernel`) — racecheck is purpose-built for exactly
this hazard class and stayed silent across 9 corrupted repros. Combined with
the prior round's `CUDA_LAUNCH_BLOCKING=1` result (host-launch-ordering ruled
out) and the single-engine-thread architecture (host-thread races ruled out),
**every race-class explanation this investigation can name is now closed**.
What's left, per this doc's own framing from the prior round: (a) a
cross-kernel memory-visibility gap (a missing fence/sync assumption between
two kernels on the same stream — CUDA's same-stream ordering guarantees
sequencing but not by itself a memory-consistency issue class racecheck
doesn't check inter-launch, only intra-launch), (b) the unbounded
`radix_topk` write spilling cross-row within one allocation (memcheck-blind
per above, not yet source-verified against the actual round-3 count
invariant), or (c) a genuine data/routing-dependent numerical edge case
(the FP8 MoE path, not re-examined this round). None of these three is a
"race" in the sense compute-sanitizer's two race-specific tools can catch —
next step is source-level (not tooling), reading `radix_topk`'s exact
round-3 termination guarantee against `topk`.

## Needle-content type A/B — NUMERIC needle drives the corruption, TEXT needle near-immune (2026-07-07)

Discriminating test between "near-tie-sensitivity" (a numeric answer is more
likely to have a close second-place logit candidate than a word completion,
making it sensitive to tiny batched-vs-solo numerical noise) vs.
"needle-content-independent" (the bug fires at a similar rate regardless of
what's being recalled).

**Setup — single-variable change.** New harness `concurrent_needle_text.py`
(mirrors `concurrent_needle_v3.py` byte-for-byte except `NEEDLE`/`PRE`/`CUE`):
swapped the numeric needle `"738291"` (tokenizes `['738','291']`, 2 tokens) for
the word `"CASTLE"` (tokenizes `['CAST','LE']`, also 2 tokens — same
prefix-token + completion-token + `.` shape, so the corrupted step's *position*
in the generation is structurally comparable). One server boot, one config
(matches `job2_ab.sh`'s default arm: `ARLE_DSV4_MOE_BACKEND=allreduce`,
`ARLE_DSV4_INCREMENTAL_KV=1`, `ARLE_DSV4_EXPERT_BACKEND=deepgemm`,
`--max-total-tokens 2048`; only the GPU set differs — 3/4/5/7 instead of
2/3/4/5, since GPUs 1/2 were occupied by another user's Qwen3.6 servers this
session), n=4, len=500, both needle sweeps run back-to-back against the same
running server. SOLO (n=1) sanity checked first for both needles — 3/3 exact
each, clean baseline, no hedging/commentary behavior in any solo response.

**Result (n=4, len=500, 15 reps = 60 requests/arm):**

| Needle | Exact | Miss | Miss rate | Truncation-class | Corruption-class (wrong final token) |
|---|---|---|---|---|---|
| Numeric (`738291`) | 26/60 | 34/60 | 56.7% | 32/60 (53.3%) | 2/60 (3.3%) — both `738292`, the exact same `17979`→`18307` substitution as the prior token-trace round |
| Text (`CASTLE`) | 59/60 | 1/60 | 1.7% | 1/60 (1.7%, `'CAST'` — dropped the final `LE` token) | 0/60 (0%) |

A 33x gap in overall miss rate, and **zero** wrong-token substitutions on the
text needle across 60 requests vs. 2 on the numeric needle in the same boot.
Full log: every numeric truncation is preceded by a hedging/meta-commentary
fragment absent from every solo numeric response and every text response —
`'The secret access code is 7382. (Note: The repeated text in'`, `'...**738**.
(The full code was stated as '` — the model starts qualifying its answer
instead of committing to it, then runs out of the fixed 16-token budget before
finishing the digits. This hedging never appears in any of the 60 text-needle
outputs or any of the 6 solo-needle sanity responses (both needle types),
which localizes it to the (n>=3, numeric-needle) joint condition, not a
general model quirk.

**Implication for the near-tie-sensitivity hypothesis: SUPPORTED for the
truncation-class failure (the majority mode, 53.3% vs 1.7%), and directionally
consistent (not separately decisive at n=2 events) for the true digit-
substitution class (3.3% vs 0%, matching the historical ~3-12% absolute
corruption rate from other rounds — this pass's elevated *overall* miss rate
is plausibly page/box contention from the two unrelated Qwen3.6 servers
sharing the node this session, but that confound applies equally to both arms
of this same-boot A/B and doesn't explain the needle-content gap).** The
numeric needle is not just "sometimes flips a digit" — under concurrent
batching it also measurably increases the model's tendency to hedge/qualify
its answer (something-drifts-then-the-model-notices-and-backpedals), which is
itself consistent with an upstream score (DSA indexer or MoE router top-k)
being subtly perturbed by batching in a way that specifically destabilizes
close-call numeric-token decisions while leaving a word completion's dominant,
non-tied candidate untouched. Does not by itself distinguish DSA-indexer vs.
MoE-router as the perturbed subsystem (out of scope this pass — diagnostic
only, no fix attempted).

**Caveat.** n=2 true wrong-token-substitution events is a small sample to rest
the whole hypothesis on in isolation; the truncation-class result (53.3% vs
1.7%, n=34 vs n=1) is the statistically solid half of this result. A future
pass could raise the numeric corruption event count (more reps, or the
fixed-tracked-prompt trace harness) to tighten the corruption-class-specific
comparison, and/or run a second text needle word to rule out `CASTLE`-specific
idiosyncrasy — not done this pass (one clean single-variable result was the
information-budget target).

## Cross-architecture check — Qwen3-4B dense shows ZERO corruption on either needle type (2026-07-07)

Steer: is the numeric-vs-text corruption gap DSv4-specific, or a general
batched-CUDA-kernel property of ARLE's inference stack? Ran the identical
harness methodology against a completely different architecture family —
**Qwen3-4B dense** (`Qwen3ForCausalLM`, 36 layers, GQA 32Q/8KV heads, no MoE,
no indexer/compressor/sliding-window-compression/MHC/Waterfill — none of
DSv4's machinery, but the same `infer-cuda` batched-decode/paged-KV/scheduler
substrate) — one boot, TP=1, GPU 1 (the shared-box-safe pin; `/host/Qwen3-4B`,
7.6 GB bf16, fits trivially on one H20).

**Harness.** Byte-for-byte copies of `concurrent_needle_v3.py`/
`concurrent_needle_text.py` (`concurrent_needle_v3_qwen.py`/
`concurrent_needle_text_qwen.py`, `/host/arle-build/`) — only change is
`wrap()`: Qwen3's ChatML (`<|im_start|>system...<|im_start|>user...
<|im_start|>assistant\n<think>\n\n</think>\n\n`, forcing a direct non-reasoning
completion) instead of DeepSeek's special tokens. Needle/PRE/CUE/TOPIC filler/
trial-salting/`max_tokens=16`/`temperature=0` identical. `--max-total-tokens
4096`, no other DSv4-only env flags (none apply to Qwen3 dense).

**Solo (n=1) sanity**, 3 reps each: 3/3 exact both needles, clean baseline —
`'The secret access code stated earlier is **738291**.'` /
`'...secret password stated earlier is **CASTLE**.'`.

**Concurrent (n=4, len=500, 15 reps = 60 requests/arm)**, same server boot,
back-to-back:

| Needle | Exact | Miss | Miss rate |
|---|---|---|---|
| Numeric (`738291`) | 60/60 | 0/60 | **0%** |
| Text (`CASTLE`) | 60/60 | 0/60 | **0%** |

Zero truncation, zero digit substitution, zero hedging/meta-commentary on
either needle across all 120 concurrent requests. Wall-clock confirms real
batching occurred (not accidental serialization masking the effect): solo
numeric 0.61-0.71s -> n=4 concurrent 2.18-2.30s (sub-4x, consistent with a
shared batched forward call, not 4 serial single-row calls); solo text
0.42s -> n=4 concurrent 1.38-1.56s, same pattern.

**Comparison to DSv4** (same n=4/len=500/15-rep design, `job3_text_needle.sh`,
TP=4 GPUs 3/4/5/7): numeric 56.7% miss / text 1.7% miss, a 33x gap. Qwen3-4B
dense: **0%/0%, no gap at all** — not "smaller gap," a clean floor on both
arms.

**Verdict: this is the "near-zero on both" outcome, not the "general property"
outcome — DSv4-specific, not a general batched-inference/floating-point-
non-associativity property of ARLE's shared CUDA batching substrate.** Qwen3
dense shares the same `infer-cuda` scheduler, paged-KV pool, continuous-batching
engine-thread architecture, and batched-decode CUDA kernel family as DSv4, and
shows no measurable corruption under the identical concurrent-batching stress
at n=4. This redirects the investigation back to DSv4-specific mechanisms —
DSA/CSA indexer top-k selection, the compressor/MHC sliding-window-compression
path, MoE routing/FP8 grouped-GEMM, or the FlashMLA/CSA split-KV
scratch-accumulator layout — none of which Qwen3 dense exercises at all. Does
**not** distinguish which DSv4-specific subsystem (out of scope this pass,
diagnostic only); the FP8 MoE decode path (flagged but not chased in the prior
source-review round) and the DSA `radix_topk` unbounded-write hypothesis
(memcheck-blind per the compute-sanitizer round) are the two concrete leads
still open, and this result doesn't discriminate between them since Qwen3
dense has neither.

**Caveat — model-scale and batch-depth not matched.** Qwen3-4B (4B dense) vs
DeepSeek-V4-Flash (much larger MoE, TP=4) differ in more than architecture
family alone: parameter count, FP8 vs BF16 compute path, TP=1 vs TP=4 (no
cross-GPU allreduce in the Qwen arm), and per-step FLOPs per row. A `0%` floor
on a small dense model at TP=1 is consistent with "no DSv4-specific mechanism
needed to see corruption" but doesn't independently prove TP/allreduce is
clean — that variable was already ruled out separately (custom one-shot
allreduce ruled out above; plain NCCL `all_reduce` on `ctx.stream`, not a
private stream). A stronger (but not run this pass, information-budget) next
step would be Qwen3.6-27B-FP8 MoE at TP>1 (already loaded/available on this
node per the process table) — same batched-decode substrate but with MoE
routing and multi-GPU TP, narrowing the "DSv4-specific" verdict from
"non-MoE-non-DSv4" down to "specifically DSv4's indexer/compressor/MHC," not
just "any MoE router."

## Rule

Concurrent DSv4 serving (n>=3, prompt length >~100-250 tokens) has a real,
unresolved silent-corruption risk today — not safe to treat as a solved
baseline when A/B-testing other DSv4 decode-path changes. Case-as-fact
paid off here: the aggregate "~40% failure" number alone pointed nowhere;
decoding actual failing text (truncation vs. digit corruption, first-3-
correct/last-N-wrong) narrowed the search to one subsystem before any code
was touched. **And the localization itself needed the same discipline**:
a plausible single-lane hypothesis (batched FlashMLA/CSA) looked clean from
source reading alone, but the licensed A/B (route around the lane entirely)
killed it in one measured pass — inference from code reading is not evidence,
a lever-gated A/B is.

**The fresh-boot-first-request test is the cheapest kill for any
cross-request-reuse hypothesis** — if the defect needs no history to
manifest, no reuse-based mechanism (cache, page, slot) can be the cause;
run it before reasoning further about *how* reuse might race, not after.

**`CUDA_LAUNCH_BLOCKING=1` is the cheapest kill for a GPU-launch-ordering
race** — run it before compute-sanitizer/nsys, but size the sample first: a
5-rep/arm pass here gave p=0.20 (looked like a real reduction, wasn't), and
only a 15-rep/arm pass resolved it to bit-identical miss rates (p=1.0). A
single-digit rep count on a ~30% baseline miss rate is not enough signal to
call a race killed or confirmed either way. It also only rules out
GPU-launch-level races — a single-engine-thread architecture (verified here
via `ServeHandle`) is what closes the host-thread-race branch, not the env
var.

**A byte-identical fixed-prompt token-ID trace (one tracked row vs. its own
solo reference) is the sharpest tool this investigation has used** — it
localizes the corruption to one fixed step with one fixed wrong-token
substitution, and directly falsifies "batch-composition churn" (join/finish
timing varied 4x across trials with zero effect on when the corruption
fired). Next: separate content-relative position from absolute-KV-depth
with a second differently-lengthed tracked prompt — the two are
confounded when every trial reuses the same fixed prompt.

**`compute-sanitizer --tool racecheck`/`synccheck` only cover intra-kernel
(cross-thread-block) hazards — verify a kernel is genuinely `<<<n,BLOCK>>>`-
batched-in-one-launch (grep the `<<<...>>>` call site, not just "looks
batched from its name") before spending a GPU-memory-constrained pass on it**;
one of this round's three priority kernels turned out to be launched once
per row in a host loop (serialized by same-stream ordering), making an
intra-launch race structurally impossible there regardless of tool output.
**A production model already near 98% VRAM (near-zero headroom even
unsanitized) will OOM under any sanitizer tool's overhead** —
`--force-synchronization-limit 1` (trade perf for lower tool memory) is the
first lever to reach for, before concluding TP needs to go up (which only
helps because it reduces per-GPU weight-shard size, not because TP/NCCL has
anything to do with the race hypothesis itself — same lesson as the
`CUDA_LAUNCH_BLOCKING` GPU-launch-vs-host-thread distinction: know exactly
which layer a mitigation operates on before reaching for it).
**Memcheck/racecheck are allocation-boundary-scoped, not layout-scoped** — a
kernel writing past its own logical slice into a NEIGHBOR's region within the
SAME parent allocation is invisible to both tools; an un-bounded-checked
`atomicAdd`-then-index write (`radix_topk`'s `output[pos]`, no `pos < topk`
guard on one branch) needs a source-level invariant check, not a sanitizer
pass, to rule in or out.

**Needle CONTENT type is not a free confound — swap it as its own controlled
variable before trusting an aggregate rate across investigation rounds.** A
same-boot, same-config A/B (numeric `738291` vs. word `CASTLE`, same n=4/
len=500/60-request sample) showed a 33x miss-rate gap (56.7% vs 1.7%) and zero
wrong-token substitutions on the text needle vs. two on the numeric — the
defect is real but strongly needle-content-dependent, consistent with a
near-tie/close-competing-candidate sensitivity (digit sequences have plausible
near-tied next-token competitors; a common word's completion doesn't). Any
future rate comparison across rounds that changed the needle string alongside
other variables should be re-read with this in mind.

**"Shares the same batched CUDA substrate" is not "shares the same bug" —
run the identical harness on a structurally different model before
generalizing a defect found in one architecture.** DSv4's 56.7%/1.7%
numeric-vs-text gap did not reproduce at all on Qwen3-4B dense (0%/0% over
60 requests/arm, same n=4/len=500 design) despite both models running through
the identical `infer-cuda` scheduler/paged-KV/continuous-batching-engine-thread
substrate. A shared-infra hypothesis needs a shared-infra control, not just a
plausible mechanism story (floating-point non-associativity in batched
kernels) — the control killed it here in one pod boot (~2 minutes of GPU time,
7.6 GB model) at negligible cost relative to the DSv4 TP=4 passes that
preceded it.

## Qwen3.6-27B-FP8 MoE control — clean, same as dense Qwen3-4B (2026-07-07)

Dense Qwen3-4B (prior round) rules out "any batched-inference substrate," but
doesn't separate "any MoE router" from "DSv4-specific indexer/compressor/MHC"
as the shared mechanism, since Qwen3-4B has no MoE routing at all. Reran the
identical numeric-vs-text needle A/B (`concurrent_needle_v3_qwen.py` /
`concurrent_needle_text_qwen.py`, ChatML wrap, n=4/len=500/15 reps = 60
requests/arm, greedy) against **Qwen3.6-27B-FP8** (qwen35-hybrid MoE path,
`sqrtsoftplus`/`noaux_tc` top-k routing — the same MoE routing family DSv4
itself uses) at TP=1/GPU=1 (fits 1×H20 per
[wins/2026-06-29-cuda-qwen36-paged-batched-decode.md](wins/2026-06-29-cuda-qwen36-paged-batched-decode.md)).
Solo (n=1) sanity 3/3 exact on both needles first.

| Needle | Requests | Exact | Miss rate |
|---|---|---|---|
| Numeric (`738291`) | 60 | 60 | **0%** |
| Text (`CASTLE`) | 60 | 60 | **0%** |

Zero misses on either needle — identical to the dense Qwen3-4B control, in
contrast to DSv4's 56.7%/1.7%. **Rules out "any MoE router" as the shared
mechanism**: Qwen3.6-27B-FP8's batched decode goes through the same
`infer-cuda` paged-KV/continuous-batching substrate AND exercises MoE top-k
routing under concurrency, yet shows no elevated numeric-vs-text gap. The
defect narrows further to something DSv4-specific that neither Qwen3
architecture has: the DSA/CSA sparse-attention indexer, the compressor
(latent KV compression), or MHC — not FP8 grouped-GEMM MoE routing in
general, and not batched-decode concurrency in general.

## Logit-lens layer diff — divergence onset at layer ~19-21 of 43, final split only at the last layer (2026-07-07)

Localizes WHICH layer first diverges between a clean and a corrupted decode,
using the built-in decode logit-lens probe (`crates/infer-cuda/src/probe.rs`,
`--probe-out`/`--probe-lens-layers 43`/`--probe-token-entropy true` — 43 =
DSv4's full `num_hidden_layers`, so this covers the entire stack, not a
partial window).

**Setup.** Booted DSv4 TP=4 (GPUs 2/3/4/5) with the probe flags, ran
`trace_probe.py`'s fixed TRACKED prompt (byte-identical every call,
prompt_tokens=456) SOLO (n=1) first as a clean reference, then looped n=4
concurrent attempts (TRACKED + 3 filler rows) until `TRACKED_MISS=True`.
Repeated as two independent server boots for a same-mechanism replication
check (not just one sample) — both hit the corruption on their 5th concurrent
attempt. Extracted the TRACKED row's `decode`/`lens` JSONL records per
capture by `pos` (concurrent fillers' prompt lengths, 471/481/491, keep their
own prefill/decode records at disjoint positions ≥467, so TRACKED's own
456-464 range is unambiguous even mid-batch — this required filtering out
`phase:"prefill"` lines too, since a longer filler's OWN prefill sweep passes
through position values that numerically overlap TRACKED's decode range).

**Alignment caveat (important, not a bug in the capture).** The two runs'
`pos` numbering is offset by a constant +1 (corrupt `pos` = solo `pos` + 1) —
confirmed by exact token-content matching, not assumed: both trajectories
emit the identical token sequence `[671, 8613, 3278, 4181, 344, 223, 30143,
...]` for their first 7 generated tokens, just one KV-slot position apart
(plausibly a cached-vs-fresh prefill boundary artifact from `trace_probe.py`'s
byte-identical TRACKED prompt hitting `RadixCache` on every call after the
first). Also found: solo generations are **not bit-reproducible across
separate server boots** for the identical prompt (boot 1's solo produced the
verbose completion `"The secret access code is 738291."` — 10 tokens
starting `671,8613,...`; boot 2's solo produced the terse `"738291"` — 3
tokens starting `30143,17979,1`, a different greedy path from token 0). This
is itself a side finding (boot-to-boot kernel/heuristic selection variance,
e.g. cuBLASLt's autotuned algorithm choice, plausibly) not chased further
here — it does NOT confound the layer-diff analysis below, since both
corrupted captures are compared against **boot 1's own solo reference**
(which shares their first-7-token trajectory exactly), not cross-boot.

**Result — two independent corrupted trials, same qualitative signature.**
Aligned by generation step `g` (0-indexed; `g=7` is the corrupted step,
solo's correct `17979` = "291" vs the corrupted `18307` = "292", matching the
prior token-ID-trace round's substitution exactly):

| layer range | solo1 top1 sequence | corrupt trial 1 | corrupt trial 2 |
|---|---|---|---|
| 0–18 (19 layers) | `69146` (constant) | `69146` (constant) — **bit-identical to solo** | `69146` (constant) — **bit-identical to solo** |
| 19 | `34366` | `34366` (agrees) | `33180` (diverges — first hint) |
| 20 | `19607` | `19607` (agrees) | `19607` (agrees) |
| 21–34 | oscillates `{0, 19607, 53869, 68468}` | **locked at `19607`**, 13/14 layers | **locked at `19607`**, 13/14 layers |
| 35–41 | mostly re-converges (`68468`/`127442`/`31942`/`17986`) | mostly re-converges, same values | mostly re-converges, same values |
| 42 (final) | `17979` (correct) | `18307` (wrong) | `18307` (wrong — same substitution) |

**Divergence onset: layer ~19-21 of 43 (mid-stack, ~46-49% depth) — not an
early layer, not a clean single late-layer culprit, and not a smoothly
accumulating drift either.** Layers 0–18 (embedding, RoPE, and the first
~18 attention/MoE blocks) are **bit-for-bit identical** between solo and
corrupted in both independent trials — rules out the embedding/RoPE/earliest
attention blocks as the entry point. From layer 21 onward the pattern is
**intermittent, not monotonic**: the corrupted run's lens top-1 becomes MORE
stable (locks onto `19607` for 13 of the next 14 layers) while the clean
run's lens top-1 is LESS stable in that exact same span (bounces between 4
different candidates) — both trials show this same inversion. The two
trajectories then mostly re-converge through layers 35-41 (same top-1 in
5 of 7 layers) before permanently forking only at the very last layer (42),
where the final unembedding projection resolves the accumulated difference
into two adjacent-vocabulary numeric tokens (`17979`="291" vs `18307`="292").

**Reading:** the onset at layer ~19-21 is a reproducible, non-noise signal
(two independent boots agree), landing squarely in DSv4's **mid-stack
CSA/DSA hybrid-attention and MoE block range** — consistent with (not yet
proof of) the indexer/compressor/MoE-routing hypotheses already on the
suspect list, and inconsistent with an embedding/RoPE-level or a
purely-final-layer-only mechanism. But the INTERMITTENT (not
monotonically-widening) pattern from layer 21 to layer 41 — re-agreement at
24-28/30/33/35-39/41 — argues against "one single kernel computes a wrong
value once, and the error propagates cleanly forward from there." It reads
more like a small, layer-local numerical perturbation that nudges the
residual stream toward a different (but not yet decisive) region of
representation space starting mid-stack, without permanently committing to
a different answer until the LM head's projection — i.e., closer to
"accumulated drift with a mid-stack onset" than to a single clean
early-layer or late-layer culprit. n=2 corrupted trials; a third/fourth
would strengthen confidence in "layer ~19-21" as the exact onset boundary
rather than "somewhere in 19-21" — not run this pass (information budget;
this round's job was localization-to-a-region, not fix-finding).

## MHC TF32-prenorm eager-fallback test — BLOCKED, premise doesn't hold in ARLE's code (2026-07-07)

Proposed test: force MHC's mixing GEMM off a "TF32-fused DeepGEMM prenorm"
path onto an eager FP32 reference path, to see if corruption disappears
(the mid-stack, whole-network-boundary signature from the layer-diff pass
matches MHC's structural footprint — present at every attn/FFN sub-layer
boundary across all 43 layers — better than a layer-localized mechanism).

**Traced every MHC call site and its underlying kernel/GEMM to check the
premise before running anything.** `crates/infer-cuda/src/hc.rs` (all 6
public functions) + `crates/cuda-kernels/csrc/misc/dsv4_mhc.cu` (every
`dsv4_mhc_*_kernel`) + the `mix_fn` weight-load path
(`crates/infer-cuda/src/loader.rs:3542` `load_dsv4_global_matrix`):

- **The MHC sinkhorn/mixing kernels themselves are scalar CUDA-core FP32,
  full stop — there is no TF32 or tensor-core variant to fall back from.**
  `dsv4_mhc_params_kernel`/`dsv4_mhc_params_pre_rms_norm_kernel`
  (`dsv4_mhc.cu:161`, `:331`) both call the *same* shared device function
  `dsv4_mhc_params_tail` (`:84`) — plain `expf`/`fmaxf` scalar sinkhorn math,
  bf16 in/out, fp32 internal accumulation. No `wmma`/`mma.` instruction, no
  `tf32` cast anywhere in the file (`grep -n "tf32\|wmma\|mma\."` — zero hits).
- **MHC's one GEMM (`mix_fn` projecting the wide stream into pre/post/comb
  weights) never touches DeepGEMM or tensor cores either.** `hc.rs`'s
  `gen_mhc_params`/`gen_mhc_params_into` route it through
  `crate::attention::dsv4_linear` → `mla_linear`
  (`attention.rs:992`) → `ffi::dsv4_fp8_gemv_batch_cuda`
  (`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu:399`) — a hand-rolled
  scalar FP8-block-scaled GEMV kernel, zero `wmma`/`mma.` instructions. This
  is forced structurally, not by a runtime flag: `load_dsv4_global_matrix`
  host-quantizes `mix_fn` to `Dsv4Fp8BlockScaled` unconditionally for BF16/F32
  source tensors (`loader.rs:3554-3567`), so `dsv4_linear`'s match arm always
  takes the `mla_linear` branch for `hc.mix_fn` — never `DenseBf16`/`gemm_batch`.
- **Repo-wide grep for the exact terms in the task brief — zero hits.**
  `hc_prenorm`, `prenorm_gemm`, `HC_PRENORM`, `SGLANG_OPT`, and `tf32`/`TF32`
  anywhere under `vendor/` or `crates/cuda-kernels/`: no matches. `docs/environment.md`
  has zero `mhc`/`hyper-connection`/`hc_mult`/`hc_eps`/`hc_sinkhorn` entries.
  The only fused-vs-unfused distinction that *does* exist in `hc.rs` is
  kernel-launch-count fusion, explicitly labeled as such in the doc comments
  (`gen_mhc_params_into` / `hc_pre`, `#[allow(dead_code)]`, "kept as the
  unfused primitive (A/B reference)") — and both variants call the identical
  shared scalar device functions as their fused counterparts
  (`dsv4_mhc_params_tail`, confirmed shared by both `dsv4_mhc_params_kernel`
  and `dsv4_mhc_params_pre_rms_norm_kernel`). Forcing the "unfused" primitives
  would produce bit-identical numerics, just more kernel launches — not a
  test of any precision hypothesis.

**Conclusion: the premise doesn't hold.** ARLE never ported (under any name)
a TF32/DeepGEMM-tensor-core prenorm path for MHC — the `SGLANG_OPT_DEEPGEMM_HC_PRENORM`
framing in the task brief describes upstream SGLang's implementation, not
ARLE's. ARLE's from-scratch `dsv4_mhc.cu` has been scalar-CUDA/"eager" by
construction since it was written; there is no faster/lower-precision sibling
path to disable. **This diagnostic is blocked — not "toggle exists but
untested," but "there is nothing to toggle."**

**Improvising a toggle is not a small, low-risk move here — declined.** The
only way to give this hypothesis a genuine A/B would be to build a *new*
TF32/tensor-core GEMM implementation of the `mix_fn` projection and/or the
sinkhorn kernel from scratch, then compare it against the existing scalar
path. That is net-new kernel development, not flipping a dormant flag — it
risks introducing a *different* bug that would confound this investigation
rather than isolate it, and doesn't fit a same-day diagnostic pass. Declined
per this round's scope (discriminating test only, no new code this pass).

**MHC not fully cleared as a suspect, just this specific angle.** MHC's own
kernels ARE genuinely batched in the multi-row decode lane (`gen_mhc_params`/
`mhc_pre_rms_norm`, used by every batched decode call site in `dsv4.rs`, e.g.
`:4783-4877`, `:5130-5304`, `:5722-5831` — `dsv4_mhc_params_kernel<<<num_tokens,
1024,...>>>` launches one block per batch row) — the same *class* of
per-row-batched kernel where the DSA topk hypothesis's row-vs-slot addressing
question was worth checking (and was killed there via trace instrumentation,
not code reading alone). A parallel row-addressing/shared-scratch-sizing
check on the MHC kernels specifically (not a precision A/B) is the more
promising next step if MHC stays on the suspect list — not run this pass
(out of scope; this pass tested one specific, now-refuted premise).

## Comprehensive substage-diff round — RadixCache-repeat confound found + KILLED for the original bug; SOLID substage localization inconclusive (2026-07-07)

Resumed a killed prior session (`af6c005853c874bf3`) mid-task on per-substage
instrumentation. **Salvaged, not redone**: `git status` on the pod tree showed
uncommitted local edits to `crates/infer-cuda/src/{dsv4,attention,probe}.rs`
adding `ARLE_PROBE_STAGES` — a per-row, per-substage fingerprint (sum + head/
tail-3 + argmin/argmax of each row's slice) at 8 call sites spanning layers
0–42 (`attn_norm`, `dsa_raw_score`, `compressor_out`, `attn_out`,
`attn_residual`, `ffn_input`, `moe_out`, `ffn_residual` — covering post-
attention, the DSA raw indexer score, compressor output, attention output,
MHC pre/post at both boundaries, and MoE output per the brief). It compiled
clean (`BUILD_EXIT=0`) and matched the brief's substage list, so it was reused
as-is. Also found partial pod-side capture files (`stage_probe.jsonl`,
`stage_manifest.txt`, `diff_stage_probe{,2}.py`, `trace_probe.py`,
`boot_stage_probe.sh`, `drive_stage_probe.sh`, `drive_more_clean.sh`) —
6/26 planned concurrent attempts had real data (conc1–conc6, conc6 corrupted),
but re-running the diff surfaced the killed session's own last note was
correct: **zero position overlap** between the corrupted attempt's captured
window (pos 462–465, where the join event/corruption tick actually falls per
this doc's earlier token-trace round) and the 5 clean attempts' window
(pos 456–460, where clean/terse completions had already hit EOS) — the
`diff_stage_probe.py` "0 divergent points" result on that partial data was
**vacuous** (nothing to compare), not a real finding.

**GPU check.** `scripts/pod.sh gpus` showed GPUs 1/2/3/4/6/7 free (0% util,
~0 MiB) and 0/5 busy (~95%, another tenant) — 6 free GPUs, well above the
TP=4 minimum; no poller, no wait, proceeded directly on GPUs 1/3/4/6.

### Attempt 1: fix the overlap gap with `ignore_eos` — surfaced a bigger, separate bug

To force every trial (clean or corrupted) to run the full 16-token budget
(guaranteeing position-window overlap regardless of completion length), added
`"ignore_eos": true` to a copy of `trace_probe.py`. First concurrent (n=4)
attempt under this harness caught corruption immediately — but so did a
**solo (n=1) sanity check run right before it**: `solo2` (the second-ever
call to the fixed `TRACKED-FIXED...` prompt on that boot) produced
`'The secret access code is 738292. 738292.'` — the exact same digit
substitution (738292 for 738291) documented in every corrupted-concurrent
case all day, but at **n=1, no concurrency at all**.

**Isolated `ignore_eos` as irrelevant.** Reran with the plain (no
`ignore_eos`) `trace_probe.py`, 20 solo (n=1) reps back-to-back on one boot:
**20/20 wrong**, byte-identical output every time
(`'The secret access code is 738292.'`). This is not racy — it's
deterministic. The first-ever call to this exact prompt on a boot is correct;
every subsequent identical-prompt call is wrong, forever, for the rest of
that boot's life.

**Decisive A/B: `ARLE_DISABLE_PREFIX_CACHE=1` eliminates it completely.**
Rebooted with the diagnostic prefix-cache-off toggle (left in place from an
earlier round) and reran the identical 20-rep solo-repeat sweep:
**20/20 correct** (`'The secret access code is 738291.'` / `'738291'`,
matching natural EOS-driven length variation, zero corruption). Cache ON
20/20 wrong vs cache OFF 20/20 correct, same prompt, same boot type, only the
toggle differs — clean, decisive, 100%-reproducible A/B.

**This is a genuinely separate bug from the one this doc has been chasing all
day**, isolated to a harness design point: `trace_probe.py`'s
`TRACKED-FIXED...` prompt is intentionally byte-identical across every
solo/concurrent call within a boot (needed for token-level tracking). Once
any call populates `RadixCache` for that exact token sequence, every later
call reading that cached prefix — solo or concurrent, doesn't matter —
produces a wrong decode, deterministically. **This directly implicates the
`Token-ID-level diff trace`, `Logit-lens layer diff`, and `MHC TF32`-blocked
rounds above**, all of which used this same fixed, repeated prompt as their
"solo reference" / "TRACKED row" — those rounds' "corrupted concurrent vs
clean solo" comparisons may have been characterizing *this* RadixCache-reuse
defect (or a superset including it), not cleanly isolating the original
n≥3-concurrency-only race. (The earlier `ARLE_DISABLE_PREFIX_CACHE` A/B in
the KV-reuse section did **not** already cover this: it ran against
`concurrent_needle_v3.py`, whose every prompt is uniquely salted per trial —
structurally incapable of ever hitting a stale self-prefix, so that test
never touched this mechanism.)

**Root cause not localized this pass** (out of scope — this was a discovery,
not a chase): whether the wrong cached value stems from an incomplete
snapshot/restore of DSv4's non-KV per-request state (compressor ring
position, DSA indexer history, MHC intermediate buffers — the exact class of
gap the `2026-06-06 DSv4 EAGLE rollback` anchor already found once, i.e.
`truncate_decode_len` restoring `compressed.seq_len` but not
`pending_kv`/`prev_overlap`) versus a lossy KV store/reload path, is an open
question for a dedicated follow-up. **Left in the tree as a live, licensed
follow-up target** — cheap to reproduce (20 reps, ~10s, one boot, one toggle).

### Confirmed the original (fresh-content, n≥3) bug is real and independent of the RadixCache-repeat defect

Wrote `concurrent_stage_probe.py` — same digit-needle design, but **every
row's prompt is uniquely salted every call** (no repeated exact prompt,
structurally immune to the RadixCache-repeat defect above). Fresh boot
(prefix cache ON, default), fired the **literal first-ever request** as a
concurrent n=4 call: clean, 4/4 exact. Five more concurrent reps against the
now-warm server: the **uniquely-salted filler rows** (never repeated, never
cache-hit) corrupted at the classic truncation rate (1–3 of 3 non-tracked
rows missing per rep, `'The secret access code is 738.'` /
`'...7382.'`-class truncations) while a row using the OLD fixed/repeated
`TRACKED-FIXED` prompt stayed content-correct throughout (only a cosmetic
leading `"\n\n"` difference after its first repeat) — confirming the
fresh-content bug and the repeat-cache bug are two distinct, independently
reproducible mechanisms, and the original one is still live and unexplained.

### Cleanest reproduction of the day: byte-identical concurrent rows, same instant

Sent **N=4 byte-identical prompts** (same exact text, same token IDs, same
KV-page-population history — a fresh boot, first admissions) concurrently,
with prefix cache OFF (avoids the repeat-cache defect while preserving the
real batching path). Result across 8 trials: a **mix of hit/miss every time**
(e.g. `miss=[0,2,3]`, `miss=[2,3]`, `miss=[1,3]`, `miss=[1,2]`) — proof the
defect is a pure per-row/slot effect, **fully independent of prompt content**
(every row has literally the same tokens), the tightest isolation of the
"requires n≥3" characterization obtained all day.

**Substage diff on this design** (5 mixed trials captured with the full
0–42-layer, full-16-step stage window): re-ran the killed session's
sum-based per-row fingerprint diff, comparing corrupted-output rows against
clean-output rows **within the same batched call** (zero cross-trial timing
noise, zero content confound — only row/slot identity differs).

- **Floor-computation caveat found**: 2 of 5 trials had only ONE clean row
  available, making the clean-vs-clean floor trivially `0` (no pair to
  compare) and the derived threshold degenerate (`max(0*5, 1e-3)`) — this
  falsely flagged nearly every stage/layer/position as "divergent" starting
  at layer 0, which is a floor-computation artifact, not a real finding
  (needs ≥2 clean instances for a meaningful floor; `diff_stage_probe.py`/
  `diff_identical.py` should skip or flag single-clean-row trials, not silently
  produce a near-zero threshold).
- **The 3 of 5 trials with a real floor (≥2 clean rows, floor ≈ 700–2100)
  showed ZERO sum-level divergent points anywhere** in the full captured
  window (all 43 layers × all 8 substages × the row's own full 9-step decode
  range) between eventually-corrupted rows and clean rows.
- A finer per-element check (comparing the `argmax`/`argmin` element-index
  fields, not just the row sum) found a weak, scattered signal in one of the
  three floor-valid trials (`attn_out` argmax cluster-split at layers 3, 5,
  and 39) and zero in the other two — not a single, clean, reproducible
  localization.

**Conclusion: the sum/index-level substage fingerprint this pass reused is
too coarse for this defect class.** This is consistent with (not
contradicting) the earlier `Logit-lens layer diff` round's own
characterization — "a small, layer-local numerical perturbation... not a
single clean early-layer or late-layer culprit" — a whole-row `sum` or a
single extremum index can miss a change confined to one or a few of a row's
~thousands of hidden dimensions. **The exact first-divergent-substage
localization asked for by this round's brief remains open** — the right next
probe is a per-dimension trajectory (max-abs-diff vector, not a scalar sum)
or the LM-head top-1 lens the earlier round already proved sensitive, applied
to *this* pass's byte-identical/same-instant design (which is a strictly
cleaner control than that round's cross-boot solo-vs-concurrent comparison,
since it removes the B=1-CUDA-graph-vs-B>1-batched structural confound
entirely — solo/n=1 decode bypasses `stage_all`'s call site altogether,
running through a separate, uninstrumented CUDA-graph-replay lane, so a
solo-vs-concurrent substage comparison was never apples-to-apples in the
first place; same-instant same-content cross-row comparison is).

**Pod-side artifacts left for reuse** (not committed — pod-only harness
scripts, consistent with this doc's existing pattern):
`/host/arle-build/concurrent_stage_identical.py` (byte-identical-prompt
harness — the cleanest repro), `concurrent_stage_probe.py` (unique-salt
harness), `concurrent_stage_probe_fixed.py`, `diff_identical.py`,
`diff_cstage_fixed.py`, `boot_stage_probe_nopfx_wide.sh` (prefix-cache-off,
wide `ARLE_PROBE_STAGE_POS_MIN/MAX=440/510` boot), `drive_cstage_fixed.sh`,
plus the captured `stage_probe.jsonl` (~32 MB, 8 identical-prompt trials'
worth of full-stack substage data) and manifests. The instrumentation itself
(`ARLE_PROBE_STAGES` in `crates/infer-cuda/src/{dsv4,attention,probe}.rs`,
off by default) is committed alongside this entry — reusable for the next
pass without re-instrumenting.

## FlashMLA split-KV `num_splits` batch-invariance hypothesis — KILLED, quantitatively (2026-07-07)

External lead (Thinking Machines Lab, "Defeating Nondeterminism in LLM
Inference"): split-KV decode kernels commonly pick `num_splits` to saturate a
*fixed* SM budget using the *aggregate* work across all concurrently-batched
rows, so a row's own reduction order (and thus its exact FP result) depends on
which other rows share its batch — a documented industry bug class, invisible
to race-detection tools (matches this investigation's own tool-negative
results). `flashmla.rs:940-982`'s `sched_meta_for_batch` doc comment ("the
cached-constant pitfall... wrong split-KV merge for n>1") pattern-matches.

**Source trace.** `sched_meta_for_batch` → vendored
`get_mla_metadata_kernel` (`vendor/flashmla/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu:30-126`):
`payload = ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks`,
where `total_num_blocks` sums `(blocks + overhead)` over **every row in the
current batch** (line 70) and `num_sm_parts` is a **per-decode-shape constant**
(`h_q`/`s_q` only, `arle_flashmla_decode_shim.cu:100-139` — no `n` term,
fixed once at layer-state construction). Rows are then walked **sequentially**
(row 0 fully consumed before row 1 starts, lines 92-110) — so a row's OWN
split boundaries depend on batch composition **only through `payload`**, i.e.
only when aggregate demand crosses a `num_sm_parts` quantization boundary.

**Quantitative kill, not a source-reading inference.** Added a temporary
env-gated readback (`ARLE_DSV4_SCHED_TRACE`, reverted after use — see below)
printing `num_sm_parts`, `topk_len`, and the resulting `num_splits` per call.
Pod-verified (TP=4, GPUs 2-5, DeepSeek-V4-Flash-FP8, `trace_probe.py` n=4/len=500,
6 reps — corruption reproduced, `miss=[2,3]` etc., consistent with the
established rate): **`num_sm_parts=78`** (H20). Observed live:

| layer_topk | n | num_blocks/row | payload | num_splits/row |
|---|---|---|---|---|
| 128 | 3 | 2 | 6 | 2 |
| 256 | 3 | 4 | 6 | 4 |
| 640 (SW 128 + index_topk 512, this checkpoint's max) | 3 | 10 | 6 | 10 |

Hand-computing the same formula at `n=1` for all three rows gives **the
identical `payload=6` and identical per-row `num_splits`** in every case —
because `total_num_blocks` (≤ 45 at `n=3`, ≤ 60 at `n=4` for the worst-case
`topk=640`) never approaches `num_sm_parts=78`, so `ceil(total/78)` floors at
1 regardless of `n`. The quantization boundary this mechanism needs only
appears at `n≳6` for this checkpoint's max `topk=640` layers (`6×15=90>78`).

**Verdict: real bug class, wrong repro.** The mechanism is genuine (confirmed
in the vendored scheduler, not fabricated) but **cannot fire at the
established n=3/4 repro conditions** — solo and n=3/4 compute byte-identical
split boundaries for every observed `topk`. This does not merely fail to
explain the corruption; it's structurally incapable of causing it at this
investigation's repro shape. Does not rule out the same mechanism mattering at
`n≥6-8` (tested elsewhere in this doc, not decisively separated by `n` in
those aggregates) — out of scope this round.

**Fallback candidates (per this doc's own DSv4-specific shortlist) — same
grep, no batch-size-dependent scheduling found:**
- `deepseek_v4_topk_transform_kernel` (DSA topk,
  `dsv4_dsa_official.cu:866`) — `<<<batch_size, FIXED_BLOCK>>>`, one block per
  row, no `num_sm`/occupancy term.
- `dsv4_compressor_update_batched_kernel` (`dsv4_attention.cu:1525`) — same
  `<<<n, FIXED_BLOCK>>>` shape.
- DSv4's actual decode-MoE kernels, `dsv4_fp8_grouped_swiglu_decode_kernel`/
  `dsv4_fp8_grouped_down_decode_kernel` (`dsv4_fp8_decode_moe.cu:332,359`) —
  fixed-tile hand-written kernels (`blockIdx.{x,y,z}` = row-tile/chunk/expert),
  no `num_sms`/wave-quantization config selection. (DeepGEMM's own
  `num_waves`-driven GEMM config selection, `deepgemm_native.cu:595-618`, is
  real but not on this decode call path — the decode lane uses the hand-written
  kernels above, not DeepGEMM's dynamically-configured grouped GEMM.)

**No fix applied.** The diagnostic readback (`flashmla.rs`,
`ARLE_DSV4_SCHED_TRACE`) was reverted after extracting the numbers above —
narrow, single-purpose, no ongoing reuse value unlike this doc's other
left-in-place traces, so not kept.

## Rule (addendum)

**A real, industry-documented bug class is not evidence without the
batch-size arithmetic.** "The vendored kernel's `num_splits` is a function of
aggregate batch demand" is true and mechanism-plausible, but the actual
quantization boundary (`ceil(total_demand / num_sm_parts)`) only bites once
aggregate demand crosses a large fixed constant (`num_sm_parts=78` on H20) —
reading the kernel source correctly still isn't a license to conclude it's
*active* at a specific `(n, context_len)` repro without plugging in the real
numbers (`num_sm_parts`, `topk_len`) pulled from the device. Same lesson as
this doc's `CUDA_LAUNCH_BLOCKING`/`compute-sanitizer` rounds, applied to
arithmetic instead of tooling: compute the actual threshold, don't infer
activity from source shape alone.

## Batch-invariance sweep 2/3 — RMSNorm KILLED structurally, `dsv4_fp8_gemv_batch_cuda` KILLED by arithmetic proof; `proj_batched`'s DeepGEMM-FP8-vs-bf16 switch CONFIRMED-CANDIDATE (2026-07-07)

Continuing the Thinking Machines / vLLM / SGLang "batch invariance" framing
past the FlashMLA `num_splits` mechanism (killed above): checked the other two
documented mechanisms — RMSNorm data-parallel↔split-reduction switching, and
GEMM Split-K/tile-config switching — plus a full grep for any other
batch-count-conditional dispatch in the DSv4 kernel call chain.

### 1. RMSNorm data-parallel↔split-reduction switching — KILLED structurally, no GPU run needed

Every RMSNorm/LayerNorm kernel DSv4's decode path touches
(`crates/cuda-kernels/csrc/misc/norm.cu`: `rms_norm_kernel`,
`rms_norm_batched_kernel`, `fused_add_rms_norm_batched_kernel`,
`rms_norm_batched_f32_in_kernel`; `crates/cuda-kernels/csrc/misc/dsv4_mhc.cu`:
`dsv4_mhc_pre_rms_norm_kernel`, `dsv4_mhc_params_pre_rms_norm_kernel`) is
launched `<<<seq_len (or num_tokens), FIXED_BLOCK>>>` — **one block per row,
always**, with a compile-time-fixed block size (`NORM_BLOCK=256` in
`norm.cu`, `1024` in `dsv4_mhc.cu`) and a fixed warp-shuffle + shared-memory
tree reduction (`warp_reduce_sum`/`block_sum`) inside every kernel, with zero
branch on `n`/`num_tokens`/`seq_len` inside the reduction itself. The Rust
call sites (`crates/infer-cuda/src/ops.rs::rms_norm_batch`/`rms_norm_vec`)
dispatch unconditionally to `rms_norm_batched_cuda`/`rms_norm_cuda` — no
batch-size-conditional kernel selection exists at the call-site level either.

Grid size (`seq_len`) only changes how many *independent, per-row* blocks run
side-by-side in the SAME kernel launch — it cannot alter a single row's own
reduction order, since each block's arithmetic (which elements it sums, in
what order, via which shuffle/shared-mem steps) is a pure function of
`hidden_dim` and `threadIdx.x`, never of `blockIdx.x`'s sibling count. This
matches the reasoning already established for the FlashMLA/scalar-kernel
distinction in this doc's KV-page-reuse section — the same *class* of
argument, verified against the actual kernel body rather than assumed.
**Verdict: KILLED, structural** — this codebase's RMSNorm kernels are
batch-invariant by construction; no code path exists that could vary with
batch size, so no GPU run was needed to rule it out.

### 2a. `dsv4_fp8_gemv_batch_cuda` (MHC `mix_fn` / low-rank GEMV path) — prior round's claim VERIFIED, refined to arithmetic proof

Re-read `crates/cuda-kernels/csrc/gemm/quantized_gemv.cu:2561-2590`
(`dsv4_fp8_gemv_batch_cuda`) directly, per the brief's instruction not to
trust the prior report's summary. **The prior claim ("never takes a
tensor-core Split-K path, always the scalar GEMV kernel") is INCOMPLETE**:
the dispatcher genuinely selects between kernels by batch size —
`B==1` → `dsv4_fp8_gemv_batch_kernel` (one column per block-column,
`grid.y=1`); `B>1` → `dsv4_fp8_gemv_batch_tiled_kernel<TILE>` with
`TILE ∈ {2,4,8,16,32}` chosen by `B`'s bracket
(`quantized_gemv.cu:2578-2582`) — a real batch-size-conditional kernel
*selection*.

**But this selection is arithmetically batch-invariant, proven at the source
level, not inferred.** `dot16_with_decoded` (the tiled kernel's per-column dot
product, `quantized_gemv.cu:357-384`) and `fp8_f32_dot16` (the untiled
kernel's, `:321-350`) are **byte-identical expression trees** — same 16 terms,
same left-to-right `+` order, same intermediate types — the comment at
`:355-356` ("Same arithmetic + accumulation order... so numerics are
identical") checks out against the actual code, not just the prose. Both
kernels use the same `GEMV_THREADS=256`/`GEMV_ROWS=4` constants, the same
`tid_in_row`/`threads_per_row` thread-to-K-range assignment, the same
`warp_reduce_sum`, and the same shared-memory inter-warp combine order — for
a FIXED (row, batch-column), the untiled and every tiled-`TILE` variant
compute the identical sequence of floating-point operations. The `TILE`
selection only changes how many *sibling* columns amortize one decoded-weight
read per k-chunk; it never reorders or reshapes one column's own reduction.
(The K-alignment fallback path, `(K%16)!=0`, does have a different loop
nesting order between tiled/untiled variants — but confirmed against the live
checkpoint's `config.json` (`hidden_size=4096`, `q_lora_rank=1024`,
`o_lora_rank=1024`, `head_dim=512`, `weight_block_size=[128,128]`) every real
K this kernel is called with is a multiple of 128, so `K%16==0` always holds
here — the fallback path is unreachable at this model's dims, moot.)
**Verdict: KILLED, structural** — real batch-size-conditional kernel
*selection* exists, but is proven arithmetically invariant per-row/per-column;
no GPU run needed since the claim is an exact code-equivalence, not an
inference.

### 2b. `proj_batched` (compressor/indexer/wqkv batched-decode projection) — CONFIRMED-CANDIDATE, pod-verified

Grepping the indexer/compressor call chain (the brief's actual target,
`crates/infer-cuda/src/attention/dsa.rs` + `dsv4.rs`) past the MHC GEMV led to
a different, genuinely precision-switching dispatcher:
`proj_batched` (`crates/infer-cuda/src/attention.rs:7700-7714`), the single
routing point for the compressor's `wkv`/`wgate` projections
(`compressor_batch_prepass`), the DSA indexer's `wq_b`/`weights_proj`
projections (`indexer_query_batch_prepass`), and (a structurally identical
sibling branch) the fused `wq_a`/`wkv` LoRA projection
(`attention.rs:4968`, `mla_attention_prepare_proj_batch`):

```rust
match (cache, scratch) {
    (Some(cache), Some(scratch)) if input.seq_len > 1 => {
        prefill_proj_deepgemm(ctx, scratch, cache, input, out)   // FP8-quantized, tensor-core DeepGEMM
    }
    _ => dsv4_linear(ctx, weight, input, out),                    // bf16 weight → cublasLt GEMM, full precision
}
```

`cache`/`scratch` are the model-wide FP8 DeepGEMM weight cache + prefill
scratch, populated whenever `dsv4_fp8_linear_deepgemm_enabled()` (= native
DeepGEMM preflight succeeds — **default ON**, `attention.rs:1630-1637`) — true
on every pod build/boot in this whole investigation (confirmed: build log
prints `DeepGEMM native enabled`). `input.seq_len` at these call sites is the
CURRENT DECODE STEP's row count `n` (the batched-decode prepass operates over
`normed`, the full N-row batch — `dsv4.rs:2820-2917`). Per this doc's own
prior substage-diff round: **solo (n=1) decode bypasses this whole batched
prepass entirely**, running through a separate cached-meta/CUDA-graph-replay
single-row lane that never calls `proj_batched`. So the real dichotomy is:

- **n=1 (solo decode):** `proj_batched` is never even called — the compressor
  KV, DSA indexer query/weights, and `wq_a`/`wkv` LoRA projections for THIS
  decode step run through the single-row lane's own bf16/fused-scalar path.
- **n≥2 (any concurrent decode):** `proj_batched` is called with
  `input.seq_len = n > 1`, ALWAYS taking the FP8-quantized DeepGEMM
  tensor-core branch (given the cache/scratch are populated, true by default)
  — a materially different numerical pipeline (bf16 activation → FP8 E4M3
  block-quantize → tensor-core GEMM → dequant) feeding the compressor's
  latent KV and the DSA/CSA sparse-attention indexer's top-k selection score,
  not just a different reduction order within the same precision.

**Pod verification (not inference from source reading alone).** Added a
temporary env-gated trace (`ARLE_DSV4_PROJ_BATCHED_TRACE=1`, reverted after
use, same precedent as `ARLE_DSV4_SCHED_TRACE`) printing which branch
`proj_batched` takes + `input.seq_len`. Built (`BUILD_EXIT=0`, log confirms
`DeepGEMM native enabled`), booted DSv4 TP=4 (GPUs 2/3/4/5, same
`ARLE_DSV4_MOE_BACKEND=allreduce`/`ARLE_DSV4_INCREMENTAL_KV=1`/
`ARLE_DSV4_EXPERT_BACKEND=deepgemm`/`--max-total-tokens 2048` config as every
prior A/B in this doc), ran `concurrent_needle_v3.py` n∈{1,2,3,4}, len=500:

- **`branch=deepgemm_fp8` fired at every observed `seq_len∈{2,3,4}`** —
  60395+ trace lines at `seq_len=2` alone across the sweep, **100% of
  observed n≥2 calls**, confirming the FP8 branch is not just reachable but
  the ONLY branch taken for every concurrent-decode step in this repro.
- **`branch=scalar_bf16` never fired, at any `n`, including n=1** — confirming
  solo decode bypasses `proj_batched` entirely (the single-row lane doesn't
  route through this function at all, matching the prior substage-diff
  round's finding).

(One instrumentation defect, same failure mode this doc already documented
once for `eprintln!` under concurrent TP-rank writers: multiple format
arguments in one `eprintln!` are not one `write()` syscall, so 4 TP-rank
processes sharing a log fd interleaved trace lines byte-wise. Cosmetic only —
`grep -o "branch=... seq_len=[0-9]"` recovers clean per-line counts regardless
of interleaving, and the counts above are read that way.)

**n=2 corrupts — and at a rate exceeding the established n≥3 baseline,
overturning this investigation's own "requires n≥3" framing.** The entire
prior investigation swept `n∈{3,4,6,8}` (`job2_ab.sh` and every derived
harness) — **n=2 was never tested**, so "requires n≥3" was an artifact of
which values happened to get swept, not a verified floor. Two independent
same-config boots, `concurrent_needle_v3.py` len=500:

| Boot | n=1 (solo) | n=2 | n=3 | n=4 |
|---|---|---|---|---|
| run1 | 2/3 exact (33% miss, n=3 tiny) | 5/30 exact (**83.3% miss**) | 16/24 exact (33.3% miss) | 13/32 exact (59.4% miss) |
| run2 (replication) | 8/10 exact (20% miss) | 25/40 exact (**37.5% miss**) | — | — |
| combined n=1 vs n=2 | 10/13 exact (23.1% miss) | 30/70 exact (**57.1% miss**) | | |

Same failure signature as every prior round in this doc — truncation
(`'The secret access code is 738.'`, both rows in `run1`'s tid=4/tid=6) and
digit corruption — confirmed by reading the actual decoded text, not just the
miss count (case-as-fact). n=2's miss rate is **2-4x n=1's floor in both
independent boots**, comparable to or exceeding n=3/n=4's rate in the same
boot (not monotonically increasing with n, but decisively non-zero and
elevated starting exactly at n=2) — matching `proj_batched`'s own structural
boundary (`input.seq_len > 1`, i.e. any n≥2) far better than a hypothesis that
requires n≥3 specifically.

**Verdict: CONFIRMED-CANDIDATE — the strongest lead this investigation has
produced.** A real, structurally-verified batch-count-conditional precision
switch (bf16 cublasLt at n=1, FP8-quantized DeepGEMM tensor-core GEMM at
n≥2) exists across four decode-batch projections feeding the compressor's
latent KV + DSA/CSA indexer's top-k selection score, is confirmed via live
trace to be the ONLY branch taken at every observed n≥2 in the real repro,
and its on/off boundary (n=1 vs n≥2) matches a newly-measured corruption
onset at n=2 that the whole prior investigation had never tested. This is
independently consistent with three signatures already established
elsewhere in this doc without this mechanism in view: the **mid-stack
(layer ~19-21) divergence onset** (squarely in DSv4's CSA/DSA-indexer +
compressor territory), the **numeric-vs-text 33x needle-content gap** (FP8
quantization noise plausibly flips a close-call numeric-token decision while
leaving a word completion's dominant candidate untouched), and the
**per-row-independent corruption on byte-identical concurrent prompts**
(FP8 block-quantization of each row's own activation is a per-row op, so
per-row-random near-tie flips are the expected signature, not
batch-position-dependent aliasing). **Not yet root-caused to "this is THE
bug"** — no fix attempted (out of scope this round) and no isolated
single-variable A/B (disabling native DeepGEMM entirely, the only available
lever, would also flip `dsv4_decode_proj_deepgemm_enabled`/
`dsv4_prefill_proj_deepgemm_enabled`/`dsv4_fused_wqkv_decode_enabled`'s
siblings and the `ARLE_DSV4_EXPERT_BACKEND=deepgemm` MoE requirement
simultaneously — a genuine multi-variable confound flagged for the next
pass, not run this round to stay within scope: "confirm, don't fix").

### 3. Other batch-count-conditional dispatch — full grep, one more hit (already covered), rest benign

Grepped `dsv4.rs`/`attention.rs`/`moe.rs`/`hc.rs`/`attention/dsa.rs` for
`if num_tokens`, `if batch_size`, `if n <`, `seq_len >`/`seq_len ==`-style
branches selecting between kernel variants:

- **`proj_batched`'s `input.seq_len > 1` branch** (attention.rs:7709) — the
  confirmed candidate above, counted once.
- **`mla_attention_prepare_proj_batch`'s `token_count == 1` (fused_wqkv) vs
  `token_count > 1 && dsv4_fp8_linear_deepgemm_enabled()`** branch
  (attention.rs:4957-4968) — the sibling of the above for the `wq_a`/`wkv`
  fused LoRA projection; same mechanism, same gate, not a separate finding.
- **`dsv4_fp8_gemv_batch_cuda`'s `B==1` vs `B>1` kernel selection**
  (quantized_gemv.cu:2569) — covered in §2a, proven arithmetically invariant.
- **`dsv4_flashmla_decode_batched_enabled`/`sched_meta_for_batch`'s
  `num_splits`** — already covered and KILLED quantitatively in the FlashMLA
  section above (this doc, same day).
- **DeepGEMM's own `num_waves`-driven grouped-GEMM config selection**
  (`deepgemm_native.cu:595-618`) — real, but (per the prior FlashMLA-round's
  own grep, re-confirmed) not on DSv4's actual decode-MoE call path; the
  decode lane uses the hand-written fixed-tile
  `dsv4_fp8_grouped_swiglu_decode_kernel`/`dsv4_fp8_grouped_down_decode_kernel`
  instead, no `num_sms`/wave-quantization term.
- **No other `if n ==`/`if batch_size ==`-style branch found** in the
  DSA/CSA/MHC/MoE kernel call chain beyond the ones above — every other
  batched kernel launched in this call chain (`deepseek_v4_topk_transform_kernel`,
  `dsv4_compressor_update_batched_kernel`, the FP8 grouped decode-MoE kernels)
  is `<<<n, FIXED_BLOCK>>>`/`<<<num_experts, FIXED_BLOCK>>>` with no
  occupancy- or batch-size-conditional launch-config selection, consistent
  with the RMSNorm finding in §1.

## Rule (addendum 2)

**"Requires n≥K" is a property of what was swept, not a verified floor, until
the boundary value is tested.** This whole investigation characterized the
bug as "n≥3" from its very first localization round and never revisited that
premise — every subsequent sweep (`job2_ab.sh`, `job3_text_needle.sh`, the
Qwen3/Qwen3.6 controls) inherited `n∈{3,4,6,8}` without re-testing n=2. n=2
turned out to corrupt at 37-83% across two independent boots (vs n=1's
20-33% floor) — a boundary value the whole investigation had silently assumed
clean. Before trusting an established repro envelope's stated boundary, test
the boundary itself, not just values comfortably inside it.

**A batch-size-conditional kernel *selection* is not automatically a
batch-invariance bug — check whether the selected variants are
arithmetically identical before or after concluding "confirmed."** §2a's
`dsv4_fp8_gemv_batch_cuda` genuinely dispatches to different kernel code by
`B`, exactly matching the industry mechanism's shape, yet is proven
byte-for-byte invariant per-column by reading the two dot-product functions
side by side — a real dispatch branch is necessary but not sufficient
evidence; the actual arithmetic (or a live trace + A/B, per §2b) decides it.

## `proj_batched` bf16-force A/B (Experiment B) — PARTIALLY CONFIRMS: precision-path is a real, measured contributor, not the sole cause (2026-07-07)

Single-variable follow-up to §2b's CONFIRMED-CANDIDATE: hold `proj_batched`'s
kernel selection **constant** (always bf16 cublasLt, never the FP8 DeepGEMM
branch) and re-run the exact same n=1×10/n=2×20/len=500 sweep
(`job_projtrace2.sh`, unmodified) that produced §2b's 20-33%(n=1)/37.5-83.3%(n=2)
table, on the same config (TP=4, GPUs 2/3/4/5, `ARLE_DSV4_MOE_BACKEND=allreduce`,
`ARLE_DSV4_INCREMENTAL_KV=1`, `ARLE_DSV4_EXPERT_BACKEND=deepgemm`,
`--max-total-tokens 2048`, `DeepSeek-V4-Flash-FP8`).

**Experiment A (force FP8 at n=1) — not run, precondition fails.** Re-checked
against the actual call graph before attempting it: `executor.rs:2865`
special-cases `batch.rows.len() == 1` into `forward_decode_row` →
`forward_tokens_decode_graph`, a CUDA-graph-replay single-row lane that never
calls `Dsv4Model::forward_decode_batch` (`dsv4.rs:2542`, "N=1 never reaches
this function") and therefore never calls `proj_batched` at all — confirmed by
static read, matching §2b's own finding. Flipping `proj_batched`'s gate is
therefore a no-op for n=1 regardless of direction; the only way to make n=1
take the FP8 branch would be to ALSO force it through the batched multi-row
lane, which confounds precision with lane-selection (graph-replay vs not,
different scratch, different keepalive discipline) — not a single-variable
test. Skipped per this doc's own "isolate confounders" discipline rather than
run a confounded experiment.

**Experiment B (force bf16 at n≥2) — one-line change, pod-verified.**
`crates/infer-cuda/src/attention.rs:7700-7714`, changed only the match guard:

```rust
(Some(cache), Some(scratch)) if false && input.seq_len > 1 => {   // was: input.seq_len > 1
```

Built (`BUILD_EXIT=0`, `--release --features cuda,nccl` — TP=4 serve needs
`nccl`, a build-config gap hit and fixed en route, not a code change), booted,
ran the sweep on a fresh port:

| Arm | n | Requests | Exact | Miss | Miss rate | Digit-corruption instances |
|---|---|---|---|---|---|---|
| bf16-forced (this experiment) | 1 | 10 | 7 | 3 | 30.0% | 0 |
| bf16-forced (this experiment) | 2 | 40 | 28 | 12 | 30.0% | 4 (10.0%) |
| default FP8 (§2b, run1+run2 combined) | 1 | 13 | 10 | 3 | 23.1% | — |
| default FP8 (§2b, run1+run2 combined) | 2 | 70 | 30 | 40 | 57.1% | — |

n=2's overall miss rate drops from the established 57.1% (default FP8) to
30.0% (bf16-forced) — landing almost exactly on n=1's own floor (30.0% here,
20-33% established) instead of 2-4x above it. Read as raw miss rate this looks
like a clean confirmation.

**But decoded at the case level (case-as-fact, not aggregate-only) it's only a
partial explanation.** Breaking n=2's 12 misses down by failure signature
(same method as this doc's original Root Cause section): 8/12 are truncation
(`'The secret access code is 738.'`-style, same signature seen throughout this
doc, mechanism outside `proj_batched`'s scope) and **4/12 are digit
corruption** — all four are the *same* wrong string, `'**7381239**'` (needle
`738291`), recurring byte-identical across four different trials with
byte-distinct salted prompts (tid=14, 20, 26, 27). Digit corruption did **not**
go to zero.

**Root cause of the residual 4/40: the untouched sibling switch, not a gap in
the hypothesis.** §2b's own text already named a second, structurally
identical FP8 gate that this experiment deliberately did not touch, per the
brief's "change ONLY the seq_len gate condition, nothing else" scope:
`mla_attention_prepare_proj_batch` (`attention.rs:4968`,
`token_count > 1 && dsv4_fp8_linear_deepgemm_enabled()`), which projects the
fused `wq_a`/`wkv` LoRA for the *same* n=2 decode step and still takes its FP8
DeepGEMM branch in this experiment (this branch's bf16 alternative is not a
one-line flip — it computes into different-shaped intermediates via a
completely different fused-decode code path sized for B=1 — genuinely out of
scope for a single-gate change). The residual corruption's determinism (same
exact wrong digit string 4/4 times, not 4 different random near-tie flips) is
consistent with a systematic FP8 quantization bias in that remaining path
producing a repeatable wrong logit at the same near-tied token position,
rather than evidence against the precision-path mechanism.

**Verdict: PARTIALLY CONFIRMS — precision-path is a real, measured, causal
contributor to the n≥2 digit-corruption failure mode (not batch-count per
se), but `proj_batched` alone is not the sole source; a second FP8 gate
(`mla_attention_prepare_proj_batch`) on the same decode step remains
unattributed.** This is the strongest evidence this investigation has
produced: a single-line, single-variable change to one of (at least) two
known FP8-switching call sites cut n=2's overall miss rate by ~1.9x (57.1% →
30.0%) and brought it to parity with n=1's own floor, while the residual
digit-corruption instances land entirely on the one sibling gate this
experiment left untouched by design. Next clean step (not run this round,
same scope-discipline reason Experiment A was skipped): force bf16 at BOTH
gates simultaneously — if digit corruption then reaches zero at n=2, that
closes the loop from "confirmed-candidate" to "confirmed root cause."
Reverted cleanly after the run (`git diff` clean, pod tree re-synced to match).

## Second FP8 gate bf16-force — BLOCKED, no bf16 weight exists for this gate (2026-07-07)

Follow-up to Experiment B's own next step: force bf16 at the sibling gate too
(`mla_attention_prepare_proj_batch`, cited last round as "attention.rs:~4968")
and see if the residual 4/40 `7381239`-vs-`738291` corruption goes to zero.
**Blocked before any code was touched** — read the gate and its fallback branch
first, per this round's own brief, and the premise doesn't hold.

**Line-number correction.** attention.rs:4968 is inside `mla_attention_prepare`
(the single-row/prefill/chunked-prefill PREPARE, called only from `mla_attention`,
`attention.rs:3918`) — **not** `mla_attention_prepare_proj_batch`, which is a
separate function starting at `attention.rs:5435` and is the *only* caller on the
n≥2 batched-decode path (`dsv4.rs:2846`, the sole call site). Last round's citation
conflated the two similarly-named functions; `mla_attention_prepare`'s
`token_count > 1` branch fires only for a single request's own multi-token
prefill, never for concurrent-decode batching. The actual gate for this
investigation is `mla_attention_prepare_proj_batch`'s `use_deepgemm`
(`attention.rs:5511-5515`:
`dsv4_fp8_linear_deepgemm_enabled()? && attention.wqkv_a_deepgemm.is_some() &&
attention.wq_b_deepgemm.is_some() && prefill_shared.is_some()`), guarding the
same `wq_a`/`wq_b`/`wkv` LoRA projections as `proj_batched`'s sibling call but for
the main MLA Q/KV path instead of the compressor/indexer.

**Premise check: does the fallback branch compute in bf16?** Read the `else`
branch (`attention.rs:5554-5588`) — it calls `dsv4_linear(ctx, &attention.wq_a,
normed, &mut c_q)` / `&attention.wq_b` / `&attention.wkv`, the same three
`DeviceMatrix`es. `dsv4_linear` (`attention.rs:1566-1579`) dispatches on
`weight.weight_format`: `DenseBf16` → `gemm_batch` (bf16 cublasLt, the true bf16
path); `Dsv4Fp8BlockScaled`/`Dsv4Fp4BlockScaled` → `mla_linear`
(`attention.rs:992-1058`), which for `Dsv4Fp8BlockScaled` calls
`ffi::dsv4_fp8_gemv_batch_cuda` — **the exact kernel this same investigation
already analyzed in "Batch-invariance sweep 2/3" §2a and proved byte-for-byte
arithmetically identical between its `B==1` and `B>1` tiled-dispatch variants**
(`quantized_gemv.cu:321-350` vs `:357-384`, same 16-term expression tree, same
accumulation order).

**Checkpoint-level verification (not source inference alone) — no bf16 raw
tensor exists for these weights.** `crates/infer-cuda/src/loader.rs:3737-3746`:
for non-GLM DSv4 (this checkpoint), `wq_a`/`wkv` load via
`load_dsv4_block_scaled` (`loader.rs:3024-3073`), which **requires the raw
checkpoint tensor dtype to be `F8_E4M3`** — `bail!`s on anything else (I8/FP4
fail-closed per #137, all other dtypes rejected outright; no `BF16`/`F32` arm
exists in this function, unlike the compressor/indexer's dtype-dispatching
`load_dsv4_global_matrix`). Read the actual pod checkpoint header directly
(`/host/DeepSeek-V4-Flash-FP8/model-00005-of-00046.safetensors`, layer 3, both
local grep of `deepseek-spec` tensor-name mapping and a pod-side Python
safetensors-header parse):

| Tensor | dtype | Sidecar `.scale`? |
|---|---|---|
| `layers.3.attn.wq_a.weight` | **F8_E4M3** | yes |
| `layers.3.attn.wq_b.weight` | **F8_E4M3** | yes |
| `layers.3.attn.wkv.weight` | **F8_E4M3** | yes |
| `layers.3.attn.compressor.wkv.weight` | **BF16** | no |
| `layers.3.attn.compressor.wgate.weight` | **BF16** | no |

Confirms the asymmetry directly from the checkpoint bytes: the compressor
weights (first gate, `proj_batched`, already forced last round) really are
bf16 in this checkpoint — Experiment B's "bf16-forced" label was accurate there.
The main MLA `wq_a`/`wq_b`/`wkv` weights (this second gate) are **F8_E4M3 in the
checkpoint with no bf16 sibling tensor at all** — there is nothing on disk to
load as a dense-bf16 alternative.

**Every code path touching `wq_a`/`wq_b`/`wkv` computes in FP8, never bf16** —
checked all three: the B=1 fused-decode path (`run_fused_wqkv_decode`,
`attention.rs:3407-3506`, quantizes into FP8 and runs DeepGEMM tensor-core);
the batched-decode `use_deepgemm=true` branch (`run_fused_wqkv_prefill` +
`prefill_proj_deepgemm`, `attention.rs:5511-5554`, same FP8 DeepGEMM); and the
`use_deepgemm=false` fallback just analyzed above (`dsv4_linear` →
`mla_linear` → `dsv4_fp8_gemv_batch_cuda`, scalar FP8 GEMV). There is no third
option and no dormant bf16 branch to flip on — matching this doc's own
`MHC TF32-prenorm eager-fallback test` precedent (a hypothesis blocked because
the premise it needed didn't exist in ARLE's code, not because the flip was
merely inconvenient).

**Verdict: BLOCKED — not "hard to flip," genuinely nothing to flip to.**
Forcing `use_deepgemm=false` on this gate would not test "bf16 vs FP8" (the
brief's stated hypothesis); it would reroute to `dsv4_fp8_gemv_batch_cuda`,
a *different* FP8 kernel this investigation already proved arithmetically
batch-invariant in an earlier round of the *same day*. Running it would not
distinguish "this gate doesn't contribute to the corruption" from "this gate's
only alternative was already proven to never vary with batch size regardless" —
a confound in the interpretation, not just a perf tradeoff, so declined per this
round's own scope guardrail (item 4: don't force a fix that introduces a new
confound). A genuine bf16 alternative for `wq_a`/`wq_b`/`wkv` would require
adding host-side FP8→bf16 block-dequantization to the loader (`loader.rs`) —
real feature work, not a same-day, single-variable correctness probe.

**Net position, unchanged from Experiment B.** `proj_batched`'s FP8 gate
(compressor/indexer projections) is a confirmed, measured, causal contributor
(57.1% → 30.0% at n=2, digit-corruption reduced but not eliminated). The
sibling gate on the main MLA Q/KV path cannot be given an equivalent bf16 A/B
without new dequantization infrastructure; its only non-DeepGEMM alternative
(scalar FP8 GEMV) was independently already shown batch-invariant, so it's a
weak suspect for the *batch-count-dependent* part of the corruption specifically
(though it remains an FP8-numerics suspect in general, untested). The residual
4/40 `7381239` corruption from Experiment B is still unattributed — next
candidates, per this doc's existing shortlist, are the FP8 grouped-GEMM
decode-MoE kernels (`dsv4_fp8_grouped_swiglu_decode_kernel`/
`dsv4_fp8_grouped_down_decode_kernel`, never precision-A/B'd) or the DSA
`radix_topk` unbounded-write hypothesis (memcheck-blind, still not
source-verified against its round-3 count invariant).

**No code changed this round** (`git diff` clean on both local and pod trees —
the blocking determination came from reading `attention.rs`/`loader.rs` plus a
direct pod-side safetensors-header parse, not from a build/run). No new
pod-side diagnostic left in place; nothing to revert.

## Rule (addendum 3)

**A "bf16 alternative" claim needs the actual weight dtype checked, not
assumed from a sibling code path's comment.** `proj_batched`'s inline comment
("the DSv4 compressor weights are bf16") was correct for *that* function's
weights but doesn't transfer to a structurally similar sibling gate
(`mla_attention_prepare_proj_batch`) guarding *different* tensors — checked
directly against the checkpoint's safetensors header (`F8_E4M3` for
`wq_a`/`wq_b`/`wkv`, `BF16` for `compressor.wkv`/`wgate`, same layer, same file)
rather than inferring from the first gate's precedent. **A same-named-looking
dispatch branch ("FP8 DeepGEMM vs scalar fallback") is not automatically an
"FP8 vs bf16" dispatch** — the fallback can itself be a *different FP8 kernel*
(as it is here, `dsv4_fp8_gemv_batch_cuda`) when the weight was never stored in
bf16 to begin with; verify the fallback's actual output precision before
building an A/B around the assumption that "not-DeepGEMM" means "bf16."

## FP8 grouped-GEMM decode-MoE — DEAD-END, no bf16 alternative in the checkpoint (2026-07-08)

Doc's own #1 residual-corruption shortlist item: the decode-MoE grouped-GEMM
kernels (`dsv4_fp8_grouped_swiglu_decode_kernel`/
`dsv4_fp8_grouped_down_decode_kernel`, `crates/cuda-kernels/csrc/gemm/dsv4_fp8_decode_moe.cu`,
called from `dsv4_moe_forward_decode_fp8`, `crates/infer-cuda/src/moe.rs:3044`)
had never been precision-A/B'd. Checked the checkpoint dtype **before**
attempting any code change, per this round's brief — same discipline as the
"Second FP8 gate" round that killed `mla_attention_prepare_proj_batch` as an
A/B target.

**Weight-load path traced.** `crates/infer-cuda/src/loader.rs:3323`
(`load_dsv4_moe_layer`) loads every routed AND shared expert's `w1`/`w2`/`w3`
(gate/down/up) via `load_fp8 = |name| self.load_dsv4_block_scaled(ctx, name)`
(non-GLM branch) — the exact same function the prior round already proved
`bail!`s on any raw tensor dtype other than `F8_E4M3` (no `BF16`/`F32` arm
exists). Only the router gate (`names.gate_weight`) loads via
`load_dsv4_bf16_matrix` — a different tensor, not on the grouped-GEMM path at
all (it feeds `dsv4_route_kernel`'s routing decision, not the expert compute).

**Checkpoint-level verification (not source inference alone).** Parsed the
live pod checkpoint's safetensors header directly
(`/host/DeepSeek-V4-Flash-FP8/model-00005-of-00046.safetensors`, layer 3):

| Tensor | dtype | shape | Sidecar |
|---|---|---|---|
| `layers.3.ffn.experts.0.w1.weight` (gate) | **F8_E4M3** | [2048,4096] | `.scale` F32 [16,32] |
| `layers.3.ffn.experts.0.w2.weight` (down) | **F8_E4M3** | [4096,2048] | `.scale` F32 [32,16] |
| `layers.3.ffn.experts.0.w3.weight` (up) | **F8_E4M3** | [2048,4096] | `.scale` F32 [16,32] |
| `layers.3.ffn.shared_experts.w{1,2,3}.weight` | **F8_E4M3** (all three) | — | `.scale` F32 |
| `layers.3.ffn.gate.weight` (router, not the GEMM) | BF16 | [256,4096] | none |

Every routed expert (checked expert 0; the loop over `local_expert_start..end`
uses the identical `load_dsv4_block_scaled` call for every index, so this
generalizes) and the shared expert are `F8_E4M3`-only, no bf16 sibling tensor
anywhere in the checkpoint. Identical dead-end shape to the second FP8 gate
(`wq_a`/`wq_b`/`wkv`): the only way to get a bf16 alternative would be adding
host-side FP8→bf16 block-dequantization to the loader — new feature work, not
a same-day precision A/B.

**Verdict: DEAD-END — confirmed via checkpoint header, no code change
attempted, no GPU run needed.** The MoE decode-FP8 grouped-GEMM kernels cannot
be given a bf16 A/B without new dequantization infrastructure, same as the
second gate. This item is closed for this investigation's remaining scope
(same pattern the doc already established once); it stays on the suspect list
only in the weaker "FP8-numerics-in-general" sense that applies to every
still-standing FP8 path here, not as an actionable next lever.

## DSA `radix_topk` round==3 tie-break — real non-invariance found, causal link INCONCLUSIVE (source-only, 2026-07-08)

Per the doc's #2 shortlist item, with #1 now cleanly dead-ended: re-read
`radix_topk` (`crates/cuda-kernels/csrc/misc/dsv4_dsa_official.cu:487-632`)
line by line to check whether the round==3 tie-break (`s_last_remain` atomic,
`:616-618`) produces a SET of selected indices invariant to thread execution
order, or only a COUNT-invariant one — the distinction the brief asked to
resolve.

**Mechanism.** `radix_topk` is a standard MSB-first radix top-k: a coarse
first pass buckets by the top byte of a float→sortable-uint16 key
(`convert_to_uint8`), then 4 further rounds (round 0-3) refine within the
tied bucket using successive bytes of the full sortable-uint32 key
(`convert_to_uint32`, `offset = 24 - round*8`), so by the end of round 3 every
remaining tied candidate has agreed on **all 32 bits** of its sortable key —
i.e. round-3 ties are genuine bit-identical-float ties, not near-ties. Within
round 3, elements landing in the final threshold bin race for the last
`R = s_last_remain` slots via `pos = atomicAdd(&s_last_remain, -1); if (pos >
0) output[topk-pos] = idx`.

**Proof 1 — COUNT is invariant to execution order (verified from the atomic's
own semantics, not a guess).** `atomicAdd` serializes: every call against the
same address returns a distinct value from a strictly-decreasing sequence
starting at `R`, regardless of which thread/idx issues which call or in what
order. Exactly `R` of the `num_input` (≥ `R`, guaranteed by the histogram
cascade that selected `threshold_bin`) calls receive `pos ∈ [1, R]` and write;
the rest get `pos ≤ 0` and are dropped. Total writes = `R` in every run,
every schedule — the kernel can never under- or over-select `topk` total
indices from a batch-composition or scheduling effect. **This kills a
"corrupts the total selected-block count" framing of the hypothesis.**

**Proof 2 — the exact SET of the R winning indices is NOT invariant to
execution order.** `atomicAdd`'s serialization guarantees a decreasing value
sequence, but CUDA gives no ordering guarantee for *which* thread's call is
issued first among threads in different warps/blocks touching the same
`__shared__` address — the assignment of `pos` values to specific `idx`s among
the tied candidates is a genuine function of physical thread-scheduling order,
which is not part of the CUDA execution model's guarantees for a `__shared__`
atomic across warps. **So: COUNT-STABLE confirmed, SET-invariance disproven at
the source level** — this is a real, not hypothetical, race-shaped
nondeterminism, distinct in kind from the already-KILLED FlashMLA/RMSNorm/
`dsv4_fp8_gemv_batch_cuda` mechanisms (those were proven arithmetically
identical regardless of dispatch; this one is proven to genuinely vary).

**Why `compute-sanitizer racecheck` never caught this.** `s_last_remain` is
accessed exclusively through `atomicAdd`, which is by definition
race-free (no unsynchronized read-modify-write) — racecheck flags
*unsynchronized* shared-memory access, not "correctly-synchronized-but-
order-dependent" outcomes. This mechanism is invisible to every tool this
investigation has already run (racecheck/synccheck/memcheck,
`CUDA_LAUNCH_BLOCKING=1`) by construction, not by bad luck — none of them
detect "the atomic is safe but its winner is schedule-dependent."

**Why this is plausible in THIS harness specifically, not asserted from
principle alone.** A round-3 tie requires bit-identical `float` DSA indexer
scores. Genuinely improbable for arbitrary content — but this investigation's
own needle harnesses (`concurrent_needle_v3.py` et al.) pad every prompt with
many repetitions of the same `TOPIC` filler text specifically to reach the
target length, i.e. by design produce many KV-cache blocks with highly
repetitive or identical underlying content. Combined with the DSA
indexer/compressor pipeline's bf16 intermediate precision (established
elsewhere in this doc), bit-identical scores across several filler blocks are
structurally plausible in this specific synthetic-benchmark shape — more so
than they would be in typical non-repetitive production prompts.

**Not batch-size-coupled by construction — a weaker mechanism than the
already-CONFIRMED `proj_batched` gate.** `s_last_remain` is `__shared__`,
scoped per-block, and `deepseek_v4_topk_transform_kernel` launches one block
per row (`bid = blockIdx.x`) — there is no cross-row/cross-block aliasing of
this variable, so batch size `n` cannot directly perturb which winner a given
row's own tie-break selects. The only route from `n` to this mechanism is
indirect: more concurrently-resident blocks (rows) change SM occupancy/
warp-scheduling pressure, which *could* shift one block's own internal thread
execution order relative to a solo (n=1) run of the identical row — but this
is a second-order GPU-scheduler effect, not proven from source, and is a
weaker causal story than `proj_batched`'s direct, unconditional `input.seq_len
> 1` branch switch.

**Verdict: INCONCLUSIVE — real, tool-invisible, structurally-plausible
mechanism confirmed present in source; not confirmed causal for the observed
corruption.** Source-only pass per the brief's own scope (no GPU run
attempted); would need a dedicated instrumented pass (thread a debug flag
through `TopKParams`/the FFI signature to count/log actual round-3
tie-arbitration events — `num_input > R` at round==3 — correlated against the
`7381239`-vs-`738291` repro) to move this from "plausible mechanism" to
"confirmed contributor," comparable effort to the `proj_batched` trace round.
Flagged as the strongest still-open lead for a future pass, ranked above
"genuine data/routing-dependent numerical edge case" (too unfalsifiable to
action directly) and below the two now-closed FP8-precision gates.

## Rule (addendum 4)

**"Race-free" (atomic-serialized) is not the same claim as "output
schedule-invariant."** `compute-sanitizer racecheck` proved `s_last_remain`'s
accesses are properly synchronized (an atomic, no raw read-modify-write hazard)
— and that is a real, correct finding — but it does not and cannot prove the
*outcome* is independent of thread execution order when multiple threads
compete for a shared, strictly-decreasing counter and the counter's exact
value picks a WINNER among candidates. A kernel can pass every race detector
in the toolbox and still be schedule-nondeterministic in its *selection*
logic; separate "is this access safe" from "is this outcome invariant" before
declaring a tie-break-shaped mechanism closed by a clean sanitizer run.

**A checkpoint-header check generalizes across "sibling" weight groups once
the loader function is shared.** The MoE expert weights (`w1`/`w2`/`w3`) use
the exact same `load_dsv4_block_scaled` call as the already-checked
`wq_a`/`wq_b`/`wkv` — reading the *loader call site*, not just the *weight
name*, is what tells you a new dead-end will match a prior one before spending
a pod round confirming it independently; the checkpoint read here was a
confirmation of an already-strong prior, not a blind probe.
