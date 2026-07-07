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
