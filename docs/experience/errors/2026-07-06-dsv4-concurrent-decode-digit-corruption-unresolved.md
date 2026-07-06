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
