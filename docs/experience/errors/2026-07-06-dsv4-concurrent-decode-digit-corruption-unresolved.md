# DSv4 concurrent-decode digit corruption — localized, not fixed

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
this lane entirely) could not be used for a clean A/B: that fallback path's
KV-pool sizing is still keyed to the FlashMLA banded layout and
admission-rejects almost every request at typical `--max-total-tokens`
ceilings — tracked separately as a blocker (task #7 in this session, needs
its own doc entry once picked up). Confirming the hypothesis needs
`cuda-memcheck`/`compute-sanitizer racecheck` or an nsys trace correlating
split-count with the corrupted row — source reading alone cannot localize a
race to the exact buffer/line.

## Rule

Concurrent DSv4 serving (n>=3, prompt length >~100-250 tokens) has a real,
unresolved silent-corruption risk today — not safe to treat as a solved
baseline when A/B-testing other DSv4 decode-path changes. Case-as-fact
paid off here: the aggregate "~40% failure" number alone pointed nowhere;
decoding actual failing text (truncation vs. digit corruption, first-3-
correct/last-N-wrong) narrowed the search to one subsystem before any code
was touched.
