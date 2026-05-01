# Longctx Phase 2 - Speculative Decode

> **Status:** Draft v1, opened after Phase 1 SGLang-row c=4 close on 2026-05-01.
> **Owner:** ckl
> **Phase 1 evidence:** `docs/experience/wins/2026-05-01-phase1-close-evictable.md`
> **Mission doc:** `docs/projects/2026-04-30-longctx-32k-128k-leadership.md`

## 1. Goal

Stack a lossless long-context speculative decode path on top of the closed
Phase 1 W1 c=4 SGLang-row foundation. The measured W1 c=4 longctx row secured
`1.469x-1.678x` SGLang. If a `2.0x-2.5x` lossless spec-decode lever stacks
cleanly on the same serving foundation, the mission-level projection is:

| Phase 1 basis | x Phase 2 conservative | x Phase 2 stretch |
|---:|---:|---:|
| 1.469x worst run | 2.94x SGLang | 3.67x SGLang |
| 1.609x mean | 3.22x SGLang | 4.02x SGLang |

The headline target from the handoff is therefore recorded as a **stack
projection**, not the W2 acceptance criterion. The observed worst W1 Phase 1
run (`1.469x`) multiplied by Phase 2's `2.0x-2.5x` lever gives
`2.94x-3.67x`; the rounded program target is `2.94x-3.68x`. For arithmetic
reference only, the older `1.30x` margin floor multiplied by `2.0x-2.5x`
would be `2.60x-3.25x`; that is not the Phase 2 promotion gate.

Phase 2 still must establish its own W2 baseline before promotion; W1 numbers
cannot substitute for W2 acceptance. The remaining W1 vLLM / TRT-LLM /
Mooncake baseline panel also remains required before a full Phase 1 endpoint
claim. The c=1 single-stream decode gap from the Phase 1 evidence entry is a
parallel work track, not a blocker for W2 spec decode.

## 2. Scope

Phase 2 targets W2: long prompt plus long decode, where speculative decode can
amortize verifier cost across a large resident KV cache.

In scope:

- CUDA verifier integration for the existing mixed-batch path.
- Lossless acceptance/rejection semantics: target-model distribution remains
  authoritative.
- Scheduler support for draft/verify accounting and adaptive acceptance rate.
- HTTP and config surface only after internal correctness and bench gates pass.
- Bench wins for L4 first; H100 follows once L4 is stable.

Out of scope:

- Multi-node disaggregated prefill/decode. That is Phase 3.
- Sparse near-lossless attention. That is Phase 4.
- A second heavyweight draft model unless self-speculation fails its gate.

## 3. Existing Assets

- `infer/src/speculative.rs` already has the CPU-side spec config and proposal
  structures (`631` LoC).
- `infer/src/speculative/cuda.rs` exists as the CUDA integration point
  (`157` LoC).
- Phase 1 mixed FP8 KV path now handles the longctx c=4 memory edge and is the
  verifier foundation.
- `docs/projects/2026-04-30-longctx-32k-128k-leadership.md` §8 lists the
  intended sources: MagicDec, TriForce, and LongSpec.

## 3.1 Scheduler / Forward Integration Design Debt

The current speculative framework has CPU verifier scaffolding but zero
scheduler/forward integration. P2.2 must produce a wiring spec before runtime
changes:

- scheduler draft KV accounting: how proposed tokens reserve pages, retire on
  rejection, and interact with prefix-cache eviction;
- verifier batch admission: how draft rows join the mixed batch without
  violating Phase 1 FP8 KV layout and admission budget invariants;
- adaptive acceptance-rate control: where accepted/rejected counters update
  speculation length and when the scheduler falls back to normal decode.

## 3.2 Draft Model Selection Decision Point

Default choice: MagicDec-style self-speculation. It avoids second-model
dependencies, keeps latency overhead smallest, and fits the existing
single-runtime deployment model.

| option | acceptance criteria | default verdict |
|---|---|---|
| MagicDec-style self-spec | no second model dependency; acceptance rate high enough for `>=2.0x` W2; W1 guard unchanged | default |
| TriForce big-small draft | only if self-spec cannot reach `1.25x` after verifier overhead tuning and a small Qwen draft can be loaded without KV/memory regressions | fallback |
| LongSpec retrieval + sliding-window draft | only if long-context acceptance collapses and retrieval/window draft improves W2 while preserving target distribution | fallback |

## 4. Design Requirements

1. **Lossless by construction.** Every accepted token must be verified by the
   target model. Draft output never bypasses the target distribution.
2. **Verifier uses Phase 1 KV format.** Do not add a parallel KV layout for
   speculative decode; use the mixed FP8 path unless profiling proves a hard
   blocker.
3. **Scheduler-visible accounting.** Expose proposed, accepted, rejected,
   verifier batch size, acceptance ratio, and fallback count in service stats
   before running headline benches.
4. **Adaptive speculation length.** Start conservative, grow when acceptance is
   high, shrink quickly when rejection or tail latency rises.
5. **No degradation on W1.** Spec decode may be disabled by default for W1
   until W2 is proven, but the code path must not regress Phase 1 admission or
   c=4 longctx throughput.

## 5. Work Slices

| slice | change | gate |
|---|---|---|
| P2.0 | c=1 single-stream decode profile side track | nice-to-have; does not block W2, but must not regress c=4 |
| P2.1 | Trace current `speculative.rs` and CUDA stubs; write exact integration design | design doc names every touched file |
| P2.2 | Add service counters and no-op config plumbing plus scheduler/forward wiring spec | `cargo test --release`, service stats expose zero counters, design names draft KV accounting + verifier admission + adaptive acceptance wiring |
| P2.3 | Implement verifier micro-batch path behind opt-in flag | deterministic 3-prompt correctness smoke |
| P2.4 | Add self-speculation proposal path | accepted-token counters non-zero; fallback works |
| P2.5 | Adaptive speculation length | acceptance ratio stable; no latency-tail explosion |
| P2.6 | W2 L4 baseline + patched bench | patched W2 `>=2.0x` SGLang and W1 guard stays within Phase 1 close variance |
| P2.7 | H100 replication | same gate on H2 |

Each runtime slice gets its own commit, its own `wins/` or `errors/` entry,
and `codex review --uncommitted` before commit.

## 6. Bench Matrix

Minimum matrix before promoting Phase 2:

| workload | prompt | output | concurrency | purpose |
|---|---:|---:|---:|---|
| W1 guard | 32768 | 256 | 4 | Phase 1 no-regression |
| W2 long-decode baseline | 32768 | 2048 | 4 | establish ARLE and SGLang pre-patch W2 anchors |
| W2 long-decode patched | 32768 | 2048 | 4 | primary Phase 2 lift |
| W2 long-decode saturation | 32768 | 2048 | 8,16 | verify MagicDec-style batch scaling |
| TTFT guard | <2048 | 256 | 1 | prevent short-request harm |
| mixed-mode | 32768 + <2048 | 256 | 4 | ensure long decode does not starve short prompts |
| agent-loop | <2048 repeated prefix | 256 | 4 | prefix-cache hit preservation |

Every row records successful-only TTFT/ITL p50, total output tokens per second,
request accounting, service trace counters, and delta vs the Phase 1 close row.

## 7. Stop Rules

- Stop and write `errors/` if target-distribution correctness fails.
- Stop and redesign if W1 guard drops below the Phase 1 close row's worst-run
  `23.893 out tok/s` without an explained measurement issue.
- Stop and reject self-speculation if W2 lift is below `1.25x` after
  verifier overhead tuning; consider an explicit small draft model only then.
- Promote Phase 2 when patched W2 L4 reaches `>=2.0x` SGLang on its own W2
  baseline and W1 guard remains within the Phase 1 close variance envelope
  (`23.893-27.307 out tok/s`, mean `26.169`).

## 8. Immediate Next Step

Start P2.1 by reading:

- `infer/src/speculative.rs`
- `infer/src/speculative/cuda.rs`
- `infer/src/scheduler/AGENTS.md`
- `infer/src/scheduler/cuda/`
- `infer/src/model/qwen3/batch_decode.rs`

The P2.1 output should be a design note with exact file ownership, failure
modes, and the first no-op counter patch.
