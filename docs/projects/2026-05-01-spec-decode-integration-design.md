# Spec Decode Integration Design

> **Status:** P2.1 design, no code changes.
> **Mission parent:** `docs/projects/2026-04-30-longctx-32k-128k-leadership.md`
> **Phase 2 plan:** `docs/plans/2026-05-01-longctx-spec-decode-phase2.md`
> **Phase 1 evidence:** `docs/experience/wins/2026-05-01-phase1-close-evictable.md`

## Goal

Wire the existing speculative-decoding scaffolding into the CUDA scheduler
without disturbing the closed W1/c4 SGLang-row admission path.

Current state:

- `infer/src/speculative.rs` is `631` LoC and already owns `SpecConfig`,
  `TokenProposal`, `VerificationResult`, `verify_tokens`,
  `AcceptanceTracker`, and `DraftModel`.
- `infer/src/speculative/cuda.rs` is `157` LoC and owns a CUDA `DraftEngine`
  skeleton, but it only drafts from a second Qwen3 model and returns
  placeholder target probabilities.
- Scheduler and model forward paths have no speculative execution branch today.

Non-goal for P2.1: implement runtime behavior. This document names the wiring
points and file ownership for P2.2-P2.5.

## Current Code Trace

### Speculative Scaffolding

| file:line | current behavior | integration gap |
|---|---|---|
| `infer/src/speculative.rs:46` | `SpecConfig` has `num_speculative_tokens`, `vocab_size`, `min_acceptance_rate`. | Needs scheduler-facing enable flag, max draft tokens, acceptance window, and per-request override policy before serving. |
| `infer/src/speculative.rs:90` | `TokenProposal` stores `tokens`, scalar `draft_probs`, scalar `target_probs`, and optional `target_bonus_dist`. | Verification needs target probabilities filled from target-model logits, not the draft skeleton. |
| `infer/src/speculative.rs:200` | `verify_tokens` implements CPU rejection sampling from scalar draft/target probabilities. | Production must keep the target-model distribution authoritative and preserve normal sampling semantics on rejection. |
| `infer/src/speculative.rs:271` | `AcceptanceTracker` has rolling mean and disable decision. | No scheduler state currently records accepted/rejected counts per request or globally. |
| `infer/src/speculative.rs:314` | `DraftModel::draft_batch` produces one `TokenProposal`. | Trait is single-prefix shaped; Phase 2 needs scheduler batch integration or an explicit per-slot loop with bounded overhead. |
| `infer/src/speculative/cuda.rs:17` | CUDA `DraftEngine` is Qwen3-only, greedy-only, stateless per call. | Production cannot rebuild draft KV from the full 32k prefix on every step. |
| `infer/src/speculative/cuda.rs:122` | `draft_batch` creates a new draft state and runs `forward_prefill(token_ids)`. | This is a correctness scaffold, not a long-context serving path. |
| `infer/src/speculative/cuda.rs:130` | Draft tokens come from repeated `select_token_with_logprob` + `forward_decode`. | Draft probabilities exist for greedy only when model logprob fast path is available. |
| `infer/src/speculative/cuda.rs:142` | Comment says target probabilities are owned by target verify pass. | The target verify pass does not exist yet. |
| `infer/src/speculative/cuda.rs:145` | `target_probs = vec![0.0; tokens.len()]`. | This hardcoded zero makes every real draft reject unless overwritten before verification. |

### Scheduler Execution

| file:line | current behavior | integration gap |
|---|---|---|
| `infer/src/scheduler/cuda/core.rs:52` | `Scheduler<M>` owns model, slot states, radix cache, paged KV pool, decode/prefill contexts, pending GPU work. | Add spec config/runtime counters here or in `state_types.rs`, not in a second scheduler. |
| `infer/src/scheduler/cuda/core.rs:138` | `paged_kv_pool` is shared by all active slots. | Draft tokens need page accounting before they are appended or verified. |
| `infer/src/scheduler/cuda/core.rs:142` | `decode_bufs` is the reusable model decode context. | Verifier logits should reuse decode/mixed buffers; do not allocate per request. |
| `infer/src/scheduler/cuda/core/state_types.rs:10` | `PendingDecode` carries decode slots, greedy launch state, spans, and optional mixed prefill. | Add pending verifier metadata here once verify launches asynchronously. |
| `infer/src/scheduler/cuda/request.rs:23` | Per-request emit/decode state is separate from scheduler runtime stats. | Add per-request spec state to `ActiveRequest`, not to emit-only structs. |
| `infer/src/scheduler/cuda/execution.rs:29` | `StepPlan` has only `Idle`, `Decode`, `Prefill`, `Split`, `Mixed`. | Add a verifier-capable decode plan or make `Decode`/`Mixed` branch internally when spec is enabled. |
| `infer/src/scheduler/cuda/execution.rs:92` | `PrefillBudget::from_scheduler` builds token/page budget for the next step. | Draft KV accounting belongs in this budget layer so speculation cannot over-admit pages. |
| `infer/src/scheduler/cuda/execution.rs:127` | Running decode reservations reserve remaining generated tokens. | Spec draft tokens must reserve only the bounded speculative window, not full max-new again. |
| `infer/src/scheduler/cuda/execution.rs:321` | `collect_prefill_candidates` gathers prefilling rows. | Do not add draft admission here; draft rows are decode-phase work for already-running requests. |
| `infer/src/scheduler/cuda/execution.rs:351` | `plan_step` decides decode/prefill/mixed from current runnable decode rows and prefill candidates. | First scheduler entry point: choose spec-eligible decode rows before launch. |
| `infer/src/scheduler/cuda/execution.rs:431` | `step` dispatches launch by `StepPlan`. | Second scheduler entry point: route decode/mixed launch to verifier path when spec is active. |
| `infer/src/scheduler/cuda/decode.rs:90` | `retract_decode_to_fit` preempts least-progressed decode if pages are short. | Draft KV allocation must use this same pressure path; no separate preemption policy. |
| `infer/src/scheduler/cuda/decode.rs:266` | `launch_decode_batch_from_tokens` calls `model.forward_decode_batch`. | Target verifier batch launch should start here for decode-only spec rows. |
| `infer/src/scheduler/cuda/decode.rs:283` | Greedy sampling launches after target decode logits exist. | Spec verifier must capture target probabilities before sampling mutates/readbacks. |
| `infer/src/scheduler/cuda/decode.rs:334` | `step_mixed_launch` combines decode rows and prefill rows. | Verifier batch admission must coexist with mixed prefill or disable spec when mixed pressure is high. |
| `infer/src/scheduler/cuda/decode.rs:498` | `MixedBatchRequest` carries decode tokens + prefill rows into `forward_mixed_batch`. | Verify rows need an analogous request shape for K draft tokens per slot, or an extension to decode batch metadata. |
| `infer/src/scheduler/cuda/decode.rs:622` | `step_decode_readback` samples tokens and appends exactly one token per decode row. | Spec acceptance must append 0..K draft tokens plus at most one target-sampled token, then update emit and finish checks for multiple tokens. |
| `infer/src/scheduler/cuda/decode.rs:677` | Current decode push path appends one sampled token to `generated_tokens`. | This is the commit point for accepted speculative tokens. |

### Forward / Logits Surface

| file:line | current behavior | integration gap |
|---|---|---|
| `infer/src/model.rs:152` | `DecodeContextOps::logprobs_host()` exposes sampled-token logprobs only. | Verification needs selected draft-token target probabilities and bonus distribution, not just sampled logprobs. |
| `infer/src/model.rs:164` | `GenerationState::logits()` exposes current per-request logits. | Useful for scalar fallback, but batch verifier should avoid per-slot D2D scatter. |
| `infer/src/model.rs:307` | `select_token_with_logprob` supports single-row greedy logprob. | Draft model can use this, but target verifier needs batch logits for arbitrary draft token ids. |
| `infer/src/model.rs:501` | `sample_batch_greedy` returns sampled token ids and records logprobs. | Add a separate verify-logits API; do not overload sampling fast path. |
| `infer/src/model.rs:550` | `forward_decode_batch` is the canonical batched target forward path. | A K-token verifier pass needs a target forward that can advance or stage K positions while preserving rejection rollback semantics. |
| `infer/src/model.rs:574` | `supports_mixed_batch` gates mixed decode+prefill. | Spec should require this gate for W1 no-regression if sharing mixed buffers. |
| `infer/src/model/qwen3/forward.rs:40` | `Qwen3State::logits()` returns prefill logits or decode logits. | Scalar verifier fallback can read from here after per-position target forwards. |
| `infer/src/model/qwen3/forward.rs:240` | `forward_prefill` computes full hidden/logits and stores last logits. | Verifier block could reuse prefill-style multi-token logits if it can avoid committing rejected KV. |
| `infer/src/model/qwen3/forward.rs:249` | `forward_decode` commits one decode token into KV and clears prefill logits. | Spec verifier cannot blindly call this K times unless it has rollback/truncate after rejection. |
| `infer/src/model/qwen3/forward.rs:358` | `select_token_with_logprob` returns greedy token + logprob from current logits. | Draft probability source exists for greedy self/second-model draft. |
| `infer/src/model/qwen3/forward.rs:439` | `sample_batch_greedy` launches argmax/logprob over `decode_ctx.logits_batch`. | Target sampled-token logprob is available, but not p(target on draft token). |
| `infer/src/model/qwen3/forward.rs:527` | `prepare_batch_sampling_fallback` can scatter batch logits into per-slot decode buffers. | A verifier fallback can scatter logits, but P2 should first design a direct gather API to avoid K x vocab D2H. |
| `infer/src/model/qwen3/forward.rs:546` | `forward_decode_batch` prepares decode context and runs paged decode. | First target verifier implementation should extend this path rather than create a separate model runner. |
| `infer/src/model/qwen3/forward.rs:580` | Qwen3 supports mixed batch for BF16/FP8/INT8 paged KV when no LoRA. | Phase 2 W1 guard must preserve this exact condition. |

## Wiring Map

### 1. Draft KV Accounting

**Where:** `infer/src/scheduler/cuda/execution.rs:92`,
`infer/src/scheduler/cuda/execution.rs:127`,
`infer/src/scheduler/cuda/decode.rs:90`,
`infer/src/scheduler/cuda/core.rs:138`.

Design:

- Treat verifier target-KV growth as a consumer of the existing running decode
  reservation, not as an additive reservation. `PrefillBudget::from_scheduler`
  already reserves `remaining_decode_reservation_tokens` for each running row;
  a target-paged verifier window should spend from that row's planned growth.
- Add a bounded speculative page reservation only when Phase 2 introduces
  genuinely separate draft-KV storage. In that case reserve
  `min(num_speculative_tokens, remaining_max_new_tokens)` for the separate
  draft state and keep the target verifier pages under the existing decode
  reservation.
- The accounting belongs beside `PrefillBudget`/`PageBudget`, because that is
  where prefill and running decode growth are already reconciled. The P2.2
  no-op counters must assert no double counting under W1/c4 memory pressure.
- If draft/verify pages do not fit, use the existing `retract_decode_to_fit`
  policy. Do not add a second victim heuristic.
- Draft pages are provisional until verification commits. Rejected suffix pages
  must be released by truncating the request state/paged KV back to the accepted
  length.

Patch outline:

- `infer/src/scheduler/types.rs`: extend `SchedulerConfig` with disabled-by-
  default spec settings.
- `infer/src/scheduler/cuda/request.rs`: add per-request spec state:
  acceptance tracker, current draft window, and disabled-until counters.
- `infer/src/scheduler/cuda/core/state_types.rs`: add pending verifier metadata
  to `PendingDecode` or a sibling `PendingSpecVerify`.
- `infer/src/scheduler/cuda/execution.rs`: make `plan_step` select
  spec-eligible decode rows only when page budget can cover target verifier
  growth through the existing decode reservation plus any separate draft-KV
  storage.
- `infer/src/scheduler/cuda/decode.rs`: reuse `retract_decode_to_fit` before
  verifier launch.

### 2. Verifier Batch Admission

**Where:** `infer/src/scheduler/cuda/execution.rs:351`,
`infer/src/scheduler/cuda/execution.rs:431`,
`infer/src/scheduler/cuda/decode.rs:266`,
`infer/src/scheduler/cuda/decode.rs:334`,
`infer/src/scheduler/cuda/decode.rs:498`,
`infer/src/model.rs:550`,
`infer/src/model/qwen3/forward.rs:546`.

Design:

- Decode-only ticks can run verifier rows directly after draft generation.
- Mixed ticks should initially disable speculation when prefill rows are present
  unless the verifier batch can share the existing mixed context without
  reducing W1/c4 throughput. This keeps Phase 1 closed behavior stable.
- Verifier admission is per running request: a row is eligible when sampling is
  greedy or when the verifier can reproduce the request's sampling params
  bit-identically.
- The target verifier pass must return mode-specific data:
  - greedy mode: target argmax token for each proposed position plus the
    post-prefix bonus argmax;
  - stochastic mode: scalar target probabilities for each proposed draft token
    plus the target distribution or sampled target token at the rejection point;
  - both modes: enough rollback metadata to truncate rejected target KV.

Patch outline:

- `infer/src/model.rs`: add a target-verifier trait method, e.g.
  `forward_verify_spec_batch(...)`, rather than overloading
  `forward_decode_batch`.
- `infer/src/model/qwen3/forward.rs`: implement Qwen3 verifier path using
  paged KV and existing decode buffers.
- `infer/src/scheduler/cuda/decode.rs`: split decode launch into normal and
  spec branches; keep normal branch byte-for-byte behavior when spec disabled.
- `infer/src/scheduler/cuda/core/state_types.rs`: carry verifier rows across
  async launch/readback.

### 3. Adaptive Acceptance Rate Feedback

**Where:** `infer/src/speculative.rs:271`,
`infer/src/scheduler/cuda/decode.rs:622`,
`infer/src/scheduler/cuda/decode.rs:677`,
`infer/src/scheduler/cuda/core/state_types.rs:45`,
`infer/src/scheduler/types.rs:16`.

Design:

- Use `AcceptanceTracker` per request first; add aggregate service counters in
  scheduler stats after the per-request semantics are correct.
- Update acceptance after `verify_tokens` returns and before tokens are appended
  in `step_decode_readback`.
- Reduce draft length when rolling acceptance falls below threshold; recover
  gradually when acceptance stays high.
- Disable speculation for a request when acceptance ratio is low enough that
  expected speedup is below `1.0x` or when latency-tail counters trip.

Patch outline:

- `infer/src/scheduler/cuda/request.rs`: store tracker and current draft length.
- `infer/src/scheduler/cuda/core/state_types.rs`: add lifetime counters:
  proposed, accepted, rejected, fallback count, verifier rows.
- `infer/src/scheduler/cuda/decode.rs`: update counters at the same point that
  generated tokens are appended.
- `infer/src/http_server` stats surface: expose counters only after the internal
  scheduler counters are stable.

### 4. Forward / Sampling Integration

**Where:** `infer/src/speculative/cuda.rs:145`,
`infer/src/model.rs:152`,
`infer/src/model.rs:501`,
`infer/src/model/qwen3/forward.rs:439`,
`infer/src/model/qwen3/forward.rs:527`,
`infer/src/ops/sampling.rs:127`.

Design:

- Target verification must come from target-model logits. The current
  `target_probs = vec![0.0; tokens.len()]` stub in
  `infer/src/speculative/cuda.rs:145` must disappear before any runtime flag is
  exposed.
- Draft probabilities should come from draft-model logits via the existing
  greedy logprob path for the first implementation, but greedy verification
  must not use stochastic `p/q` acceptance.
- Rejection sampling must be bit-identical to normal target sampling at the
  rejection point. For greedy requests this is simple: accepted tokens must
  equal the target argmax token at each proposed position, and the first
  mismatch falls back to the target argmax at that position. For stochastic
  requests, P2 must use the same RNG stream and GPU sampling path as normal
  decode or keep speculation disabled.

Patch outline for the target-probs stub:

1. Add a model-side verifier output type in `infer/src/model.rs` with separate
   greedy and stochastic variants. Greedy output must contain target argmax
   tokens, a bonus/fallback argmax token, and rollback length. Stochastic output
   may contain `target_probs`, optional `target_bonus_dist` or sampled fallback
   token, and rollback length.
2. Implement Qwen3 verifier in `infer/src/model/qwen3/forward.rs`, reusing
   `BatchDecodeBuffers` logits. Start with greedy-only argmax verification for
   each proposed position; do not route greedy drafts through `verify_tokens`
   until stochastic sampling semantics are implemented.
3. Change `infer/src/speculative/cuda.rs` so `DraftEngine` returns only draft
   tokens and draft probabilities. Target probabilities are filled by the
   scheduler after target verify forward.
4. In `infer/src/scheduler/cuda/decode.rs`, build `TokenProposal` from
   draft output + target verifier output only for stochastic mode. Greedy mode
   should commit the longest prefix where `draft_token[i] == target_argmax[i]`
   and then append the target fallback/bonus argmax.

## Draft Model Selection Decision

Current constraint: W1/c4 FP8 KV pool is already effectively full. Phase 1
evidence recorded an FP8 pool of `136976` tokens / `8561` pages, and the c=4
row only became stable after admission counted evictable prefix pages. Any
Phase 2 draft path that adds a second large KV footprint risks reopening the
same memory edge.

| option | acceptance criteria | expected cost | verdict |
|---|---|---|---|
| MagicDec-style self-spec | No second model dependency; W2 lift `>=2.0x`; W1/c4 guard stays within Phase 1 variance; acceptance ratio should stay `>=0.6` because below that the verifier overhead eats most gain. | Smallest latency overhead and no second-model KV pool. Requires model-side self-draft design but fits current memory envelope. | **Default.** Best match for a 96%-full FP8 KV pool. |
| TriForce big-small draft | Use only if self-spec cannot reach `1.25x` after verifier overhead tuning; small Qwen draft must fit without reducing W1/c4 below `23.893 out tok/s`; target acceptance should be materially higher than self-spec. | Adds roughly `+0.5 GB` weights/KV/workspace for a small draft and more scheduler state. Likely better long-context acceptance, but memory pressure is risky. | Fallback. |
| LongSpec retrieval + sliding-window draft | Use only if long-context acceptance collapses and retrieval/window draft improves W2 without hurting target distribution correctness. | Benefits long context specifically, but adds retrieval/window policy, cache interactions, and workload-specific complexity. | Late fallback / research track. |

Default: **MagicDec-style self-spec**. It avoids a second model dependency,
keeps latency overhead smallest, and does not add a second draft KV pool while
W1 already runs close to the FP8 pool limit.

Fallback triggers:

- Self-spec W2 lift stays below `1.25x` after verifier overhead tuning.
- Rolling acceptance ratio is below `0.6` on W2 for three valid bench runs.
- W1 guard drops below Phase 1 worst-run `23.893 out tok/s`.
- Verifier rollback or sampling equivalence cannot be made deterministic.

## P2.2-P2.5 Dependency Graph

```text
P2.2 no-op counters + wiring spec
  ├─ adds disabled-by-default config/counters
  ├─ proves service stats and tests see zero spec activity
  └─ names exact scheduler/model file ownership
        ↓
P2.3 target verifier batch path
  ├─ removes target_probs stub dependency
  ├─ exposes target probabilities/logits through model forward
  └─ keeps normal decode path unchanged when spec disabled
        ↓
P2.4 draft proposal path
  ├─ implements MagicDec-style self-spec first
  ├─ builds TokenProposal from draft + target verifier output
  └─ commits accepted tokens with rollback on reject
        ↓
P2.5 adaptive speculation length
  ├─ wires AcceptanceTracker into ActiveRequest
  ├─ tunes draft length and fallback thresholds
  └─ promotes only after W2 lift + W1 guard benches pass
```

## Every Touched File For Implementation

P2.2 expected write set:

- `infer/src/scheduler/types.rs`
- `infer/src/scheduler/cuda/request.rs`
- `infer/src/scheduler/cuda/core/state_types.rs`
- `infer/src/scheduler/cuda/core.rs`
- `infer/src/scheduler/cuda/decode.rs`
- `infer/src/speculative.rs`
- `infer/src/metrics.rs`
- `infer/src/http_server/`
- `docs/experience/wins/YYYY-MM-DD-bench-guidellm-*.md`

P2.3 expected write set:

- `infer/src/model.rs`
- `infer/src/model/qwen3/forward.rs`
- `infer/src/model/qwen3/batch_decode.rs`
- `infer/src/model/qwen3/decode_buffers.rs`
- `infer/src/ops/sampling.rs`
- `infer/src/scheduler/cuda/decode.rs`
- `infer/src/speculative.rs`
- `infer/src/speculative/cuda.rs`
- `infer/tests/greedy_consistency.rs`
- `docs/experience/wins/YYYY-MM-DD-bench-guidellm-*.md` or
  `docs/experience/errors/YYYY-MM-DD-*.md`

P2.4 expected write set:

- `infer/src/speculative.rs`
- `infer/src/speculative/cuda.rs`
- `infer/src/scheduler/cuda/request.rs`
- `infer/src/scheduler/cuda/decode.rs`
- `infer/src/scheduler/cuda/core/state_types.rs`
- `infer/src/model.rs`
- `infer/src/model/qwen3/forward.rs`
- `infer/tests/e2e.rs` or a new focused CUDA integration test
- `docs/experience/wins/YYYY-MM-DD-bench-guidellm-*.md` or
  `docs/experience/errors/YYYY-MM-DD-*.md`

P2.5 expected write set:

- `infer/src/speculative.rs`
- `infer/src/scheduler/types.rs`
- `infer/src/scheduler/cuda/request.rs`
- `infer/src/scheduler/cuda/decode.rs`
- `infer/src/scheduler/cuda/core/state_types.rs`
- `infer/src/metrics.rs`
- `infer/src/http_server/`
- `docs/experience/wins/YYYY-MM-DD-bench-guidellm-*.md`

P2.5 reuses the P2.2 stats files only to tune/adapt non-zero counters; P2.2
must already expose disabled-by-default zero spec activity in service stats.

Files intentionally not touched in Phase 2 first pass:

- `crates/cuda-kernels/csrc/`: no new kernel until verifier data movement is
  measured.
- `infer/src/backend/metal/`: Metal speculative decode is a separate path.
- `crates/qwen35-spec/` and `infer/src/model/qwen35/`: Qwen3.5 follows only
  after Qwen3-4B W2 is stable.

## Correctness Gates

- Spec disabled must preserve normal decode output and W1/c4 Phase 1 guard.
- Greedy spec path must be bit-identical to target greedy decode for accepted
  sequences and rejection fallback.
- Stochastic spec path stays disabled until RNG and target distribution sampling
  can reuse the normal decode sampling path exactly.
- Any runtime slice under `infer/src/` needs a `wins/` or `errors/` bench entry
  before commit per `AGENTS.md`.
