# Agent-first architecture priorities

**Status**: Active — opened 2026-04-13 as the top-level architecture track for
turning `agent-infer` into the strongest inference engine for agent sequences.

**Goal**: every agent turn reuses the maximum possible prior work, every tool
call is syntactically valid on first try, and every session can survive a
process restart without paying the cold-prefill tax.

This doc is a **priority ledger**, not a design spec. Each item is scoped to
land as a single PR under the Phase 1 PR discipline (one main topic per PR);
the original "Phase 1 split" plan was reverted by Route-A on 2026-04-15 — see
`docs/archives/art-grade-architecture-for-long-agent-infer.md` for the dead
proposal and `docs/architecture.md` / `docs/codebase-map.md` for the current
canonical workspace shape.

> **Status update — 2026-04-15** (post M1+M2a, post Codex review)
>
> The original P1/P2/P3 phase numbers on items in this doc are **superseded**
> by the M0–M5 milestone scheme in
> [`tiered-kv-cache.md`](tiered-kv-cache.md) §6. Crosswalk:
>
> - **A1** (Wire RadixCache into scheduler) — **partially shipped**.
>   M1b (`323aee0`) wired RadixCache as a shadow observer; M2a (`4402ab0`)
>   added per-page refcount + watermark eviction so radix-held pages
>   survive `free_slot`. M2b (selector flip + resurrect read path) is the
>   next milestone — see `tiered-kv-cache.md` §6 M2b.
> - **A2** (HTTP `session_id`) — plumbed in `http_server/openai_v1.rs`
>   and `scheduler/types.rs::IncomingRequest`; **scheduler does not consume
>   it yet**. Still open.
> - **B1** (Session save/load + disk tier) — `infer/src/session_store.rs`
>   was never created; the disk tier landed instead as
>   `infer/src/kv_tier/transport/disk.rs::DiskStore`. The HTTP routes
>   are scoped under `tiered-kv-cache.md` §6 M4.
> - **B3** (Session-aware eviction) — landed as
>   `infer/src/scheduler/policy.rs::SessionBiasedLru` (and 3 sibling
>   `EvictionPolicy` impls). **Trait shipped, zero call sites** —
>   convergence onto the policy trait is `tiered-kv-cache.md` §5.4.1
>   under M3b.
> - **C6** (Agent workload benchmark) — **shipped** as
>   `scripts/bench_agent_trace.py`.
>
> The §1 "current gap diagnosis" below was written before M1b/M2a
> shipped and uses pre-Route-A file paths in some places (e.g.
> `infer/src/metal_scheduler.rs` is now
> `infer/src/backend/metal/scheduler.rs`,
> `infer/src/server_engine.rs:437-475` no longer matches). Treat §1 as
> the original-state record; the canonical post-M2a state lives in
> [`tiered-kv-cache.md`](tiered-kv-cache.md) §3.

---

## 1 · Current gap diagnosis

The codebase already contains the building blocks for an agent-grade engine,
but the four most valuable pieces are **unused in the production data path**.
The review that produced this list is summarized below as evidence — not as
blame. The parts that are wired are genuinely good; the parts that are not
wired are a connection problem, not a capability problem.

### Gap 1 — `RadixCache` is orphaned
- **Implementation**: `infer/src/prefix_cache.rs` (552 lines: radix tree +
  LRU + refcount + block granularity).
- **Consumers in data path**: zero.
  - `infer/src/scheduler/cuda/runtime.rs:134-151` uses `cached_prompts:
    Vec<Vec<u32>>` — a `num_slots`-entry (default 4) linear last-prompt
    compare, not a radix tree.
  - `infer/src/metal_scheduler.rs` never imports `MetalPrefixCache`, even
    though `infer/src/metal_prefix_cache.rs` correctly wraps `RadixCache`.
  - `infer/src/server_engine.rs:437-475` has a third, independent
    single-entry `cached_prompt: Vec<u32>` compare used only by the CLI path.
- **Consequence**: README claims "radix-tree prefix cache" + "100% cache hit
  on multi-turn agent benchmarks". The 100% number only holds for the
  single-session CLI path where `server_engine.cached_prompt` happens to
  equal the previous turn. Cross-request / cross-session KV reuse is
  effectively zero.

### Gap 2 — HTTP API has no session identity
- `openai_v1::CompletionRequest` / `ChatCompletionRequest` carry no
  `session_id`, `conversation_id`, or `prefix_key`.
- The scheduler's only slot-affinity logic is
  `runtime.rs::best_prefix_slot`, which scans `num_slots` cached prompts and
  picks the best linear prefix match. Same session → same slot is not
  guaranteed under concurrency or eviction.

### Gap 3 — Speculative decoding is CPU math only
- `infer/src/speculative.rs` (625 lines) has `SpecConfig`, `TokenProposal`,
  `verify_tokens`, `AcceptanceTracker`, and a `DraftModel` trait whose only
  implementation is `MockDraftModel`.
- `grep speculative infer/src/` outside `speculative.rs` returns a single
  `pub mod speculative;` declaration. No scheduler call site, no draft
  forward pass, no KV writeback.

### Gap 4 — No grammar-constrained / structured decoding
- `grep -rE 'grammar|xgrammar|outlines|json_schema|constrained' infer/src/`
  returns zero matches.
- Tool-call reliability currently depends on
  `infer_chat::openai_parse_tool_calls` running a regex + JSON bailout on fully
  generated text (`http_server/openai_v1.rs:296-309`). This is below the
  modern bar for serving 4B–8B agent models.

### Gap 5 — Session save/load is chat-only, not KV
- `AgentSession::save_to_file` / `load_from_file` persist only
  `Vec<OpenAiChatMessage>` JSON. On reload, the engine retokenizes and reruns the
  full prefill from scratch. No KV block persistence.

### Gap 6 — Tool-call streaming is parsed post-hoc
- `http_server::openai_v1.rs:296` calls `openai_parse_tool_calls(&output.text)` on
  the fully assembled response. Streaming clients cannot observe incremental
  `delta.tool_calls[].function.arguments`. Agent UX loses progressive
  tool-call animation.

### Gap 7 — Policy signals are not agent-aware
- `infer::scheduler::policy::SchedulerSignals { queued_requests, active_decodes }` is
  the entire input surface. It cannot express "prefer requests with prefix
  hits" or "the Nth turn of session S should not be preempted by a cold
  request".

---

## 2 · Work items, ordered by impact on the goal

Every item lists: **what** (one-line description), **why** (what it unlocks),
**where** (concrete file paths), **exit criterion** (how we know it shipped).

### Tier A — defines whether we are agent-grade at all

#### A1. Wire `RadixCache` into the CUDA scheduler (blocker for A2, A3, A4)
> **Implementation spec**: [`tiered-kv-cache.md`](tiered-kv-cache.md) §6 M1
> (formerly P1, renumbered in the 2026-04-15 revision — see tiered-kv-cache.md
> §13 revision log).
> A1 is folded into the Tiered KV Cache project — it ships as the first
> behavior milestone (M1) there. When M1 lands, move A1 to the Done
> section with a pointer to the merged PRs.

- **What**: Replace `scheduler/cuda/runtime.rs::best_prefix_slot`'s
  `num_slots`-entry linear compare with a `RadixCache` lookup backed by
  block-refcounted `BlockManager` allocations. Cross-request CoW sharing.
- **Why**: Makes the README-level claim true; unlocks every other tier-A
  optimization that expects a real prefix tree.
- **Where**:
  - `infer/src/prefix_cache.rs` — already complete.
  - `infer/src/block_manager.rs` — extend with token-prefix → block lookup
    and refcount-aware free.
  - `infer/src/scheduler/cuda/core.rs` — drop `cached_prompts: Vec<Vec<u32>>`,
    hold `Arc<Mutex<RadixCache>>` or a scheduler-owned instance.
  - `infer/src/scheduler/cuda/runtime.rs` — rewrite slot admission as
    "radix lookup → block ref grant → emit prefill chunk for suffix only".
  - `infer/src/scheduler/cuda/prefill.rs` — consume the prefix hit length
    instead of the per-slot linear compare at `prefill.rs:14`.
- **Exit**: Multi-session concurrent bench shows ≥70% prefix hit rate on
  agent traces; README architecture diagram becomes accurate; `RadixCache`
  has at least one production consumer inside `scheduler/`.

#### A2. `session_id` in the HTTP API + session-sticky routing (needs A1)
- **What**: Add optional `session_id` to `CompletionRequest` and
  `ChatCompletionRequest`. Hash → slot / radix subtree affinity. When a
  session presents, preferentially schedule its request to the slot whose
  radix subtree already holds its prior prefix.
- **Why**: Agent clients that send a growing conversation array should get
  deterministic reuse, not "whichever slot happens to win the linear match".
- **Where**:
  - `infer/src/http_server/openai_v1.rs` — add field.
  - `infer/src/scheduler/types.rs::IncomingRequest` — forward the id.
  - `infer/src/scheduler/cuda/runtime.rs` — session-aware admission.
- **Exit**: Same `session_id` across N turns keeps TTFT ratio
  `ttft[N] / ttft[1] < 0.1` for long histories.

#### A3. JSON-schema constrained decoding
- **What**: FSM-based logit mask during sampling. Minimum viable:
  xgrammar-style JSON-schema compiler + per-step mask application in
  `ops/sampling.rs`. Expose via OpenAI-compatible
  `response_format: {"type": "json_schema", "schema": ...}`.
- **Why**: Agent tool calls must be syntactically valid on first try.
  Current regex-bailout parse in `parse_tool_calls` does not scale below 7B.
- **Where**:
  - New crate or module: `infer/src/constrained/` with the compiler + FSM.
  - `infer/src/ops/sampling.rs` — inject logit mask before top-k / top-p.
  - `infer/src/http_server/openai_v1.rs` — accept `response_format`.
  - `crates/infer-chat` — consume constrained output (skip regex fallback
    when a schema was enforced).
- **Exit**: On a tool-calling bench (τ-bench or BFCL), syntax-error rate
  for 7B-class models drops ≥10× compared to the current free-form path.

#### A4. Speculative decoding end-to-end (needs A1 for KV branching)
- **What**: Connect the existing `verify_tokens` algorithm to a real draft
  model forward pass. Draft model = smaller variant of the target
  (e.g. Qwen3-0.5B drafting Qwen3-4B). Batched verify writes accepted
  tokens to the target KV and reclaims rejected-branch blocks.
- **Why**: 2–3× decode-phase throughput without changing greedy output.
- **Where**:
  - `infer/src/speculative.rs` — promote from CPU math module to an
    integrated path.
  - `infer/src/scheduler/cuda/decode.rs` — split decode step into
    draft-forward / target-verify / writeback.
  - `infer/src/model/` — add a `DraftModel` impl for Qwen3-0.5B.
- **Exit**: Greedy numerical parity with non-speculative path, ≥2× decode
  throughput on a single-request bench, accept-rate tracked in
  `/v1/stats`.

### Tier B — defines whether we are professional

#### B1. Session KV snapshot persistence (needs A1)
> **Implementation spec**: [`tiered-kv-cache.md`](tiered-kv-cache.md) §6 M4
> (formerly P3, renumbered 2026-04-15). B1 is folded into the Tiered KV
> Cache project. The proposed `infer/src/session_store.rs` does not land
> as a standalone module; its functionality ships as
> `kv_tier::transport::disk` (T2 tier after the 2026-04-15 T0/T2/T3/T4 →
> T0/T1/T2/T3 renumber) plus the HTTP save/load handlers. Radix `serde`
> lands in M1 as an M4 precondition.

- **What**: `POST /v1/sessions/{id}/save` serializes radix nodes + their
  KV blocks to local storage. `POST /v1/sessions/{id}/load` re-attaches.
  Pair with a graceful shutdown hook to persist on SIGTERM.
- **Why**: Agent processes restart (deploys, OOM, preemption); cold-start
  prefill tax on a 30k-token system prompt should be payable once.
- **Where**:
  - `infer/src/prefix_cache.rs` — serde for nodes.
  - New: `infer/src/session_store.rs` — LMDB or file-based block storage.
  - `infer/src/http_server.rs` — session routes.
- **Exit**: Restart → first request for saved session has TTFT within 20%
  of the pre-restart warm-state baseline.

#### B2. Streaming tool-call delta protocol
- **What**: Incremental JSON parser tracks the decode stream and emits
  OpenAI-shaped `delta.tool_calls[].function.arguments` chunks as they
  form, instead of post-hoc splitting the full output text.
- **Why**: Agent UI depends on progressive tool-call rendering. Current
  implementation emits content deltas then "un-emits" at completion.
- **Where**:
  - `crates/infer-chat/src/protocol.rs` — streaming parser.
  - `infer/src/http_server/openai_v1.rs` — emit `delta.tool_calls`.
  - `infer/src/http_server.rs::delta_sse_events` — branch on tool-call
    state.
- **Exit**: A streaming client observes monotonically-growing tool-call
  arguments; no mid-stream content↔tool_call rewrites.

#### B3. Policy signals for prefix / session awareness
> **Implementation spec**: [`tiered-kv-cache.md`](tiered-kv-cache.md) §6 M3
> (formerly P2, renumbered 2026-04-15). The signal extension itself has
> already shipped (commit `3e1d35f`); the remaining `EvictionPolicy`
> trait + `SessionBiasedLru` default is already defined in
> `scheduler/policy.rs:179-189` but has zero scheduler callers — it
> lands as the M3b coordinator+watermarks stacked PR. When M3 ships,
> move B3 to Done.

- **What**: Extend `infer::scheduler::policy::SchedulerSignals` with
  `prefix_hit_tokens`, `session_affinity_slot`, `turn_depth`. Add a built-in
  `PrefixAwareAdmission` that deprioritizes cold requests when warm ones
  are waiting.
- **Why**: The policy crate currently cannot express the most basic
  agent-first admission rule.
- **Where**:
  - `infer/src/scheduler/policy.rs` — extend struct + new policy.
  - `infer/src/scheduler/batch.rs` and `infer/src/metal_scheduler.rs` —
    fill in the new signals at call sites.
- **Exit**: Benchmarks show warm (session-continuation) requests do not
  get starved behind bursts of cold requests.

#### B4. Generic HuggingFace layer fallback
- **What**: A slow-path model loader that can serve any HF architecture
  by composing existing `ops::{linear, norm, attention, embedding}` with
  a config-driven layer graph. No new CUDA kernels required per model.
- **Why**: `model_registry::is_implemented` covers 3 architectures
  (`Qwen3 | Qwen35 | GLM4`). Agent ecosystem demands Llama / Mistral /
  Gemma / Phi / DeepSeek. The fast path stays specialized; the slow path
  is the on-ramp.
- **Where**:
  - New: `infer/src/model/generic_hf.rs` + `infer/src/model/generic_hf/`.
  - `infer/src/model_registry.rs::is_implemented` — lift to
    "fast-implemented" and "generic-implemented".
- **Exit**: Llama-3-8B loads and serves via the generic path with
  correct numerics, at whatever throughput the slow path reaches.

### Tier C — DX: defines whether newcomers can succeed

#### C1. Single `agent-infer serve` command with runtime backend detection
- **What**: One CLI entry point that picks CUDA / Metal / CPU at runtime
  via feature detection (`nvidia-smi` / macOS sysctl / fallback). Hides
  the `--no-default-features --features metal,no-cuda,cli` matrix.
- **Why**: Current quickstart asks a newcomer to reason about feature
  flags before they have a working token.
- **Where**:
  - `crates/infer-cli/src/args.rs` — add `serve` subcommand.
  - `infer/src/server_engine.rs::LoadedInferenceEngine::load` — runtime
    dispatch rather than compile-time cfg walls.
- **Exit**: `cargo install agent-infer && agent-infer serve` produces a
  working server on Linux+CUDA, macOS+Metal, and CPU fallback, with no
  feature flags.

#### C2. Rust library quickstart
- **What**: `examples/agent_loop.rs` showing ≤15 lines: load engine,
  loop `engine.complete(...)`. Link from README as example one.
- **Why**: The current pub API (`LoadedInferenceEngine` + `InferenceEngine` trait)
  is clean but undocumented; newcomers must read `infer-cli::repl` to
  learn the pattern.
- **Where**:
  - New: `examples/agent_loop.rs` (at workspace root).
  - `README.md` — replace Docker-first example with the Rust one.
- **Exit**: `cargo run --example agent_loop` works on all three backends.

#### C3. `GET /v1/trace/<req_id>` + `agent-infer doctor`
- **What**:
  - Trace endpoint returns the per-request event timeline captured by
    `infer::events::EventSink`: enqueue / prefill_start / first_token /
    decode_step count / completed, with per-phase milliseconds and
    `prefix_hit_tokens`.
  - `agent-infer doctor` prints startup self-check: CUDA version, free
    VRAM, `hf_hub` cache location, detected model architecture, tokenizer
    sanity, recommended `--num-slots`.
- **Why**: Agent developers cannot answer "why was this turn 6s" without
  per-request breakdown; they cannot answer "why won't the engine start"
  without a doctor output.
- **Where**:
  - `infer/src/http_server.rs` — new route + a ring buffer sink
    implementing `EventSink`.
  - `crates/infer-cli/src/args.rs` — `doctor` subcommand.
  - `infer/src/metrics.rs` — reuse or extend for trace storage.
- **Exit**: Both endpoints exist, document a per-turn slow-path
  investigation that would have been impossible before.

#### C4. Pyo3 embedding crate
- **What**: `crates/infer-python` exposing `infer.Engine`,
  `engine.complete`, `engine.stream` via pyo3 + maturin. Replaces the
  currently non-existent `agent_infer/` package (CLAUDE.md references a
  directory that is not on disk).
- **Why**: Research users cannot adopt `agent-infer` if they cannot
  `import` it. vLLM, SGLang, and llama.cpp all ship Python bindings.
- **Where**:
  - New crate: `crates/infer-python/`.
  - Root `pyproject.toml` — convert from pure-HTTP-client to
    maturin-driven wheel build.
- **Exit**: `pip install -e crates/infer-python` installs a wheel that
  exposes a working `Engine` class.

#### C5. CPU backend: real inference, not a stub
- **What**: Replace `CpuBackend::build_response`'s hard-coded string with
  a minimal real CPU inference path (e.g. CPU BF16 matmul via `ndarray` /
  `blas-src`, or GGUF Q4_K CPU dequant + GEMM). No production claim, but
  it must produce actual tokens.
- **Why**: Newcomers try CPU first on laptops. Seeing
  `"CPU backend development response from ..."` makes them conclude the
  engine is broken.
- **Where**:
  - `infer/src/cpu_backend.rs` — replace the stub path.
  - `infer/src/ops/linear.rs`, `infer/src/ops/norm.rs` — CPU fallbacks.
- **Exit**: CPU backend emits real tokens for at least Qwen3-0.6B,
  labeled as "slow, development only", and passes the existing
  `cpu_runtime_handle_loads_and_streams` smoke test with genuine output.

#### C6. Agent workload benchmark script
> **Implementation spec**: shipped as `scripts/bench_agent_trace.py` +
> `scripts/data/agent_trace_default.jsonl` (renamed from the originally
> proposed `bench_agent.py` because that file already exists as a
> binary-subprocess benchmark; the two measure different things and
> cohabit). Landed 2026-04-13 under the Tiered KV Cache task split
> (`docs/plans/tiered-kv-cache-tasks.md` §7.1).

- **What**: `scripts/bench_agent_trace.py` — replays a multi-turn
  tool-calling trace against a running server, reports TTFT per turn,
  inter-token latency, wall time, and optional JSON snapshot output.
  Matches `bench_throughput_sweep.py` CLI conventions.
- **Why**: The README publishes agent numbers without a reproducible
  script. That is non-falsifiable and blocks honest iteration on A1–A4.
- **Where**:
  - `scripts/bench_agent_trace.py` and an input trace under
    `scripts/data/agent_trace_default.jsonl`.
  - `docs/experience/wins/` — update the template for agent bench
    snapshots.
- **Exit**: A fresh clone can reproduce the README table with one
  command, or the README table is removed.

---

## 3 · Suggested execution order

This order respects blocking dependencies and defers work that rewrites
after A1 lands.

1. **A1** — radix cache into CUDA scheduler *(blocks A2/A4/B1)*
2. **C6** — agent bench script *(gives A1 a scoreboard)*
3. **A2** — session-sticky routing
4. **A3** — constrained decoding *(independent of A1, ship in parallel if
   bandwidth allows)*
5. **B3** — prefix/session policy signals
6. **B2** — streaming tool-call delta
7. **A4** — speculative decoding end-to-end
8. **B1** — session KV snapshot persistence
9. **C1** — single-command serve
10. **C2** — Rust library quickstart
11. **C3** — trace endpoint + doctor
12. **C5** — CPU backend real inference
13. **B4** — generic HF fallback
14. **C4** — Pyo3 embedding crate

A1+A2+A3 together are what take `agent-infer` from "building blocks" to
"actually agent-grade". Everything after that compounds, but nothing after
A4 is on the critical path for the stated goal.

---

## 4 · Non-goals for this project

- **New kernel microoptimization**. Every item here is connection or
  protocol work. Kernel perf is a separate track (see
  `experience/wins/` for the ongoing optimization log).
- **New model families via hand-written fast paths**. B4 handles the
  breadth need via a generic slow path; hand-written fast paths remain
  opportunistic.
- **Training / fine-tuning**. Out of scope; `agent-infer` is an
  inference engine.
- **Multi-replica distributed serving**. A2's session-sticky routing is
  the groundwork, but distributed scheduling is deferred until the
  single-node story is done.

---

## 5 · How this doc is kept honest

- Each item above is the skeleton of **one PR**. When it lands, move its
  entry to a "Done" section with a pointer to the merged PR.
- README marketing claims must point at a Tier-A item that has actually
  shipped. If a claim does not match reality, either ship the item or
  remove the claim — never paper over with roadmap language.
- Quarterly: re-check the "Gap diagnosis" grep commands in §1 and verify
  that the arch claims still hold. If `grep -r RadixCache infer/src/scheduler/`
  goes back to empty, A1 has regressed and takes priority over new work.

---

## 6 · Related docs

- [`tiered-kv-cache.md`](tiered-kv-cache.md) — Hierarchical KV cache project.
  Owns the implementation shape for A1, B1, and B3. Any contract change
  affecting those three items lands there first and propagates here.
- [`../archives/art-grade-architecture-for-long-agent-infer.md`](../archives/art-grade-architecture-for-long-agent-infer.md) —
  archived 8-crate decomposition proposal. Its §六 governance rules and §七
  acceptance criteria still inform PR discipline; its §一 / 二 / 三 crate
  topology was reverted by Route-A.
- `docs/architecture.md` — current workspace/package topology.
- `docs/projects/mlx-backend-roadmap.md` — Metal-side work that must stay
  consistent with tier-A contract changes.
- `docs/projects/kv-quantization-long-context.md` — complements A1/B1:
  radix cache plus quantized KV is the long-context story.
