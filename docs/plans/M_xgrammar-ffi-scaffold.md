# M_xgrammar — xgrammar FFI scaffold for grammar-constrained generation

> Master strategy §7.5 P1.2: grammar-constrained generation for tool-call
> JSON output. ARLE agent workload (W3/W4 per master §2.1) requires
> structured output 70%+ of time → grammar masking is binding for
> agent-shape acceptance.
>
> Per master §6.1 5-cap moat: xgrammar is one of 5 capability axis.
> ARLE currently has zero grammar substrate. **FFI integration of
> existing xgrammar C++ library is the LOC-low path** (vs Rust rewrite).

## Phase 1 — Target

| Field | Value |
|---|---|
| Metric | tool-call JSON validity rate (target = 100%) AND decode overhead (target ≤ 10% vs unconstrained) |
| Baseline | unconstrained Qwen3-4B at agent W3/W4 shape — JSON validity ~70-90% (literature) |
| **License** | 100% JSON validity + ≤ 10% decode overhead on simple schemas |
| Soft win | 95% validity + ≤ 20% overhead — proceed but flag for tuning |
| Kill | < 90% validity OR > 30% overhead — xgrammar integration broken |
| Wall-clock budget | 3-5 days (codex substrate ~400-600 LOC + Claude bench) |

## Phase 2 — Hardware constraints

xgrammar is **CPU-side mask computation** (not GPU kernel). Per-step
overhead is mask compute time + memory copy to GPU sampler.

xgrammar architecture (per upstream `mlc-ai/xgrammar`):
- Compile EBNF/JSON-schema → finite-state machine (FSM) at request start
- At each decode step: compute valid-token bitmask from FSM state
- Mask multiplied into logits before sampling (zeros out invalid tokens)
- Update FSM state with sampled token

Compute cost per step:
- Mask compute: O(grammar_complexity), typically < 50us for simple schemas
- Bitmask transfer host→device: O(vocab_size / 8), ~16 KB for 128k vocab
- Total per-step: ~50-100us added to ITL

For Qwen3-4B at ITL ~12 ms (W4A16 Marlin), 50-100us = **0.4-0.8% overhead** —
well below 10% license threshold.

## Phase 3 — Binding constraint

ARLE workload binding for grammar:
- **CPU sampler latency** (small): grammar mask add ~50us per step (skill v1.3.0 §3 — kernel time)
- **JSON validity** (large): unconstrained sampling makes invalid JSON in
  high-entropy contexts (tool call argument values, missing brackets, etc.)
- **Schema compile cost** (one-time per schema): ~10-100ms; cache per
  unique schema. Not in steady-state critical path.

Production binding constraint = JSON validity (correctness gate). Decode
overhead is comfortably below threshold.

## Phase 4 — Formula prediction

```
xgrammar mask compute: ~50us per step (constant, vocab-dependent)
ARLE Qwen3-4B ITL: ~12 ms per step (W4A16 Marlin)
Overhead = 0.05 / 12 = 0.4% (well under 10% license)

JSON validity:
  unconstrained: ~80% (literature on Qwen-class models)
  with xgrammar masking: 100% (mathematically guaranteed)

Throughput effect: tok/s decrease ≤ 1% (ITL bumps 12.0 → 12.05 ms)
```

## Phase 5 — Implementation (codex own, ~400-600 LOC)

### 5.1 Cargo workspace + xgrammar build (~80 LOC)

`crates/xgrammar-sys/`:
- `Cargo.toml` — crate definition with `cc` build dep
- `build.rs` — clone + build xgrammar as static lib OR git submodule
- `src/lib.rs` — extern C declarations + safe Rust wrappers
- xgrammar version pinning (e.g., v0.1.x stable)

### 5.2 Rust wrappers (~150 LOC)

`crates/xgrammar-sys/src/lib.rs`:

```rust
pub struct GrammarCompiler {
    inner: *mut xgrammar_compiler_t,  // opaque handle
}

impl GrammarCompiler {
    pub fn new() -> Result<Self> { ... }
    pub fn compile_json_schema(&mut self, schema: &str) -> Result<CompiledGrammar> { ... }
    pub fn compile_ebnf(&mut self, grammar: &str) -> Result<CompiledGrammar> { ... }
}

pub struct CompiledGrammar { ... }  // FSM ready for masking

pub struct GrammarMatcher {
    inner: *mut xgrammar_matcher_t,
    grammar: Arc<CompiledGrammar>,
}

impl GrammarMatcher {
    pub fn new(grammar: Arc<CompiledGrammar>, vocab_size: usize) -> Result<Self> { ... }
    pub fn fill_bitmask(&mut self, bitmask: &mut [u32]) -> Result<()> { ... }
    pub fn accept_token(&mut self, token_id: u32) -> Result<bool> { ... }
    pub fn is_terminated(&self) -> bool { ... }
}
```

### 5.3 ARLE integration (~150 LOC)

`infer/src/sampling.rs` (or equivalent):
- Add `Sampler::with_grammar(matcher: GrammarMatcher)` constructor variant
- During `sample_logits`: AND in bitmask before softmax; sample; update matcher
- Per-request grammar matcher (in scheduler state)

`infer/src/scheduler/cuda/runtime/admission.rs` (or where requests enter):
- Parse `--response-format=json_schema` from request
- Compile grammar (cached) + create matcher
- Attach matcher to request state

### 5.4 OpenAI-compatible HTTP (~50-100 LOC)

`infer/src/http_server/openai.rs` (or equivalent):
- Accept `response_format` field per OpenAI Structured Outputs
- Compile schema → grammar
- Pass to scheduler as request meta

### 5.5 Tests (~50-100 LOC)

`infer/tests/grammar_consistency.rs`:
- JSON schema validity at end of generation
- Decode overhead vs unconstrained baseline (≤ 10%)
- EBNF grammar acceptance (e.g., simple expression grammar)

## Phase 6 — Combinational A/B (post-license)

| Config | Expected validity | Expected overhead |
|---|---|---|
| unconstrained | ~80% | 0% (baseline) |
| simple JSON schema | 100% | < 1% |
| complex nested JSON schema | 100% | 1-3% |
| EBNF math expression | 100% | 1-2% |

Multi-shape validation: tool-call workloads (W3/W4) + code completion +
strict JSON output.

## Phase 7 — Tradeoffs

| Axis | Status | Note |
|---|---|---|
| LOC | ⚠ ~400-600 (codex) | substrate-LOC heavy |
| HW specificity | ✅ none | CPU-side mask compute |
| Compiler/runtime | ⚠ xgrammar version pinning | track upstream |
| Maintainability | ⚠ FFI maintenance | xgrammar API may evolve |
| Correctness | ⚠ FSM compile bugs | upstream xgrammar test coverage |
| Generality | ✅ schema-agnostic | works for any JSON / EBNF |
| Memory | ✅ small | bitmask ~16 KB, FSM ~100 KB |
| Scheduling | ⚠ per-request grammar state | fits into existing slot state |
| Throughput | ✅ overhead < 1% predicted | well under license |
| Latency variance | ✅ minimal | mask compute deterministic per step |

## Phase 8 — License decision

| Result | Action |
|---|---|
| 100% validity AND ≤ 10% overhead | LAND HARD — xgrammar default for `response_format=json_schema` |
| 100% validity AND 10-20% overhead | LAND incremental — opt-in flag `--enable-grammar` |
| 95-100% validity | debug — schema compile path may have edge cases |
| < 95% validity | KILL — FSM matcher broken |
| Throughput regression > 30% | KILL hard — overhead unacceptable |

## Pre-execution checklist

- [ ] xgrammar upstream stable version selected (e.g., v0.1.x)
- [ ] ARLE `crates/` workspace configured for native cc build
- [ ] OpenAI Structured Outputs API surface decided (`response_format` field)
- [ ] greedy_consistency-style test added for grammar correctness

## Cross-references

- Master strategy §6.1 5-cap moat: capability 5 — xgrammar
- Master §2.1 W3/W4 agent shape: tool-call output ~70% of bytes (workload alignment)
- xgrammar upstream: <https://github.com/mlc-ai/xgrammar>
- xgrammar paper: <https://arxiv.org/abs/2411.15100> (xgrammar: Flexible and Efficient Structured Generation)
- Skill v1.3.0: [`.claude/skills/kernel-optimization/SKILL.md`](../../.claude/skills/kernel-optimization/SKILL.md) (`d09480b`)
- Existing speculative substrate (parallel axis): `infer/src/speculative.rs`

## Rule

xgrammar FFI integration is **substrate-heavy** (~400-600 LOC, codex
territory) but has minimal correctness risk because xgrammar upstream is
production-tested at MLC-LLM. The fastest path to LAND is FFI thin wrapper
+ minimal sampler integration. Do NOT rewrite the FSM in Rust (~5-10× LOC
+ correctness risk for zero perf gain since CPU side).

Per skill v1.3.0 anti-pattern #11 (typedef ambiguity): xgrammar is C++ →
Rust FFI ABI. Pin xgrammar version in `Cargo.toml`; CI verify against
upstream API stability before bumping.
