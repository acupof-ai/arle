<!-- /autoplan restore point: /path/to/.gstack/projects/cklxx-agent-infer/main-autoplan-restore-20260502-004054.md -->
# Agent-Specific Scheduler Optimizations

**Context**: Current c=1 performance is 9.83 tok/s vs SGLang's 11.57 tok/s (15% gap). Agent workloads are often single-session and can benefit from bypassing batching overhead.

**Task**: Add fast-path optimization for single-request agent workloads in the CUDA scheduler.

**Files to modify**:
1. `infer/src/scheduler/cuda/execution.rs:351` - Add agent fast-path detection in `plan_step`
2. `infer/src/scheduler/cuda/decode.rs:266` - Optimize single-agent launch path
3. `infer/src/scheduler/types.rs` - Add agent workload configuration flags
4. `crates/agent/src/lib.rs` - Mark agent sessions for fast-path scheduling

**Optimizations to implement**:
1. **Single-Agent Fast Path**: Bypass batching when only one active agent session
2. **Tool Schema Caching**: Cache tool encodings between agent turns
3. **Agent State Streaming**: Reduce allocations in turn processing
4. **Prefix Cache Optimization**: Improve hit rate for agent conversation patterns

**Requirements**:
- Preserve existing behavior for non-agent workloads
- No regression on batch workloads (c=4)
- Target 15-20% latency reduction for c=1 agent sessions
- Follow scheduler architecture in `infer/src/scheduler/AGENTS.md`

**Acceptance criteria**:
- `scripts/bench_guidellm.sh` shows improvement on `WORKLOAD=longctx-32k` c=1
- No regression on batch benchmarks
- `cargo test --workspace` passes
- Produces bench entry per AGENTS.md requirements

<!-- AUTONOMOUS DECISION LOG -->
## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|-------|----------|-----------|-----------|----------|
| 1 | CEO | Premises valid | Mechanical | P6 | Performance gap is measurable, use case is real | Alternative framings |
| 2 | CEO | Leverage existing bypass | Mechanical | P4 | Don't rebuild short prompt bypass logic | Parallel implementation |
| 3 | CEO | Approach B selection | Taste | P1+P3 | Balances completeness with manageable risk | A: insufficient gain, C: too complex |
| 4 | CEO | Defer most expansions | Mechanical | P2+P3 | Keep scope manageable, include streaming fix only | Tool caching, prefix warming |
| 5 | CEO | SELECTIVE EXPANSION mode | Mechanical | Based on plan analysis | Fits plan scope and expansion opportunities | Other modes |