# Task #24 substrate cleanup audit — BF16 split-KV identified

> Per task #24 "3 KILLED substrate half-state cleanup(1 周观察期,due
> 2026-05-14)",doing pre-deadline grep audit to identify candidate
> substrates with half-state code in tree。

## Confirmed half-state substrate (1 of 3 expected)

### M_b.2.2 BF16 split-KV(KILLED 2026-05-07)

**KILL evidence**:`docs/experience/errors/2026-05-07-m_b22-bf16-splitkv-killed-regression-and-hang.md`
- ITL +31.6% regression
- out tok/s -18.8% regression
- 33m+ hang (correctness/runtime bug)

**Half-state code in tree**(grep confirmed 2026-05-09 EOD+94):
- `infer/src/ops.rs`:re-export `tilelang_bf16_split_kv_requested`
- `infer/src/ops/attention.rs`:
  - `pub(crate) fn tilelang_bf16_split_kv_requested() -> bool`
  - `fn tilelang_bf16_split_kv_enabled(max_kv_tokens: usize) -> bool`
  - `if is_pure_decode && tilelang_bf16_split_kv_enabled(max_kv_tokens) { ... }`
- `infer/src/model/qwen3/forward.rs`:2 call sites for
  `include_hd128_split_workspace`

**Total**:5 references across 3 files。

**Gating**:`std::env::var("INFER_TILELANG_BF16_SPLIT_KV")`(opt-in env var)。

**Cleanup options**:
- **Option A(remove)**:Delete all 5 references + env var check。Per CLAUDE.md
  "no half-states" rule,this is preferred if substrate is dead-on-arrival。
  ~30-50 LOC delta。
- **Option B(retain with KILL marker)**:Add `#[deprecated(note = "KILLED 2026-05-07
  per docs/experience/errors/...")]` + log warning if env var set。Preserves
  re-experimentation capability。~10 LOC delta。

**Recommendation**:**Option A**。KILL evidence is conclusive(perf regression
+ correctness bug)。No evidence retention needed beyond docs/errors/ entry。
Re-experimentation can re-introduce the path if hardware/architecture
changes。

## Other candidates(unfound via simple grep)

Checked for these KILLED substrates(per `docs/experience/errors/` recent
entries):
- `m_pgc-phase0-killed-ttft-under-threshold.md`(2026-05-08)→ no `m_pgc` refs in `infer/` or `crates/cuda-kernels/csrc/`
- `m_quant-cutlass-fp8-smoke-killed-sm89.md`(2026-05-08)→ no `cutlass.*fp8` refs
- `r4-hybrid-dispatch-killed-batch4-decode-regression.md`(2026-05-08)→ no `r4_hybrid_dispatch` refs
- `spec-decode-32k-self-spec-kill-axis-level.md`(2026-05-08)→ no `magicdec` / `self_spec_k5` refs

**Hypothesis**:these 4 KILLED substrates were either(a)cleanly removed
in their respective KILL commits,or(b)live behind different naming I
haven't matched。

**Recommended action**:codex/Claude tomorrow do deeper grep with each
KILL entry's specific code-pointer references(commit message + entry
"## Substrate" section)to confirm complete removal vs hidden half-state。

If only BF16 split-KV has half-state:**task #24 scope shrinks to 1 substrate**
not 3。Still valuable cleanup,but smaller。

## Observation window status

- KILL dates 2026-05-07 + 2026-05-08
- Observation 1-week ends 2026-05-14 / 2026-05-15
- Today is 2026-05-09 → **5-6 days remaining**
- No re-bench evidence has surfaced post-KILL that would invalidate the KILL
  decisions

→ Cleanup is on track for 2026-05-14+ execution。Pre-staging this audit
saves discovery time at cleanup-day。

## §0 first principle applied

The "3 KILLED substrates" task description assumes 3 are present。Empirical
audit finds **only 1 confirmed half-state**。Per §0:**hypothesis ≠ evidence**
applies to task scope itself。

Either:
- (a) Task description was approximate(actual count = 1,not 3)
- (b) Other 2 substrates have non-obvious naming I missed
- (c) Other 2 substrates already cleaned in prior commits(post-KILL hygiene)

→ Tomorrow's deeper grep should disambiguate。If(a)or(c),task #24
scope confirms = 1 substrate(BF16 split-KV)。If(b),deeper audit
reveals others。

## Cross-references

- Task #24:"3 KILLED substrate half-state cleanup(1 周观察期)"
- KILL record:`docs/experience/errors/2026-05-07-m_b22-bf16-splitkv-killed-regression-and-hang.md`
- Half-state code:`infer/src/ops/attention.rs` + `infer/src/ops.rs` +
  `infer/src/model/qwen3/forward.rs`
- Opt-in env var:`INFER_TILELANG_BF16_SPLIT_KV`
- §0 first principle:CLAUDE.md "求真务实,追求极致"
- "no half-states" rule:CLAUDE.md "feedback_no_half_states.md"

## Status

Pre-staged cleanup audit for task #24。1 substrate confirmed(BF16 split-KV
with 5 references across 3 files)。Other 2 expected substrates need deeper
grep with KILL-entry-specific code pointers。

Recommendation:Option A removal(~30-50 LOC delta)at observation window
end(2026-05-14)。Pre-staging documentation reduces cleanup-day time-to-action
from ~30 min audit + execute to ~5 min execute-only。
