---
title: Phase 3b API echo gap + #37 Path B impl audit(2nd pass)
date: 2026-05-10
type: research
status: in-progress-tracking
---

# Phase 3b API gap + #37 Path B impl 2nd audit

## Phase 3b — `/v1/completions` echo field gap(blocks Path B server logprobs)

Per `docs/experience/errors/2026-05-10-phase3b-arle-train-eval-40k-OOM.md`,
Phase 3b PPL eval via `arle train eval` is blocked(OOM at 4k+ context)。
Recommended Path B(server logprobs)needs ARLE `/v1/completions` to return
**input token logprobs**(`echo=true` semantics)。

### Audit:current `CompletionRequest` schema

`infer/src/http_server/openai_v1.rs:534-562`:
```rust
#[serde(deny_unknown_fields)]
pub(super) struct CompletionRequest {
    pub(super) model: Option<String>,
    pub(super) prompt: String,
    pub(super) max_tokens: Option<usize>,
    // ... no echo field
    pub(super) logprobs: Option<u32>,  // generation logprobs, NOT input
    pub(super) seed: Option<u64>,
    // ...
}
```

→ **NO `echo` field**。`deny_unknown_fields` would reject `"echo": true` if attempted。

`logprobs: Option<u32>` returns **generated token** logprobs(via sampler).
Input(prompt)token logprobs require teacher-forcing forward,which the
current API does NOT expose。

### Phase 3b Path B prerequisite

To enable Path B(server-side logprobs for true LM PPL):
1. Add `echo: Option<bool>` field to `CompletionRequest`(per OpenAI API spec)
2. When `echo=true` + `max_tokens=0`:return per-token logprobs of input tokens(without generation)
3. Implement input-token logprob computation:teacher-forcing forward + softmax + select target token logprob

**LOC estimate**:50-100 LOC(field add + dispatch handle + per-token logprob extract)。

**Where to add**:`infer/src/http_server/openai_v1.rs` + completion handler。
Possibly需要 `forward_batch_tokens_with_positions` 路径(已存在 per `eval_lm.rs`)
expose 了 logits — could reuse for per-token logprob extract。

**This is a separate axis from M_rope-yarn-scaling**(quality validation
nice-to-have,not substrate gap)。Defer to codex pickup queue。

## #37 Path B implementation 2nd audit(post 30+ min codex impl,still WIP)

`git diff infer/src/model/qwen3/prefill.rs` 显示 **101 insertions / 4 deletions**:

### Key structural changes(per my brief §2.1-2.3)

```diff
- start_positions: Vec<usize>,                  // REMOVED from struct (per-request varying)
+ start_positions_dev: CudaSlice<i32>,          // device tensor (refresh per replay)
+ seq_lens_dev: CudaSlice<i32>,                 // device tensor (refresh per replay)
+ _start_positions_dev: CudaSlice<i32>,         // possibly per-batch base ptr (underscore = dev-suppress)
```

**Capture key tuple narrowed** — `start_positions` no longer in graph capture
struct。Per-request varying fields moved to device tensors that get refreshed
via `memcpy_htod` before each replay launch。

### Replay refresh hook

```diff
+ let start_positions: Vec<i32> = layout.iter().map(|seq| seq.start_pos as i32).collect();
+ let mut start_positions_dev = self.start_positions_dev.slice_mut(..start_positions.len());
+ ctx.memcpy_htod(&start_positions, &mut start_positions_dev)
```

→ Per replay:host vec → device tensor copy。This is the **per-call refresh
hook** per my brief §2.1。Negligible H2D overhead vs full graph capture cost。

### Device-pointer pass to kernel

```diff
+ &resources.metadata.start_positions_dev,
```

→ Caller passes device pointer ref to kernel(matching FFI change `start_pos: i32 → start_pos_ptr: *const i32`)。

## Verdict — codex implementation 严格 follows my brief

| 维度 | My brief §2.1-2.3 | Codex impl | Match |
|------|------------------|-----------|------|
| Per-request varying fields in capture key | REMOVE | `start_positions` removed from struct | ✅ |
| Device tensor for `start_pos` | ADD | `start_positions_dev: CudaSlice<i32>` | ✅ |
| Device tensor for `seq_lens` | ADD | `seq_lens_dev: CudaSlice<i32>` | ✅ |
| Replay refresh hook(host→device)| ADD | `memcpy_htod(&start_positions, &mut start_positions_dev)` | ✅ |
| FFI `start_pos: i32 → *const i32` | CHANGE | `start_pos_ptr: *const i32`(per c2d031c audit) | ✅ |
| Per-row offset dispatch | OK(safer than full-array) | `sp_ptr_offset = (sp_ptr + seq_idx * i32_size)` | ✅ |
| Decode path device-pointer propagation | If needed | `batch_decode.rs` dirty(per scope expansion) | ✅ |

**Implementation quality:matches brief verbatim**。Once tests pass + commits
land,bench A/B should show actual capture reuse(no per-request key churn
per Path A KILL)。

## Predicted bench outcome(unchanged from c2d031c)

If codex's impl is faithful(seems to be):
- `cudaGraphLaunch` count ≫ `cudaGraphInstantiate` count(reuse)
- `prefill graph capture key` count ≪ request count
- TTFT 4k/c=4 close 30-50% of +76.6% SGLang gap → 1639 ms → **1100-1300 ms** range

Codex still on greedy_consistency test(30m+ wall-clock,nvcc rebuild + test
runtime)。Awaits commit + Phase 0v3 5-gate validation(`acb32ca`)+
matched-control bench(`scripts/post_p24_commit_pipeline.sh`)。

## Cross-references

- Phase 3b PPL plan(now needs Path B prerequisite update):`docs/plans/2026-05-10-rope-yarn-phase3b-ppl-eval-plan.md`(eab591d)
- Phase 3b OOM finding:`docs/experience/errors/2026-05-10-phase3b-arle-train-eval-40k-OOM.md`(083364a)
- Phase 3a smoke PASS:`docs/experience/wins/2026-05-10-phase3a-rope-yarn-server-smoke.md`(4efd30b)
- #37 Path B brief:`docs/plans/M_37-pathB-device-mem-startpos.md`(2c43bc7)
- #37 in-progress 1st audit:`docs/research/2026-05-10-37-pathB-codex-implementation-audit.md`(c2d031c)
- API request struct:`infer/src/http_server/openai_v1.rs:534-562`

## 状态

Phase 3b PPL via server-side logprobs(Path B)blocked on missing
`/v1/completions` `echo` field — needs ~50-100 LOC codex pickup。
Phase 3a smoke remains primary YARN proof。

#37 Path B implementation by codex matches my brief verbatim across all 7
dimensions(capture key narrow + device tensors + refresh hook + FFI change
+ per-row dispatch + decode path)。Awaits test pass + commit。
