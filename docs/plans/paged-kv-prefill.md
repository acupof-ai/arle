# Plan — paged-KV prefill (structural fix for `CONTIGUOUS_KV_TOKENS=512`)

> **Partially superseded (2026-04-17):** Phase 1 (kernel FFI), Phase 2 (Rust
> ops), and Phase 3a (Qwen3 scheduler wiring) are complete. Phase 3b (Qwen3.5),
> 3c (GLM4), and Phase 4 (delete `CONTIGUOUS_KV_TOKENS`) are re-sequenced under
> [`p99-unified-mixed-batch.md`](p99-unified-mixed-batch.md) §Phase 1. Refer
> there for ordering and acceptance.

**Status:** revised after Codex review (`docs/reviews/2026-04-17-paged-prefill-codex-review.md`). Execution delegated to Codex per CLAUDE.md §Delegation.
**Motivation doc:** `docs/experience/wins/2026-04-17-prefill-ttft-root-cause-contiguous-kv-cap.md`
**Codex review:** `docs/reviews/2026-04-17-paged-prefill-codex-review.md`
**Sibling plan (re-sequenced after this):** `docs/plans/qwen35-single-graph-prefill.md`

## Why not "Path A first" (raise `CONTIGUOUS_KV_TOKENS` to 2048)?

The root-cause doc offered a tactical Path A (bump the cap, keep the
contiguous scratch) as a faster de-risking step. Per the user's
architectural-preference rule
(`memory: feedback_architecture_ideal.md`, "架构永远按照理想态做事"):
for architectural problems the structural fix is pursued directly
regardless of diff size. Path A is rejected. This plan is Path B
unconditionally.

## Problem (one sentence)

Our prefill attention (`SinglePrefillWithKVCacheDispatched`, HD128 + HD256
variants) requires a **contiguous** per-slot K/V working buffer, sized
at engine start to `CONTIGUOUS_KV_TOKENS = 512` tokens. That constant
simultaneously caps every prefill chunk to 512 tokens, so any prompt
> 512 tokens is sliced into N = ⌈prompt_len/512⌉ forwards — each
paying the full kernel-launch storm and scheduler cycle. sglang avoids
this by calling `BatchPrefillWithPagedKVCacheDispatched` directly
against the paged KV pool — no contiguous scratch, no cap.

## End-state design

Prefill runs one forward per prompt, writing K/V **directly to the
paged pool** via the same page-table indirection already used by batched
decode (`flashinfer_tc_decode_run`). No contiguous K/V scratch exists
for the prefill forward path. `CONTIGUOUS_KV_TOKENS` is deleted.
`migrate_kv_range_to_paged` is deleted from the prefill hot path.
Prefill chunking becomes purely a *scheduling* concern (fairness across
concurrent requests), not a memory-layout concern, and is governed
entirely by `SchedulerConfig::prefill_chunk_size` (default 4096).

## Architecture — call-site shape parity with sglang

sglang's `BatchPrefillWithPagedKVCacheWrapper.run(q, kv_cache)` takes:

- `q` : ragged `[total_qo_tokens, num_qo_heads, head_dim]`
- `qo_indptr` : `[batch_size+1]` (cumulative query-token starts)
- `kv_cache` : paged layout `[num_pages, 2, num_kv_heads, page_size,
  head_dim]` with per-request `(kv_indptr, kv_indices,
  kv_last_page_len)` triple
- `o` : `[total_qo_tokens, num_qo_heads, head_dim]`

We already emit exactly this shape for decode at
`crates/cuda-kernels/csrc/attention/flashinfer_tc_decode.cu` with
qo_len=1 per request. Prefill is the same call with real qo_len ≥ 1
per request, and with K/V tokens **written via page-table
indirection** before the attention kernel fires (not into a contiguous
buffer that's later migrated).

## File-level change list

| # | file | change |
|---|------|--------|
| 1 | `crates/cuda-kernels/csrc/attention/flashinfer_prefill_paged.cu` (NEW) | HD128 paged-prefill wrapper — structural analogue of `flashinfer_prefill.cu` but using `BatchPrefillPagedParams` + `paged_kv_t`. Includes `flashinfer_batch_prefill_paged_plan` and `flashinfer_batch_prefill_paged_run`. |
| 2 | `crates/cuda-kernels/csrc/attention/flashinfer_prefill_paged_hd256.cu` (NEW) | Same as #1 for HD256 (Qwen3.5 full-attn layers). |
| 3 | `crates/cuda-kernels/csrc/attention/prefill_attention_paged_prep.cu` (NEW) | New QK-norm + RoPE + **paged** K/V write. Adapts `prefill_attention_prep_dual_write_cuda` → paged-only (drop contiguous write). HD128 + HD256 variants. |
| 4 | `crates/cuda-kernels/src/ffi/attention.rs` | Declare FFIs for #1–3. Keep old FFIs until #8–11 land. |
| 5 | `crates/cuda-kernels/src/flashinfer.rs` | Expose the paged-prefill wrapper with workspace allocation mirroring the decode planner. |
| 6 | `infer/src/ops/attention.rs` | Add `prefill_attention_paged_batch` (HD128) + `prefill_attention_hd256_paged_batch` (HD256). Signature consumes `PagedKVMeta` (already exists for decode) + `qo_indptr` + `kv_indptr` + `kv_indices` + `kv_last_page_len`. Keep old variants until migration completes. |
| 7 | `infer/src/ops.rs` | Re-export new ops. |
| 8 | `infer/src/model/qwen3/prefill.rs` | Switch both call sites (`prefill_attention_batch`@278,460) to `prefill_attention_paged_batch`. Pass paged metadata threaded through from the scheduler. |
| 9 | `infer/src/model/qwen35/prefill.rs` | Switch both call sites (`prefill_attention_hd256_batch`@164,356) to `prefill_attention_hd256_paged_batch`. Qwen3.5 linear-attn layers are unaffected (they use GDN chunkwise, no contiguous K/V issue). |
| 10 | `infer/src/model/glm4/prefill.rs` | Switch both call sites (`prefill_attention_batch`@230,365). |
| 11 | `infer/src/scheduler/cuda/prefill.rs` | Pre-allocate paged pages for the full prompt length *before* the first forward (currently done per-chunk at L253–L270). After forward, no migration step — delete `migrate_kv_range_to_paged` call and the retry/error path around it for prefill. Prefix-hit migration (L174–L194) still needs to migrate prefix-cache hits into paged pool — that stays. |
| 12 | `infer/src/scheduler/cuda/core.rs` | Delete `CONTIGUOUS_KV_TOKENS`; delete `state.set_max_seq_len(CONTIGUOUS_KV_TOKENS)` at line 304; delete `.min(CONTIGUOUS_KV_TOKENS)` at line 886. `prefill_chunk_size()` now returns the DecodeAwareChunking result capped only by `self.config.prefill_chunk_size`. |
| 13 | `infer/src/scheduler/types.rs` | `prefill_chunk_size` default stays 4096 in `runtime_defaults`; the `Default` for cpu-only scheduling stays 512 (it's an arbitrary value since no GPU buffer is sized from it anymore). Update the `// Default 512 is chunking cap …` doc comment to reflect the new role (scheduler fairness knob only). |
| 14 | `infer/src/model/kv_cache.rs` | `KVCache` still backs single-token-decode contiguous K/V read/write for Qwen3/GLM4 (`fused_attention_decode_into` at `model/qwen3/decode.rs:11-38`, `model/glm4/decode.rs:11-38`), AND the migrate_kv_range_to_paged dispatch path for BF16/FP8/INT8/TurboQuant (`model/generation_state.rs:121-165`). Downsize `max_seq_len` only where it was inflated for prefill chunk capacity; do NOT remove the struct. |
| 15 | `infer/src/model.rs` | Re-examine `ModelForward` trait surface — `set_max_seq_len` (`model.rs:98`), `migrate_kv_to_paged` / `migrate_kv_range_to_paged` (`model.rs:105-120`), `forward_prefill_with_pool` (`model.rs:249-272`). The paged-prefill path should plug into the **existing** `forward_prefill_with_pool` contract that Qwen3/GLM4 already implement (`model/qwen3/forward.rs:163-188`, `model/glm4/forward.rs:164-180`) — not add a parallel entry point. |
| 16 | `infer/src/model/generation_state.rs` | `migrate_kv_range_to_paged` dispatch stays (still used for prefix-cache hits and INT8 quant migration). Audit which branches are hit from which call sites; nothing here is deleted as part of this refactor. |
| 17 | `infer/src/model/qwen3/batch_decode.rs` | Mixed decode+prefill path currently requires contiguous caches via `kv_cache.k_caches().is_empty()` + `max_seq_len()` guards (`batch_decode.rs:360-364,543-566`) and `prefill_attention_prep_dual_write_cuda`. Must either migrate this path to paged-only or keep its contiguous dependency explicitly (documented carve-out). |
| 18 | `infer/src/model/{qwen3,qwen35,glm4}/forward.rs` | `fn set_max_seq_len` stays (INT8 + single-token-decode contiguous path still uses it). Value passed in reduces from `CONTIGUOUS_KV_TOKENS` constant to a dynamically-sized value matching single-token-decode needs. |
| 19 | `infer/src/ops/tests.rs` | Existing `test_prefill_attention_*` already use **tolerances, not bitwise** (`ops/tests.rs:596-606,816-818`). Add parallel `test_prefill_attention_paged_*` that compare to the same CPU reference within the same tolerance. Do NOT tighten to bitwise — the kernel template is different (`BatchPrefillPagedParams` vs `SinglePrefillParams`). |
| 20 | `infer/test_data/*.json` | Regenerate greedy baselines iff **greedy-token-id parity fails** — i.e. sampled token IDs must match bitwise even if intermediate tensor values differ within tolerance. Numerical drift at the tensor level within existing tolerance is acceptable; different greedy token sequences are not. |
| 21 | `infer/benches/ops/ops_attention_bench.rs` + `ops_qwen35_state_bench.rs` | Add paged variants alongside existing benches to measure the contiguous → paged delta directly. |
| 22 | `docs/support-matrix.md` + `infer/src/model/AGENTS.md` + `infer/src/scheduler/AGENTS.md` | Update architecture notes: prefill runs against paged pool, no contiguous scratch. |

## Phased delivery (each phase = one commit on `main`)

Codex review flagged the original ordering as buggy: Phase 3 (model
switch) was blocked by Phase 4 (scheduler unlock) — you can't "one
forward per prompt" while `.min(CONTIGUOUS_KV_TOKENS)` still caps the
chunk. Revised ordering puts scheduler plumbing and model switches in
the **same** commit, per-model.

**Phase 1 — kernel wrappers** (files 1–5)
- Implement paged-prefill CU wrappers + prep kernels.
- No Rust op exposure yet. Self-test via a standalone Rust test that
  constructs a paged-KV fixture and calls the FFI directly; compares
  output against the existing `SinglePrefill` path on the same
  logical tokens. Within per-tensor tolerance from `ops/tests.rs`.
- Acceptance: `cargo build --release -p cuda-kernels` clean;
  FFI-level test passes.

**Phase 2 — Rust ops + scheduler plumbing** (files 6, 7, 11 partial, 19 partial)
- Expose `prefill_attention_paged_batch` + HD256 variant.
- Scheduler: expose a path that passes `qo_indptr`, `kv_indptr`,
  `kv_indices`, `kv_last_page_len` to the forward call. Do NOT yet
  remove `CONTIGUOUS_KV_TOKENS` cap. The old contiguous path is still
  the default.
- Unit test per-op parity (tolerance-based, not bitwise).
- Acceptance: test module in `infer/src/ops/tests.rs` passes — e.g.
  `cargo test --release -p infer -- test_prefill_attention_paged`.

**Phase 3 — one model migration at a time** (Qwen3 → Qwen3.5 → GLM4)
Each model = one commit that includes:
1. Model `prefill.rs` switched to paged variant.
2. Scheduler `prefill_chunk_size()` lifts the `.min(CONTIGUOUS_KV_TOKENS)`
   cap **for that model** (gate via the model's
   `forward_prefill_with_pool` availability at `model.rs:249-272`,
   which Qwen3/GLM4 already implement).
3. Scheduler `prefill.rs:253-277` skips `migrate_kv_range_to_paged`
   when the model uses paged-prefill.
4. Prefix-cache hit path (`prefill.rs:99-156,174-194`) audited — same-slot
   resurrection today depends on contiguous KV as the source. Before
   switching each model, prefix hits must either (a) write into paged
   pool directly during prefix materialization, or (b) keep the
   contiguous source for this narrow path. Decide per-model based on
   what's least invasive.

For each model:
- Run `cargo test --release --test e2e` (Qwen3 / GLM4) or
  `cargo test --release --test e2e_qwen35` (Qwen3.5).
- **Greedy-token parity required**: sampled token IDs must match the
  `infer/test_data/*.json` baseline exactly. Per-tensor values can
  differ within existing tolerance; sampled-token sequence cannot.
- Run matching guidellm sweep vs the 2026-04-17 baseline.

**Phase 4 — deprecated kernel deletion + struct cleanup** (files 12, 13, 14, 15, 17, 18, 22)
- Last model migrated → delete `CONTIGUOUS_KV_TOKENS` entirely.
- Delete `flashinfer_prefill.cu`, `flashinfer_prefill_hd256.cu`,
  `prefill_attention_prep_cuda`, `prefill_attention_hd256_prep_cuda`
  and their FFI declarations.
- Downsize `KVCache::max_seq_len` to the size actually required by
  single-token decode (Qwen3/GLM4) + INT8 working buffer — these are
  the surviving uses per Codex review.
- Update architecture docs.
- Acceptance: grep shows no remaining references to deleted constants
  / FFI / kernels; build + e2e + guidellm all green.

## Acceptance criteria (final, for the full refactor)

1. `grep -r CONTIGUOUS_KV_TOKENS infer/ crates/` returns zero matches.
2. `grep -r SinglePrefillWithKVCache infer/ crates/` returns zero
   matches in Rust/FFI; CU files that wrapped it are deleted.
3. All e2e tests pass with **bitwise-identical greedy output** against
   existing `infer/test_data/*.json` baselines. If not bitwise
   identical, the PR is rejected and the numerical drift is root-caused
   before landing.
4. `cargo clippy --workspace -- -D warnings` clean.
5. `cargo check -p infer --no-default-features --features cuda,no-cuda`
   clean (Mac CUDA-Rust typecheck).
6. guidellm sweep deltas vs
   `docs/experience/wins/2026-04-17-bench-guidellm-qwen3-4b-infer-l4-p99.md`
   and `2026-04-17-bench-guidellm-qwen35-4b-infer-l4-p99.md`.
   **Realism caveat** (Codex review §5): the timing-breakdown doc
   shows 93% of prefill-chunk wall-clock is inside the 32-layer forward
   loop, not in setup/migration. Pure layout change eliminates the
   8× setup overhead (~50–80 ms on a 4096-tok prompt) but does NOT
   reduce kernel work per forward. Acceptance targets are therefore
   modest; peak-tput gains beyond +5–10% require P1 graph capture on
   top:
   - Qwen3-4B sync TTFT p99 ≤ **810 ms** (baseline 871, −7%).
   - Qwen3-4B @0.135 r/s TTFT p99 ≤ **1150 ms** (baseline 1234, −7%).
   - Qwen3-4B peak throughput ≥ **103 tok/s** (baseline 98, +5%).
   - Qwen3.5-4B sync TTFT p99 ≤ **900 ms** (baseline **982.6 ms**
     per benchmark file, −8%).
   - Qwen3.5-4B peak throughput ≥ **96 tok/s** (baseline **91.43
     tok/s**, +5%).
   The +26% Qwen3.5 throughput and −15% TTFT numbers in the pre-review
   draft were not defensible from the timing breakdown and have been
   corrected. Hitting sglang's 134 tok/s peak requires P1 graph
   capture on top of this refactor, not this refactor alone.

## Risks

1. **Numerical drift.** If the paged write introduces race hazards
   with attention reads (wrong page table on kernel entry), output
   drifts silently. Mitigation: Phase 3 lands one model at a time and
   e2e tests must produce bitwise-identical output vs existing JSON
   baselines. First drift = stop, root-cause, fix.

2. **Prefix-cache same-slot resurrection depends on contiguous KV today.**
   Codex review (§4) corrected my earlier assumption: admission paths
   at `scheduler/cuda/runtime.rs:343-356` treat a radix hit as reusable
   only when a free slot still *materializes* the prefix via
   `block_owner_slots` + `slot_materialized_prompt_lens`, and
   `step_new()` then truncates/restores local state and migrates the
   prefix range from **contiguous** KV into the pool
   (`scheduler/cuda/prefill.rs:99-156,174-194`). The contiguous source
   is not a vestige — it's the live path for same-slot prefix reuse.
   Migration to paged-only requires either: (a) writing prefix pages
   into the paged pool at radix-hit time (not at admission time), or
   (b) keeping a smaller contiguous scratch specifically for this
   path. Decide during Phase 3.

3. **GDN chunkwise state isolation.** Qwen3.5 linear-attn layers don't
   touch K/V at all — they use recurrent state in `RecurrentState`.
   Phase 3 Qwen3.5 must verify linear-attn layers are untouched; only
   the 8 full-attn layers change.

4. **Workspace allocation growth.** `BatchPrefillPaged` requires a plan
   workspace (float_workspace + int_workspace + page_locked
   workspace). Sglang allocates ~64 MB for this. We do too, for
   decode (`flashinfer_tc_decode_plan`). Prefill plan can share the
   same workspace pool — single allocation, reused. **Do not allocate
   a second workspace.**

5. **FlashInfer version compatibility.** Our vendored FlashInfer
   (pinned in `crates/cuda-kernels/third_party/flashinfer/`) must
   expose `PrefillPlanInfo` + `BatchPrefillPagedParams` with the
   signatures we already use in decode. Both are present — confirmed
   via `flashinfer_tc_decode.cu:30,53`.

## What this plan does NOT touch

- P1 full-forward CUDA Graph capture — separate lever (P1 plan doc).
  Sequenced after this refactor lands.
- Qwen3.5 attn-output gate fusion (P3) — separate optimization.
- Linear-attention GDN chunkwise state — unchanged.
- Single-token prefill path — already graph-safe, already bypasses
  chunking (seq_len=1 < 512 trivially).
- INT8 KV decode — unchanged attention, unchanged working-buffer
  mechanism.

## Delegation

This plan is too large for Claude to hand-write per CLAUDE.md
§Delegation. Execution brief for Codex will be drafted after this plan
is approved, covering Phase 1 end-to-end. Each subsequent phase = its
own Codex brief, so Claude can integrate + verify bitwise parity
between phases rather than delegating all at once.

## Plan adjustments (post-commit, live log)

### 2026-04-17 — `forward_prefill_with_pool` already exists as the integration hook (discovered during Phase 2 scouting)

`ModelForward::forward_prefill_with_pool` at `infer/src/model.rs:262-272`
is an already-declared trait method with a default impl that delegates
to `forward_prefill`. Qwen3 (`model/qwen3/forward.rs:163-192`) and GLM4
(`model/glm4/forward.rs:164-180`) already implement it via
`process_all_layers_batch_with_pool` — **a dual-write path** that
writes K/V to contiguous cache AND token pool.

**But no caller exists yet**: scheduler only calls `forward_prefill`
at `scheduler/cuda/prefill.rs:234`. The Qwen3/GLM4 dual-write
implementations are dead code today.

This reshapes Phase 3:
- **Integration point**: rewrite `process_all_layers_batch_with_pool`
  (in Qwen3 + GLM4) to call the new paged-prefill kernel
  **paged-only** (drop the contiguous write). Qwen3.5 doesn't have
  this method yet — it needs to be added, following the same pattern.
- **Scheduler switch**: change `scheduler/cuda/prefill.rs:234` from
  `forward_prefill(chunk, state)` to `forward_prefill_with_pool(chunk,
  state, pool, slot_idx, new_token_indices)`. The `new_token_indices`
  slice is already computed one level up (`prefill.rs:174-194` for
  prefix-hit allocation, `prefill.rs:253-277` for post-forward
  allocation — the latter moves BEFORE the forward call).
- **Migration deletion**: `migrate_kv_range_to_paged` at
  `scheduler/cuda/prefill.rs:264` becomes obsolete for the post-
  forward path. The prefix-hit migration at `prefill.rs:181` stays
  for now (separate issue; same-slot resurrection from contiguous KV —
  Codex review §4).
- This is **cleaner than the original plan** because it reuses an
  existing trait contract; the plan's "new `prefill_attention_paged_batch`
  Rust op" wrapper still exists but is called *from inside*
  `process_all_layers_batch_with_pool`, not as a standalone alternative.

### 2026-04-17 — Baseline number clarification

Qwen3.5 sync TTFT p99 = **982.6 ms** (from
`2026-04-17-bench-guidellm-qwen35-4b-infer-l4-p99.md:48`) measures a
guidellm-sync workload (1 r/s, default prompt length). The 820ms
figure in the timing-breakdown doc is from a **separate probe** at
seq_len=4096 single-request streaming. Both measurements are valid
at their respective workloads; plan acceptance targets use the 982.6ms
sync number as the canonical guidellm baseline.

## Rule (pre-registered)

This refactor requires **greedy-token-ID parity** with existing
`infer/test_data/*.json` baselines — not bitwise tensor parity.
Codex review (§3) established that the two FlashInfer entry points
(`SinglePrefillWithKVCacheDispatched` vs
`BatchPrefillWithPagedKVCacheDispatched`) are different template
instantiations; our existing unit tests already compare within
tolerance, not bitwise (`ops/tests.rs:596-606,816-818`). So the
acceptance bar is:
- Per-tensor values within existing unit-test tolerance.
- Greedy sampled-token sequence exactly matches JSON baseline.
- Any greedy-token drift = bug to root-cause, not baseline to rebase.
