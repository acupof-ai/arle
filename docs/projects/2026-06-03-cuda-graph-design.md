# CUDA Graph capture/replay design for the ARLE rewrite

Status: design + research, pre-implementation. Branch `arch/ideal-inference-engine`.
Scope: decode-path CUDA graph capture/replay for the new `infer-cuda` executor,
integrated into the host-only `infer-seam` / `infer-core` rewrite. Design-only —
no code lands here. Sequencing and gating are stated in §9.

This doc is the pre-design the rewrite asked for so that CUDA graph work is fully
specified before R6a parity is reached, and can be ported the moment the eager
CUDA parity gate passes ([`2026-06-03-r6-cuda-port-plan.md`](2026-06-03-r6-cuda-port-plan.md)
§"CUDA Graph": *"GraphRunner should be ported only after eager parity is proven"*).

---

## 1. Why this matters and what we already proved

A decode step on the hot path launches one kernel per operator per layer: for a
~28-layer dense Qwen3, embedding + per-layer (RMSNorm, QKV GEMM, RoPE, paged
attention, O-proj, residual+RMSNorm, gate/up GEMM, SwiGLU, down GEMM) + final
norm + LM-head GEMM. That is ~250-400 individual `cuLaunchKernel` calls, each
with fixed CPU-side launch overhead (1-5 µs). At batch=1 the kernels are tiny
(M=1 GEMV-shaped), so per-token wall-clock is **launch-bound, not compute-bound**
— exactly the regime CUDA graphs target. Replaying a captured graph collapses
those hundreds of launches into a **single `cuGraphLaunch`**.

This is not a hypothesis on this codebase — the legacy tree shipped it and the
proven shape is documented below. The rewrite must preserve the win, not
re-derive it.

### Legacy proven approach (cite)

`infer/src/model/cuda_graph.rs` — `CudaGraphState::run_or_capture`
([L38-67](../../infer/src/model/cuda_graph.rs)):

- First decode call: `stream.begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)`
  → run the kernel closure → `stream.end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)`
  stores a `cudarc::driver::safe::CudaGraph`, then `graph.launch()`.
- Every subsequent call: `graph.launch()` only.
- Contract comment (verbatim): *"`kernels` must be a pure GPU kernel sequence —
  no CPU-GPU sync, no allocation."* This is the load-bearing invariant for the
  whole design (§4).
- `unsafe impl Send` is justified only because capture and replay both run on the
  single blocking inference thread with the same stream
  ([L16-28](../../infer/src/model/cuda_graph.rs)). The rewrite's executor is
  likewise single-threaded behind `BackendExecutor`; this invariant carries over
  unchanged.

The canonical two-stage-metadata example is the Qwen3 dense decode path
`infer/src/model/qwen3/decode.rs` `decode_one_token`
([L11-44](../../infer/src/model/qwen3/decode.rs)):

```text
// Stage 1: write this step's metadata into a FIXED device buffer (decode_meta)
stream.memcpy_htod(&[token_id as i32, pos as i32, seq_len as i32], &mut bufs.decode_meta)
// Stage 2: replay (or capture-then-replay) the kernel sequence that reads decode_meta
graph_state.run_or_capture(&ctx, || self.decode_kernels(kv_cache, bufs))
```

`bufs.decode_meta` is a fixed `CudaSlice<i32>` of `[token_id, current_pos, seq_len]`
(`infer/src/model/qwen3/decode_buffers.rs` L52-53). The captured `decode_kernels`
reads it via `embedding_decode_into(&embed_tokens, &bufs.decode_meta, …)`
(L56-60). The graph never embeds the token value — it embeds a *pointer* to
`decode_meta`; the host overwrites the contents each step before replay. That is
the entire two-stage trick.

### Legacy warmup / bucketing (cite)

`infer/src/scheduler/cuda/core/warmup.rs` `warmup_cuda_graphs`
([L26-195](../../infer/src/scheduler/cuda/core/warmup.rs)):

- `cuda_graph_batch_sizes(max_bs)` (L446-464): **dense 1..=min(64, max_bs)**, then
  **sparse step-16 from 80 onward**, plus `max_bs` itself. The dense band exists
  specifically because graph-miss eager fallback on batch-composition churn was
  *"the primary source of p99 ITL spikes (100-150 ms outliers at B=16)"* (L437-445).
- `max_bs = num_slots.min(config.cuda_graph_max_bs)` (L39); `cuda_graph_max_bs`
  defaults to **256** (`infer/src/scheduler/types.rs` L564).
- Two-pass capture (L107-168): Pass 1 captures with cublasLt **heuristic** algos
  and populates the algo cache; `autotune_all_cached_gemms_cuda` then picks the
  measured-fastest algo per shape; Pass 2 **invalidates** (`invalidate_graph_cache`)
  and **re-captures** with the autotuned algos. The graph bakes in the chosen
  cublasLt algorithm, so re-tuning *requires* re-capture.
- Warmup uses dummy tokens (`vec![0; max_bs]`), real slot indices `0..max_bs`,
  drives `forward_decode_batch` under `with_synthetic_decode_warmup_scope`, and
  `sync()`s after each size (L414-432). Failure at any size → `break` (skip larger
  sizes), never a hard error.
- Per-size metadata is set up before the captured forward exactly as decode does:
  `set_batch_size(bs)` → `upload_token_ids` → `update_metadata` (returns
  `reallocated` → invalidate cache) → `plan_attention` (L372-412).

Qwen3.5 uses a **piecewise** variant — one graph per consecutive linear-attention
group, cached `[group_idx][batch_size - 1]` (`infer/src/model/qwen35/batch_decode.rs`
`run_linear_group_graphed` L956-1047). The rewrite's first CUDA target is dense
Qwen3 (R6a), which uses the **whole-decode** graph, so the v1 design below is the
whole-decode form; piecewise is a §10 extension.

Invocation point: `warmup_cuda_graphs()` runs once at scheduler-loop start, before
the HTTP bind signal, so readiness never races capture
(`infer/src/scheduler/cuda/runtime/scheduler_loop.rs` L117-120).

---

## 2. SGLang reference design (cite)

SGLang's `CudaGraphRunner`
([cuda_graph_runner.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py))
is the industry baseline and confirms every structural choice the legacy ARLE
tree made independently.

**Fixed static input buffers** (`DecodeInputBuffers.create()`): `input_ids`,
`req_pool_indices`, `seq_lens`, `out_cache_loc` (per-token KV cache location),
`positions`, `mrope_positions`, `num_token_non_padded`, `custom_mask`, and the
output `next_token_logits_buffer`. All allocated once on the weight device, sized
to `max_num_token` / `max_bs`.

**Bucketing** (`get_batch_sizes_to_capture()`): a server-configured `capture_bs`
list filtered by an alignment constraint (`bs * num_tokens_per_bs % mul_base == 0`)
and `bs <= num_max_requests`. Tunable via `--cuda_graph_bs` and `--cuda_graph_max_bs`.
Capture iterates **reverse (largest-first)** so smaller graphs reuse the memory
pool allocated by the largest; SGLang uses a **single shared memory pool** across
all captured graphs. Each size runs the forward **twice** (one warmup, one
capture) for allocation stability.

**Replay** (`populate_from_forward_batch()` → `graph.replay()`): if `bs != raw_bs`
it pads — fill `seq_lens` with `seq_len_fill_value`, zero `out_cache_loc` /
`req_pool_indices` (dummy rows point to reserved pool slot 0); batch-copy the live
metadata into the static buffer slices via `_grouped_foreach_copy_()`; then
`graph.replay()`. `can_run()` gates: with padding disabled the batch must hit an
exact captured key, else `cuda_graph_bs <= self.max_bs`. Output is sliced back to
the real token count.

**Paged-KV page table into the graph**: the attention backend's
`init_forward_metadata_capture_cuda_graph` allocates the page-table metadata
(`kv_indices` / `kv_indptr` for FlashInfer/FA3) as **static buffers** captured by
the graph; `init_forward_metadata_replay_cuda_graph` rewrites those buffers with
the current step's page table before replay. The page table is *never* baked as a
constant — it is a fixed buffer the captured attention kernel dereferences. This
is the exact constraint §4 enforces for ARLE's `PagedKVPool`.

**Decode-only**: `capture_forward_mode = ForwardMode.DECODE` (or `TARGET_VERIFY`
for spec draft). **Prefill/extend stays eager** because token count varies per
step. Variable-length prefill is the domain of SGLang's separate *Piecewise CUDA
Graph* feature ([docs](https://docs.sglang.io/advanced_features/piecewise_cuda_graph.html)),
explicitly out of scope here.

**Sampling is outside the graph**: the graph fills `next_token_logits_buffer`;
token selection runs in eager code afterward. **Overlap**: under
`enable_two_batch_overlap` a `TboCudaGraphRunnerPlugin` and a `can_run_tbo` check
manage the interaction; the graph replay is one launch the overlap scheduler
issues, not a special case.

Sources: [cuda_graph_runner.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py),
[Piecewise CUDA Graph](https://docs.sglang.io/advanced_features/piecewise_cuda_graph.html),
[CUDA Graphs (DeepWiki)](https://deepwiki.com/zhanxxxxxxx/sglang/6.2-cuda-graphs-and-torch.compile),
[GPU bubble between decode tokens (issue #5593)](https://github.com/sgl-project/sglang/issues/5593).

---

## 3. The `GraphRunner` seam decision

`infer-seam` today declares (hypothesis-grade, two methods):

```rust
pub trait GraphRunner {                       // crates/infer-seam/src/lib.rs L83-89
    fn capture(&mut self, batch: usize) -> anyhow::Result<()>;
    fn replay(&mut self, batch: usize) -> anyhow::Result<()>;
}
```

**Decision: keep `GraphRunner` backend-internal; do NOT route it through the
engine-core ↔ executor seam. Delete it from `infer-seam`'s public surface and move
it inside `infer-cuda`.**

Rationale, grounded in the rewrite's own altitude principle
([`2026-06-03-backend-seam-redesign.md`](2026-06-03-backend-seam-redesign.md):
*"graph → executor-internal"*, and the seam-scan win
[`wins/2026-06-03-sglang-backend-seam-scan.md`](../experience/wins/2026-06-03-sglang-backend-seam-scan.md):
*"orthogonal sampling/comm/graph layers"*):

1. **Engine-core has no device concept.** `infer-seam`'s own module doc states the
   engine-facing seam is *"host-only … Device tensors remain inside backend
   executors"* (lib.rs L1-6, L33-37). A `GraphRunner` is pure device-buffer
   plumbing — it has nothing host-side to expose. Putting it on the public seam
   leaks the implementation altitude.
2. **`capture(batch)/replay(batch)` is the wrong signature for the engine.** The
   engine never decides "capture batch N then replay batch M"; it hands the
   executor a `ForwardPlan` and calls `submit`. Whether that `submit` internally
   replays a graph or runs eager is the executor's private decision keyed off
   `plan.decode_rows.len()`. The two-arg trait is a speculative interface with no
   real engine-side caller — exactly the pattern the project bans
   ([`feedback_no_speculative_interface_shaping.md`](../../../.claude/projects/-Users-user-code-agent-infer/memory/feedback_no_speculative_interface_shaping.md):
   *"traits/handles trail real callers"*).
3. **Warmup is already a seam method.** `BackendExecutor::warmup(&mut self)`
   exists (infer-seam L50-52). Graph capture *is* warmup. Engine-core calls
   `executor.warmup()` once at startup; the CUDA executor's `warmup` body runs the
   bucket-capture loop. No new seam method is needed.

**Net seam contract (unchanged public surface):**

```text
BackendExecutor::warmup(&mut self)            // CUDA: capture decode graph buckets
BackendExecutor::submit(&mut self, plan, kv)  // CUDA: replay if bucket-match, else eager
BackendExecutor::poll(&mut self, inflight)    // CUDA: a graph replay completes like any submit
```

Engine-core does **not** trigger warmup beyond the one existing `executor.warmup()`
call, and does **not** know graphs exist. All capture/replay logic is private to
`infer-cuda`'s `CudaExecutor`. Inside `infer-cuda`, a private `GraphRunner`-shaped
helper struct may exist for code organisation, but it is not a `pub trait` on the
crate boundary.

---

## 4. Where capture/replay lives: inside `infer-cuda::CudaExecutor`

Today `RealCudaExecutor::submit` (`crates/infer-cuda/src/executor.rs` L82-177) is
**single-row, eager** (R6a): it asserts `rows == 1`, allocates one KV token,
calls `model.forward_tokens(...)`, returns one `SlotToken`. The graph design slots
*under* this without changing the seam.

### Dispatch rule (the one branch the executor adds)

On `submit`, after the existing validation:

```text
if plan.prefill_rows.is_empty()                         // pure decode
   && self.graph_cache.contains_key(plan.decode_rows.len())   // bucket hit
   && all rows greedy-or-graph-safe (§7)
{
    write current-step metadata into FIXED buffers (Stage 1, §5)
    graph.replay()                                      // ONE cuGraphLaunch
} else {
    eager forward                                       // existing path, unchanged
}
```

- **Decode-only is graphed.** A `ForwardPlan` whose `decode_rows.len()` equals a
  captured bucket size and whose `prefill_rows` is empty → replay.
- **Prefill stays eager — always.** Prefill `tokens.len()` is variable per row;
  `ForwardMode::Prefill` and `ForwardMode::Mixed` plans never replay. This matches
  both the legacy gate (Qwen3 prefill graph is *"explicitly not a default because
  shape churn regressed c=4/c=8/c=16"*,
  [r6-cuda-port-plan](2026-06-03-r6-cuda-port-plan.md) §CUDA Graph) and SGLang
  (decode-only, prefill eager).
- **`Mixed` mode**: if a tick mixes decode + prefill rows, run fully eager. (A
  later optimization can split the decode subset onto a graph and run prefill
  eager in the same submit; deferred — §10.)
- **Bucket miss** (e.g. `decode_rows.len() == 70` but only dense≤64 + sparse-80
  captured): fall back to eager for that step, OR pad up to the next captured
  bucket (§6 decides — v1 = **pad-up like SGLang**, since the dense band already
  covers 1..64 and padding avoids the eager-fallback p99 spike the legacy doc
  called out).

The graph cache is keyed by decode batch size: `HashMap<usize, CapturedDecodeGraph>`
(or a `Vec<Option<_>>` indexed `bs-1`, matching legacy `graph_cache[g][bs-1]`).
Each entry owns the `CudaGraph` plus the slice handles into the fixed buffers it
captured.

---

## 5. Two-stage metadata — the fixed device buffer list

This is the heart of the design. The captured graph contains kernels that hold
**device pointers** to fixed buffers. The host rewrites the *contents* of those
buffers each step, then replays. Buffer **addresses must be stable for the
lifetime of the graph** (never reallocated, never resized — invalidate + recapture
if a realloc is forced, exactly as legacy `update_metadata` returns `reallocated`
→ `invalidate_graph_cache`, warmup.rs L383-396).

### Fixed buffers the decode graph references (CUDA, dense Qwen3, bucket size B)

| Buffer | Type / size | Written per step (Stage 1) | Read by (in-graph) | Legacy / SGLang anchor |
|--------|-------------|----------------------------|--------------------|------------------------|
| `input_token_ids` | `i32 × B` | H2D copy of the B `last_token`s from `plan.decode_rows` | embedding kernel | qwen3 `decode_meta[0]` / qwen35 `token_ids_gpu` (batch_decode.rs L630-637); SGLang `input_ids` |
| `positions` | `i32 × B` | per-row `kv_seq_len + 1` (decode position) | RoPE, attention | qwen3 `decode_meta[1..2]`; SGLang `positions` |
| `seq_lens` | `i32 × B` | per-row `kv_seq_len + 1` (new KV length after append) | paged attention (loop bound) | qwen35 metadata `seq_lens`; SGLang `seq_lens` |
| **`page_table` (kv page indices)** | `i32 × B × max_pages_per_seq` (row-major, or `kv_indices` + `kv_indptr` CSR) | per-row physical page ids from `PagedKVPool::page_indices(slot)` | **paged attention kernel** | qwen35 `page_indices_gpu` (prefill_buffers.rs L154); SGLang `kv_indices`/`kv_indptr` static buffers |
| `slot_mapping` / `out_cache_loc` | `i32 × B` | physical KV location for *this* step's appended token (tail page slot) | KV-append kernel | SGLang `out_cache_loc` |
| `logits_out` | `bf16/f32 × B × vocab` | (output — graph writes it) | LM-head GEMM output | qwen3 `logits` (decode_buffers.rs L51); SGLang `next_token_logits_buffer` |

Plus the **already-fixed activation scratch** the eager path allocates once
(`normed`, `q/k/v`, `attn_out`, `mlp_*`, `hidden` — qwen3 `DecodeBuffers` L103-112).
These are stable by construction (allocated at decode-context creation, never
resized per step), so the captured graph holds them directly with no per-step write.

### The page-table feed (the subtle one)

The paged attention kernel must read the page table from a **fixed device buffer**,
not from a host slice passed by value. Today `PagedKVPool::page_indices(slot)`
returns a **host** `&[u32]` (paged_kv.rs L1273-1275); the eager path uploads it per
call. For the graph, the executor maintains a fixed device `page_table` buffer of
shape `[B][max_pages_per_seq]` (or CSR `kv_indices`+`kv_indptr`):

1. **Stage 1, per step**: for each of the B rows, copy `pool.page_indices(slot)`
   into row `i` of the fixed `page_table` device buffer (one batched H2D, or a
   small gather kernel). Zero/`-1`-pad the unused tail of each row up to
   `max_pages_per_seq`. `seq_lens[i]` tells the attention kernel how many of those
   page slots are valid.
2. **Stage 2**: replay. The captured attention kernel dereferences the fixed
   `page_table` base pointer (constant address) and indexes it by `blockIdx`/row;
   it sees the freshly-written page ids.

This is precisely SGLang's `init_forward_metadata_replay_cuda_graph` rewriting
`kv_indices`/`kv_indptr` before `graph.replay()` (§2). The page *table* is dynamic;
its *storage address* is fixed.

**Sizing `max_pages_per_seq`**: bounded by `effective_max_seq_len / page_size`
(page_size = 16, the TileLang HD128 invariant, warmup.rs L212). The buffer is
allocated once at decode-context creation to the worst case so it never reallocs.

### Why "no CPU-GPU sync, no allocation" matters

`run_or_capture`'s contract (cuda_graph.rs L36-37) is non-negotiable: the captured
closure must be a pure kernel sequence. Concretely the captured region must contain
**no** `to_host`, **no** `synchronize`, **no** `cudaMalloc`/`DeviceVec::zeros`, and
**no** cublasLt heuristic query (that is why warmup pre-populates and freezes the
algo cache, then re-captures, warmup.rs L130-168). All H2D metadata writes happen in
**Stage 1, before** `graph.replay()`, on the same stream — they are issued, not
captured. Any sync/alloc inside the captured region either errors at `end_capture`
or, worse, captures stale behavior.

---

## 6. Bucketing, warmup capture flow, padding, fallback

### Bucket schedule (port legacy `cuda_graph_batch_sizes`, warmup.rs L446-464)

- **Dense 1..=min(64, max_bs)** — every size, to kill batch-composition-churn
  graph misses (the documented p99 ITL spike source).
- **Sparse step-16 from 80** up to `max_bs`, plus `max_bs` itself.
- `max_bs = num_slots.min(cuda_graph_max_bs)`, `cuda_graph_max_bs` default 256.
- **AI-PC c=1 note**: on the single-user AI-PC target the only bucket that matters
  is **B=1** (§7). The dense band still captures cheaply; B=1 is the headline win.

### Warmup capture flow (inside `CudaExecutor::warmup`)

Port the proven two-pass shape (warmup.rs L107-168), adapted to the new executor's
own KV pool and decode buffers:

1. Reserve B dummy slots in the executor's `PagedKVPool` (`alloc_tokens(slot, 1)`),
   lazy-init the decode context if absent.
2. **Pass 1** — for each bucket size B (ascending or, like SGLang, descending to
   share the memory pool): set batch size, write dummy metadata into the fixed
   buffers (§5), run `forward_decode` **in capture mode** → store `CudaGraph`,
   then `sync()`. Failure at a size → log + `break` (skip larger), never hard-fail.
   This also populates the cublasLt heuristic algo cache.
3. **Autotune** all cached GEMM shapes (`autotune_all_cached_gemms_cuda`), unless
   `INFER_DETERMINISTIC=1` (which skips autotune to keep batch-invariant numerics —
   warmup.rs L126-147; relevant because the greedy-parity gate in §8 needs
   determinism).
4. **Pass 2** — invalidate the Pass-1 graphs and **re-capture** with autotuned
   algos (only when autotune ran). Skip in deterministic mode (no autotune → no
   re-capture needed).
5. Free all dummy slots; reset state. Runs before the HTTP-ready signal (legacy
   ordering, scheduler_loop.rs L117-120), so capture never races serving.

### Padding & fallback

- **Pad-up (v1 default, SGLang-style)**: a pure-decode plan with `B'` rows where
  `B' < max_bs` and `B'` is not an exact captured key → pick the smallest captured
  bucket `B >= B'`, fill rows `B'..B` with dummy padding (page table → reserved
  slot, `seq_lens` → fill value, dummy token 0), replay the B-graph, discard the
  pad rows' outputs. With dense 1..64 this only triggers in the sparse band.
- **Eager fallback (always available)**: if no captured bucket fits (`B' > max_bs`),
  or the plan is prefill/mixed, or a row is graph-unsafe (§7) → run the existing
  eager `forward`. Eager is the correctness floor; graph is the fast path. A graph
  bug can never produce a wrong answer that eager wouldn't — same kernels, same
  buffers.

---

## 7. Interactions

### Submit/poll overlap (infer-core L324-358)

The engine's overlap is: poll the previous in-flight step first; if `NotReady`,
exit the tick keeping the handle; once `Ready`, apply output, admit, build plan
N+1, `submit`. A graph replay is **one `cuGraphLaunch`** — strictly fewer CPU
instructions than the eager path's hundreds of launches, so it *helps* overlap:
the CPU finishes issuing plan N's work faster and has more slack to build plan N+1
behind the seam. `poll` is unchanged — a replayed step completes via the same
stream-event / sync mechanism as an eager `submit`. No overlap redesign needed;
this is the SGLang property that graph replay is "just one launch the scheduler
issues" (§2).

### Sampling: OUTSIDE the graph

The captured graph ends at `logits_out` (the fixed LM-head output buffer). Token
selection runs in eager code after replay, exactly as the legacy executor's
`sample_cuda_token` (executor.rs L195-209): greedy → `argmax(ctx, logits)`;
non-greedy → `to_host` + `sample_token`. Reasons sampling must stay out:

- Non-greedy sampling does a `to_host` readback (executor.rs L207) — a CPU-GPU
  sync, which §5 forbids inside the captured region.
- Sampling params (`temperature`, `top_k`, penalties) vary per request per step;
  baking them into a graph would force a recapture per param-set. SGLang reaches
  the same conclusion (`next_token_logits_buffer` captured, selection eager).
- **Graph-safe row test (§4 dispatch)**: a row is graph-safe if its sampling is
  greedy *or* if non-greedy sampling is applied to the graph's logits output
  *after* replay. Since sampling is always post-graph, **all** decode rows are
  graph-safe with respect to the graph itself; the only true gate is decode-only +
  bucket-fit. (Penalty kernels needing per-request token history — the executor.rs
  L205-206 TODO — also run post-graph on `logits_out`, no graph impact.)

### AI-PC c=1 focus — the headline win

[`2026-06-03-aipc-pivot-and-northstar.md`](2026-06-03-aipc-pivot-and-northstar.md)
makes single-user interactive the north star, and the project memory pins the
Metal local focus at c=1. The CUDA AI-PC story is identical: **a B=1 decode graph
is the single biggest per-token latency win available**, because at B=1 the decode
step is purely launch-bound (tiny GEMV-shaped kernels, hundreds of them). Replacing
~250-400 launches with one `cuGraphLaunch` removes essentially all per-token CPU
launch overhead — the dominant cost at B=1. This is why B=1 is the must-capture
bucket and the primary verification target (§8). On a busy server the per-token
launch overhead is amortised across a large batch's compute; at c=1 it is the whole
story.

### `ResourceGovernor` / step budget

Orthogonal. The governor (infer-seam L152-161) gates *admission* and *step budget*;
once a decode plan is admitted and built, whether it replays a graph or runs eager
is the executor's internal choice. No interaction.

---

## 8. Verification

Gated behind the eager CUDA parity gate (R6a). Verification is two claims:
**correctness (graph == eager)** and **the c=1 win (graph < eager latency)**.

### Correctness: greedy parity vs eager

- Capture decode graphs for B ∈ {1, 2, 4}. Run the **same greedy prompt** twice
  through the executor: once forcing eager (`force_eager`, the legacy hook exists —
  qwen35 `force_eager_once`, batch_decode.rs L682-684), once allowing replay.
- **Assert the full generated greedy token sequence is identical** — this is the
  exact bar R6a sets for eager parity
  ([r6-cuda-port-plan](2026-06-03-r6-cuda-port-plan.md) §R6a: *"compares the full
  generated greedy token sequence"*). The graph reuses the same kernels and the
  same cublasLt algos (re-captured post-autotune), so token-exact parity is the
  requirement, not approximate.
- Run with `INFER_DETERMINISTIC=1` to pin batch-invariant cublasLt algos so B=1 vs
  B>1 don't diverge on fp accumulation order (warmup.rs L126-147 documents this
  exact failure mode).
- Multi-bucket: verify pad-up (B'=3 padded into a B=4 graph) yields the same first-3
  tokens as the unpadded B=3 graph and as eager.

### Performance: decode tok/s at c=1 on H20

- Per [bench-and-trace-spec](../bench-and-trace-spec.md) and the mandatory
  wins-entry rule: a matched A/B, **same binary, same shell, two flips**
  (`--cuda-graph` on vs off, equivalently `--disable-cuda-graph`), side-by-side,
  c=1, production prompt shape. Report decode tok/s + ITL p50/p99 Δ%.
- Expected direction: c=1 decode tok/s **up** (fewer launches), ITL p99 **down**
  (no per-token launch jitter). Magnitude is the open question the bench answers —
  the legacy tree's dense-band rationale was specifically p99 ITL spike removal.
- **Framing discipline** (CLAUDE.md §0, M_pf-graph framing trap): report the win as
  **per-token wall-clock / per-request ITL**, not "X% of an nsys decode window". A
  narrow-window percentage is not the license; the per-request ITL Δ at c=1 is.
- Cross-check on H20 (the project's TP=8 / DSv4 pod) for the server-side multi-bucket
  case, but the **c=1 single-GPU number is the headline** per the AI-PC north star.
- nsys cross-check: confirm decode region shows **one `cuGraphLaunch`** replacing
  the hundreds of `cuLaunchKernel` calls — this is the mechanism-level evidence
  that the win is real and attributable (not a confounded measurement).

### Negative / safety checks

- Forcing a realloc of a fixed buffer mid-serve must invalidate + recapture, never
  replay a stale-pointer graph (mirror legacy `reallocated → invalidate_graph_cache`).
- Bucket miss above `max_bs` must cleanly hit eager fallback, not panic.

---

## 9. Effort estimate & sequencing

**Hard gate: this lands only after the R6a eager CUDA parity gate passes.**
The rewrite plan is explicit:
[r6-cuda-port-plan](2026-06-03-r6-cuda-port-plan.md) §CUDA Graph — *"R6a should
disable graph or use the existing eager path. GraphRunner should be ported only
after eager parity is proven."* and §R6a target — *"no graph"*. The clean executor
must produce token-exact greedy output eagerly first; the graph is a pure latency
optimization layered on a proven-correct path, and its own correctness test (§8) is
*defined as* "matches that eager output".

Sequencing slot: **after R6a (single-slot Qwen3 greedy eager parity on V100/H20),
before or alongside R6b (batched decode)**. B=1 graph needs only single-row decode
working; multi-bucket graphs need batched decode. So:

- **R6a+**: B=1 decode graph (single-row, the AI-PC headline). Smallest viable
  increment, highest c=1 ROI.
- **R6b+**: dense 1..64 + sparse buckets, pad-up, two-pass autotune recapture.

**Effort (engineering, post-gate):**

| Work item | Est. | Notes |
|-----------|------|-------|
| Fixed-buffer struct + Stage-1 writers in `infer-cuda` (input ids, positions, seq_lens, page_table, slot_mapping, logits) | ~1.5 d | Mostly porting qwen3 `DecodeBuffers` + qwen35 metadata buffer shapes into the new executor |
| `CudaExecutor::submit` dispatch branch (decode-only + bucket-fit → replay, else eager) | ~0.5 d | One branch; eager path already exists |
| Capture helper (begin/end capture, `CudaGraph` cache keyed by B) | ~0.5 d | Direct port of `run_or_capture` |
| `CudaExecutor::warmup` bucket-capture + two-pass autotune recapture | ~1.5 d | Port `warmup_cuda_graphs` to the new executor + KV pool |
| Page-table fixed-buffer feed + pad-up logic | ~1 d | The subtle piece (§5); needs a per-step gather/copy into the fixed buffer |
| Greedy-parity test (eager vs replay, full sequence) + multi-bucket pad test | ~1 d | Reuse R6a's greedy-sequence harness |
| c=1 H20 bench A/B + wins entry (mandatory, §8) | ~0.5 d | Matched same-binary A/B |
| nsys cross-check (one cuGraphLaunch) | ~0.5 d | Mechanism evidence |

**Total ≈ 7 engineer-days** post-gate, of which **B=1-only is ≈ 3 days** (fixed
buffers + dispatch + capture + B=1 parity + c=1 bench) and is the recommended first
landing.

---

## 10. Deferred / out of scope (explicit, per §0 no-silent-deferral)

- **Piecewise (per-linear-group) graphs** — needed for Qwen3.5 hybrid linear-attn
  and DSv4; port the qwen35 `run_linear_group_graphed` scheme after the whole-decode
  form ships. DSv4 additionally needs the synthetic-warmup substrate fix (its body
  graph capture is currently zero because warmup doesn't materialize the
  compressed/FP8/FlashMLA cache — r6-cuda-port-plan §CUDA Graph). Tracked separately.
- **Prefill graphs** — killed in legacy (shape churn regressed c=4/8/16); SGLang
  routes variable-length prefill through its separate Piecewise CUDA Graph feature.
  Not in v1.
- **Mixed-mode partial graphing** (graph the decode subset, eager the prefill rows
  in one submit) — possible later; v1 runs Mixed fully eager.
- **In-graph sampling** — rejected (§7): readback sync + per-request param variance.
- **Graph capture under NCCL/DeepEP collectives** — the r6 plan flags
  *"graph capture safety for NCCL and DeepEP"* as an open item; multi-GPU graphs are
  a TP-track concern, gated separately from the single-GPU AI-PC win this doc targets.
- **TileLang paged-attention capture compatibility** — assumed (the legacy paged
  decode path captures today); to be re-verified on the new executor's attention
  call during R6b, since the new path may issue the attention kernel differently.
  Flagged, not assumed-proven.
