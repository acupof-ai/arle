# Tiered KV Cache — execution plan (local / remote-CUDA / parallel-GPU)

**Status**: Active execution split for [`../projects/tiered-kv-cache.md`](../projects/tiered-kv-cache.md).

**Revised 2026-04-15**: see §0.5 for the P0-P5 → M0-M5 remapping. The
project doc was re-architected after an internal survey + 7-system industry
comparison found three corrections (BlockId unification, RadixCache and
TierDirectory merge, page_size=1 blocks tier transfer bandwidth). The
detailed task lists per phase below are still largely correct; only the
phase grouping and the file paths (post Route-A) change. Section §0.5
translates old P0–P5 references into the new M0–M5 plan.

This doc carves every phase from the project plan into three lanes so the
local Mac dev box and the remote CUDA host can both stay busy at all times.
The project doc is the **what + why**; this doc is the **where you do it**,
**what production systems do for this exact problem**, and **what changed
in our plan after the research**.

Every phase below has three subsections:
- **Lane breakdown** — checklist tagged `[L]` (Mac-only), `[R]` (remote CUDA-only), `[L+R]` (local edits, remote verify)
- **Industry references** — concrete file:line refs from vLLM / SGLang / FlashInfer / LMCache / Mooncake / NIXL / KVFlow
- **Course corrections from research** — places where the project plan must change because production systems do it differently than we initially designed

---

## 0 · Build matrix recap

| Lane | Build invocation | Runs | Does NOT run |
|---|---|---|---|
| **Mac · no-cuda** | `cargo check --no-default-features --features no-cuda` | type/borrow check, non-GPU unit tests | CUDA kernels, FlashInfer link, e2e |
| **Mac · metal** | `cargo build --release --no-default-features --features metal` | Metal backend, `metal_kv_pool` tests, `mlx-sys` bindings | scheduler/cuda/*, FlashInfer |
| **Remote CUDA** | `cargo build --release` (default features = `["cuda"]`) | full stack, `e2e`, `e2e_qwen35`, `greedy_consistency`, `bench_throughput_sweep.py` | n/a |

**`#[cfg(feature = "cuda")]` gating that affects local-checkability** (read from
`infer/src/lib.rs:1-17`):

| Module | Gated under cuda? | Local cargo check sees it? |
|---|---|---|
| `paged_kv` | YES | NO |
| `flashinfer_metadata` | YES | NO |
| `model/*` | YES | NO |
| `ops/*` | YES | NO |
| `tensor` | YES | NO |
| `prefix_cache` | NO (always-on) | **YES** ✓ |
| `block_manager` | NO | YES ✓ |
| `metal_kv_pool` | NO | YES ✓ |
| `metal_prefix_cache` | NO | YES ✓ |
| `scheduler` | NO (but `scheduler/cuda/*` files import cuda-gated types) | partial |
| `infer/src/scheduler/policy.rs` | internal module | YES ✓ |
| `infer/src/events.rs` | internal module | YES ✓ |
| `infer/mlx-sys` | always built when `metal` feature on | YES (with `--features metal`) |

**Lane rules**:
- A task is `[L]` only if both `cargo check --features no-cuda` AND `cargo check --features metal` validate it. Any change inside `paged_kv` / `flashinfer_metadata` / `scheduler/cuda/*` / `model/*` / `ops/*` is automatically `[R]` or `[L+R]`.
- `[L+R]` = local can write the diff; only remote CUDA can verify it. Used for cuda-gated files where the change is mechanical and review-able by eye on Mac.
- A task is `[R]` only when its exit gate is a benchmark / e2e test that needs a GPU.

---

## 0.5 · 2026-04-15 revision — P0-P5 → M0-M5 remapping

Project doc §6 was re-organised into M0–M5 after the internal survey +
industry comparison. The detailed per-phase task lists in §1–§6 below are
still correct at the per-task level; only the phase groupings and a few
file paths change. This table translates old phase references into the new
milestone plan.

| Old phase | New milestone | What changed | Task doc section |
|---|---|---|---|
| P0 page_size=16 | **M0.3** | Unchanged intent; now gated on the in-flight `infer-cuda-kernels` crate extraction landing. Per-format dispatch still applies (BF16 → 16, quantized → stay at 1). | §1 below |
| P1 (a) structural (BlockId, kv_tier module, serde) | **retired** | The old `kv_tier::BlockId(u64)` / `TierDirectory` / `BlockDescriptor` skeleton is being deleted in M1. Do **not** extend it. The bug fixes from P1(a) (three `prefix_cache.rs` correctness bugs) are now **M0.2** — see §2.1 below. | §2 below, fragment |
| P1 (b) behavior (scheduler wire) | **M1** | The atomic PR. Expanded: no longer "wire RadixCache then merge directory" as two steps — instead, **delete `kv_tier/directory.rs`** and move its fields onto `RadixNode` in the same PR. | §2 below |
| — (new) | **M0.1** | `BlockId` unification: `types::BlockId(u32)` canonical, `types::BlockFingerprint([u8; 16])` separate. Deletes `kv_tier/id.rs` and `block_manager::BlockId`. Blocks M1. | (new, §2.1 addendum) |
| — (new) | **M2** | Dual residency (T0-only): `RadixCache::evict_into_free_queue`, pool reuses free-queue slots, `lookup` can resurrect. Was implicitly absorbed into "P2 behavior"; now first-class because it is the single biggest prefix-hit lever and is orthogonal to tiering. | (new, §3.5 addendum) |
| P2 T2 host pinned + coordinator | **M3** (renamed to T1 host pinned) | Tier numbering T0/T2/T3/T4 → T0/T1/T2/T3 for industry alignment. Content unchanged. **Coordinator is an OS thread + crossbeam** (task doc §3.3 course correction, now committed in the project doc §4.4). Split into M3a transport / M3b coordinator / M3c promote. | §3 below |
| P3 T3 disk + session save/load | **M4** (renamed to T2 disk) | Same renumber. Content unchanged. MLX wired-memory bindings still required for Metal bounding. | §4 below |
| P4 KVFlow-lite reuse-distance + cache-aware routing | **post-M4 experiment** | Dropped from critical path. LRU / SessionBiasedLru ship in M3; priority-bucket LRU (TRT-LLM style) is the more promising post-M3 experiment. Reuse-distance is deferred until M3's default policy is proven insufficient. | §5 below (keep for reference) |
| P5 NIXL trait freeze + stub | **M5 (stub only)** | `NixlTransport` stub and trait shape are already shipped as of 2026-04-15 (`infer/src/kv_tier/transport.rs` + `transport/nixl.rs`, 144 + 205 lines). The "real RDMA" half of M5 is **deferred** until a trigger fires (prefill/decode disaggregation, cross-node session roaming, second consumer of the kernel crate). | §6 below |

**New M0 scope** (prereqs for M1, all independent PRs):

- **M0.1** — BlockId unification. New `infer/src/types.rs::BlockId(u32)` +
  `BlockFingerprint([u8; 16])`. Delete `infer/src/kv_tier/id.rs`. Update
  `prefix_cache::BlockId` and `block_manager::BlockId` to re-export the
  canonical type. Pure rename + `use` path updates, no algorithmic change.
- **M0.2** — Three `prefix_cache.rs` correctness bugs
  (`_split_node` ref_count inherit, `lookup` ancestor walk, `evict`
  iterate orphans). Unit-tested, no scheduler changes. Details in §2.1 below.
- **M0.3** — page_size lift from 1 to 16 with per-format dispatch (the old
  P0). Blocks on the in-flight `infer-cuda-kernels` crate extraction
  landing — after that extraction, the kernel files move from
  `infer/csrc/cuda/kv/*.cu` to `crates/infer-cuda-kernels/csrc/kv/*.cu`.

**Paths updated post Route-A (structural, not content)**:

All "cuda-gated" file paths in the detailed task sections below shifted
during the Route-A workspace rewrite (commit `d902090`) and the CUDA
internal hygiene pass (commit `26c8f39`). The table below lists the
renames that affect this doc:

| Pre Route-A path | Current path |
|---|---|
| `infer/src/paged_kv.rs` | `infer/src/backend/cuda/paged_kv.rs` |
| `infer/src/flashinfer_metadata.rs` | `infer/src/backend/cuda/flashinfer.rs` |
| `infer/src/tensor.rs` | `infer/src/backend/cuda/tensor.rs` |
| `infer/src/metal_kv_pool.rs` | `infer/src/backend/metal/kv_pool.rs` |
| `infer/src/metal_prefix_cache.rs` | `infer/src/backend/metal/prefix_cache.rs` |
| `infer/src/metal_gdr.rs` | `infer/src/backend/metal/gdr.rs` |
| `infer/mlx-sys/src/lib.rs` | `crates/mlx-sys/src/lib.rs` |

When reading §1–§6 below, apply these renames mentally. After the
in-flight `infer-cuda-kernels` extraction PR lands, additional renames
will apply (`infer/src/backend/cuda/*.rs` → `crates/infer-cuda-kernels/src/*.rs`,
`infer/csrc/cuda/*.cu` → `crates/infer-cuda-kernels/csrc/*.cu`,
`infer/tools/triton/*.py` → `crates/infer-cuda-kernels/tools/triton/*.py`);
this doc will get another path update pass at that point.

---

## 1 · Milestone M0.3 (formerly P0) — `page_size = 16` with per-format dispatch

### 1.1 Lane breakdown

#### Local (Mac) — `[L+R]` because all touched files are cuda-gated
- [ ] `[L+R]` Edit `infer/src/paged_kv.rs:7,76-87,482-511,549-562,614,760` — add `page_size: usize` field on `TokenKVPool`, rewrite `alloc_tokens` as **two-level allocator** modeled on vLLM (see §1.2). Keep `Result<Vec<u32>>` signature even though all 4 callers discard the Vec — return `Vec::new()` for now to avoid touching call sites in the same PR
- [ ] `[L+R]` Edit `infer/src/flashinfer_metadata.rs:124-180` — `indptr` becomes **cumulative pages** (was tokens); rewrite incremental update to distinguish "new token lands inside last page (bump `last_page_len`)" vs "spills to new page (append page id, reset `last_page_len = 1`)"
- [ ] `[L+R]` Edit `infer/src/model/qwen3/batch_decode.rs:384` — drop `let page_size = 1;`, read from `kv_pool.page_size`
- [ ] `[L+R]` Edit `infer/src/model/qwen35/batch_decode.rs:724` — same
- [ ] `[L+R]` Edit `infer/src/model/glm4/batch_decode.rs:338` — same
- [ ] `[L+R]` Edit `infer/src/scheduler/cuda/decode.rs:193` — literal `1` → `paged_kv_pool.page_size`
- [ ] `[L]` Run `cargo check --no-default-features --features no-cuda` — verify nothing in the always-on modules broke
- [ ] `[L]` Run `cargo check --no-default-features --features metal` — verify Metal build untouched
- [ ] **Hand off to remote GPU** with the diff and the §1.3 caveat list

#### Remote GPU — `[R]`
- [ ] `[R]` `cargo build --release` — CUDA compile + FlashInfer link green
- [ ] `[R]` `cargo test --release` — unit tests pass
- [ ] `[R]` `cargo test --release --test e2e` — qwen3 greedy parity unchanged
- [ ] `[R]` `cargo test --release --test e2e_qwen35` — qwen35 greedy parity unchanged
- [ ] `[R]` `cargo test --release --test greedy_consistency` — multi-config sanity unchanged
- [ ] `[R]` `scripts/bench_throughput_sweep.py --label page16` — perf snapshot
- [ ] `[R]` Compare to `--label page1` baseline (parallel-GPU §7.1); write `docs/experience/wins/2026-04-13-bench-page16.md`
- [ ] `[R]` Watch the short-context tail in the sweep — FlashInfer split-KV scheduler can shed parallelism with larger pages at very short contexts

### 1.2 Industry references

**Default page_size in production**:
- **vLLM v1**: `DEFAULT_BLOCK_SIZE: ClassVar[int] = 16` at [`vllm/config/cache.py:19`](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py#L19), unchanged since v0.5. Resolved by `_apply_block_size_default` at [`cache.py:108-114`](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py#L108).
- **SGLang**: `TokenToKVPoolAllocator` (non-paged) hard-codes `page_size=1` at [`allocator.py:121`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/allocator.py#L121). `PagedTokenToKVPoolAllocator` at [`allocator.py:341-465`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/allocator.py#L341) takes `page_size` as a ctor arg; typical 64 (TRT-LLM parity) or 1.
- **FlashInfer**: `paged_kv_t::page_size` is a runtime `uint_fastdiv` field, no template parameter, no static_assert. Tests parametrize `page_size ∈ {1, 5, 8, 16}`. No upper bound.
- **HF TGI**: same as vLLM — block_size 16.
- **Recommendation: 16.** Schelling point of the entire FlashInfer ecosystem.

**Two-level allocator design**:
- **vLLM v1 `BlockPool`** ([`vllm/v1/core/block_pool.py:146-177`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py#L146)) — pool of `KVCacheBlock` *objects* (not raw indices), each carrying `{block_id, ref_cnt, _block_hash, prev_free_block, next_free_block, is_null}`. Free list = **doubly-linked list with sentinel head/tail** (`FreeKVCacheBlockQueue` at [`kv_cache_utils.py:147-359`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py#L147)). `get_new_blocks(n)` pops `popleft_n()` then bumps `ref_cnt`; `free_blocks` decrements and appends only when `ref_cnt == 0`.
- **SGLang `PagedTokenToKVPoolAllocator`** ([`allocator.py:341`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/allocator.py#L341)) — two tensors `free_pages` (live) + `release_pages` (pending), `merge_and_sort_free()` lazy fold. **LIFO** by default. `release_pages` prepends so freed blocks reused first.
- **Spill-to-new-page**: SGLang's `alloc_extend_kernel` ([`allocator.py:367-409`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/allocator.py#L367)) splits growth into **3 parts** — (1) fill the old partial page from `last_loc`, (2) pop `num_new_pages` from free list for whole-page middle, (3) fill the trailing partial. `alloc_decode` ([`allocator.py:411-436`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/allocator.py#L411)) is the hot path: pops a new page only if `(last_loc + 1) % page_size == 0`.
- **vLLM hides the boundary** — `KVCacheManager.allocate_slots` just appends new `KVCacheBlock` whenever `num_computed_tokens` crosses a `block_size` multiple.

**`last_page_len` semantics — FlashInfer hard constraint**:
- **`1 ≤ kv_last_page_len ≤ page_size`**. **Zero is invalid** — empty sequences MUST NOT appear in the FlashInfer batch. We must filter empty slots in `build_flashinfer_metadata` before passing to FlashInfer's `plan()`.
- **`indptr` is cumulative PAGE count, not token count.** Currently `paged_kv.rs:518-528` cumulates `token_indices.len()`, correct only because `page_size=1`. Must change to `div_ceil(seq_len, page_size)`.
- Formula: `last_page_len = ((seq_len - 1) % page_size) + 1` (assuming `seq_len > 0`).

**Migration path**:
- **vLLM/SGLang prefill DIRECTLY into paged blocks.** No contiguous intermediate. `attn_impl.forward` calls FlashInfer `append_paged_kv_cache` with the slot-mapping from `KVCacheManager.allocate_slots` before the kernel runs.
- **agent-infer's `migrate_from_contiguous`** at [`paged_kv.rs:571-624`](file:///path/to/code/agent-infer/infer/src/paged_kv.rs#L571) is a **legacy artifact** of the old `ContiguousKVCache`. It works at page_size=1 because `kv_cache_to_paged.cu:41-47` uses `logical_page = pos / page_size`.
- **Recommendation**: P0 keeps the migration path. Mark `migrate_from_contiguous` as `#[deprecated = "P1: prefill direct into paged blocks"]`. Schedule P1+ follow-up to remove contiguous prefill entirely (matches vLLM/SGLang).

**Kernel readiness** (verified by reading `infer/csrc/cuda/*.cu`):
- ✅ [`paged_kv_append.cu:43-57`](file:///path/to/code/agent-infer/infer/csrc/cuda/paged_kv_append.cu) — `logical_page = pos / page_size`, `physical_page = page_indices[indptr[b] + logical_page]`, `stride_page = num_kv_heads * page_size * head_dim` (computed inside kernel). **Fully page_size-parametric.**
- ✅ [`decode_prep_paged.cu:138-155`](file:///path/to/code/agent-infer/infer/csrc/cuda/decode_prep_paged.cu) (HD128) — uses `last_page_len - 1` for in-page offset; HND addressing correct. Caller must set `stride_page = num_kv_heads * page_size * head_dim`.
- ✅ [`decode_prep_paged_hd256.cu:157-160`](file:///path/to/code/agent-infer/infer/csrc/cuda/decode_prep_paged_hd256.cu) — identical paging logic for Qwen3.5 full attention.
- ✅ [`kv_cache_to_paged.cu:41-47`](file:///path/to/code/agent-infer/infer/csrc/cuda/kv_cache_to_paged.cu) (bf16 path) — full `pos / page_size` decomposition.
- ✅ All FlashInfer wrapper files (`flashinfer_decode.cu`, `flashinfer_decode_hd256.cu`, `flashinfer_tc_decode.cu`) just forward `page_size` into FlashInfer's `paged_kv_t<>`.
- ⚠️ **`kv_cache_to_paged_int8_kernel` at [`kv_cache_to_paged.cu:64-103`](file:///path/to/code/agent-infer/infer/csrc/cuda/kv_cache_to_paged.cu#L64) hardcodes page_size=1.** Computes `pool_idx = page_indices[pos]` directly. No `page_size` parameter in signature.
- ⚠️ **`kv_quant.cu:184,193,207,211`** (`quantize_paged_kv_fp8_kernel`, `quantize_scatter_kv_fp8_kernel`) — same NHD per-token assumption.
- ⚠️ **`scatter_kv.cu`** — entirely page_size=1.

### 1.3 Course corrections from research

1. **INT8/FP8 quantized paths stay at page_size=1 in P0.** The bf16 HND path is clean; the NHD quantized kernels are not. Two options:
   - **Recommended**: gate `page_size` per-format. `TokenKVPool::page_size` becomes `match format { BF16 => 16, INT8 | FP8E4M3 | TurboQuant => 1 }`. Document the divergence prominently. Schedule P1.5 to either rewrite the INT8 kernels with `pos / page_size` decomposition or move INT8 to a separate HND format. **Single-line change in `with_format`**: `let page_size = if format.uses_paged_layout() { 16 } else { 1 };`
   - Alternative: rewrite `kv_cache_to_paged_int8_kernel` and the FP8 quantize kernels in P0. Bigger blast radius, kernel work, +1 PR. **Defer.**
2. **`indptr` semantics change** — `build_indptr` cumulates pages now, not tokens. This is the most subtle change in P0; every caller of `pool.build_indptr()` must be re-audited. Currently 1 caller: `flashinfer_metadata.rs:124`.
3. **Empty-slot filter** — `build_flashinfer_metadata` must skip slots with `seq_len == 0`. Currently it emits `last_page_len[i] = 0` for empty slots ([`paged_kv.rs:503`](file:///path/to/code/agent-infer/infer/src/paged_kv.rs#L503)), which violates FlashInfer's `[1, page_size]` invariant. The fact this hasn't crashed at page_size=1 is luck — FlashInfer treats `last_page_len=0` as "empty pages array, length 0", which the test path tolerates because `indptr[i+1] - indptr[i] = 0` for empty slots. At page_size>1 this no longer reliably aligns.
4. **Two-level allocator data model** — slot-level state changes from `token_indices: Vec<Vec<u32>>` to `(page_indices: Vec<Vec<u32>>, last_page_len: Vec<u32>)`. The free list changes from `free_slots: Vec<u32>` to `free_pages: Vec<u32>` (pages, not tokens). Total pool bytes unchanged: `max_total_pages = max_total_tokens / page_size`.

---

## 2 · Milestones M0.1, M0.2, M1 (formerly P1) — BlockId unify + prefix bug fixes + scheduler wire

### 2.1 Lane breakdown

#### Structural PR (a) — Local (Mac), `[L]` mostly
- [ ] `[L]` Edit `infer/src/prefix_cache.rs` — **fix existing bugs** (see §2.3): (1) `_split_node` ref_count inheritance, (2) `lookup` path-bump, (3) iterative parent eviction
- [ ] `[L]` Edit `infer/src/prefix_cache.rs` — add `session_id: Option<SessionId>` field on `Node`
- [ ] `[L]` Edit `infer/src/prefix_cache.rs` — derive `Serialize / Deserialize` (gates B1 / P3)
- [ ] `[L]` Edit `infer/src/prefix_cache.rs` — change `lookup` return shape to surface block ids (still token-indexed under the hood, but a strongly-typed wrapper for downstream)
- [ ] `[L]` Add `infer/src/kv_tier.rs` + `infer/src/kv_tier/{id,directory,tier}.rs` (flat layout). **Do NOT cuda-gate this module** — keep it in the always-on set so `cargo check --features no-cuda` validates it
- [ ] `[L]` Add unit tests: serde round-trip, recursive eviction, session-tagged lookup, `_split_node` refcount inheritance regression test
- [ ] `[L]` `cargo check --no-default-features --features no-cuda` green
- [ ] `[L]` `cargo check --no-default-features --features metal` green
- [ ] `[L]` `cargo test --no-default-features --features no-cuda kv_tier prefix_cache` — pure-Rust tests pass

#### Structural PR (a) — Remote GPU
- [ ] `[R]` `cargo build --release` green (no behavior change expected)
- [ ] `[R]` `cargo test --release` — entire suite still green; new tests included

#### Behavior PR (b) — `[L+R]`
- [ ] `[L+R]` Edit `infer/src/scheduler/cuda/core.rs:114-141` — hold `Arc<TieredKvCache>`; remove `cached_prompts: Vec<Vec<u32>>`
- [ ] `[L+R]` Edit `infer/src/scheduler/cuda/runtime.rs:75-151` — admission rewrite using radix lookup
- [ ] `[L+R]` Edit `infer/src/scheduler/cuda/request.rs` — `ActiveRequest` carries `Vec<BlockId>` accumulator
- [ ] `[L+R]` Edit `infer/src/scheduler/cuda/prefill.rs:14` — read prefix hit length from radix lookup result, drop linear compare
- [ ] `[L+R]` Edit `infer/src/server_engine.rs:437-475` — drop the second-source-of-truth `cached_prompt: Vec<u32>` (third one if you count `metal_prefix_cache.rs`'s wrapper that's never called)
- [ ] `[L+R]` Edit `infer/src/paged_kv.rs` — expose `alloc_slot / free_slot / read_into / write_from` as the T0 physical layer
- [ ] `[L]` `cargo check --no-default-features --features no-cuda` and `--features metal` — Rust types still align across the boundary
- [ ] `[R]` `cargo build --release` green
- [ ] `[R]` Full e2e + greedy_consistency
- [ ] `[R]` **Regression gates** (from `agent-first-architecture.md` §5): `grep -r cached_prompts infer/src/` returns empty; `grep -r RadixCache infer/src/scheduler/` returns non-empty
- [ ] `[R]` Cross-session benchmark on `scripts/bench_agent_trace.py` (built in §7.1): 2-session alternating trace, prefix hit rate ≥70%
- [ ] `[R]` Bench markdown in `docs/experience/wins/`

### 2.2 Industry references

**SGLang radix integration** (the gold reference because we already match its tree-based paradigm):
- `match_prefix(MatchPrefixParams) -> MatchResult` at [`radix_cache.py:334`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py#L334) — returns `device_indices: torch.int64`, `last_device_node`, `last_host_node`.
- **Path-based `lock_ref` semantics** — `inc_lock_ref` ([radix_cache.py:789-801](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py#L789)) walks **upward from leaf to root**. On each `0→1` edge it moves `len(node.key)` tokens from `evictable_size_` to `protected_size_`. Locking a leaf protects its **entire ancestor chain**.
- **Iterative upward eviction** — `evict()` at [lines 756-777](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py#L756) uses a heap, pops lowest-priority leaf, calls `_delete_leaf`, then if `len(parent.children) == 0 and parent.lock_ref == 0`, pushes the parent back onto the heap.
- **Split node refcount inheritance** — `_split_node` at [lines 830-851](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py#L830) does `new_node.lock_ref = child.lock_ref` (line 841). The intermediate node inherits the same count that was protecting the original child.

**vLLM block hash chain** (the paradigm we explicitly chose NOT to use):
- Hash formula at [`kv_cache_utils.py:375-397`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py#L375): `hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))`. `extra_keys` carries multimodal/LoRA/cache-salt context.
- **Pluggable hash function**: `sha256_cbor` or `xxhash_cbor` (`_CBOR_HASH_FUNCTIONS` at line 138). **Not blake3, not Python builtin.**
- Storage: `dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]` — flat hash table, not a tree.
- **No collision detection**: vLLM trusts hash strength. `xxhash64` is NOT cryptographic — implicit user choice.
- Cached-but-idle blocks live **simultaneously** in `_cache` (hash → block) AND in `FreeKVCacheBlockQueue` (eviction candidates).

**Two paradigms compared**:

| | SGLang RadixCache | vLLM BlockPool |
|---|---|---|
| Structure | Radix tree, keys = token sequences | Flat `dict[hash] → block` |
| Match cost | `O(prompt_len)` tree descent | `O(num_blocks_in_prompt)` lookups |
| Split cost | Real (tree node split) | None (hash chain forks naturally) |
| Collision | Impossible (literal token compare per edge) | Possible; hash-strength-dependent |
| Locking | Path-based, propagates through ancestors | Per-block, caller touches every block |

**Session awareness in production** — vLLM and SGLang are both **session-agnostic**. Mooncake's "Conductor" layers prefix-hit-length + load distribution into a global routing decision but within a single instance still uses content addressing. **`session_id` is NEVER a cache key in production**.

### 2.3 Course corrections from research

1. **KEEP the radix tree, do NOT switch to a hash chain.** Our existing `prefix_cache.rs` (552 lines) IS the correct paradigm. The `BlockId` content-hash design from `tiered-kv-cache.md` §5.1 was overspec — keep block ids as opaque scheduler-assigned IDs in P1; only introduce content-hash form for P3 disk persistence and P5 RDMA cross-node alignment, where it's actually load-bearing.
2. **Three existing bugs in `prefix_cache.rs` that P1 (a) MUST fix** — these are **pre-existing** issues in the orphaned radix cache; P1 is the wiring step where they would start mattering:
   - **Bug 1**: `_split_node` does NOT inherit `ref_count` from the old child. Under concurrent admission, this lets `evict()` steal a path another request is actively using. SGLang's pattern is `new_node.lock_ref = child.lock_ref` (`radix_cache.py:841`).
   - **Bug 2**: `lookup` only bumps the matched leaf's `ref_count`, not the ancestor path. This leaves intermediate nodes evictable while a request holds a deep leaf. SGLang's `inc_lock_ref` walks up to the root.
   - **Bug 3**: `evict()` only considers leaves; after removing a leaf, the now-childless parent should become a candidate in the same pass. SGLang re-pushes the orphaned parent onto its heap.
3. **`session_id` does NOT become a cache key.** It stays in `SchedulerSignals` (already plumbed via commit `3e1d35f`) for slot affinity. The radix cache stays content-addressed.
4. **`_cache` + free queue dual residency** is the vLLM pattern we'll need in P2 — cached blocks must remain reachable by hash even after their refcount drops to 0, until physically reassigned. Note this for the `TieredKvCache` directory shape.
5. **Third independent cached_prompt site** at `server_engine.rs:437-475` (`prepare_with_prefix_cache`) — a single-slot linear compare against `self.cached_prompt`. P1 (b) should thread the `RadixCache` handle through `ModelInferenceEngine` too so there is **one** prefix cache per engine, not three.

---

## 3 · Milestones M2, M3 (formerly P2) — dual residency + T1 host pinned tier + coordinator

### 3.1 Lane breakdown

#### Structural PR (a) — Local (Mac)
- [ ] `[L]` `infer/src/scheduler/policy.rs` — add `EvictionPolicy` trait, `EvictionCandidate`, `SessionState`, `LruEviction` + `ReuseBiasedLru` + `HitCountLru` + `SessionBiasedLru` default. Pure scoring functions, no IO, no state mutation in `score()`.
- [ ] `[L]` Unit tests for all 4 default policy impls — pure Rust, fully local
- [ ] `[L+R]` Add `infer/src/kv_tier/transport.rs` — `KVTransport` trait surface (this trait gets frozen in P5; P2 introduces it under a draft form)
- [ ] `[L+R]` Add `infer/src/kv_tier/transport/local_cuda.rs` — `LocalCudaTransport` over `cudarc` async copy. Code-writeable on Mac, only `cargo build --release` on remote can verify
- [ ] `[L+R]` Add `infer/src/kv_tier/host_pool.rs` — `HostPinnedPool` over `cudaHostAlloc`; **allocation-stable base pointer** (P5 RDMA precondition)
- [ ] `[L+R]` Add `infer/src/kv_tier/coordinator.rs` — **OS thread (NOT tokio task)**, owns 2 dedicated CUDA streams, in-flight queue, drains via `crossbeam::channel::Receiver::recv_timeout(1ms)`
- [ ] `[L+R]` `TieredKvCache::demote / promote` — route to coordinator (no-op until behavior PR sets watermarks)
- [ ] `[L]` `cargo check --features no-cuda` — non-CUDA bits (the policy crate, the trait, channel types) compile
- [ ] `[L]` `cargo check --features metal` — Metal build untouched

#### Structural PR (a) — Remote GPU
- [ ] `[R]` `cargo build --release` — `cudarc` + `cudaHostAlloc` + dedicated copy stream all link
- [ ] `[R]` Unit smoke: register a host pinned region, do one async D2H copy on `write_stream`, do one H2D on `load_stream`, verify checksums
- [ ] `[R]` `cargo test --release` — full suite still green (no behavior change yet)

#### Behavior PR (b) — `[L+R]`
- [ ] `[L+R]` Edit `infer/src/scheduler/cuda/runtime.rs` — eviction hook at admission (`evict_if_needed`) and post-decode (`stamp_keepalive`)
- [ ] `[L+R]` Edit `infer/src/scheduler/cuda/core.rs` — pass watermark thresholds (high=0.90, low=0.75 per §3.2) into `TieredKvCache`
- [ ] `[L+R]` **Diff before delete** — confirm `grep -r 'offload_if_needed\|ensure_on_gpu\|k_host\|v_host' infer/src/` returns ONLY `infer/src/model/kv_cache.rs` (21 internal hits) + `infer/src/model/generation_state.rs` (1 hit, audit + strip) + `infer/src/ops/tests.rs` (6 unit-test local-variable name collisions, keep). Confirmed by P2 research grep.
- [ ] `[L+R]` **Delete** `infer/src/model/kv_cache.rs:130-168` — `k_host`, `v_host`, `ensure_on_gpu`, `offload_if_needed`, `OFFLOAD_BLOCK_SIZE = 64`, `gpu_has_full_seq`, `offloaded_len`, `max_gpu_seq_len`
- [ ] `[L+R]` Delete the matching mirror in `tests/test_kv_cache.py:135,138,262,367,373,382,383,394`
- [ ] `[L]` `cargo check --features no-cuda` and `--features metal` still green
- [ ] `[R]` `cargo build --release`, full e2e suite, `greedy_consistency`
- [ ] `[R]` Long-context bench (32k+ cumulative tokens, num_slots=4) that OOMs on main now runs to completion
- [ ] `[R]` `scripts/bench_throughput_sweep.py --label tier-T2`; ≤3% steady-state regression vs P1 baseline
- [ ] `[R]` Bench markdown in `docs/experience/wins/`

### 3.2 Industry references

**Coordinator ownership model** (the most opinionated section, because the design changed):
- **SGLang `HiCacheController`** ([`cache_controller.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/cache_controller.py)) — **3 OS daemon threads**: `prefetch_thread` (~L520), `backup_thread` (~L521), `prefetch_io_aux_thread` (~L519). Communication: plain Python lists + `queue.Queue` with 1-second `get(timeout=1.0)` loops. **No asyncio, no event loop.** The scheduler polls `ack_load_queue`/`ack_write_queue` (~L219) each step.
- **Two dedicated CUDA streams** at init (~L223):
  ```python
  self.write_stream = device_module.Stream()   # D2H
  self.load_stream  = device_module.Stream()   # H2D
  ```
  Explicitly separate from the compute stream → hardware copy-engine overlap.
- **Layer-granular `LayerLoadingEvent`** (~L48) + `LayerDoneCounter` (~L65) + `HiCacheAck` (~L104) — per-layer producer/consumer overlap. Copy stream records a `cudaEvent` per layer; compute stream `wait_event(ev)` before touching that layer. **Forward never blocks on whole-request transfer.**
- **SGLang host pool** ([`memory_pool_host.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool_host.py)) — pinned via `torch.cuda.cudart().cudaHostRegister()` (~L95) (PyTorch idiom). Sized by `--hicache-ratio` × device pool or explicit `--hicache-size` GB; reserves 10 GB system RAM as floor (~L162).
- **LMCache `StorageManager`** ([source](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/storage_backend/storage_manager.py)) — asyncio loop on a background thread named `"storage-manger-event-loop"` (~L209-216). **Backpressure via `WeightedSemaphore`** (~L118-148): `_concurrent_budget_cap = chunk_budget // 2`. `LMCacheEngine.store_layer/retrieve_layer` does **3-stage layerwise pipeline**: allocate → D2H layer i → put layer i-1 (~L585-687, ~L832-980).
- **No prior Rust LLM serving project has offload.** mistral.rs and candle-vllm both lack any host tier. **agent-infer would be the reference Rust impl.**

**Watermarks**:
- **SGLang has NO explicit high-water %.** Eviction is on-demand inside `evict()` when `evictable_size_` exceeds available memory — "triggered by allocator failure, not by threshold". `load_back_threshold = 10 tokens` ([`hiradix_cache.py:186`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/hiradix_cache.py#L186)) is hysteresis-like: don't promote prefixes shorter than 10 tokens — cheaper to recompute.
- **LMCache `WeightedSemaphore`** with `chunk_budget // 2` cap — in-flight backpressure, not a watermark.
- **vLLM `gpu_memory_utilization`** default 0.9 ([cache config](https://docs.vllm.ai/en/stable/api/vllm/config/cache/)). KV preemption kicks in when block pool exhausted. No hysteresis.
- **Mooncake**: batch capped at 128 pages empirically — bandwidth-derived.
- **Recommendation for agent-infer**: `high_water = 0.90`, `low_water = 0.75`, evict in chunks of `max(64, 0.05 * pool)` tokens. Conservative; first bench-sweep tunes. Add `min_promote_tokens = 16` matching SGLang's `load_back_threshold` spirit.

**Write policies** (SGLang's three from [`hiradix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/hiradix_cache.py)):
- `write_through` — every cache insertion copies to host immediately. Strongest hit rate; highest bandwidth.
- `write_through_selective` — only when `node.hit_count >= write_through_threshold` (~L195 default 2). **The blog's recommended default** when bandwidth-constrained.
- `write_back` — only on GPU eviction. Protects bandwidth at the cost of losing un-backed-up data on crash.

**Cancel-safe Rust pipeline** — references:
- [Tokio cancel-safety RFD 400](https://rfd.shared.oxide.computer/rfd/400) — production patterns
- [Cybernetist — Rust tokio task cancellation patterns](https://cybernetist.com/2024/04/19/rust-tokio-task-cancellation-patterns/) — `select! { biased; ... }` shape

**Existing scheduler-policy shape** ([`infer/src/scheduler/policy.rs:64-78,133-152`](file:///path/to/code/agent-infer/infer/src/scheduler/policy.rs)):
```rust
pub trait AdmissionPolicy: Send + Sync {
    fn allow(&self, signals: SchedulerSignals) -> bool;
}
pub trait ChunkingPolicy: Send + Sync {
    fn next_chunk_size(&self, mode: InferenceMode, signals: SchedulerSignals) -> usize;
}
```
Both pure scoring functions: `&self`, snapshot in → decision out. No mutable state, no IO, no allocation in hot path.

### 3.3 Course corrections from research

1. **Use OS thread + crossbeam, NOT tokio.** Major change from the project doc which said "tokio task". Three reasons:
   - `cudarc::driver::CudaStream` is `!Send` until wrapped in `Arc`; tokio's `spawn` requires `Send + 'static`, forcing pointer dance OS threads avoid.
   - the relevant `infer` modules currently pull **zero runtime dependency**;
     the scheduler runs on a raw `std::thread`. Staying off tokio keeps the
     layering clean.
   - Cancel-safety becomes trivial: every `await` becomes `recv_timeout(1ms)`. No `select!` foot-guns.
2. **Two dedicated streams, not one.** `write_stream` (D2H) + `load_stream` (H2D). The project doc's "one dedicated copy stream" was incomplete.
3. **Layer-granular `cudaEvent` sync, not whole-request.** Record event on copy stream per layer; attention kernel waits on `compute_stream.wait_event(ev)` before reading that layer. Matches SGLang `LayerLoadingEvent`.
4. **Default policy: `write_back`** (matches our current dormant behavior) with `write_through_selective` (`hit_count >= 2`) as opt-in toggle.
5. **`EvictionPolicy` trait shape — pure scoring** (matches existing `AdmissionPolicy` / `ChunkingPolicy` siblings). Final shape:
   ```rust
   #[derive(Debug, Clone, Copy)]
   pub struct EvictionCandidate {
       pub slot: u32,
       pub tokens: u32,
       pub last_access_step: u64,
       pub hit_count: u32,
       pub prefix_depth: u32,    // radix-tree depth
       pub is_sealed: bool,      // request still producing tokens?
   }
   pub trait EvictionPolicy: Send + Sync {
       /// Lower score → evict first. f32::INFINITY = pinned.
       fn score(&self, candidate: EvictionCandidate, signals: SchedulerSignals) -> f32;
   }
   ```
   Default impls: `LruEviction`, `ReuseBiasedLru`, `HitCountLru`, `SessionBiasedLru`. The coordinator owns the candidate set, calls `policy.score()` on each, sorts, evicts down to `low_water`.

---

## 4 · Milestone M4 (formerly P3) — T2 disk tier + session save/load + Metal first contact

### 4.1 Lane breakdown

P3 has the most local content of any phase — **all of P3 is `[L]` except the linux io_uring path**.

#### Disk transport — Local (Mac)
- [ ] `[L]` Add `infer/src/kv_tier/transport/disk.rs`
  - **Default = `tokio::fs` on all platforms** (research correction; see §4.3)
  - `cfg(target_os = "linux")` + `cfg(feature = "disk-io-uring")` → optional io_uring path via `compio` or `monoio`, **off by default**
  - macOS: `fcntl(F_NOCACHE)` for cache bypass (NOT `O_DIRECT` which doesn't exist on Darwin)
- [ ] `[L]` Allocation-stable region: one large pre-extended file per node, indexed by `(file_id, offset)`. Filename = blake3 of `(model_id, tokenizer_id, token_prefix)` (LMCache pattern)
- [ ] `[L]` Wire format (per §4.2): postcard-encoded versioned header + raw bf16/f16 trailer matching `MetalKVPool` row stride
- [ ] `[L]` Unit tests fully local: write block, read back, hash match, version round-trip
- [ ] `[L]` Add `postcard = "1"` and `blake3 = "1.5"` to `infer/Cargo.toml`
- [ ] `[L]` **Move `memmap2` out of cuda feature gate** so disk tier can mmap on Mac too

#### HTTP session routes — Local (Mac)
- [ ] `[L]` Add `infer/src/http_server/sessions.rs` — `POST /v1/sessions/:id/save`, `POST /v1/sessions/:id/load`, `GET /v1/sessions/:id/manifest`, `DELETE /v1/sessions/:id` handlers
- [ ] `[L]` Edit `infer/src/http_server.rs:422-427` — register routes alongside `/v1/completions`
- [ ] `[L]` Idempotency-Key header support; ETag = content_hash for skip-re-upload
- [ ] `[L]` `crates/infer-agent/src/lib.rs:166-188` — extend `save_to_path / load_from_path` with optional `Option<KvBlobRef>` on `SessionSnapshot` pointing at content hash; existing JSON-only path remains
- [ ] `[L]` `cargo test -p infer http_server::sessions` — green on Mac

#### MLX wired memory bindings — Local (Mac)
- [ ] `[L]` Edit `infer/mlx-sys/src/mlx_bridge.cpp` — add 6 new C bridges (see §4.2)
- [ ] `[L]` Edit `infer/mlx-sys/src/lib.rs:468` — extern declarations for the 6 new functions
- [ ] `[L]` Edit `infer/src/metal_kv_pool.rs:223-262` — read `mlx_metal_device_max_recommended_working_set_size()` at init, cap `max_total_tokens` per §4.2 formula
- [ ] `[L]` Edit `infer/src/metal_prefix_cache.rs` — disk tier hook through `TieredKvCache` façade
- [ ] `[L]` `cargo build --release --no-default-features --features metal` — Metal build green
- [ ] `[L]` `cargo test --release --no-default-features --features metal` — Metal tests green
- [ ] `[L]` Long-context Metal smoke test on Mac: 16k+ token session with bounded pool — must NOT panic (mlx-lm #883 mitigation gate)

#### Remote GPU
- [ ] `[R]` CUDA build with disk transport on
- [ ] `[R]` (Optional) build with `--features disk-io-uring` to verify the linux io_uring path
- [ ] `[R]` Restart smoke test: save 30k-token system prompt session, kill process, restart, reload, TTFT recovery within 20% of pre-restart warm baseline
- [ ] `[R]` Bench markdown

#### P3 cleanup gate
- [ ] `[L]` `grep -r session_store infer/` should reference `kv_tier` paths only (the proposed standalone `session_store.rs` never lands)

### 4.2 Industry references

**Disk tier file format**:
- **LMCache** ([`local_disk_backend.py`](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/storage_backend/local_disk_backend.py)) — per-key `.pt` files in flat directory, filename = `key.replace("/", "-") + ".pt"`. **In-memory index only** (`self.dict`); **rebuilt on startup by scanning the directory**. O_DIRECT path exists for aligned block I/O. **No LMDB/RocksDB.**
- **SGLang HiCache `HiCacheFile`** ([`hicache_storage.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/hicache_storage.py)) — per-page `.bin` files, key = `"{key}_{model_name}_{tp_rank}_{tp_size}_{pp_rank}_{pp_size}.bin"`. **No persistent index** — state rebuilt on startup via `os.scandir() + batch_exists_v2`. Default `/tmp/hicache`.
- **Mooncake**: disk variant not shipped as a reference impl. LMCache reaches Mooncake via generic remote backend.
- **vLLM `--swap-space`**: pinned host RAM for preempted sequences, **NOT a disk tier**. vLLM tracks "load/save KV from disk" in [issue #10611](https://github.com/vllm-project/vllm/issues/10611) as **unshipped**. Production vLLM disk-tier delegated to LMCache via KV connector.

**io_uring vs tokio::fs**:
- **`tokio-uring` is effectively dead** — no releases since 2022, stale issues, still has `epoll → io_uring` indirection. [Source](https://users.rust-lang.org/t/status-of-tokio-uring/114481).
- Active alternatives: **`compio`** (broadest coverage, actively maintained) or **`monoio`** (fastest, redacted production).
- **Production Rust LLM serving uses NEITHER.** mistral.rs, candle-serve don't touch io_uring. TGI is Python. Nobody bothers because the latency win at 1 MB block reads is <10% (NVMe read dominates).
- **macOS has no io_uring.** Use `tokio::fs` + `fcntl(F_RDADVISE)` / `fcntl(F_NOCACHE)`.
- **O_DIRECT pitfalls**: requires 512 B / 4 KB alignment on buffer, offset, and length (else `EINVAL`); bypasses page cache. Doesn't exist on Darwin.

**Session save/load HTTP semantics — no prior art**:
- OpenAI, Anthropic, Google: none expose a "save KV state" endpoint. Anthropic's Files API + Memory tool operate on **message content**, not attention state.
- LangChain memory: persists **messages** (`BaseChatMessageHistory`), never KV.
- Closest: Anthropic prompt caching, Gemini "cached content" — **server-side only**, client cannot push/pull blobs.
- **Recommendation**: define our own. Idempotent + content-addressing + ETag pattern from blob storage best practice.

**MLX wired memory C++ API** (`mlx::core::metal::MetalAllocator` in [`mlx/backend/metal/allocator.h`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.h)):
```cpp
namespace mlx::core::metal {
  class MetalAllocator : public allocator::Allocator {
    size_t get_active_memory();
    size_t get_peak_memory();
    size_t get_cache_memory();
    size_t set_cache_limit(size_t limit);
    size_t set_memory_limit(size_t limit);
    size_t set_wired_limit(size_t limit);   // macOS 15.0+ only
    void   clear_cache();
  };
}
```
- **`set_wired_limit` semantics**: macOS 15.0+ only (no-op on older); default `0` (no guarantee); mlx-lm sets to `max_recommended_working_set_size` ≈ 75 % of RAM at boot; returns previous limit.
- **`set_wired_limit` failure modes**: exceeds system hard limit → error; raising system ceiling needs `sysctl` + admin.
- **mlx-lm #883 failure mode**: mlx_lm.server boots with wired_limit ≈ 75 % RAM. Wired memory **cannot be swapped**. KV cache grows unboundedly because mlx_lm.server **does not honor `--max-kv-size`** (issue #615). When KV + weights crosses wired ceiling, `IOGPUMemory.cpp` hits "prepare count underflow" → **kernel panic** (not process crash).
- **agent-infer's existing `mlx-sys` binding** ([`infer/mlx-sys/src/lib.rs:468`](file:///path/to/code/agent-infer/infer/mlx-sys/src/lib.rs#L468)): only `mlx_metal_clear_cache()`. Need to add 6 new C bridges.

**Bounded MLX KV pool formula**:
```
bytes_per_token  = num_layers * 2 * num_kv_heads * head_dim * sizeof(dtype)   // K + V
wired_budget     = mlx_metal_device_max_recommended_working_set_size()
wired_for_kv     = wired_budget * 0.60   // 40 % for weights + activations + MLX cache
max_total_tokens = wired_for_kv / bytes_per_token
```
**Use `max_recommended_working_set_size`, NOT `set_wired_limit`'s return value** (which is `0` until we call it). Call `mlx_metal_set_wired_limit(wired_budget)` at startup, log both, emit a metric when `available_tokens()` < 10 %.

**Wire format proposal** — postcard-encoded header + raw KV trailer:
```
[u32 magic "AISV"][u16 version=1][u16 flags]
[u64 model_uid_hash][u32 tokenizer_uid_hash]
[u32 n_layers][u32 n_kv_heads][u32 head_dim][u16 dtype]
[u32 n_tokens][u32 token_ids...][u8 kv_bytes...]
```
Filename = blake3 of `(model_uid, tokenizer_uid, token_ids)`.

### 4.3 Course corrections from research

1. **Default to `tokio::fs` everywhere; defer io_uring to a feature flag.** Project doc said "io_uring on Linux + tokio::fs fallback"; research says nobody in production Rust LLM serving uses io_uring, and tokio-uring is dead. Save the dependency.
2. **macOS uses `F_NOCACHE` not `O_DIRECT`** — Darwin has no `O_DIRECT`. Add explicit `cfg(target_os = "macos")` branch.
3. **No persistent disk index in v1.** Both LMCache and SGLang rebuild via directory scan on startup. Add a `manifest.bin` (postcard) only when scan time > 100 ms.
4. **6 new mlx-sys bindings, not 3.** Project doc only mentioned `set_wired_limit + get_active_memory`. Add: `set_memory_limit`, `set_cache_limit`, `get_peak_memory`, `get_cache_memory`, `device_max_recommended_working_set_size`. The last is the critical one — it's the source for our pool sizing formula because `set_wired_limit` returns `0` until called.
5. **Cargo deps to add**: `postcard`, `blake3`. **And move `memmap2` out of the cuda feature gate** — the disk tier can mmap on Mac too. Currently `infer/Cargo.toml:50` has `cuda = ["dep:cudarc", "dep:memmap2"]`.
6. **Session save/load wire format**: no prior art to copy. We define our own. Use postcard (already in serde ecosystem; we have `serde = "1.0"`).

---

## 5 · Post-M4 experiment (formerly P4) — KVFlow-lite reuse-distance + cache-aware routing

### 5.1 Lane breakdown

#### Local (Mac)
- [ ] `[L]` `infer/src/scheduler/policy.rs` — `ReuseDistancePolicy` impl of `EvictionPolicy` (the trait shape from P2 accommodates this without modification; see §5.3)
- [ ] `[L]` `SessionArrivalTracker { rings: HashMap<SessionId, ArrivalRing>, alpha: f32 }` — internal state, RwLock or Mutex
- [ ] `[L]` Unit tests with synthetic turn histories: predicted next-access-time, eviction ordering, cold-fallback to LRU
- [ ] `[L]` `infer/src/kv_tier/directory.rs` — per-session ring buffer integration (mirror in pure-Rust under non-cuda mod for local check)
- [ ] `[L+R]` Edit `infer/src/scheduler/cuda/runtime.rs:135-152` — replace `best_prefix_slot` linear scan with Mooncake Algorithm 1: `cost(slot) = T_prefill_est + slot.residual_decode_cost + spill_penalty(slot)`
- [ ] `[L+R]` Edit `infer/src/scheduler/cuda/core.rs` — plumb `session_id` and `now` into slot selection (already partially threaded per commit `82a19b1`)
- [ ] `[L]` `cargo check` both `--features no-cuda` and `--features metal`
- [ ] `[L]` Pure-Rust test in `infer/src/scheduler/policy.rs`: cross-session hit rate ≥85% on synthetic interleaved-session trace

#### Remote GPU
- [ ] `[R]` Cross-session bench, 2-session alternating: prefix hit rate ≥85% (vs ≥70% in P1)
- [ ] `[R]` `scripts/bench_throughput_sweep.py --label p4-reuse-dist` on 4-agent interleaved workload
- [ ] `[R]` Target: ≥1.5× TTFT improvement vs P1's `SessionBiasedLru` (scaled-down from KVFlow paper's 1.83× because we don't have the CPU-tier prefetch leg)
- [ ] `[R]` If reproduction fails: keep `SessionBiasedLru` as default, ship `ReuseDistancePolicy` behind a flag

### 5.2 Industry references

**KVFlow paper deep-dive** ([arXiv 2507.07400](https://arxiv.org/abs/2507.07400), [PDF](https://arxiv.org/pdf/2507.07400)):
- **Step graph** (§3.1): **user-provided static topology, not learned.** "Each `sgl.function` corresponds to an independent agent"; JIT substitution embeds workflow metadata into HTTP requests. The frontend is the source of truth.
- **ETA computation** (Figure 3a):
  - AND-join (sync barrier): `eta(n) = max(eta(pred_i)) + 1`
  - OR-join (conditional branch): `eta(n) = min(eta(pred_i)) + 1`
  - Currently executing agent: `eta = 0`
- **Eviction order** (§3.1 "Workflow-Aware Eviction Priority Assignment"): priority map over a radix tree. ETA attached to last KV node of each agent's fixed prompt; propagated upward. Internal node priority = **min ETA across children**. Varying suffixes get `+∞` (evict first). Evict suffixes first, then prefix nodes in **descending ETA order**.
- **Prefetch trigger** (§3.2): proactive, exploits PCIe full-duplex. While agent A_i forward-passes (GPU-bound), background CPU→GPU thread loads agents with smallest upcoming ETA (`eta=1`). State machine `{in_gpu, cpu_backup, loading, offloading}`.
- **Numbers**: 1.83× single-workflow large-prompt (10-agent sequential, Llama-3.1-8B/A10G, Qwen2.5-32B/H100); 2.19× high-concurrency (512/20-task, 1024/10-task; Figure 6).
- **Open source**: NO public code release at paper submission. We're reverse-engineering from the paper.

**Mooncake cache-aware routing** ([FAST'25 paper](https://www.usenix.org/system/files/fast25-qin.pdf)):
- Conductor (global scheduler) indexes `{prefix_hash_chain → prefill_instance_set}` per-instance.
- **Algorithm 1** (§5.1) verbatim:
  1. `block_keys = PrefixHash(R.prompt_tokens, B)`
  2. `best_prefix_len, best_matched_instance = FindBestPrefixMatch(P, block_keys)`
  3. For each instance, candidate TTFT:
     - If `best_prefix_len / instance.prefix_len < kvcache_balancing_threshold`: stay local; `TTFT = T_queue + T_prefill(len, instance.prefix_len)`
     - Else: transfer from `best_matched_instance`; `TTFT = T_transfer + T_queue + T_prefill(len, best_prefix_len)`
  4. Pick minimum TTFT. Reject if `TTFT > TTFT_SLO`.
  5. Migrate hot blocks via `TransferKVCache` when ratio trips threshold.
- **Cost model**: pure TTFT minimization, three additive components (transfer, queue, compute). `T_prefill` from offline-fit predictive model keyed by `(len, prefix_len)`. NOT a hand-tuned `α·cache + β·load` blend.

**llm-d KV indexer** ([llm-d-kv-cache-manager](https://github.com/llm-d/llm-d-kv-cache-manager)):
- Signals: ZMQ PUB events from each vLLM pod on topic `kv@{podIP}@{model}`: `BlockStored`, `BlockRemoved`, `AllBlocksCleared`. Msgpack-encoded.
- Index: `kvblock.Index: {ModelName, ChunkHash} → []PodEntry{PodIdentifier, DeviceTier}`. Chunk hash = chained SHA-256 over 16-token blocks (mirrors vLLM APC).
- **Routing winner**: cache affinity wins decisively. Score = consecutive matching blocks from position 0; pick highest-score pod. **Red Hat report: 99.92% traffic concentration onto warm pod.** Load balancing delegated to external scheduler.
- Stricter than Mooncake: cache-first, load-balance second. Justified because "prefix warmth is multiplicative on TTFT but load imbalance is additive on queue."

**Per-engine multi-slot routing context**:
- vLLM v1: NO per-slot affinity. Single global radix `BlockPool`; blocks **shared across requests** via refcounting. Our `--num-slots` distinct slot arenas don't have an analog there.
- SGLang: same single-radix model. KVFlow extends SGLang's radix with workflow priorities, still one shared tree.
- agent-infer's `--num-slots` = **distributed-mini-Mooncake**. Apply Algorithm 1 scaled down at slot granularity.

### 5.3 Course corrections from research

1. **KVFlow needs a DAG we don't have.** Replace with **per-session EWMA inter-arrival** as a proxy: `Δ_ewma(s) = α·(now − last) + (1−α)·Δ_prev`, predicted `next(s) = last(s) + Δ_ewma(s)`. Block score = `next(session_owner_of_block)`. Suffix nodes (no session) get `+∞`. Cold sessions (`ring.len() < 2`) fall back to LRU. **No new infrastructure beyond `turn_depth` already plumbed**.
2. **Use Mooncake Algorithm 1 for slot selection.** Replace `best_prefix_slot`'s linear scan with `cost(slot) = T_prefill_est(len, prefix_hit_blocks[slot]) + slot.residual_decode_steps × T_step_est + spill_penalty(slot)`. With 4–16 slots, O(num_slots) is fine.
3. **`EvictionPolicy` trait shape stays the same as P2.** Verified: `ReuseDistancePolicy` reads `expected_next_access` from internal state (the ring buffer it owns), not from `EvictionCandidate`. No new fields needed. Score = `next(s) − now` in millis (saturating to `u64::MAX` for cold/suffix nodes).
4. **llm-d-style affinity bias is justified** — prefix warmth is multiplicative on TTFT, load imbalance is additive. We weight cache-hit much more heavily than slot occupancy.

---

## 6 · Milestone M5 (formerly P5) — KVTransport trait freeze + NixlTransport stub

### 6.1 Lane breakdown

#### Local (Mac) — entirely `[L]`
- [ ] `[L]` Add `infer/src/kv_tier/transport/nixl.rs` — `NixlTransport` skeleton; `register / deregister` fully implemented (just calls into `nixl-sys::Agent::register_memory`); `put_batch / get_batch / poll / abort` return `todo!("P6")`
- [ ] `[L]` Edit `infer/Cargo.toml` — add `nixl-sys = { version = "1.0", optional = true, features = ["stub-api"] }`; add features `rdma-nixl = ["dep:nixl-sys"]` (default = stub-api active) and `rdma-nixl-real = ["dep:nixl-sys"]` (no stub-api, links real libnixl.so on remote CUDA box)
- [ ] `[L]` `infer/src/events.rs` — add new variant of `RequestEventKind` for `TierTransition { from: Tier, to: Tier, bytes: u64, micros: u64 }`. Keep `EventSink::emit` signature stable
- [ ] `[L]` Verify trait surface has not changed since P2 — if it has, go back and fix P2 instead of forking
- [ ] `[L]` `cargo check --no-default-features --features no-cuda` — default still compiles (no nixl-sys dep)
- [ ] `[L]` `cargo check --no-default-features --features no-cuda,rdma-nixl` — stub-api path compiles on Mac WITHOUT `libnixl.so`

#### Remote (optional manual smoke; not a CI gate)
- [ ] `[R]` On a CUDA + NIXL native lib box: `cargo check --features cuda,rdma-nixl-real` with `NIXL_PREFIX=/opt/nvidia/nvda_nixl`

### 6.2 Industry references

**nixl-sys public API** (v1.0 stable, [docs.rs](https://docs.rs/nixl-sys/latest/nixl_sys/)):

```rust
// Agent — the central handle
pub struct Agent { /* opaque */ }
impl Agent {
    pub fn new(name: &str) -> Result<Self, NixlError>;
    pub fn new_configured(name: &str, cfg: &AgentConfig) -> Result<Self, NixlError>;
    pub fn create_backend(&self, plugin: &str, params: &Params) -> Result<Backend, NixlError>;
    pub fn register_memory(&self, descriptor: &impl NixlDescriptor, opt_args: Option<&OptArgs>)
        -> Result<RegistrationHandle, NixlError>;  // RegistrationHandle is drop-guarded
    pub fn get_local_md(&self) -> Result<Vec<u8>, NixlError>;
    pub fn load_remote_md(&self, metadata: &[u8]) -> Result<String, NixlError>;
    pub fn create_xfer_req(&self, op: XferOp, local: &XferDescList,
                           remote: &XferDescList, remote_agent: &str,
                           opt_args: Option<&OptArgs>) -> Result<XferRequest, NixlError>;
    pub fn post_xfer_req(&self, req: &XferRequest, opt_args: Option<&OptArgs>) -> Result<bool, NixlError>;
    pub fn get_xfer_status(&self, req: &XferRequest) -> Result<XferStatus, NixlError>;
    pub fn get_notifications(&self, notifs: &mut NotificationMap, opt_args: Option<&OptArgs>) -> Result<(), NixlError>;
    pub fn invalidate_local_md(&self) -> Result<(), NixlError>;
    pub fn invalidate_remote_md(&self, agent_name: &str) -> Result<(), NixlError>;
}

// Enums
pub enum XferOp     { Read, Write }
pub enum XferStatus { Success, InProgress }
pub enum MemType    { Dram, Vram, Block, Object, File, Unknown }
```

**Critical shape facts**:
- **`XferRequest` has NO async methods.** Only `get_telemetry()`. Completion is **always polled** via `Agent::get_xfer_status(&req) -> Result<XferStatus, NixlError>`.
- **Notifications also polled**: `Agent::get_notifications(&mut NotificationMap)`. **No internal reactor.**
- **`XferOp = {Read, Write}`** — symmetric; get/put collapse to `Read`/`Write`.
- **`stub-api` feature**: empty flag (`stub-api = []` in `nixl-sys`'s Cargo.toml). When enabled, build skips linking `libnixl.so`, prints `"Building with stub API - NIXL functions will be resolved at runtime via dlopen"`. Without `stub-api`, `NIXL_NO_STUBS_FALLBACK=0` (default) → missing `libnixl.so` falls back to stubs with warning; `=1` → build panics.
- **Env vars consumed by `nixl-sys`'s `build.rs`**: `NIXL_PREFIX` (default `/opt/nvidia/nvda_nixl`), `HAVE_ETCD`, `NIXL_NO_STUBS_FALLBACK`, `CPLUS_INCLUDE_PATH`.

**NIXL backend dispatch** ([BackendGuide.md](https://github.com/ai-dynamo/nixl/blob/main/docs/BackendGuide.md)):
- Backends: **UCX** (full local + remote + notifications), **GDS** (local-only, disk → VRAM), **POSIX**, **OBJ**, **Mooncake**.
- Selection is **programmatic, not env/config**. Call `Agent::create_backend("UCX", &params)` with `params: Params` key-value map.
- Memory type tags: `DRAM` (host), `VRAM` (GPU, by GPU id), `BLK` (block device by volume id), `FILE` (fd + offset), `OBJ` (key + offset).

**UCX prerequisites**:
- NIXL README pins **UCX 1.20.x**.
- **CUDA ≥12** (PyPI wheels publish `nixl[cu12]` and `nixl[cu13]`).
- **`nvidia-peermem`** ([NVIDIA docs](https://download.nvidia.com/XFree86/Linux-x86_64/525.78.01/README/nvidia-peermem.html)): ships with NVIDIA driver, replaces deprecated `nv_peer_mem`. **Not autoloaded** — must `modprobe nvidia-peermem` (or systemd unit). Required for GPUDirect RDMA via UCX `cuda_ipc`/`rdma` transports.
- **Hugepages**: not required by NIXL — recommended by UCX for faster MR registration but optional.

**Mooncake TransferEngine** ([`transfer_engine.h`](https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-transfer-engine/include/transfer_engine.h)):
```cpp
int registerLocalMemory(void* addr, size_t length, const std::string& location, ...);
int unregisterLocalMemory(void* addr, ...);
SegmentHandle openSegment(const std::string& segment_name);
BatchID allocateBatchID(size_t batch_size);
Status submitTransfer(BatchID batch_id, const std::vector<TransferRequest>& entries);
Status getTransferStatus(BatchID batch_id, size_t task_id, TransferStatus& status);
int getNotifies(std::vector<NotifyDesc>& notifies);
```
- **Same shape as NIXL**: register MR, build descriptor list, submit batch, poll/notify completion. **No native Future.** A single trait covers both.
- Rust binding directory exists at `mooncake-transfer-engine/rust/` but actual `lib.rs` returned 404 on direct fetch. **Defer Mooncake to P6.** Bind NIXL only in P5; Mooncake reaches us through NIXL's `"Mooncake"` plugin.

**`metal_gdr.rs` reality check** ([`infer/src/metal_gdr.rs:1-20`](file:///path/to/code/agent-infer/infer/src/metal_gdr.rs)): module doc reads "Implements the Qwen3.5 linear attention decode step using MLX high-level ops". **Confirmed: Gated Delta Rule, NOT GPUDirect RDMA.** Filename is misleading; rename to `metal_qwen35_gdr.rs` is a separate cleanup PR (NOT part of P5).

**Register-once requirement**:
- All three stacks (UCX, NIXL, Mooncake) require **allocation-stable MRs**. Hardware constraint (IOMMU page-table pinning + HCA memory key caches), not library policy.
- T0 GPU pool and T2 host pool must be allocated up-front to maximum size and pinned for process lifetime.
- Growing → unregister-then-reregister round-trip + peer metadata refresh. Acceptable for rare rebalance, never on hot path.

### 6.3 Course corrections from research

1. **MAJOR: trait shape changes from `type Completion: Future` to `type Op: Send` + explicit `poll`.** NIXL has no native Future; everything is polling. Cancel-on-drop is unsound for in-flight RDMA — buffer must stay pinned until the op actually completes. **Final trait** (this is the version that lands in P2, gets frozen in P5):
   ```rust
   pub trait KVTransport: Send + Sync {
       type Region: Send + Sync;          // drop-guarded MR handle
       type Op: Send;                     // per-request handle, NOT a Future

       fn register(&self, ptr: *mut u8, len: usize, kind: MemKind)
           -> Result<Self::Region, TransportError>;

       fn invalidate_region(&self, region: &Self::Region)
           -> Result<(), TransportError> { Ok(()) }   // default no-op

       fn put_batch(&self, ops: &[TransferOp]) -> Result<Self::Op, TransportError>;
       fn get_batch(&self, ops: &[TransferOp]) -> Result<Self::Op, TransportError>;

       /// Non-blocking poll. Returns Pending while in-flight, Ready(Ok) on success.
       fn poll(&self, op: &mut Self::Op) -> Poll<Result<(), TransportError>>;

       /// Best-effort abort; buffer must stay pinned until poll returns Ready.
       fn abort(&self, op: &mut Self::Op);
   }

   pub enum MemKind { Host, Vram { device: u32 }, Block { volume: u32 } }
   ```
   `MemKind` maps 1:1 with NIXL's `MemType` for future-compat with GDS without enlarging the surface. A `Future` adapter (`TransportFuture<T>`) lives in `infer`, NOT in the trait — keeps the trait `no_std`-friendly.
2. **`BlockLocation::Remote` opaque + tag** (final shape):
   ```rust
   pub struct RemoteBlockDesc {
       pub transport: TransportId,        // u8 tag: Nixl | Mooncake | …
       pub payload: SmallVec<[u8; 32]>,   // opaque, serialized by transport impl
   }
   ```
   - NIXL payload = bincode of `(remote_agent_name, addr, len, mem_type, dev_id)` (~24-32 B for short names)
   - Mooncake payload = `(SegmentHandle u64, offset u64, length u64)` = 24 B
   - Cross-backend code never parses it.
3. **Two cargo features for nixl-sys, not one.** `rdma-nixl = ["dep:nixl-sys"]` with stub-api active by default (Mac CI gate). `rdma-nixl-real = ["dep:nixl-sys"]` for production CUDA boxes that link real `libnixl.so`. Without this split, Mac CI can't validate the trait surface.
4. **Defer Mooncake to P6.** Rust binding is unverified; Mooncake reaches us via NIXL's `"Mooncake"` plugin in the meantime.
5. **`nvidia-peermem` is NOT autoloaded.** Document as a deployment requirement in P6 docs; P5 doesn't run it.

---

## 7 · Parallel-GPU work pool (independent of P0–P5)

Work the remote GPU host can run **while** the local Mac is busy. None of this is on the tiered-kv critical path; it keeps the GPU productive and produces data we will need anyway.

### 7.1 — High value, do BEFORE P0

These should already be running by the time P0 edits land:

- [ ] `[R]` **`--label page1` baseline**: explicit pre-P0 snapshot under current `page_size=1` regime. **Required** for P0 delta comparison.
- [ ] `[R]` **Baseline collection**: `scripts/bench_throughput_sweep.py --label baseline-main-2026-04-13` — every model + slot config we ship. Becomes regression gate from P0 onwards. Save to `docs/experience/wins/2026-04-13-bench-baseline.md`.
- [ ] `[R]` **Long-context agent baseline**: 32k+ token agent trace, num_slots=4, current main. Numbers we compare against in P2's "must run to completion" gate.
- [ ] `[R]` **Greedy regression sample**: full `e2e + e2e_qwen35 + greedy_consistency` on current main, capture pass/fail counts as post-merge baseline.
- [x] `[L+R]` **`scripts/bench_agent_trace.py`** (item C6 from `agent-first-architecture.md`, renamed from `bench_agent.py` to avoid collision with the existing binary-subprocess benchmark of that name): multi-turn tool-calling replayer + input trace under `scripts/data/agent_trace_default.jsonl`. Mostly local Python; GPU validation only. **P1 needs this as a scoreboard.** Landed 2026-04-13.

### 7.2 — Medium value, parallel tracks

- [ ] `[L+R]` **A3 constrained decoding prototype** (independent of KV tier work): xgrammar-style JSON-schema FSM compiler in `infer/src/constrained/`, sampling-time logit mask in `infer/src/ops/sampling.rs`. Local dev → GPU validation.
- [ ] `[R]` **A4 speculative decoding scaffolding** (depends on P1; can stage standalone draft model load and CPU-math verify until P1 ships).
- [ ] `[R]` **Active KV quantization track** — if `docs/projects/kv-quantization-long-context.md` is still active, format-comparison benches keep running on the GPU.

### 7.3 — Background (continuous)

- [ ] `[R]` **Greedy regression sweep on every main commit**: `cargo test --release --test e2e e2e_qwen35` against a known input set; log timing.
- [ ] `[R]` **nsys profile collection** for current decode hot path. Useful as before-snapshot when P0 lands.
- [ ] `[L]` **`docs/experience/wins/` benchmark hygiene** — anything missing the environment table per `CLAUDE.md` benchmark rules should be backfilled.

### 7.4 — Lane discipline for parallel work

- Parallel-GPU tasks **never block the critical path**. If a parallel task starts to need the same files as the active phase, pause it, do not race it.
- Every parallel run produces a recorded artifact (test pass count, bench markdown). No invisible runs.
- If the GPU box is genuinely idle (no active phase, no parallel work), pull the next item from §7.2 instead of waiting.

---

## 8 · Concrete next action (revised 2026-04-15)

**Status update (2026-04-15 revision + execution)**:

1. **M0.1 · `BlockId` unification** — **done**, shipped in the same
   commit as this revision. `infer/src/types.rs` now holds the canonical
   `BlockId(u32)` and `BlockFingerprint([u8; 16])`.
   `infer/src/kv_tier/id.rs` is a re-export stub;
   `prefix_cache::BlockId` and `block_manager::BlockId` also re-export.
   `BlockHashCtx` deleted. Exactly one `pub struct BlockId` in the tree.
   All three feature combos compile (CUDA-gated check clean; Metal
   check has pre-existing unrelated Codex mid-edit errors outside M0.1
   scope).
2. **M0.2 · three `prefix_cache.rs` bug fixes** — **no-op, already done
   in commit `5da8b67 fix(prefix_cache): split must inherit ref_count +
   evict must cascade`**. The 2026-04-13 research flagged these as open;
   they were fixed in the same 2026-04-13 work batch that wrote the
   research. 22 prefix_cache unit tests green. See project doc §8 items
   10–12 for the code locations.

**M0.3 · `page_size = 1 → 16` with per-format dispatch** blocks on the
in-flight `infer-cuda-kernels` extraction (kernel files move from
`infer/csrc/cuda/` to `crates/infer-cuda-kernels/csrc/`). After the
extraction lands, re-target the file edits to their new paths and
execute §1 below with the per-format dispatch course correction from
§1.3. **M0.3 is a prereq for M3, not for M1** — M1's exit gate is T0-only
`cached_prompts` parity and does not depend on `page_size`. See project
doc §6 M0.3 "Sequencing clarification" for the rationale.

**M1 · wire `RadixCache` + delete `kv_tier/directory.rs`** depends on
M0.1 (BlockId unification) and M0.2 (prefix_cache bug fixes). It does
**not** depend on M0.3 or on the in-flight extraction — the file paths
it touches (`infer/src/prefix_cache.rs`, `infer/src/scheduler/cuda/*.rs`,
`infer/src/kv_tier/directory.rs`, `infer/src/server_engine.rs`) are all
outside the extraction scope. Execute §2 below.

**M1 may ship as one atomic PR or split into M1a+M1b**. The 2026-04-15
draft said "must be one atomic PR"; Codex review showed a safe
compilable split exists:
- **M1a · structural** — extend the private `Node` struct in
  `prefix_cache.rs` with new fields (`tier_location`, `session_id`,
  `soft_pin_until`, `byte_len`, optional `fingerprint`). The struct is
  private so no API surface changes. `cached_prompts` and `TierDirectory`
  remain untouched; full test suite passes unchanged.
- **M1b · behavior** — swap `scheduler/cuda/*.rs` and
  `server_engine.rs:475-547` to use `radix_cache.lookup` instead of the
  linear `cached_prompts` compare; add refcount inc/dec; delete
  `kv_tier/directory.rs` and the `TierDirectory` re-export. Apply the
  full M1 benchmark gate to this PR.
Pick single-PR when reviewer bandwidth is high; split when it is not.

**M2 · dual residency** is a new milestone added in the 2026-04-15
revision; see §3.5 for details. It is a single-PR behaviour change on
top of M1 with an agent-trace benchmark gate.

**After M2**: proceed through §3 (M3 host pinned + coordinator, 3 stacked
PRs), §4 (M4 disk + session save/load, 2 stacked), and keep M5 (NIXL
real) as a deferred trigger-gated task.

**Remote CUDA host, in parallel**: keep collecting baseline benches
(§7.1 `--label page1` and `--label baseline-main-2026-04-13`) so that
M0.3's exit gate has a comparison point when it unblocks.

---

## 9 · How this doc is kept honest

- Each task above is a checkbox with a lane tag. Mark `[x]` when done; do not delete.
- When a phase fully ships, its section header gets a `**Done — see PR #N**` marker. Tasks stay visible as audit trail.
- If a task moves between lanes (e.g. "we thought this was `[L]` but a hidden cuda dep leaked"), update the lane tag and add a one-line note.
- If research from a future agent finds a course correction, update the **§N.3 Course corrections** subsection and bump a `## 10 · Changelog` entry below.

## 10 · Changelog

- **2026-04-15**: Major revision after internal survey + 7-system industry comparison. Project doc re-architected around three corrections; this doc gains §0.5 remapping + §8 rewrite + path updates. Summary of changes:
  - **P0-P5 → M0-M5**: phase remapping. M0 now holds three pre-reqs (M0.1 BlockId unify, M0.2 prefix_cache bug fixes, M0.3 page_size lift). M1 is the single atomic PR that wires RadixCache into the scheduler **and** deletes `kv_tier/directory.rs` (the old P1(a) structural / P1(b) behavior split is superseded because the midway state is uncompilable).
  - **New M2 (dual residency)** — the SGLang / vLLM / TRT-LLM "evict into free queue, pool reuses, lookup resurrects" pattern. Previously absorbed into "P2 behavior"; now first-class because it is the single biggest prefix-hit lever and is orthogonal to tiering. Industry reference: SGLang Novita 40% → 80% prefix hit rate, -56% TTFT.
  - **BlockId unification** (new M0.1). Three incompatible types existed (`kv_tier::BlockId(u64)`, `prefix_cache::BlockId(u32)`, `block_manager::BlockId(u32)`). Canonical is `types::BlockId(u32)`; content hash is a separate `types::BlockFingerprint([u8; 16])` only constructed when persistence / cross-node reuse is needed.
  - **RadixCache ↔ TierDirectory merge**. The 2026-04-13 two-layer topology (radix tree → directory resolve) was not industry-proven. 7 of 7 surveyed systems merge them. The revised design puts `block_id`, `location: Cell<TierLocation>`, `last_access`, `session_id`, `soft_pin_until`, `byte_len`, `fingerprint` fields directly on `RadixNode`; `infer/src/kv_tier/directory.rs` (322 lines, 0 production callers) is deleted in M1.
  - **Tier numbering** T0/T2/T3/T4 → T0/T1/T2/T3 for alignment with vLLM / SGLang / Mooncake / Dynamo KVBM documentation. No semantic change.
  - **Coordinator threading model** commits to OS thread + crossbeam (the §3.3 course correction, now project doc §4.4), not tokio.
  - **P4 KVFlow-lite** dropped from critical path. Ship LRU / SessionBiasedLru in M3; revisit reuse-distance only if M3's default policy proves insufficient.
  - **P5 NIXL real RDMA** deferred. Trigger criteria documented in project doc §6 M5: prefill/decode disaggregation, cluster-wide session roaming, or second consumer of the kernel crate.
  - **File paths updated for Route-A**: `infer/src/paged_kv.rs` → `infer/src/backend/cuda/paged_kv.rs`, `infer/src/metal_*` → `infer/src/backend/metal/*`, etc. See §0.5 table for the full rename list.
  - **New sources** added: 14 industry research references (vLLM KV offloading connector blog, SGLang HiCache design, LMCache/CacheBlend, Mooncake FAST'25 paper, Dynamo KVBM, TRT-LLM KV reuse, etc.). Full list in project doc §12.
- **2026-04-13**: Initial split (without per-phase research).
- **2026-04-13** (later): Enriched with 6-agent industry research. Added §N.2 Industry references and §N.3 Course corrections subsections per phase. Major design changes:
  - P0: bf16-only at `page_size=16`; INT8/FP8 quantized paths stay at `page_size=1` behind `format.uses_paged_layout()`
  - P1: keep radix tree, fix 3 existing bugs in `prefix_cache.rs`, drop blake3+chain BlockId design until P3/P5
  - P2: OS thread + crossbeam (NOT tokio); 2 dedicated streams; layer-granular cudaEvents
  - P3: tokio::fs default everywhere; 6 mlx-sys bindings (not 3); deps `postcard`+`blake3`; move `memmap2` out of cuda gate
  - P4: KVFlow-lite via per-session EWMA proxy (no DAG needed); Mooncake Algorithm 1 for slot routing
  - P5: trait uses `type Op` + explicit `poll`/`abort` (NOT `type Completion: Future`); two cargo features (`rdma-nixl` for Mac stub, `rdma-nixl-real` for CUDA prod)

---

## 11 · References (consolidated)

### vLLM
- [`vllm/config/cache.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py) — `DEFAULT_BLOCK_SIZE = 16`
- [`vllm/v1/core/block_pool.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)
- [`vllm/v1/core/kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py)
- [vLLM Hybrid KV Cache Manager design](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/)
- [vLLM Automatic Prefix Caching design](https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html)
- [vLLM cache config](https://docs.vllm.ai/en/stable/api/vllm/config/cache/)
- [vLLM issue #10611 — load/save KV from disk](https://github.com/vllm-project/vllm/issues/10611)
- [vLLM KV offloading connector blog 2026-01](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)

### SGLang
- [SGLang `radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- [SGLang `allocator.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/allocator.py)
- [SGLang `memory_pool.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py)
- [SGLang `memory_pool_host.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool_host.py)
- [SGLang `cache_controller.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/cache_controller.py)
- [SGLang `hicache_storage.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/hicache_storage.py)
- [SGLang `hiradix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/hiradix_cache.py)
- [SGLang HiCache blog (LMSYS 2025-09-10)](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
- [SGLang HiCache design docs](https://docs.sglang.io/advanced_features/hicache_design.html)

### FlashInfer
- [`include/flashinfer/page.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/page.cuh)
- [batch_prefill kernel tests (page_size ∈ {1,5,16})](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/attention/test_batch_prefill_kernels.py)
- [batch_decode kernel tests (page_size ∈ {1,8,16})](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/attention/test_batch_decode_kernels.py)
- [FlashInfer paged KV docs (last_page_len constraint)](https://docs.flashinfer.ai/api/cascade.html)

### LMCache
- [LMCache repo](https://github.com/LMCache/LMCache)
- [`storage_backend/local_disk_backend.py`](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/storage_backend/local_disk_backend.py)
- [`storage_backend/storage_manager.py`](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/storage_backend/storage_manager.py)
- [`storage_backend/local_cpu_backend.py`](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/storage_backend/local_cpu_backend.py)
- [LMCache tech report](https://lmcache.ai/tech_report.pdf)
- [LMCache × Mooncake integration](https://blog.lmcache.ai/en/2025/05/08/lmcache-x-mooncake-unite-to-pioneer-kvcache-centric-llm-serving-system/)

### Mooncake
- [Mooncake repo](https://github.com/kvcache-ai/Mooncake)
- [`mooncake-transfer-engine/include/transfer_engine.h`](https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-transfer-engine/include/transfer_engine.h)
- [Mooncake FAST '25 paper](https://www.usenix.org/system/files/fast25-qin.pdf)
- [Mooncake × SGLang HiCache design](https://kvcache-ai.github.io/Mooncake/design/hicache-design.html)

### NIXL / NVIDIA
- [`ai-dynamo/nixl`](https://github.com/ai-dynamo/nixl)
- [`nixl-sys` v1.0 on docs.rs](https://docs.rs/nixl-sys/latest/nixl_sys/)
- [NIXL BackendGuide.md](https://github.com/ai-dynamo/nixl/blob/main/docs/BackendGuide.md)
- [NVIDIA NIXL blog](https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/)
- [NVIDIA `nvidia-peermem` README](https://download.nvidia.com/XFree86/Linux-x86_64/525.78.01/README/nvidia-peermem.html)
- [GPUDirect RDMA overview](https://docs.nvidia.com/cuda/gpudirect-rdma/)

### KVFlow / cache-aware routing papers
- [KVFlow paper (arXiv 2507.07400)](https://arxiv.org/abs/2507.07400)
- [llm-d KV Cache Manager](https://llm-d.ai/docs/architecture/Components/kv-cache-manager)
- [llm-d KVCache Indexer Core (DeepWiki)](https://deepwiki.com/llm-d/llm-d-kv-cache-manager/2.2-kvcache-indexer-core)
- [Red Hat — Master KV cache aware routing with llm-d](https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference)

### MLX / Metal
- [MLX `mlx/backend/metal/allocator.h`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.h)
- [MLX Metal memory management Python docs](https://ml-explore.github.io/mlx/build/html/python/metal.html)
- [`mlx.core.set_wired_limit`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)
- [mlx-lm #883 — wired memory kernel panic](https://github.com/ml-explore/mlx-lm/issues/883)
- [mlx-lm #615 — `--max-kv-size` not in server](https://github.com/ml-explore/mlx-lm/issues/615)
- [llama.cpp #20697 — `--cache-disk` for UMA](https://github.com/ggml-org/llama.cpp/issues/20697)

### Rust async / cancel-safety / IO
- [Tokio cancel-safety RFD 400 (Oxide)](https://rfd.shared.oxide.computer/rfd/400)
- [Cybernetist — Rust tokio task cancellation patterns](https://cybernetist.com/2024/04/19/rust-tokio-task-cancellation-patterns/)
- [`tokio_util::sync::CancellationToken`](https://docs.rs/tokio-util/latest/tokio_util/sync/struct.CancellationToken.html)
- [Status of tokio-uring (rust-lang forum)](https://users.rust-lang.org/t/status-of-tokio-uring/114481)
- [Monoio benchmark docs](https://github.com/redacted/monoio/blob/master/docs/en/benchmark.md)

### Local files (referenced repeatedly)
- `infer/src/paged_kv.rs` (cuda-gated)
- `infer/src/flashinfer_metadata.rs` (cuda-gated)
- `infer/src/prefix_cache.rs` (always-on, local-checkable)
- `infer/src/scheduler/cuda/{prefill,decode,core,runtime,request}.rs` (cuda-gated)
- `infer/src/server_engine.rs`
- `infer/src/model/kv_cache.rs:130-168` — legacy CPU offload (deleted in P2)
- `infer/src/metal_kv_pool.rs`
- `infer/src/metal_prefix_cache.rs`
- `infer/src/metal_gdr.rs` — **Gated Delta Rule, not GPUDirect** (rename suggested in separate cleanup PR)
- `infer/mlx-sys/src/lib.rs`
- `infer/csrc/cuda/{paged_kv_append,decode_prep_paged,decode_prep_paged_hd256,kv_cache_to_paged,kv_quant,scatter_kv}.cu`
- `infer/src/scheduler/policy.rs:64-78,133-152` — existing `AdmissionPolicy` / `ChunkingPolicy`
- `infer/src/events.rs:7-24` — existing `EngineEvent` / `EventSink`
- `crates/infer-agent/src/lib.rs:166-188` — `AgentSession::save_to_path / load_from_path` (JSON-only today)
