//! Session KV-recall on the dense-Qwen3 CUDA paged decode path (the
//! infinite-memory feature; mirror of the Metal design in
//! `infer-metal/src/executor.rs` + `mlx_qwen35_model.cpp`).
//!
//! CUDA decode already runs **paged attention over a page table**
//! (`decode_attention` consumes `meta.kv_indices` / `kv_indptr` /
//! `kv_last_page_len`, with the KV length derived entirely from the page table —
//! no per-KV-token position array). So recall needs **no gather kernel**: each
//! step, instead of writing ALL resident pages to `kv_indices`, write only the
//! SELECTED pages (sink ∪ top-k recalled ∪ local). The existing TileLang decode
//! kernel then attends exactly the working set, and RoPE is already baked into
//! the cached K, so the non-contiguous page subset is correct.
//!
//! Under the write-through model (supersedes the swap
//! plan), this module owns the **decode attend-resident** verb (the restricted
//! page table), the **R6 reps** (mean-pooled K kept resident, computed once a
//! block freezes — exactly the write-through-time rep), AND the **evict-drop
//! selection** ([`CudaRecallState::take_evict_pages`]): the cold middle pages
//! outside the working set whose HBM page is returned to the pool. The other
//! write-through verbs (mirror at page-fill / prefetch at prefill) live on the
//! executor as the `infer_seam::KvTier` impl.
//!
//! **The page actually frees now (flat VRAM).** The deferred blocker — that the
//! host single-allocator re-publishes a contiguous page table every step via
//! `mirror_slot`, so a live slot's middle page could not be freed without
//! breaking the `SlotProgress` continuity guard — is resolved by decoupling
//! *physical residency* from the *logical* page table: an evicted middle page
//! leaves a sentinel ([`cuda_kernels::prelude::EVICTED_PAGE`]) in its logical
//! slot while its physical page returns to both pools' free lists. The logical
//! page count (and `seq_len`) is unchanged, so `mirror_slot`'s
//! `pages.len() == ceil(seq_len/page_size)` and the `SlotProgress` contiguity
//! guard both still hold — the slot is sparse without violating either contract.
//! The executor drives the free against the host (`KvAllocator::evict_slot_page`)
//! and device (`PagedKVPool::evict_slot_page`) pools.
//!
//! Everything here is `#[cfg(feature = "cuda")]`: the planner
//! ([`infer_core::plan_recall`]) is device-neutral, but the rep/score machinery
//! reads back device K/Q. Default off → the decode hot path never touches this.

/// Validated session KV-recall budget (per the offline Qwen3.6 quality gate +
/// `wins/2026-06-23-kv-recall-arle-core-e2e.md`): sink 32, local 256, block 32,
/// top-k 8 → working set 32 + 256 + 8·32 = 544 tokens (9.6% KV in the e2e). The
/// same defaults the Metal executor uses (`default_recall_config`). Recall only
/// restricts attention once `cache_len` exceeds this budget; below it
/// `plan_recall` returns the full contiguous range (no-op).
#[cfg(feature = "cuda")]
#[must_use]
pub(crate) fn default_recall_config() -> infer_core::RecallConfig {
    infer_core::RecallConfig {
        n_init: 32,
        n_local: 256,
        l_bs: 32,
        top_k: 8,
    }
}

#[cfg(feature = "cuda")]
pub(crate) use cuda_impl::{CudaRecallState, RecallShardReduce};

#[cfg(feature = "cuda")]
mod cuda_impl {
    use anyhow::Result;
    use cuda_kernels::prelude::{DeviceContext, EVICTED_PAGE, PagedKVPool};
    use infer_seam::ShardSpec;

    /// Cross-shard reductions that keep every 2D rank planning identically.
    /// Without them a rank scores only its own sequence slice and head shard,
    /// selects a different top-k, and evicts pages its peers kept — no block ends
    /// up fully resident anywhere.
    pub(crate) struct RecallShardReduce<'a> {
        /// Element-wise min over `lo` and max over `hi` across the sequence-shard
        /// (`attn_cp`) group.
        pub(crate) widen: &'a dyn Fn(&mut [f32], &mut [f32]) -> Result<()>,
        /// Sum across the head-shard (`attn_tp`) group.
        pub(crate) sum_scores: &'a dyn Fn(&mut [f32]) -> Result<()>,
    }

    /// Per-slot session KV-recall state for the dense-Qwen3 paged decode path.
    ///
    /// Holds the resident per-middle-block mean-key reps (so offloaded blocks
    /// stay scorable) and the page plan for the NEXT decode step (stale-Q:
    /// this step's query selects next step's pages — licensed). Empty/`None`
    /// unless recall is enabled and the session has exceeded the budget.
    #[derive(Default)]
    pub(crate) struct CudaRecallState {
        /// CP shard ownership (rank, size). Default = no sharding (rank 0, size 1).
        shard: ShardSpec,
        /// Resident per-middle-block key envelopes: layer-0 K reduced over the
        /// block's `l_bs` tokens to a per-channel `[min | max]` pair, flattened
        /// `2 * nkv * hd` f32 (min half first). Index = middle block index (token
        /// base `n_init + i * l_bs`). Keeps offloaded blocks scorable.
        ///
        /// An interval, not a mean: K is cached post-RoPE, and averaging over
        /// `l_bs` consecutive positions rotates each key by a different angle, so
        /// the high-frequency channels cancel toward zero and the score carries no
        /// ranking signal (measured 2026-08-18: 0/16 at every context length, flat
        /// in how much of the middle was retained). The envelope survives rotation
        /// and makes `max q·k` boundable, which is what top-k needs.
        ///
        /// Under CP each envelope covers only this shard's local tokens.
        block_reps: Vec<Vec<f32>>,
        /// Selected page IDs for the next decode step (sink ∪ recalled ∪ local),
        /// ascending temporal order. `None` = session fits budget → full
        /// contiguous page table (byte-identical default).
        recall_pages: Option<Vec<u32>>,
        /// Working-set token ranges of the next step's plan. Kept so the
        /// executor can re-resolve `recall_pages` after prefetching L3-resident
        /// blocks back (sentinels patched to fresh page ids). Empty when the
        /// plan is the full contiguous range.
        recall_ranges: Vec<(usize, usize)>,
        /// Logical page indices to evict-drop (cold middle pages outside the
        /// working set). Drained via [`Self::take_evict_pages`]; the executor
        /// frees them from both host and device pools.
        evict_pages: Vec<usize>,
        /// Logical page indices to prefetch back from the L3 tier: working-set
        /// pages currently `EVICTED_PAGE` sentinels. Drained via
        /// [`Self::take_prefetch_pages`]; the executor allocs a fresh page,
        /// H2D-copies the tier payload, patches the sentinel, re-resolves.
        prefetch_pages: Vec<usize>,
    }

    impl CudaRecallState {
        /// Selected page list for this step; `None` keeps the default full-cache page table.
        pub(crate) fn recall_pages(&self) -> Option<&[u32]> {
            self.recall_pages.as_deref()
        }

        /// Set the CP shard ownership. Called once when recall is enabled; the
        /// shard is fixed for the executor's lifetime.
        pub(crate) fn set_shard(&mut self, shard: ShardSpec) {
            self.shard = shard;
        }

        /// Reset on slot reuse (new occupant): drop the prior session's reps and
        /// plan so a fresh request starts from the byte-identical default.
        pub(crate) fn reset(&mut self) {
            self.block_reps.clear();
            self.recall_pages = None;
            self.recall_ranges.clear();
            self.evict_pages.clear();
            self.prefetch_pages.clear();
        }

        /// Grow resident mean-key reps for newly-frozen middle blocks.
        ///
        /// A block is "frozen" once it has left the local window
        /// (`base + l_bs <= cache_len - n_local`), so its K is final and the rep
        /// is computed exactly once. Only newly-completed blocks are read back.
        fn update_block_reps(
            &mut self,
            ctx: &DeviceContext,
            pool: &PagedKVPool,
            slot: usize,
            cache_len: usize,
            cfg: &infer_core::RecallConfig,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            if cfg.l_bs == 0 || cache_len <= cfg.n_init + cfg.n_local {
                return Ok(());
            }
            let mid_span = cache_len - cfg.n_init - cfg.n_local;
            let frozen_blocks = mid_span / cfg.l_bs;
            if frozen_blocks <= self.block_reps.len() {
                return Ok(());
            }
            let page_size = pool.page_size;
            let pages = pool.page_indices(slot);
            // Layer-0 BF16 K plane: [max_pages, num_kv_heads, page_size, head_dim].
            let k0 = pool.k_data_slice(0);
            let hd_len = num_kv_heads * head_dim;
            let page_elems = num_kv_heads * page_size * head_dim;
            let page_bytes = page_elems * 2;

            for block in self.block_reps.len()..frozen_blocks {
                let base = cfg.n_init + block * cfg.l_bs;
                // A just-frozen block is still resident this step (it only left the
                // local window now), so its rep is computed before any evict-drop.
                // Guard defensively: if a span page is already an evict sentinel,
                // stop growing reps here (we cannot reconstruct a freed block's
                // mean key; it stays prefix-only-recallable per R1). Never advance
                // `block_reps` past a sentinel block — that would misalign scoring.
                // Under CP, only this shard's pages are checked/read.
                let span_resident = (base..base + cfg.l_bs)
                    .filter_map(|pos| self.shard.local_index(pos / page_size))
                    .all(|li| pages[li] != EVICTED_PAGE);
                if !span_resident {
                    break;
                }
                let mut lo = vec![f32::INFINITY; hd_len];
                let mut hi = vec![f32::NEG_INFINITY; hd_len];
                // One readback per PAGE, not per token: `n_init`/`n_local`/`l_bs`
                // are forced to page multiples, so a block is a whole number of
                // pages and the per-token form re-read each page `page_size` times.
                for gp in (base / page_size)..(base + cfg.l_bs).div_ceil(page_size) {
                    let Some(li) = self.shard.local_index(gp) else {
                        continue;
                    };
                    let page = pages[li] as usize;
                    let start = page * page_bytes;
                    let bytes = ctx
                        .stream
                        .clone_dtoh(&k0.slice(start..start + page_bytes))
                        .map_err(|e| anyhow::anyhow!("recall K page dtoh failed: {e}"))?;
                    // [num_kv_heads, page_size, head_dim] → token `row` is
                    // [num_kv_heads, head_dim] at stride page_size over the
                    // middle axis.
                    for row in 0..page_size {
                        let pos = gp * page_size + row;
                        if pos < base || pos >= base + cfg.l_bs {
                            continue;
                        }
                        for h in 0..num_kv_heads {
                            let head_base = (h * page_size + row) * head_dim;
                            let out = h * head_dim;
                            for d in 0..head_dim {
                                let off = (head_base + d) * 2;
                                let v = half::bf16::from_le_bytes([bytes[off], bytes[off + 1]])
                                    .to_f32();
                                lo[out + d] = lo[out + d].min(v);
                                hi[out + d] = hi[out + d].max(v);
                            }
                        }
                    }
                }
                // A block with no local page (cp_size >= 4, l_bs < cp * page_size)
                // keeps the min/max identity, so a cross-shard widen leaves the
                // covering ranks' values untouched. Scoring masks it when there is
                // no widen to come.
                self.block_reps.push([lo, hi].concat());
            }
            Ok(())
        }

        /// Widen this rank's block envelopes over the sequence-shard group so
        /// every rank scores the same intervals.
        ///
        /// Sized from `recall_block_count`, never from `block_reps.len()`: rep
        /// growth stops early at a sentinel span and that point differs per rank,
        /// which would mismatch the collective payload and hang the group.
        fn widen_envelopes(
            &mut self,
            cache_len: usize,
            cfg: &infer_core::RecallConfig,
            hd_len: usize,
            reduce: &RecallShardReduce<'_>,
        ) -> Result<()> {
            let nb = infer_core::recall_block_count(cache_len, cfg);
            if nb == 0 || hd_len == 0 {
                return Ok(());
            }
            let mut lo = vec![f32::INFINITY; nb * hd_len];
            let mut hi = vec![f32::NEG_INFINITY; nb * hd_len];
            for (i, rep) in self.block_reps.iter().take(nb).enumerate() {
                if rep.len() != 2 * hd_len {
                    continue; // no rep yet: the min/max identity stands in
                }
                let at = i * hd_len;
                lo[at..at + hd_len].copy_from_slice(&rep[..hd_len]);
                hi[at..at + hd_len].copy_from_slice(&rep[hd_len..]);
            }
            (reduce.widen)(&mut lo, &mut hi)?;
            self.block_reps.truncate(nb);
            self.block_reps.resize(nb, Vec::new());
            for (i, rep) in self.block_reps.iter_mut().enumerate() {
                let at = i * hd_len;
                *rep = [&lo[at..at + hd_len], &hi[at..at + hd_len]].concat();
            }
            Ok(())
        }

        /// Recompute the recall page plan for the next decode step.
        ///
        /// Scores block reps against this step's GQA-mean layer-0 query
        /// (`q · rep`, one step stale — licensed). `recall_pages` is left `None`
        /// when the plan is the single contiguous range (session fits budget) so
        /// the default page table stays byte-identical.
        ///
        /// `query_layer0`: post-RoPE layer-0 decode query, `[num_q_heads, head_dim]`
        /// row-major f32, GQA-mean pooled here to `[num_kv_heads, head_dim]`.
        ///
        /// `allow_prefetch` selects the tier policy:
        /// - `false` (dense-Qwen3 arm, no L3 tier): evicted blocks masked to `-inf`
        ///   (cannot be re-attended); page list resolved eagerly.
        /// - `true` (Qwen3.6 recall tier): all blocks compete; top-k evicted blocks
        ///   go to `prefetch_pages`; page-list resolution deferred until
        ///   [`Self::resolve_recall_pages`] runs after the prefetch patches sentinels.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn recompute_recall_plan(
            &mut self,
            ctx: &DeviceContext,
            pool: &PagedKVPool,
            slot: usize,
            cache_len: usize,
            cfg: &infer_core::RecallConfig,
            num_q_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            query_layer0: &[f32],
            allow_prefetch: bool,
            reduce: Option<&RecallShardReduce<'_>>,
        ) -> Result<()> {
            self.update_block_reps(ctx, pool, slot, cache_len, cfg, num_kv_heads, head_dim)?;
            let hd_len = num_kv_heads * head_dim;
            if let Some(r) = reduce {
                self.widen_envelopes(cache_len, cfg, hd_len, r)?;
            }

            // A short query is never legitimate — it scores every block identically
            // and the planner's index tie-break then keeps blocks `0..top_k`, which
            // looks like a working plan. The 2D ring prefill path fed an empty vec
            // here for months (2026-08-18).
            let group = num_q_heads / num_kv_heads.max(1);
            anyhow::ensure!(
                group > 0 && query_layer0.len() >= num_q_heads * head_dim,
                "recall scoring query missing: got {} floats, need {} ({num_q_heads} q heads × {head_dim})",
                query_layer0.len(),
                num_q_heads * head_dim
            );
            // GQA-mean the query: average the `num_q_heads / num_kv_heads` query
            // heads in each KV group into one `[head_dim]` vector per KV head.
            let mut q = vec![0.0_f32; num_kv_heads * head_dim];
            for kv in 0..num_kv_heads {
                for g in 0..group {
                    let qh = kv * group + g;
                    for d in 0..head_dim {
                        q[kv * head_dim + d] += query_layer0[qh * head_dim + d];
                    }
                }
                for d in 0..head_dim {
                    q[kv * head_dim + d] /= group as f32;
                }
            }

            let nb = self.block_reps.len();
            let page_size = pool.page_size;
            let table = pool.page_indices(slot);
            let mut scores = vec![0.0_f32; nb];
            for (i, rep) in self.block_reps.iter().enumerate() {
                let base = cfg.n_init + i * cfg.l_bs;
                let local_pages = (base..base + cfg.l_bs)
                    .filter_map(|pos| self.shard.local_index(pos / page_size))
                    .count();
                let resident = local_pages > 0
                    && (base..base + cfg.l_bs)
                        .filter_map(|pos| self.shard.local_index(pos / page_size))
                        .all(|li| table.get(li).copied() != Some(EVICTED_PAGE));
                // Without an L3 tier (dense arm, `allow_prefetch=false`) a block
                // whose pages were evict-dropped cannot be re-attended mid-decode,
                // so mask it to `-inf` (only resident blocks compete). With the L3
                // recall tier (`allow_prefetch=true`) an evicted block IS scorable
                // via its resident rep and will be prefetched back if it ranks
                // top-k — the re-recall coverage win.
                // After a widen the envelope covers every shard's tokens, so a
                // block this rank holds no page of is still scorable — and masking
                // it would diverge this rank's selection from its peers'.
                scores[i] = if reduce.is_some() || (local_pages > 0 && (resident || allow_prefetch))
                {
                    // Upper bound on the block's max `q·k`: per channel take the
                    // better end of the key interval. A bound is what makes top-k
                    // admissible — a mean gives none.
                    let half = rep.len() / 2;
                    let (lo, hi) = rep.split_at(half);
                    (0..half.min(q.len()))
                        .map(|k| (q[k] * lo[k]).max(q[k] * hi[k]))
                        .sum()
                } else {
                    f32::NEG_INFINITY
                };
            }
            // Each rank scored only its own KV heads; the sum is the full-head
            // score, so every rank now ranks blocks identically.
            if let Some(r) = reduce {
                (r.sum_scores)(&mut scores)?;
            }

            let plan = infer_core::plan_recall(cache_len, &scores, cfg);
            // A single contiguous full range == today's default read; keep `None`
            // so the decode hot path stays byte-identical when the session fits.
            let is_full = plan.ranges.len() == 1 && plan.ranges[0] == (0, cache_len);
            if is_full {
                self.recall_pages = None;
                self.recall_ranges.clear();
                self.evict_pages.clear();
                self.prefetch_pages.clear();
            } else {
                // Write-through evict-drop (the flat-VRAM win): the logical pages
                // the plan does NOT attend, restricted to the still-resident middle
                // (outside the pinned sink + local windows). The executor frees
                // these physical pages from BOTH pools after this step; their KV is
                // already mirrored to the tier (write_through), so the drop is free.
                self.evict_pages = evict_logical_pages(
                    &plan.ranges,
                    pool.page_indices(slot),
                    page_size,
                    cfg,
                    self.shard,
                );
                if allow_prefetch {
                    // Re-recall: logical pages in the working set that are currently
                    // evicted sentinels. The executor pulls them back H2D from the
                    // tier (reinstating real page ids) BEFORE the next step, then
                    // calls `resolve_recall_pages`. Defer page resolution until
                    // then so a sentinel never reaches the kernel page table.
                    self.prefetch_pages = prefetch_logical_pages(
                        &plan.ranges,
                        pool.page_indices(slot),
                        page_size,
                        self.shard,
                    );
                    self.recall_ranges = plan.ranges;
                    // `recall_pages` is intentionally left at its previous value;
                    // `resolve_recall_pages` rebuilds it after the executor's
                    // prefetch + evict patch the table for the next step.
                } else {
                    self.recall_pages = Some(token_ranges_to_pages(
                        &plan.ranges,
                        pool.page_indices(slot),
                        page_size,
                        self.shard,
                    ));
                    self.recall_ranges.clear();
                    self.prefetch_pages.clear();
                }
            }
            Ok(())
        }

        /// Logical page indices to evict-drop (complement of the working set). Drained by the executor.
        pub(crate) fn take_evict_pages(&mut self) -> Vec<usize> {
            std::mem::take(&mut self.evict_pages)
        }

        /// Logical page indices to prefetch back from the L3 tier. Drained by the executor.
        pub(crate) fn take_prefetch_pages(&mut self) -> Vec<usize> {
            std::mem::take(&mut self.prefetch_pages)
        }

        /// Re-resolve `recall_pages` from the stored plan ranges against the pool's
        /// CURRENT page table — called by the executor AFTER it has prefetched the
        /// evicted blocks back (patching their sentinels to fresh page ids) and
        /// evict-dropped the cold ones. Only used on the prefetch-enabled
        /// (Qwen3.6) arm, where `recompute_recall_plan` deferred page resolution.
        ///
        /// Every working-set page must now be resident (the executor prefetched all
        /// `take_prefetch_pages`), so a surviving sentinel is a bug — `errors`-loud
        /// in debug, skipped in release rather than feeding the kernel a sentinel.
        pub(crate) fn resolve_recall_pages(&mut self, pool: &PagedKVPool, slot: usize) {
            if self.recall_ranges.is_empty() {
                return;
            }
            self.recall_pages = Some(token_ranges_to_pages(
                &self.recall_ranges,
                pool.page_indices(slot),
                pool.page_size,
                self.shard,
            ));
        }
    }

    /// Pages covered by a working-set range that are currently `EVICTED_PAGE`
    /// sentinels (KV lives only in the L3 tier) — the re-recalled-block pages
    /// the executor must reinstate before the next step attends them.
    /// Returns LOCAL page-table indices.
    fn prefetch_logical_pages(
        ranges: &[(usize, usize)],
        pages: &[u32],
        page_size: usize,
        shard: ShardSpec,
    ) -> Vec<usize> {
        if page_size == 0 || pages.is_empty() {
            return Vec::new();
        }
        let mut out: Vec<usize> = Vec::new();
        for &(s, e) in ranges {
            if s >= e {
                continue;
            }
            let first = s / page_size;
            let last = (e - 1) / page_size;
            for global_page in first..=last {
                if let Some(li) = shard.local_index(global_page) {
                    if li < pages.len() && pages[li] == EVICTED_PAGE && out.last() != Some(&li) {
                        out.push(li);
                    }
                }
            }
        }
        out
    }

    /// Pages NOT covered by any working-set range, excluding the pinned sink
    /// (first `n_init` tokens) and local (last `n_local` tokens) regions, and
    /// skipping already-evicted sentinels — the cold middle pages whose HBM
    /// page can be returned to the pool. Returns LOCAL page-table indices.
    fn evict_logical_pages(
        ranges: &[(usize, usize)],
        pages: &[u32],
        page_size: usize,
        cfg: &infer_core::RecallConfig,
        shard: ShardSpec,
    ) -> Vec<usize> {
        if page_size == 0 || pages.is_empty() {
            return Vec::new();
        }
        let mut keep = vec![false; pages.len()];
        for &(s, e) in ranges {
            if s >= e {
                continue;
            }
            let first = s / page_size;
            let last = (e - 1) / page_size;
            for global_page in first..=last {
                if let Some(li) = shard.local_index(global_page) {
                    if li < keep.len() {
                        keep[li] = true;
                    }
                }
            }
        }
        // Pin regions: leading sink pages and trailing local pages never evict.
        let sink_pages = cfg.n_init.div_ceil(page_size);
        // The local window is the last `n_local` tokens; pin every page from the
        // local window's first page to the end (the current partial page included).
        let total_tokens = ranges.iter().map(|&(_, e)| e).max().unwrap_or(0);
        let local_start = total_tokens.saturating_sub(cfg.n_local);
        let first_local_page = local_start / page_size;
        (0..pages.len())
            .filter(|&li| {
                let global_page = shard.global_page(li);
                global_page >= sink_pages
                    && global_page < first_local_page
                    && !keep[li]
                    && pages[li] != EVICTED_PAGE
            })
            .collect()
    }

    /// Page granularity (16) is finer than the recall block (`l_bs`=32) and the
    /// sink/local windows are multi-page, so each range covers whole pages; the
    /// final range ends at `cache_len`, so the last page is the current
    /// (partially filled) local page — exactly what `kv_last_page_len` describes.
    /// Under CP, only this shard's pages are included.
    fn token_ranges_to_pages(
        ranges: &[(usize, usize)],
        pages: &[u32],
        page_size: usize,
        shard: ShardSpec,
    ) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for &(s, e) in ranges {
            if s >= e {
                continue;
            }
            let first_page = s / page_size;
            let last_page = (e - 1) / page_size;
            for global_page in first_page..=last_page {
                if let Some(li) = shard.local_index(global_page) {
                    if let Some(&page) = pages.get(li) {
                        // The working set is built only from resident blocks (scoring
                        // masks evicted ones to -inf), so a sentinel must never reach
                        // the kernel page table. Guard the invariant and skip if hit.
                        debug_assert_ne!(
                            page, EVICTED_PAGE,
                            "recall working set referenced an evicted page (local {li}, global {global_page})"
                        );
                        if page == EVICTED_PAGE {
                            continue;
                        }
                        // Ranges are ascending and page-aligned at the block grain,
                        // but the sink end and a recalled-block start can land in the
                        // same page; dedup the boundary so a page is never doubled.
                        if out.last() != Some(&page) {
                            out.push(page);
                        }
                    }
                }
            }
        }
        out
    }
}
