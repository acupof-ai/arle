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
//! This is the **resident variant** (mirroring the Metal landing): the full KV
//! stays in the device pool and recall restricts *attention* to the selected
//! pages (a decode-compute saving). Under the write-through model
//! (`docs/plans/2026-06-23-writethrough-tiered-kv-memory.md`, supersedes the swap
//! plan), this module is the **decode attend-resident** verb (the restricted page
//! table) and the **R6 reps** (mean-pooled K kept resident, computed once a block
//! freezes — exactly the write-through-time rep). The other write-through verbs
//! (mirror at page-fill / prefetch at prefill) live on the executor as the
//! `infer_seam::KvTier` impl. The remaining gap — freeing the non-selected middle
//! pages out of HBM for the flat-VRAM win — is the same single-allocator blocker:
//! the host `CudaKvPool` owns the page free and the executor re-publishes the
//! slot's full page table every step via `mirror_slot`, so a live slot's middle
//! page cannot be freed without breaking the `SlotProgress` continuity guard. The
//! reps + scoring + restricted page table are fully live, so the budget-bounded
//! decode-attention working set works now; the VRAM flattening is deferred.
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
pub(crate) use cuda_impl::CudaRecallState;

#[cfg(feature = "cuda")]
mod cuda_impl {
    use anyhow::Result;
    use cuda_kernels::prelude::{DeviceContext, PagedKVPool};

    /// Per-slot session KV-recall state for the dense-Qwen3 paged decode path.
    ///
    /// Holds the resident per-middle-block mean-key reps (so offloaded blocks
    /// stay scorable) and the page plan for the NEXT decode step (stale-Q:
    /// this step's query selects next step's pages — licensed). Empty/`None`
    /// unless recall is enabled and the session has exceeded the budget.
    #[derive(Default)]
    pub(crate) struct CudaRecallState {
        /// Resident per-middle-block mean-key reps (#2). Each entry is the
        /// layer-0 K mean-pooled over its `l_bs` tokens, GQA-shaped to
        /// `[num_kv_heads, head_dim]` flattened to `nkv * hd` f32 — the resident
        /// representative that keeps a (future-)offloaded block scorable
        /// (`q · rep`). Index = middle block index (token base
        /// `n_init + i * l_bs`). Grown incrementally as whole blocks complete.
        block_reps: Vec<Vec<f32>>,
        /// Selected page IDs for the NEXT decode step (sink ∪ recalled ∪ local),
        /// in ascending temporal page order, ending with the current local page.
        /// `None` = the session still fits the budget → today's full contiguous
        /// page table (byte-identical default).
        recall_pages: Option<Vec<u32>>,
    }

    impl CudaRecallState {
        /// The selected page list for this step, if recall is active. `None`
        /// keeps the default full-cache page table.
        pub(crate) fn recall_pages(&self) -> Option<&[u32]> {
            self.recall_pages.as_deref()
        }

        /// Reset on slot reuse (new occupant): drop the prior session's reps and
        /// plan so a fresh request starts from the byte-identical default.
        pub(crate) fn reset(&mut self) {
            self.block_reps.clear();
            self.recall_pages = None;
        }

        /// Grow the resident mean-key reps for any newly-frozen middle blocks (#2).
        ///
        /// A block is "frozen" once it has left the local window
        /// (`base + l_bs <= cache_len - n_local`), so its K is final and the rep
        /// is computed exactly once. Mean-pools layer-0 K over each frozen block's
        /// `l_bs` tokens into a `[num_kv_heads, head_dim]` rep (flattened). Cheap:
        /// only newly-completed blocks are read back. Reads layer-0 K from the
        /// paged pool by page (the BF16 K plane is laid out
        /// `[max_pages, num_kv_heads, page_size, head_dim]`).
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
            let page_elems = num_kv_heads * page_size * head_dim; // bf16 elems / page
            let page_bytes = page_elems * 2;
            let l_bs_f = cfg.l_bs as f32;

            for block in self.block_reps.len()..frozen_blocks {
                let base = cfg.n_init + block * cfg.l_bs;
                let mut rep = vec![0.0_f32; num_kv_heads * head_dim];
                // Walk this block's `l_bs` tokens; each token lives at page
                // `pages[pos / page_size]`, intra-page row `pos % page_size`.
                for pos in base..base + cfg.l_bs {
                    let page = pages[pos / page_size] as usize;
                    let row = pos % page_size;
                    let start = page * page_bytes;
                    let bytes = ctx
                        .stream
                        .clone_dtoh(&k0.slice(start..start + page_bytes))
                        .map_err(|e| anyhow::anyhow!("recall K page dtoh failed: {e}"))?;
                    // [num_kv_heads, page_size, head_dim] → token `row` is
                    // [num_kv_heads, head_dim] at stride page_size over the
                    // middle axis.
                    for h in 0..num_kv_heads {
                        let head_base = (h * page_size + row) * head_dim;
                        let out = h * head_dim;
                        for d in 0..head_dim {
                            let off = (head_base + d) * 2;
                            let v =
                                half::bf16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
                            rep[out + d] += v;
                        }
                    }
                }
                for v in &mut rep {
                    *v /= l_bs_f;
                }
                self.block_reps.push(rep);
            }
            Ok(())
        }

        /// Recompute the recall page plan for the NEXT decode step (#3/#5).
        ///
        /// Scores the resident block reps against this step's GQA-mean layer-0
        /// query (`q · rep`, one step stale — licensed), runs
        /// [`infer_core::plan_recall`], converts the chosen token ranges to a
        /// page subset, and stashes it. `recall_pages` is left `None` when the
        /// plan is the single contiguous range (session fits the budget) so the
        /// default page table stays byte-identical.
        ///
        /// `query_layer0` is the post-RoPE layer-0 decode query read back from
        /// `q_batch` as `[num_q_heads, head_dim]` row-major f32. It is GQA-mean
        /// pooled here (query-heads-per-KV-group → `[num_kv_heads, head_dim]`) to
        /// match the rep shape — the validated `--shared` per-slot scoring.
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
        ) -> Result<()> {
            self.update_block_reps(ctx, pool, slot, cache_len, cfg, num_kv_heads, head_dim)?;

            // GQA-mean the query: average the `num_q_heads / num_kv_heads` query
            // heads in each KV group into one `[head_dim]` vector per KV head.
            let group = num_q_heads / num_kv_heads.max(1);
            let mut q = vec![0.0_f32; num_kv_heads * head_dim];
            if group > 0 && query_layer0.len() >= num_q_heads * head_dim {
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
            }

            let nb = self.block_reps.len();
            let mut scores = vec![0.0_f32; nb];
            for (i, rep) in self.block_reps.iter().enumerate() {
                let n = rep.len().min(q.len());
                let mut acc = 0.0_f32;
                for k in 0..n {
                    acc += q[k] * rep[k];
                }
                scores[i] = acc;
            }

            let plan = infer_core::plan_recall(cache_len, &scores, cfg);
            // A single contiguous full range == today's default read; keep `None`
            // so the decode hot path stays byte-identical when the session fits.
            let is_full = plan.ranges.len() == 1 && plan.ranges[0] == (0, cache_len);
            self.recall_pages = if is_full {
                None
            } else {
                Some(token_ranges_to_pages(
                    &plan.ranges,
                    pool.page_indices(slot),
                    pool.page_size,
                ))
            };
            // TODO(write-through evict-drop): this is the RESIDENT variant — full
            // KV stays in the device pool and recall restricts *attention* to the
            // selected pages (the restricted page table), saving decode compute.
            // The write-through verbs (`infer_seam::KvTier` on the executor)
            // already mirror pages to the tier (`write_through`) and can prefetch
            // them back (`prefetch`); the missing step for the flat-VRAM-vs-history
            // win is freeing the non-selected middle pages OUT of HBM, which needs
            // the executor to own mid-decode device-page lifecycle (currently the
            // host CudaKvPool is the single allocator and re-publishes the full
            // page table via `mirror_slot` every step). Reps + scoring + restricted
            // page table are fully live.
            Ok(())
        }
    }

    /// Convert ascending, merged token ranges (from `plan_recall`) to the
    /// deduplicated, ascending physical page subset. Page granularity (16) is
    /// finer than the recall block (`l_bs`=32) and the sink/local windows are
    /// multi-page, so each range covers whole pages; the final range ends at
    /// `cache_len`, so the last page is the current (partially filled) local
    /// page — exactly what `kv_last_page_len` must describe.
    fn token_ranges_to_pages(
        ranges: &[(usize, usize)],
        pages: &[u32],
        page_size: usize,
    ) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for &(s, e) in ranges {
            if s >= e {
                continue;
            }
            let first_page = s / page_size;
            let last_page = (e - 1) / page_size;
            for p in first_page..=last_page {
                if let Some(&page) = pages.get(p) {
                    // Ranges are ascending and page-aligned at the block grain,
                    // but the sink end and a recalled-block start can land in the
                    // same page; dedup the boundary so a page is never doubled.
                    if out.last() != Some(&page) {
                        out.push(page);
                    }
                }
            }
        }
        out
    }
}
