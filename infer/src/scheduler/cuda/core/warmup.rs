//! CUDA-graph + cublasLt heuristic warmup methods on `Scheduler<M>`.
//!
//! Split out of `core.rs` (pure structural refactor — no behavior change).

use log::{error, info};

use super::super::Scheduler;
use crate::model::{GenerationState, ModelForward};

impl<M: ModelForward> Scheduler<M> {
    /// Pre-capture CUDA Graphs for batched decode at common batch sizes.
    ///
    /// Uses SGLang-style batch size schedule: 1, 2, 4, 8, 12, 16, 24, 32, 40, ...
    /// up to min(num_slots, 256). This covers the most common concurrent request
    /// counts without capturing every single size.
    ///
    /// Two-pass warmup:
    /// 1. Pass 1 drives forward_decode_batch per batch size, which populates the
    ///    cublasLt heuristic algo cache for every shape. In graph-capture mode
    ///    it also records a graph per batch size.
    /// 2. `autotune_all_cached_gemms_cuda` benchmarks all heuristic candidates and
    ///    replaces each shape's algo with the measured-fastest one.
    /// 3. Pass 2 (graph-capture mode only) re-captures graphs with the autotuned
    ///    algorithms. Eager decode (e.g. LoRA) skips pass 2 since no graphs
    ///    were cached.
    pub(in crate::scheduler::cuda) fn warmup_cuda_graphs(&mut self) {
        let num_slots = self.states.len();
        if !self.paged_kv_pool.is_active() {
            return;
        }

        let graph_capture_enabled = self.model.supports_cuda_graph_decode();
        let max_bs = num_slots.min(256);
        let warmup_sizes = Self::cuda_graph_batch_sizes(max_bs);

        if graph_capture_enabled {
            info!(
                "Warming up CUDA Graphs for {} batch sizes (max {})...",
                warmup_sizes.len(),
                max_bs,
            );
        } else {
            info!(
                "Graph capture disabled (eager decode, e.g. LoRA); running \
                 eager warmup + cublasLt autotune for {} batch sizes (max {})...",
                warmup_sizes.len(),
                max_bs,
            );
        }
        let t0 = std::time::Instant::now();

        // Track how many slots we actually allocated so any early exit below
        // still frees them in the cleanup loop. Previously, a failing
        // `alloc_tokens` or `create_decode_context` would `return` with slots
        // still holding warmup tokens — `free_slots()` would then consider
        // them free while the pool still had dirty state, and the first real
        // request could inherit stale paged-KV entries.
        let mut allocated: usize = 0;
        let mut warmed: usize = 0;
        debug_assert!(
            self.paged_kv_pool.page_size > 0,
            "paged KV pool page size must be non-zero"
        );

        'warmup: {
            for slot in 0..max_bs {
                if let Err(e) = self.paged_kv_pool.alloc_tokens(slot, 1) {
                    error!("Warmup: pool alloc for slot {} failed: {}", slot, e);
                    break 'warmup;
                }
                allocated = slot + 1;
            }

            // Lazy-init decode context before warmup.
            if self.decode_bufs.is_none() {
                match self.model.create_decode_context(
                    self.states.len(),
                    self.effective_max_seq_len,
                    &self.paged_kv_pool,
                ) {
                    Ok(ctx) => self.decode_bufs = Some(ctx),
                    Err(e) => {
                        error!("Warmup: failed to create decode context: {}", e);
                        break 'warmup;
                    }
                }
            }

            let dummy_tokens: Vec<u32> = vec![0; max_bs];
            let slot_indices: Vec<usize> = (0..max_bs).collect();

            // Pass 1: drive forward for each warmup batch size. Populates the
            // cublasLt heuristic algo cache for all GEMM shapes used by decode.
            // In graph-capture mode, also captures a graph per batch size.
            warmed = self.warmup_graphs_pass(&warmup_sizes, &dummy_tokens, &slot_indices);

            // Autotune: benchmark all heuristic candidates, replace with measured best.
            // Runs regardless of graph mode so eager LoRA decode lands on the same
            // tuned algorithms as graph-mode decode.
            if warmed > 0 {
                info!("Autotuning cublasLt GEMM algorithms ({} shapes)...", warmed);
                let t_at = std::time::Instant::now();
                unsafe {
                    cuda_kernels::ffi::autotune_all_cached_gemms_cuda(
                        self.model.device_context().stream.cu_stream(),
                    );
                }
                info!(
                    "cublasLt autotune done in {:.0}ms",
                    t_at.elapsed().as_secs_f64() * 1e3,
                );

                if graph_capture_enabled {
                    // Invalidate graphs captured with heuristic algos.
                    {
                        use crate::model::DecodeContextOps;
                        let decode_ctx = self
                            .decode_bufs
                            .as_mut()
                            .expect("invariant: decode_bufs initialized above");
                        for &bs in &warmup_sizes[..warmed] {
                            decode_ctx.invalidate_graph_cache(bs);
                        }
                    }

                    // Pass 2: re-capture with autotuned algorithms.
                    let recaptured =
                        self.warmup_graphs_pass(&warmup_sizes, &dummy_tokens, &slot_indices);
                    info!(
                        "Re-captured {} graphs with autotuned GEMM algorithms",
                        recaptured,
                    );
                }
            }
        }

        // Always reached: frees any slots the warmup body allocated, whether
        // the body ran to completion or bailed on an error above.
        for slot in 0..allocated {
            self.paged_kv_pool.free_slot(slot);
            let _ = self.states[slot].reset();
        }

        let mode = if graph_capture_enabled {
            "CUDA Graph warmup"
        } else {
            "Eager warmup + cublasLt autotune"
        };
        info!(
            "{} done in {:.0}ms ({} batch sizes, max {})",
            mode,
            t0.elapsed().as_secs_f64() * 1e3,
            warmed,
            warmup_sizes.last().copied().unwrap_or(0),
        );
    }

    /// Single pass of graph warmup: set up metadata and forward for each batch size.
    fn warmup_graphs_pass(
        &mut self,
        warmup_sizes: &[usize],
        dummy_tokens: &[u32],
        slot_indices: &[usize],
    ) -> usize {
        let mut captured = 0;
        for &bs in warmup_sizes {
            let tokens = &dummy_tokens[..bs];
            let si = &slot_indices[..bs];
            let page_size = self.paged_kv_pool.page_size;
            let decode_ctx = self
                .decode_bufs
                .as_mut()
                .expect("invariant: decode_bufs initialized in warmup block above");

            {
                use crate::model::DecodeContextOps;
                let ctx = self.model.device_context();
                decode_ctx.set_batch_size(bs);
                if let Err(e) = decode_ctx.upload_token_ids(ctx, tokens) {
                    info!(
                        "Warmup: upload_token_ids for B={} failed ({}), skipping",
                        bs, e
                    );
                    break;
                }
                match decode_ctx.update_metadata(ctx, &self.paged_kv_pool, si) {
                    Ok(reallocated) => {
                        if reallocated {
                            decode_ctx.invalidate_graph_cache(bs);
                        }
                    }
                    Err(e) => {
                        info!(
                            "Warmup: update_metadata for B={} failed ({}), skipping",
                            bs, e
                        );
                        break;
                    }
                }
                if let Err(e) = decode_ctx.plan_attention(
                    ctx,
                    bs,
                    self.model.num_q_heads(),
                    self.model.num_kv_heads(),
                    page_size,
                    self.model.head_dim(),
                    self.paged_kv_pool.format,
                ) {
                    info!(
                        "Warmup: plan_attention for B={} failed ({}), skipping",
                        bs, e
                    );
                    break;
                }
            }

            if let Err(e) = self.model.forward_decode_batch(
                tokens,
                &mut self.states,
                si,
                Some(&mut self.paged_kv_pool),
                decode_ctx,
                false,
            ) {
                info!(
                    "Warmup: graph capture for B={} failed ({}), skipping larger sizes",
                    bs, e
                );
                break;
            }
            let _ = self.model.device_context().sync();
            captured += 1;
        }
        captured
    }

    /// Generate batch size schedule for CUDA Graph warmup.
    ///
    /// Warm up EVERY batch size from 1..=min(max_bs, 64). This eliminates
    /// graph-miss eager fallbacks when the batch composition changes during
    /// request transitions, which was the primary source of p99 ITL spikes
    /// (100-150ms outliers at B=16).
    ///
    /// Beyond 64 we use a sparse schedule (step by 16) since the marginal
    /// difference between B=65 and B=64 graphs is negligible.
    fn cuda_graph_batch_sizes(max_bs: usize) -> Vec<usize> {
        let mut sizes = Vec::new();
        // Dense: every size from 1 to min(64, max_bs)
        let dense_limit = 64.min(max_bs);
        for bs in 1..=dense_limit {
            sizes.push(bs);
        }
        // Sparse: from 80 onward, step by 16
        let mut bs = 80;
        while bs <= max_bs {
            sizes.push(bs);
            bs += 16;
        }
        // Ensure max_bs itself is included
        if sizes.last() != Some(&max_bs) && max_bs > 8 {
            sizes.push(max_bs);
        }
        sizes
    }
}
