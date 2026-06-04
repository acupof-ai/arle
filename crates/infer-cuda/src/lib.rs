//! CUDA backend executor.
//!
//! [`CudaKvPool`] is the host-side page manager implementing the host-only
//! [`KvPool`] seam (alloc/grow/truncate + prefix retain/release), structurally
//! identical to [`infer_metal::MetalKvPool`]. [`CudaExecutor`] implements
//! [`BackendExecutor`]: a CPU-testable placeholder without `cuda`, the real
//! cuda-kernels path with it. Scope: dense BF16 Qwen3, safetensors, single
//! scheduled row, device argmax greedy / host temperature sampling.
//!
//! Depends only on `infer-plan` + `infer-seam`, never engine-core.

use std::collections::HashMap;
use std::fmt;
#[cfg(feature = "cuda")]
use std::path::Path;

use infer_plan::{ForwardPlan, SlotToken, StepOutput};
use infer_seam::{BackendExecutor, KvAllocator, KvPool, KvPrefixStore, KvQuery, PollResult};

#[cfg(feature = "cuda")]
mod attention;
#[cfg(feature = "cuda")]
mod decode_graph;
// Not cuda-gated: pure host capture-key math, CPU-testable without nvcc.
mod decode_graph_key;
// DSv4-Flash FP8 model (loader + structs + MLA KV arena). cuda-gated: holds
// device weight matrices + the shared DSv4 FP8 DeepGEMM caches.
#[cfg(feature = "cuda")]
mod dsv4;
#[cfg(feature = "cuda")]
mod executor;
#[cfg(feature = "cuda")]
pub mod graph;
// DSv4 hyper-connections (`hc_mult > 1`): the wide residual stream wrap. cuda-
// gated (device kernels + DSv4 weight matrices).
#[cfg(feature = "cuda")]
mod hc;
#[cfg(feature = "cuda")]
mod loader;
#[cfg(feature = "cuda")]
mod model;
#[cfg(feature = "cuda")]
mod ops;

// Not cuda-gated: env→TpConfig resolution is CPU-testable; only the NCCL comm
// variant is feature-gated.
pub mod tp;

// Not cuda-gated: pure-CPU per-rank weight-shard byte slicing; the device upload
// that consumes it stays in `loader`.
pub mod shard_slice;

// Not cuda-gated: Qwen35Config → infer_moe::MoeConfig bridge + per-rank expert
// split arithmetic.
pub mod moe_config;

// Not cuda-gated: the host route→assignment flattening is CPU-tested; the device
// `moe_forward` lives in the inner `cuda`-gated module.
mod moe;

/// Host-side paged KV bookkeeping for the CUDA backend.
///
/// Pages are logical `u32` indices; the device-side KV buffers they map to are
/// allocated by the cuda-kernels layer.
#[derive(Debug)]
pub struct CudaKvPool {
    page_size: usize,
    total_pages: usize,
    /// Free page ids, used as a LIFO stack.
    free: Vec<u32>,
    /// Per-slot page ids in logical order.
    slot_pages: Vec<Vec<u32>>,
    /// Per-slot logical token length.
    slot_len: Vec<usize>,
    /// Per-slot occupant epoch (bumped on free/attach).
    slot_epoch: Vec<u64>,
    /// Ref counts for pages retained by an external owner (e.g. the prefix cache).
    /// A page with a positive ref count survives `free_slot`.
    page_refs: HashMap<u32, u32>,
}

impl CudaKvPool {
    /// Build a pool with `num_slots` logical slots and `total_pages` physical pages.
    #[must_use]
    pub fn new(num_slots: usize, total_pages: usize, page_size: usize) -> Self {
        let page_size = page_size.max(1);
        // LIFO stack: pop yields ascending ids first.
        let free: Vec<u32> = (0..total_pages as u32).rev().collect();
        Self {
            page_size,
            total_pages,
            free,
            slot_pages: vec![Vec::new(); num_slots],
            slot_len: vec![0; num_slots],
            slot_epoch: vec![0; num_slots],
            page_refs: HashMap::new(),
        }
    }

    fn pages_for_tokens(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.page_size)
    }
}

impl KvQuery for CudaKvPool {
    fn is_active(&self) -> bool {
        self.total_pages > 0
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn free_pages(&self) -> usize {
        self.free.len()
    }

    fn free_tokens(&self) -> usize {
        self.free.len() * self.page_size
    }

    fn seq_len(&self, slot: usize) -> usize {
        self.slot_len.get(slot).copied().unwrap_or(0)
    }

    fn slot_epoch(&self, slot: usize) -> u64 {
        self.slot_epoch.get(slot).copied().unwrap_or(0)
    }

    fn append_pages_needed(&self, slot: usize, tokens: usize) -> usize {
        let have = self.slot_pages.get(slot).map_or(0, Vec::len);
        let after = self.pages_for_tokens(self.seq_len(slot) + tokens);
        after.saturating_sub(have)
    }

    fn page_indices(&self, slot: usize) -> &[u32] {
        self.slot_pages.get(slot).map_or(&[], Vec::as_slice)
    }

    fn page_indices_for_token_range(&self, slot: usize, start: usize, len: usize) -> &[u32] {
        let Some(pages) = self.slot_pages.get(slot) else {
            return &[];
        };
        let start_page = start / self.page_size;
        let end_page = (start + len).div_ceil(self.page_size).min(pages.len());
        if start_page >= end_page {
            return &[];
        }
        &pages[start_page..end_page]
    }
}

impl KvAllocator for CudaKvPool {
    fn alloc(&mut self, slot: usize, tokens: usize) -> anyhow::Result<()> {
        let need = self.append_pages_needed(slot, tokens);
        if need > self.free.len() {
            anyhow::bail!(
                "CudaKvPool out of pages: slot {slot} needs {need}, free {}",
                self.free.len()
            );
        }
        for _ in 0..need {
            let page = self.free.pop().expect("checked free >= need");
            self.slot_pages[slot].push(page);
        }
        self.slot_len[slot] += tokens;
        Ok(())
    }

    fn alloc_detached_pages(&mut self, pages: usize) -> anyhow::Result<Vec<u32>> {
        if pages > self.free.len() {
            anyhow::bail!(
                "CudaKvPool out of pages: detached request {pages}, free {}",
                self.free.len()
            );
        }
        Ok((0..pages)
            .map(|_| self.free.pop().expect("checked free >= pages"))
            .collect())
    }

    fn free_slot(&mut self, slot: usize) {
        let Some(pages) = self.slot_pages.get_mut(slot) else {
            return;
        };
        let taken = std::mem::take(pages);
        for page in taken {
            // Retained pages (held by the prefix cache) survive the slot's release.
            if self.page_refs.get(&page).copied().unwrap_or(0) == 0 {
                self.free.push(page);
            }
        }
        self.slot_len[slot] = 0;
        self.slot_epoch[slot] = self.slot_epoch[slot].wrapping_add(1);
    }

    fn truncate_slot(&mut self, slot: usize, new_len: usize) -> anyhow::Result<()> {
        let keep_pages = self.pages_for_tokens(new_len);
        let pages = self
            .slot_pages
            .get_mut(slot)
            .ok_or_else(|| anyhow::anyhow!("truncate_slot: slot {slot} out of range"))?;
        let cut = keep_pages.min(pages.len());
        let removed: Vec<u32> = pages.split_off(cut);
        for page in removed {
            // Only physically free pages not retained by a prefix owner.
            if self.page_refs.get(&page).copied().unwrap_or(0) == 0 {
                self.free.push(page);
            }
        }
        self.slot_len[slot] = new_len;
        Ok(())
    }

    fn migrate(&mut self, _slot: usize, _start: usize, _len: usize) -> anyhow::Result<()> {
        // No-op: migration leaves the host page mapping unchanged; the
        // device-buffer copy is a cuda-kernels concern.
        Ok(())
    }
}

impl KvPrefixStore for CudaKvPool {
    fn retain_pages(&mut self, pages: &[u32]) {
        for &page in pages {
            *self.page_refs.entry(page).or_insert(0) += 1;
        }
    }

    fn release_pages(&mut self, pages: &[u32]) {
        for &page in pages {
            if let Some(c) = self.page_refs.get_mut(&page) {
                *c = c.saturating_sub(1);
                if *c == 0 {
                    self.page_refs.remove(&page);
                    self.free.push(page);
                }
            }
        }
    }

    fn retained_count(&self) -> usize {
        self.page_refs.values().filter(|&&c| c > 0).count()
    }

    fn attach_pages(
        &mut self,
        slot: usize,
        pages: &[u32],
        token_count: usize,
    ) -> anyhow::Result<()> {
        // Prefix-reuse: a fresh slot adopts already-allocated (retained) pages.
        let dst = self
            .slot_pages
            .get_mut(slot)
            .ok_or_else(|| anyhow::anyhow!("attach_pages: slot {slot} out of range"))?;
        dst.extend_from_slice(pages);
        self.slot_len[slot] = self.slot_len[slot].max(token_count);
        self.slot_epoch[slot] = self.slot_epoch[slot].wrapping_add(1);
        Ok(())
    }
}

/// In-flight handle for a submitted CUDA step. Resolves synchronously today.
#[derive(Debug)]
pub struct CudaInflight {
    output: StepOutput,
}

/// CUDA backend executor.
///
/// `new()` is the no-GPU placeholder for host tests;
/// [`CudaExecutor::from_qwen3_bf16_safetensors`] (feature `cuda`) is the real
/// dense BF16 Qwen3 path.
#[derive(Default)]
pub struct CudaExecutor {
    inner: CudaExecutorInner,
}

#[derive(Default)]
enum CudaExecutorInner {
    #[default]
    Placeholder,
    #[cfg(feature = "cuda")]
    Real(Box<executor::RealCudaExecutor>),
}

impl fmt::Debug for CudaExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            CudaExecutorInner::Placeholder => f
                .debug_struct("CudaExecutor")
                .field("inner", &"placeholder")
                .finish(),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => {
                f.debug_struct("CudaExecutor").field("inner", real).finish()
            }
        }
    }
}

impl CudaExecutor {
    /// Build a CUDA executor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: CudaExecutorInner::Placeholder,
        }
    }

    /// Build the real CUDA executor for dense BF16 Qwen3 safetensors.
    ///
    /// `total_pages` must match the host [`CudaKvPool`] so device page
    /// allocation mirrors host logical pages.
    #[cfg(feature = "cuda")]
    pub fn from_qwen3_bf16_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            inner: CudaExecutorInner::Real(Box::new(
                executor::RealCudaExecutor::from_qwen3_bf16_safetensors(
                    model_path,
                    num_slots,
                    total_pages,
                )?,
            )),
        })
    }

    /// Build the real CUDA executor for a single-GPU BF16 Qwen3.5/3.6 MoE
    /// checkpoint (all experts local). BF16 only; the W4/4-bit canonical needs
    /// the W4 grouped-GEMM follow-up.
    #[cfg(feature = "cuda")]
    pub fn from_qwen35_moe_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            inner: CudaExecutorInner::Real(Box::new(
                executor::RealCudaExecutor::from_qwen35_moe_safetensors(
                    model_path,
                    num_slots,
                    total_pages,
                )?,
            )),
        })
    }

    /// Build the real CUDA executor for a DSv4-Flash FP8 checkpoint (MLA + HC +
    /// FP8 DeepGEMM MoE). DSv4 is multi-GPU only (TP=8/EP=8): the per-rank EP
    /// expert split + NCCL TP groups resolve from the env (`INFER_NCCL_UNIQUE_ID`,
    /// `INFER_CUDA_DEVICES`/world-size), so the launcher binds one rank per GPU.
    /// DSv4 owns its MLA KV state inside the forward, so no `total_pages`/
    /// `CudaKvPool` page budget is needed (a host pool is still attached for slot
    /// bookkeeping).
    #[cfg(feature = "cuda")]
    pub fn from_dsv4_fp8_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            inner: CudaExecutorInner::Real(Box::new(
                executor::RealCudaExecutor::from_dsv4_fp8_safetensors(model_path, num_slots)?,
            )),
        })
    }

    /// Placeholder forward — produces one deterministic token per scheduled row.
    fn placeholder_forward(plan: &ForwardPlan) -> StepOutput {
        let mut tokens = Vec::with_capacity(plan.decode_rows.len() + plan.prefill_rows.len());
        for row in &plan.decode_rows {
            tokens.push(SlotToken {
                slot: row.slot,
                token: row.last_token.wrapping_add(1),
                logprob: None,
                finish: None,
            });
        }
        for row in &plan.prefill_rows {
            let token = row.tokens.last().copied().unwrap_or(0).wrapping_add(1);
            tokens.push(SlotToken {
                slot: row.slot,
                token,
                logprob: None,
                finish: None,
            });
        }
        StepOutput { tokens }
    }
}

impl BackendExecutor for CudaExecutor {
    type Inflight = CudaInflight;

    fn submit(
        &mut self,
        plan: &ForwardPlan,
        _kv: &mut dyn KvPool,
    ) -> anyhow::Result<Self::Inflight> {
        let output = match &mut self.inner {
            CudaExecutorInner::Placeholder => Self::placeholder_forward(plan),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.submit(plan, _kv)?,
        };
        Ok(CudaInflight { output })
    }

    fn poll(&mut self, inflight: Self::Inflight) -> anyhow::Result<PollResult<Self::Inflight>> {
        Ok(PollResult::Ready(inflight.output))
    }

    fn warmup(&mut self) -> anyhow::Result<()> {
        match &mut self.inner {
            CudaExecutorInner::Placeholder => Ok(()),
            #[cfg(feature = "cuda")]
            CudaExecutorInner::Real(real) => real.warmup(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_plan::{DecodeRow, ForwardMode, PrefillRow};

    #[test]
    fn kvpool_alloc_grows_and_free_returns_pages() {
        let mut pool = CudaKvPool::new(2, 8, 16);
        assert_eq!(pool.free_pages(), 8);
        pool.alloc(0, 16).unwrap(); // exactly 1 page
        assert_eq!(pool.seq_len(0), 16);
        assert_eq!(pool.page_indices(0).len(), 1);
        assert_eq!(pool.free_pages(), 7);
        pool.alloc(0, 1).unwrap(); // crosses into a 2nd page
        assert_eq!(pool.page_indices(0).len(), 2);
        pool.free_slot(0);
        assert_eq!(pool.seq_len(0), 0);
        assert_eq!(pool.free_pages(), 8);
    }

    #[test]
    fn kvpool_out_of_pages_errors() {
        let mut pool = CudaKvPool::new(1, 1, 16);
        assert!(pool.alloc(0, 17).is_err()); // needs 2 pages, only 1
        assert_eq!(pool.seq_len(0), 0);
    }

    #[test]
    fn kvpool_retained_pages_survive_free_slot_then_release() {
        let mut pool = CudaKvPool::new(2, 8, 16);
        pool.alloc(0, 32).unwrap(); // 2 pages
        let prefix: Vec<u32> = pool.page_indices(0).to_vec();
        pool.retain_pages(&prefix);
        assert_eq!(pool.retained_count(), 2);
        let free_before = pool.free_pages();
        pool.free_slot(0); // retained pages must NOT return to free
        assert_eq!(pool.free_pages(), free_before);
        // A fresh slot can adopt the retained prefix pages.
        pool.attach_pages(1, &prefix, 32).unwrap();
        assert_eq!(pool.page_indices(1), prefix.as_slice());
        assert_eq!(pool.seq_len(1), 32);
        // Releasing returns them to the free pool once unreferenced.
        pool.free_slot(1);
        pool.release_pages(&prefix);
        assert_eq!(pool.retained_count(), 0);
        assert_eq!(pool.free_pages(), 8);
    }

    #[test]
    fn kvpool_truncate_frees_tail_pages() {
        let mut pool = CudaKvPool::new(1, 8, 16);
        pool.alloc(0, 48).unwrap(); // 3 pages
        assert_eq!(pool.page_indices(0).len(), 3);
        pool.truncate_slot(0, 16).unwrap(); // keep 1 page
        assert_eq!(pool.page_indices(0).len(), 1);
        assert_eq!(pool.seq_len(0), 16);
        assert_eq!(pool.free_pages(), 7);
    }

    #[test]
    fn executor_decode_plumbing_returns_one_token_per_row() {
        let mut exec = CudaExecutor::new();
        let mut pool = CudaKvPool::new(2, 8, 16);
        let plan = ForwardPlan {
            mode: ForwardMode::Decode,
            decode_rows: vec![
                DecodeRow {
                    slot: 0,
                    last_token: 10,
                    kv_seq_len: 4,
                    params: infer_plan::SamplingParams::default(),
                },
                DecodeRow {
                    slot: 1,
                    last_token: 20,
                    kv_seq_len: 7,
                    params: infer_plan::SamplingParams::default(),
                },
            ],
            prefill_rows: Vec::new(),
            microbatch: None,
            spec: None,
        };
        let inflight = exec.submit(&plan, &mut pool).unwrap();
        match exec.poll(inflight).unwrap() {
            PollResult::Ready(out) => {
                assert_eq!(out.tokens.len(), 2);
                assert_eq!(out.tokens[0].token, 11);
                assert_eq!(out.tokens[1].token, 21);
            }
            PollResult::NotReady(_) => panic!("skeleton resolves synchronously"),
        }
    }

    #[test]
    fn executor_prefill_plumbing_returns_completion_token() {
        let mut exec = CudaExecutor::new();
        let mut pool = CudaKvPool::new(1, 8, 16);
        let plan = ForwardPlan {
            mode: ForwardMode::Prefill,
            decode_rows: Vec::new(),
            prefill_rows: vec![PrefillRow {
                slot: 0,
                tokens: vec![1, 2, 3],
                start_pos: 0,
                total_tokens: 3,
                params: infer_plan::SamplingParams::default(),
            }],
            microbatch: None,
            spec: None,
        };
        let inflight = exec.submit(&plan, &mut pool).unwrap();
        match exec.poll(inflight).unwrap() {
            PollResult::Ready(out) => {
                assert_eq!(out.tokens.len(), 1);
                assert_eq!(out.tokens[0].slot, 0);
                assert_eq!(out.tokens[0].token, 4); // last prompt token (3) + 1
            }
            PollResult::NotReady(_) => panic!("skeleton resolves synchronously"),
        }
    }
}
