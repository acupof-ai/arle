//! Metal backend executor (Apple Silicon) — the primary AI-PC backend.
//!
//! Scope of this crate at R2:
//! - [`MetalKvPool`] is a **complete, real** host-side page manager: it implements
//!   the host-indexed [`KvPool`] seam (page allocation, slot growth/truncation,
//!   prefix-share retain/release). Because the seam is host-only, none of this
//!   needs a device tensor — it is identical in spirit to any backend's pool.
//! - [`MetalExecutor`] implements the [`BackendExecutor`] seam plumbing
//!   (submit/poll overlap shape). The actual MLX forward + on-device KV buffers
//!   are wired in R3 (model port via `crates/mlx-sys`); until then `submit`
//!   runs a clearly-marked **placeholder** forward so the seam is testable.
//!
//! Nothing here references engine-core; this crate depends only on the stable
//! `infer-plan` + `infer-seam` contracts.

use std::collections::HashMap;
#[cfg(feature = "metal")]
use std::path::Path;

use infer_plan::{ForwardPlan, SlotToken, StepOutput};
use infer_seam::{BackendExecutor, KvPool, PollResult};

#[cfg(feature = "metal")]
mod config;
#[cfg(feature = "metal")]
mod loader;
#[cfg(feature = "metal")]
mod mlx;
#[cfg(feature = "metal")]
mod model_source;
#[cfg(feature = "metal")]
mod qwen35;
#[cfg(feature = "metal")]
mod weights;

#[cfg(feature = "metal")]
const KV_CACHE_CHUNK: i32 = 256;

/// Host-side paged KV bookkeeping for the Metal backend.
///
/// Pages are logical indices (`u32`); the device-side KV buffers they map to are
/// allocated by the MLX layer in R3. Page lifetime, slot growth, and the
/// prefix-cache retain/release protocol are fully handled here.
#[derive(Debug)]
pub struct MetalKvPool {
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

impl MetalKvPool {
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

impl KvPool for MetalKvPool {
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

    fn slot_epoch(&self, slot: usize) -> u64 {
        self.slot_epoch.get(slot).copied().unwrap_or(0)
    }

    fn append_pages_needed(&self, slot: usize, tokens: usize) -> usize {
        let have = self.slot_pages.get(slot).map_or(0, Vec::len);
        let after = self.pages_for_tokens(self.seq_len(slot) + tokens);
        after.saturating_sub(have)
    }

    fn alloc(&mut self, slot: usize, tokens: usize) -> anyhow::Result<()> {
        let need = self.append_pages_needed(slot, tokens);
        if need > self.free.len() {
            anyhow::bail!(
                "MetalKvPool out of pages: slot {slot} needs {need}, free {}",
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
                "MetalKvPool out of pages: detached request {pages}, free {}",
                self.free.len()
            );
        }
        Ok((0..pages)
            .map(|_| self.free.pop().expect("checked free >= pages"))
            .collect())
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

    fn migrate(&mut self, _slot: usize, _start: usize, _len: usize) -> anyhow::Result<()> {
        // Host page mapping is unchanged by migration; the device-buffer copy is
        // an MLX concern wired in R3. No-op at the host-indexing layer.
        Ok(())
    }

    fn retained_count(&self) -> usize {
        self.page_refs.values().filter(|&&c| c > 0).count()
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

    fn retain_pages(&mut self, pages: &[u32]) {
        for &page in pages {
            *self.page_refs.entry(page).or_insert(0) += 1;
        }
    }
}

/// In-flight handle for a submitted Metal step.
///
/// The R2 skeleton resolves synchronously, so this carries the resolved output.
/// R3 replaces this with an MLX async handle (command-buffer + future tokens)
/// to keep CPU scheduling overlapped with the GPU forward.
pub enum MetalInflight {
    /// CPU placeholder output.
    Ready(StepOutput),
    /// Real MLX greedy sample. `poll` materializes this scalar token.
    #[cfg(feature = "metal")]
    Sampled { slot: usize, sampled: mlx::MlxArray },
}

impl std::fmt::Debug for MetalInflight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready(output) => f.debug_tuple("Ready").field(output).finish(),
            #[cfg(feature = "metal")]
            Self::Sampled { slot, sampled } => f
                .debug_struct("Sampled")
                .field("slot", slot)
                .field("sampled", sampled)
                .finish(),
        }
    }
}

/// Metal backend executor.
///
/// `new()` keeps the R2 CPU placeholder for seam tests. `from_model_path()`
/// builds the R3a real MLX Qwen3.5 executor.
#[derive(Default)]
pub struct MetalExecutor {
    #[cfg(feature = "metal")]
    real: Option<RealMetalExecutor>,
}

impl std::fmt::Debug for MetalExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("MetalExecutor");
        #[cfg(feature = "metal")]
        debug.field("real", &self.real.is_some());
        debug.finish()
    }
}

impl MetalExecutor {
    /// Build a Metal executor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "metal")]
            real: None,
        }
    }

    /// Build a real single-row greedy MLX Qwen3.5 executor from a local model
    /// path or HuggingFace id.
    #[cfg(feature = "metal")]
    pub fn from_model_path(model_path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let model_source = model_path.as_ref().to_string_lossy();
        let resolved = model_source::resolve_model_path(&model_source)?;
        let _guard = mlx_sys::mlx_guard();
        let config = config::load_metal_config(&resolved)?;
        let weights = qwen35::load_qwen35_metal_weights(&resolved, &config)?;
        Ok(Self {
            real: Some(RealMetalExecutor {
                config,
                weights,
                slots: HashMap::new(),
                active_session_slot: None,
            }),
        })
    }

    /// Placeholder forward — produces one deterministic token per scheduled row.
    ///
    /// TODO(R3): replace with the real MLX forward via `crates/mlx-sys`
    /// (Qwen3.6 MoE step), reading KV pages from `kv.page_indices(slot)` and
    /// sampling real logits. This identity-ish stub exists only so the
    /// submit/poll seam is exercisable on CPU.
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

impl BackendExecutor for MetalExecutor {
    type Inflight = MetalInflight;

    fn submit(
        &mut self,
        plan: &ForwardPlan,
        kv: &mut dyn KvPool,
    ) -> anyhow::Result<Self::Inflight> {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_mut() {
            return real.submit(plan, kv);
        }
        #[cfg(not(feature = "metal"))]
        let _ = kv;

        Ok(MetalInflight::Ready(Self::placeholder_forward(plan)))
    }

    fn poll(&mut self, inflight: Self::Inflight) -> anyhow::Result<PollResult<Self::Inflight>> {
        match inflight {
            MetalInflight::Ready(output) => Ok(PollResult::Ready(output)),
            #[cfg(feature = "metal")]
            MetalInflight::Sampled { slot, sampled } => {
                let _guard = mlx_sys::mlx_guard();
                mlx::eval(&[&sampled]);
                let token = sampled.item_i32() as u32;
                Ok(PollResult::Ready(StepOutput {
                    tokens: vec![SlotToken {
                        slot,
                        token,
                        logprob: None,
                        finish: None,
                    }],
                }))
            }
        }
    }
}

#[cfg(feature = "metal")]
struct RealMetalExecutor {
    config: config::MetalModelConfig,
    weights: qwen35::Qwen35MetalWeights,
    slots: HashMap<usize, MetalSlotState>,
    active_session_slot: Option<usize>,
}

#[cfg(feature = "metal")]
impl RealMetalExecutor {
    fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> anyhow::Result<MetalInflight> {
        let _guard = mlx_sys::mlx_guard();
        let row_count = plan.prefill_rows.len() + plan.decode_rows.len();
        anyhow::ensure!(
            row_count == 1,
            "R3a MetalExecutor supports exactly one prefill or decode row, got {row_count}"
        );

        if let Some(row) = plan.prefill_rows.first() {
            return self.submit_prefill(row, kv);
        }
        if let Some(row) = plan.decode_rows.first() {
            return self.submit_decode(row, kv);
        }
        anyhow::bail!("R3a MetalExecutor received a non-idle plan with no rows")
    }

    fn submit_prefill(
        &mut self,
        row: &infer_plan::PrefillRow,
        kv: &mut dyn KvPool,
    ) -> anyhow::Result<MetalInflight> {
        anyhow::ensure!(
            row.start_pos == 0,
            "R3a MetalExecutor does not support prefix reuse or chunked prefill yet"
        );
        anyhow::ensure!(
            !row.tokens.is_empty(),
            "R3a MetalExecutor prefill row must contain at least one token"
        );
        if let Some(active) = self.active_session_slot {
            anyhow::ensure!(
                active == row.slot,
                "R3a scalar Qwen3.5 C++ sessions support only one active slot"
            );
        }

        self.reset_slot_if_epoch_changed(row.slot, kv)?;
        if !self.slots.contains_key(&row.slot) {
            let reservation = kv
                .seq_len(row.slot)
                .max(row.total_tokens.saturating_add(512))
                .max(row.tokens.len().saturating_add(1));
            self.slots.insert(
                row.slot,
                MetalSlotState::new(row.slot, kv.slot_epoch(row.slot), &self.config, reservation),
            );
        }

        let model = self.weights.cpp_model()?;
        let slot = self.slots.get_mut(&row.slot).expect("slot inserted above");
        slot.ensure_session_active(model)?;
        let token_values: Vec<i32> = row.tokens.iter().map(|&token| token as i32).collect();
        let token_arr = mlx::MlxArray::from_slice_i32(&token_values, &[token_values.len() as i32]);
        let logits = model.prefill_session(&token_arr, token_values.len() as i32, 0)?;
        mlx::async_eval(&[&logits]);
        slot.cache_len = row.tokens.len();

        let sampled = mlx::argmax(&logits);
        mlx::async_eval(&[&sampled]);
        self.active_session_slot = Some(row.slot);
        Ok(MetalInflight::Sampled {
            slot: row.slot,
            sampled,
        })
    }

    fn submit_decode(
        &mut self,
        row: &infer_plan::DecodeRow,
        kv: &mut dyn KvPool,
    ) -> anyhow::Result<MetalInflight> {
        if let Some(active) = self.active_session_slot {
            anyhow::ensure!(
                active == row.slot,
                "R3a scalar Qwen3.5 C++ sessions support only one active slot"
            );
        }
        self.reset_slot_if_epoch_changed(row.slot, kv)?;
        let model = self.weights.cpp_model()?;
        let slot = self
            .slots
            .get_mut(&row.slot)
            .ok_or_else(|| anyhow::anyhow!("decode for slot {} before prefill", row.slot))?;
        anyhow::ensure!(
            row.kv_seq_len == slot.cache_len,
            "decode kv_seq_len mismatch for slot {}: plan={}, metal_state={}",
            row.slot,
            row.kv_seq_len,
            slot.cache_len
        );
        slot.ensure_session_active(model)?;
        let token_arr = mlx::MlxArray::from_slice_i32(&[row.last_token as i32], &[1]);
        let logits = model.step_session(&token_arr, slot.cache_len as i32)?;
        mlx::async_eval(&[&logits]);
        slot.cache_len = slot.cache_len.saturating_add(1);

        let sampled = mlx::argmax(&logits);
        mlx::async_eval(&[&sampled]);
        self.active_session_slot = Some(row.slot);
        Ok(MetalInflight::Sampled {
            slot: row.slot,
            sampled,
        })
    }

    fn reset_slot_if_epoch_changed(&mut self, slot: usize, kv: &dyn KvPool) -> anyhow::Result<()> {
        let epoch = kv.slot_epoch(slot);
        let stale = self
            .slots
            .get(&slot)
            .is_some_and(|state| state.slot_epoch != epoch);
        if stale {
            // TODO(R3b): replace this host-epoch observation with an explicit
            // executor slot-release callback owned by the seam.
            if let Some(mut state) = self.slots.remove(&slot)
                && state.session_active
            {
                let model = self.weights.cpp_model()?;
                state.drain_session(model)?;
            }
            if self.active_session_slot == Some(slot) {
                self.active_session_slot = None;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "metal")]
struct MetalSlotState {
    _slot: usize,
    slot_epoch: u64,
    cache_len: usize,
    kv_flat: Vec<mlx::MlxArray>,
    gdr_flat: Vec<mlx::MlxArray>,
    session_active: bool,
}

#[cfg(feature = "metal")]
impl MetalSlotState {
    fn new(
        slot: usize,
        slot_epoch: u64,
        config: &config::MetalModelConfig,
        capacity_tokens: usize,
    ) -> Self {
        let capacity = round_up_capacity(capacity_tokens);
        let cache_shape = [
            1,
            config.num_key_value_heads as i32,
            capacity,
            config.head_dim as i32,
        ];
        let mut kv_flat = Vec::with_capacity(config.arch.num_full_attention_layers() * 2);
        for _ in 0..config.arch.num_full_attention_layers() {
            kv_flat.push(mlx::zeros(&cache_shape, mlx::Dtype::Bfloat16));
            kv_flat.push(mlx::zeros(&cache_shape, mlx::Dtype::Bfloat16));
        }

        let mut gdr_flat = Vec::with_capacity(config.arch.num_linear_attention_layers() * 2);
        for _ in 0..config.arch.num_linear_attention_layers() {
            gdr_flat.push(mlx::zeros(
                &[
                    1,
                    config.arch.linear.num_value_heads as i32,
                    config.arch.linear.value_dim as i32,
                    config.arch.linear.key_dim as i32,
                ],
                mlx::Dtype::Float32,
            ));
            gdr_flat.push(mlx::zeros(
                &[
                    1,
                    (config.arch.linear.conv_kernel - 1) as i32,
                    config.arch.linear.qkv_dim() as i32,
                ],
                mlx::Dtype::Bfloat16,
            ));
        }

        Self {
            _slot: slot,
            slot_epoch,
            cache_len: 0,
            kv_flat,
            gdr_flat,
            session_active: false,
        }
    }

    fn ensure_session_active(&mut self, model: &qwen35::CppQwen35Model) -> anyhow::Result<()> {
        if self.session_active {
            return Ok(());
        }
        model.begin_session(&self.kv_flat, &self.gdr_flat)?;
        self.session_active = true;
        Ok(())
    }

    fn drain_session(&mut self, model: &qwen35::CppQwen35Model) -> anyhow::Result<()> {
        if !self.session_active {
            return Ok(());
        }
        let (kv_flat, gdr_flat) = model.end_session(self.kv_flat.len(), self.gdr_flat.len())?;
        self.kv_flat = kv_flat;
        self.gdr_flat = gdr_flat;
        self.session_active = false;
        Ok(())
    }
}

#[cfg(feature = "metal")]
fn round_up_capacity(tokens: usize) -> i32 {
    let tokens = tokens.max(1) as i32;
    ((tokens + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK) * KV_CACHE_CHUNK
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_plan::{DecodeRow, ForwardMode, PrefillRow};

    #[test]
    fn kvpool_alloc_grows_and_free_returns_pages() {
        let mut pool = MetalKvPool::new(2, 8, 16);
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
        let mut pool = MetalKvPool::new(1, 1, 16);
        assert!(pool.alloc(0, 17).is_err()); // needs 2 pages, only 1
        assert_eq!(pool.seq_len(0), 0);
    }

    #[test]
    fn kvpool_retained_pages_survive_free_slot_then_release() {
        let mut pool = MetalKvPool::new(2, 8, 16);
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
        let mut pool = MetalKvPool::new(1, 8, 16);
        pool.alloc(0, 48).unwrap(); // 3 pages
        assert_eq!(pool.page_indices(0).len(), 3);
        pool.truncate_slot(0, 16).unwrap(); // keep 1 page
        assert_eq!(pool.page_indices(0).len(), 1);
        assert_eq!(pool.seq_len(0), 16);
        assert_eq!(pool.free_pages(), 7);
    }

    #[test]
    fn executor_decode_plumbing_returns_one_token_per_row() {
        let mut exec = MetalExecutor::new();
        let mut pool = MetalKvPool::new(2, 8, 16);
        let plan = ForwardPlan {
            mode: ForwardMode::Decode,
            decode_rows: vec![
                DecodeRow {
                    slot: 0,
                    last_token: 10,
                    kv_seq_len: 4,
                },
                DecodeRow {
                    slot: 1,
                    last_token: 20,
                    kv_seq_len: 7,
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
        let mut exec = MetalExecutor::new();
        let mut pool = MetalKvPool::new(1, 8, 16);
        let plan = ForwardPlan {
            mode: ForwardMode::Prefill,
            decode_rows: Vec::new(),
            prefill_rows: vec![PrefillRow {
                slot: 0,
                tokens: vec![1, 2, 3],
                start_pos: 0,
                total_tokens: 3,
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
