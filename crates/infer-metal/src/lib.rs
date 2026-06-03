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
mod wired_limit;

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
        if let Some(limit) = wired_limit::auto_wired_limit_bytes(&resolved) {
            let previous = mlx::set_wired_limit_bytes(limit as u64);
            log::info!(
                "Metal executor wired limit set to {} bytes (previous {})",
                limit,
                previous
            );
        }
        let config = config::load_metal_config(&resolved)?;
        let weights = qwen35::load_qwen35_metal_weights(&resolved, &config)?;
        Ok(Self {
            real: Some(RealMetalExecutor {
                config,
                weights,
                slots: HashMap::new(),
                page_store: MetalPageStore::default(),
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
    page_store: MetalPageStore,
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
            !row.tokens.is_empty(),
            "MetalExecutor prefill row must contain at least one token"
        );
        self.ensure_no_other_active_session(row.slot)?;

        self.reset_slot_if_epoch_changed(row.slot, kv)?;
        if !self.slots.contains_key(&row.slot) {
            let reservation = kv
                .seq_len(row.slot)
                .max(row.total_tokens.saturating_add(512))
                .max(row.tokens.len().saturating_add(1));
            let state = if row.start_pos == 0 {
                MetalSlotState::new(row.slot, kv.slot_epoch(row.slot), &self.config, reservation)
            } else {
                self.page_store.materialize_slot_from_prefix(
                    row.slot,
                    kv.slot_epoch(row.slot),
                    kv,
                    row.start_pos,
                    reservation,
                )?
            };
            self.slots.insert(row.slot, state);
        }

        let model = self.weights.cpp_model()?;
        let slot = self.slots.get_mut(&row.slot).expect("slot inserted above");
        anyhow::ensure!(
            row.start_pos == slot.cache_len,
            "prefill start_pos mismatch for slot {}: plan={}, metal_state={}",
            row.slot,
            row.start_pos,
            slot.cache_len
        );
        slot.ensure_session_active(model)?;
        self.active_session_slot = Some(row.slot);
        let token_values: Vec<i32> = row.tokens.iter().map(|&token| token as i32).collect();
        let token_arr = mlx::MlxArray::from_slice_i32(&token_values, &[token_values.len() as i32]);
        let logits =
            model.prefill_session(&token_arr, token_values.len() as i32, row.start_pos as i32)?;
        mlx::async_eval(&[&logits]);
        slot.cache_len = row.start_pos + row.tokens.len();
        slot.drain_session(model)?;
        self.active_session_slot = None;
        self.page_store.publish_slot(slot, kv)?;

        let sampled = mlx::argmax(&logits);
        mlx::async_eval(&[&sampled]);
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
        self.ensure_no_other_active_session(row.slot)?;
        self.reset_slot_if_epoch_changed(row.slot, kv)?;
        let model = self.weights.cpp_model()?;
        if !self.slots.contains_key(&row.slot) {
            anyhow::ensure!(
                row.kv_seq_len > 0,
                "decode for slot {} before prefill with empty host prefix",
                row.slot
            );
            let reservation = kv.seq_len(row.slot).max(row.kv_seq_len.saturating_add(512));
            let state = self.page_store.materialize_slot_from_prefix(
                row.slot,
                kv.slot_epoch(row.slot),
                kv,
                row.kv_seq_len,
                reservation,
            )?;
            self.slots.insert(row.slot, state);
        }
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
        self.active_session_slot = Some(row.slot);
        let token_arr = mlx::MlxArray::from_slice_i32(&[row.last_token as i32], &[1]);
        let logits = model.step_session(&token_arr, slot.cache_len as i32)?;
        mlx::async_eval(&[&logits]);
        slot.cache_len = slot.cache_len.saturating_add(1);
        slot.drain_session(model)?;
        self.active_session_slot = None;
        self.page_store.publish_slot(slot, kv)?;

        let sampled = mlx::argmax(&logits);
        mlx::async_eval(&[&sampled]);
        Ok(MetalInflight::Sampled {
            slot: row.slot,
            sampled,
        })
    }

    fn ensure_no_other_active_session(&self, slot: usize) -> anyhow::Result<()> {
        if let Some(active) = self.active_session_slot {
            anyhow::ensure!(
                active == slot,
                "scalar Qwen3.5 C++ sessions support only one active slot"
            );
        }
        Ok(())
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
#[derive(Default)]
struct MetalPageStore {
    pages: HashMap<u32, MetalPageBlock>,
    prefixes: HashMap<Vec<u32>, MetalPrefixSnapshot>,
}

#[cfg(feature = "metal")]
struct MetalPageBlock {
    kv_flat: Vec<mlx::MlxArray>,
}

#[cfg(feature = "metal")]
struct MetalPrefixSnapshot {
    cache_len: usize,
    gdr_flat: Vec<mlx::MlxArray>,
}

#[cfg(feature = "metal")]
impl MetalPageStore {
    fn publish_slot(&mut self, slot: &MetalSlotState, kv: &dyn KvPool) -> anyhow::Result<()> {
        let page_size = kv.page_size().max(1);
        let full_pages = slot.cache_len / page_size;
        if full_pages == 0 {
            return Ok(());
        }

        let page_ids = kv.page_indices(slot.slot);
        let publish_pages = full_pages.min(page_ids.len());
        for (page_idx, page_id) in page_ids.iter().take(publish_pages).enumerate() {
            let start = page_idx * page_size;
            let end = start + page_size;
            let mut kv_flat = Vec::with_capacity(slot.kv_flat.len());
            for array in &slot.kv_flat {
                kv_flat.push(slice_kv_tokens(array, start, end)?);
            }
            // Host page ids may be reused after the seam frees a slot. Overwrite
            // with the current slot's contents; retained/shared pages cannot be
            // reallocated by the host pool, so this does not corrupt live reuse.
            self.pages.insert(*page_id, MetalPageBlock { kv_flat });
        }

        // GDR state is prefix-wide, not page-local. Only publish a hot-prefix
        // snapshot at an exact page boundary where the exported recurrent/conv
        // state corresponds to the same token length as the page-id prefix.
        if slot.cache_len % page_size == 0 && publish_pages == full_pages {
            let key = page_ids[..full_pages].to_vec();
            if key.iter().all(|page| self.pages.contains_key(page)) {
                self.prefixes.insert(
                    key,
                    MetalPrefixSnapshot {
                        cache_len: slot.cache_len,
                        gdr_flat: slot.gdr_flat.clone(),
                    },
                );
            }
        }

        Ok(())
    }

    fn materialize_slot_from_prefix(
        &self,
        slot: usize,
        slot_epoch: u64,
        kv: &dyn KvPool,
        prefix_tokens: usize,
        capacity_tokens: usize,
    ) -> anyhow::Result<MetalSlotState> {
        let page_size = kv.page_size().max(1);
        anyhow::ensure!(
            prefix_tokens % page_size == 0,
            "Metal prefix attach requires page-aligned prefix: prefix_tokens={}, page_size={}",
            prefix_tokens,
            page_size
        );
        let prefix_pages = prefix_tokens / page_size;
        let slot_pages = kv.page_indices(slot);
        anyhow::ensure!(
            slot_pages.len() >= prefix_pages,
            "Metal prefix attach for slot {slot} needs {prefix_pages} pages, host slot has {}",
            slot_pages.len()
        );
        let key = slot_pages[..prefix_pages].to_vec();
        let snapshot = self.prefixes.get(&key).ok_or_else(|| {
            anyhow::anyhow!(
                "Metal prefix attach missing GDR snapshot for slot {slot}, prefix_tokens={prefix_tokens}, pages={key:?}"
            )
        })?;
        anyhow::ensure!(
            snapshot.cache_len == prefix_tokens,
            "Metal prefix snapshot length mismatch for slot {slot}: requested={}, snapshot={}",
            prefix_tokens,
            snapshot.cache_len
        );

        let first_page = key
            .first()
            .ok_or_else(|| anyhow::anyhow!("Metal prefix attach got empty page key"))?;
        let first_block = self.pages.get(first_page).ok_or_else(|| {
            anyhow::anyhow!("Metal prefix attach missing K/V page {first_page} for slot {slot}")
        })?;

        let mut kv_flat = Vec::with_capacity(first_block.kv_flat.len());
        let capacity = round_up_capacity(capacity_tokens.max(prefix_tokens)) as usize;
        for array_idx in 0..first_block.kv_flat.len() {
            let mut page_arrays = Vec::with_capacity(key.len());
            for page in &key {
                let block = self.pages.get(page).ok_or_else(|| {
                    anyhow::anyhow!("Metal prefix attach missing K/V page {page} for slot {slot}")
                })?;
                let array = block.kv_flat.get(array_idx).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Metal prefix attach K/V page {page} is missing array index {array_idx}"
                    )
                })?;
                page_arrays.push(array.clone());
            }
            let prefix_array = concatenate_or_single(page_arrays);
            let shape = prefix_array.shape().to_vec();
            anyhow::ensure!(
                shape.len() == 4 && shape[2] as usize == prefix_tokens,
                "Metal prefix K/V materialization shape mismatch for slot {slot}: shape={shape:?}, prefix_tokens={prefix_tokens}"
            );
            if capacity > prefix_tokens {
                let mut zero_shape = shape;
                zero_shape[2] = usize_to_i32(capacity - prefix_tokens)?;
                let zeros = mlx::zeros(&zero_shape, prefix_array.dtype());
                kv_flat.push(mlx::concatenate_axis(&[prefix_array, zeros], 2));
            } else {
                kv_flat.push(prefix_array);
            }
        }

        Ok(MetalSlotState::from_arrays(
            slot,
            slot_epoch,
            prefix_tokens,
            kv_flat,
            snapshot.gdr_flat.clone(),
        ))
    }
}

#[cfg(feature = "metal")]
struct MetalSlotState {
    slot: usize,
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
            slot,
            slot_epoch,
            cache_len: 0,
            kv_flat,
            gdr_flat,
            session_active: false,
        }
    }

    fn from_arrays(
        slot: usize,
        slot_epoch: u64,
        cache_len: usize,
        kv_flat: Vec<mlx::MlxArray>,
        gdr_flat: Vec<mlx::MlxArray>,
    ) -> Self {
        Self {
            slot,
            slot_epoch,
            cache_len,
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
fn slice_kv_tokens(
    array: &mlx::MlxArray,
    start_token: usize,
    end_token: usize,
) -> anyhow::Result<mlx::MlxArray> {
    let shape = array.shape().to_vec();
    anyhow::ensure!(
        shape.len() == 4,
        "expected Qwen3.5 flat K/V array to be rank-4, got shape={shape:?}"
    );
    anyhow::ensure!(
        start_token <= end_token && end_token <= shape[2] as usize,
        "K/V slice token range [{start_token}, {end_token}) exceeds shape={shape:?}"
    );
    let start = [0, 0, usize_to_i32(start_token)?, 0];
    let stop = [shape[0], shape[1], usize_to_i32(end_token)?, shape[3]];
    let strides = [1, 1, 1, 1];
    Ok(mlx::slice(array, &start, &stop, &strides))
}

#[cfg(feature = "metal")]
fn concatenate_or_single(mut arrays: Vec<mlx::MlxArray>) -> mlx::MlxArray {
    debug_assert!(!arrays.is_empty());
    if arrays.len() == 1 {
        arrays.pop().expect("len checked")
    } else {
        mlx::concatenate_axis(&arrays, 2)
    }
}

#[cfg(feature = "metal")]
fn usize_to_i32(value: usize) -> anyhow::Result<i32> {
    i32::try_from(value).map_err(|_| anyhow::anyhow!("value {value} exceeds i32::MAX"))
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
