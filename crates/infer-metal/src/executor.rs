//! Metal backend executor + session machinery.
//!
//! `new()` keeps a CPU placeholder so the submit/poll seam stays testable without
//! the `metal` feature; `from_model_path()` builds the real MLX Qwen3.5 executor.
//! `RealMetalExecutor` and all MLX-touching session state are gated behind
//! `#[cfg(feature = "metal")]`.

#[cfg(feature = "metal")]
use std::collections::HashMap;
#[cfg(feature = "metal")]
use std::path::Path;

use infer_plan::{ForwardPlan, SlotToken, StepOutput};
use infer_seam::{BackendExecutor, KvPool, PollResult};

#[cfg(feature = "metal")]
use crate::{config, mlx, model_source, qwen35, wired_limit};

#[cfg(feature = "metal")]
const KV_CACHE_CHUNK: i32 = 256;

/// In-flight handle for a submitted Metal step.
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

/// Turn a logits array into an in-flight result under `params`.
///
/// Greedy keeps the device `argmax` + async path; `temperature > 0` materializes
/// host f32 logits and draws via the shared `infer_plan::sample_token` (one D2H
/// per token, no GPU sampling kernel).
#[cfg(feature = "metal")]
fn sample_inflight(
    slot: usize,
    logits: &mlx::MlxArray,
    params: &infer_plan::SamplingParams,
    position: u64,
) -> MetalInflight {
    if params.is_greedy() {
        let sampled = mlx::argmax(logits);
        mlx::async_eval(&[&sampled]);
        return MetalInflight::Sampled { slot, sampled };
    }
    let logits_f32 = mlx::as_dtype(logits, mlx::Dtype::Float32);
    mlx::eval(&[&logits_f32]);
    let token = infer_plan::sample_token(logits_f32.as_slice_f32(), params, position);
    MetalInflight::Ready(StepOutput {
        tokens: vec![SlotToken {
            slot,
            token,
            logprob: None,
            finish: None,
        }],
    })
}

/// Metal backend executor.
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

    /// Feature-free placeholder forward: one deterministic token per scheduled
    /// row, so the submit/poll seam is exercisable on CPU without MLX.
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

    fn model_stop_token_ids(&self) -> Vec<u32> {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_ref() {
            return real.config.stop_token_ids.clone();
        }
        Vec::new()
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
        let position = slot.cache_len as u64;
        slot.drain_session(model)?;
        self.active_session_slot = None;
        self.page_store.publish_slot(slot, kv)?;

        Ok(sample_inflight(row.slot, &logits, &row.params, position))
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
        let position = slot.cache_len as u64;
        slot.drain_session(model)?;
        self.active_session_slot = None;
        self.page_store.publish_slot(slot, kv)?;

        Ok(sample_inflight(row.slot, &logits, &row.params, position))
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
            // Host-epoch bump is the slot-release signal until the seam grows an
            // explicit executor release callback.
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
        if slot.cache_len.is_multiple_of(page_size) && publish_pages == full_pages {
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
            prefix_tokens.is_multiple_of(page_size),
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
    use crate::kv_pool::MetalKvPool;
    use infer_plan::{DecodeRow, ForwardMode, PrefillRow};

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
