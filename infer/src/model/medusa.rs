//! Medusa draft-head substrate.
//!
//! This module intentionally stops at the model/head boundary: it can capture
//! Qwen3.5 post-norm hidden states and run top-1 Medusa heads, while scheduler
//! routing (`--spec-draft-model medusa:<path>`) lands in the next tranche.

#[path = "medusa/weights.rs"]
pub mod weights;

use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, bail};
use cudarc::driver::CudaSlice;
use half::bf16;

use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};

pub type SharedHiddenStateCapture = Arc<Mutex<HiddenStateCapture>>;

#[derive(Clone, Debug)]
pub struct MedusaConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_heads: usize,
    pub num_hidden_layers: usize,
    pub max_paths: usize,
    pub topk: usize,
}

impl MedusaConfig {
    pub fn qwen35_default(hidden_size: usize, vocab_size: usize) -> Self {
        Self {
            hidden_size,
            vocab_size,
            num_heads: 5,
            num_hidden_layers: 1,
            max_paths: 64,
            topk: 10,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            bail!("Medusa hidden_size must be > 0");
        }
        if self.vocab_size == 0 {
            bail!("Medusa vocab_size must be > 0");
        }
        if self.num_heads == 0 {
            bail!("Medusa num_heads must be > 0");
        }
        if self.num_hidden_layers == 0 {
            bail!("Medusa num_hidden_layers must be > 0");
        }
        Ok(())
    }
}

pub struct HiddenStateCapture {
    hidden_dim: usize,
    slots: Vec<Option<DeviceVec>>,
    ring_slots: Vec<Vec<Option<DeviceVec>>>,
}

// SAFETY: The CUDA scheduler owns and mutates this capture from its single
// inference thread. The mutex only connects the target decode path to the draft
// wrapper; it is not intended for concurrent CUDA stream use.
unsafe impl Send for HiddenStateCapture {}

impl HiddenStateCapture {
    pub fn new(hidden_dim: usize, max_slots: usize) -> Self {
        Self {
            hidden_dim,
            slots: (0..max_slots).map(|_| None).collect(),
            ring_slots: (0..max_slots).map(|_| Vec::new()).collect(),
        }
    }

    pub fn shared(hidden_dim: usize, max_slots: usize) -> SharedHiddenStateCapture {
        Arc::new(Mutex::new(Self::new(hidden_dim, max_slots)))
    }

    pub fn store_batch(
        &mut self,
        ctx: &DeviceContext,
        slot_indices: &[usize],
        hidden_batch: &HiddenStates,
    ) -> Result<()> {
        anyhow::ensure!(
            hidden_batch.hidden_dim == self.hidden_dim,
            "Medusa capture hidden dim mismatch: batch={} capture={}",
            hidden_batch.hidden_dim,
            self.hidden_dim
        );
        anyhow::ensure!(
            hidden_batch.seq_len >= slot_indices.len(),
            "Medusa capture batch has {} rows for {} slots",
            hidden_batch.seq_len,
            slot_indices.len()
        );
        for (row_idx, &slot_idx) in slot_indices.iter().enumerate() {
            if slot_idx >= self.slots.len() {
                bail!(
                    "Medusa capture slot {} exceeds capture capacity {}",
                    slot_idx,
                    self.slots.len()
                );
            }
            if self.slots[slot_idx].is_none() {
                self.slots[slot_idx] = Some(
                    DeviceVec::zeros(ctx, self.hidden_dim)?.with_label("medusa_hidden_capture"),
                );
            }
            let dst = self.slots[slot_idx]
                .as_mut()
                .expect("allocated Medusa hidden capture slot");
            crate::ops::extract_vec_into(ctx, hidden_batch, row_idx, dst)?;
        }
        Ok(())
    }

    pub fn get_last(&self, slot_idx: usize) -> Result<DeviceVec> {
        let Some(Some(hidden)) = self.slots.get(slot_idx) else {
            bail!("Medusa hidden capture missing for slot {slot_idx}");
        };
        Ok(hidden.clone())
    }

    pub fn push_ring_slot(
        &mut self,
        ctx: &DeviceContext,
        slot_idx: usize,
        ring_idx: usize,
        ring_depth: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            ring_idx < ring_depth,
            "Medusa hidden ring index {ring_idx} exceeds depth {ring_depth}"
        );
        let src = self.get_last(slot_idx)?;
        let ring = self
            .ring_slots
            .get_mut(slot_idx)
            .ok_or_else(|| anyhow!("Medusa hidden ring slot {slot_idx} is out of range"))?;
        if ring.len() < ring_depth {
            ring.resize_with(ring_depth, || None);
        }
        if ring[ring_idx].is_none() {
            ring[ring_idx] =
                Some(DeviceVec::zeros(ctx, self.hidden_dim)?.with_label("medusa_hidden_ring"));
        }
        let dst = ring[ring_idx]
            .as_mut()
            .expect("allocated Medusa hidden ring slot");
        dst.copy_region_from_device(ctx, 0, &src, 0, self.hidden_dim)
    }

    pub fn restore_ring_slot(
        &mut self,
        ctx: &DeviceContext,
        slot_idx: usize,
        ring_idx: usize,
    ) -> Result<()> {
        let src = self
            .ring_slots
            .get(slot_idx)
            .and_then(|ring| ring.get(ring_idx))
            .and_then(|entry| entry.as_ref())
            .ok_or_else(|| anyhow!("Medusa hidden ring missing slot {slot_idx} row {ring_idx}"))?
            .clone();
        let dst_slot = self
            .slots
            .get_mut(slot_idx)
            .ok_or_else(|| anyhow!("Medusa hidden capture slot {slot_idx} is out of range"))?;
        if dst_slot.is_none() {
            *dst_slot =
                Some(DeviceVec::zeros(ctx, self.hidden_dim)?.with_label("medusa_hidden_capture"));
        }
        let dst = dst_slot
            .as_mut()
            .expect("allocated Medusa hidden capture slot");
        dst.copy_region_from_device(ctx, 0, &src, 0, self.hidden_dim)
    }
}

pub struct ResidualBlock {
    pub layers: Vec<DeviceMatrix>,
}

impl ResidualBlock {
    pub fn new(layers: Vec<DeviceMatrix>) -> Self {
        Self { layers }
    }

    fn validate(&self, config: &MedusaConfig, head_idx: usize) -> Result<()> {
        anyhow::ensure!(
            self.layers.len() == config.num_hidden_layers,
            "Medusa head {head_idx} has {} residual layers, expected {}",
            self.layers.len(),
            config.num_hidden_layers
        );
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            anyhow::ensure!(
                layer.rows == config.hidden_size && layer.cols == config.hidden_size,
                "Medusa head {head_idx} residual layer {layer_idx} shape [{}, {}], expected [{}, {}]",
                layer.rows,
                layer.cols,
                config.hidden_size,
                config.hidden_size
            );
        }
        Ok(())
    }
}

pub struct Medusa {
    pub config: MedusaConfig,
    pub blocks: Vec<ResidualBlock>,
    pub lm_heads: Vec<DeviceMatrix>,
}

// SAFETY: Medusa weights are immutable after load and are launched on the same
// CUDA stream as the owning target model.
unsafe impl Send for Medusa {}

impl Medusa {
    pub fn new(
        config: MedusaConfig,
        blocks: Vec<ResidualBlock>,
        lm_heads: Vec<DeviceMatrix>,
    ) -> Result<Self> {
        config.validate()?;
        anyhow::ensure!(
            blocks.len() == config.num_heads,
            "Medusa block count {} != num_heads {}",
            blocks.len(),
            config.num_heads
        );
        anyhow::ensure!(
            lm_heads.len() == config.num_heads,
            "Medusa lm_head count {} != num_heads {}",
            lm_heads.len(),
            config.num_heads
        );
        for (head_idx, block) in blocks.iter().enumerate() {
            block.validate(&config, head_idx)?;
            let head = &lm_heads[head_idx];
            anyhow::ensure!(
                head.rows == config.vocab_size && head.cols == config.hidden_size,
                "Medusa head {head_idx} lm_head shape [{}, {}], expected [{}, {}]",
                head.rows,
                head.cols,
                config.vocab_size,
                config.hidden_size
            );
        }
        Ok(Self {
            config,
            blocks,
            lm_heads,
        })
    }

    pub fn create_scratch(&self, ctx: &DeviceContext) -> Result<MedusaScratch> {
        MedusaScratch::new(ctx, &self.config)
    }

    pub fn propose_top1(
        &self,
        ctx: &DeviceContext,
        hidden: &DeviceVec,
        scratch: &mut MedusaScratch,
        num_draft_tokens: usize,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(
            hidden.len == self.config.hidden_size,
            "Medusa hidden len {} != hidden_size {}",
            hidden.len,
            self.config.hidden_size
        );
        let heads = num_draft_tokens.min(self.config.num_heads);
        scratch.input.data = hidden
            .data
            .try_clone()
            .map_err(|e| anyhow!("clone Medusa hidden handle: {e}"))?;
        scratch.input.seq_len = 1;

        let mut out = Vec::with_capacity(heads);
        for head_idx in 0..heads {
            let head_scratch = &mut scratch.heads[head_idx];
            let block_out_is_a = self.blocks[head_idx].forward(
                ctx,
                &scratch.input,
                head_scratch,
                self.config.num_hidden_layers,
            )?;
            let block_out = if block_out_is_a {
                &head_scratch.residual_a
            } else {
                &head_scratch.residual_b
            };
            crate::ops::gemm_into(
                ctx,
                &self.lm_heads[head_idx],
                block_out,
                &mut head_scratch.logits,
            );
            crate::ops::extract_vec_into(
                ctx,
                &head_scratch.logits,
                0,
                &mut head_scratch.logits_vec,
            )?;
            let (token, _logprob) = crate::ops::argmax_with_logprob(
                ctx,
                &head_scratch.logits_vec,
                &mut head_scratch.argmax_out,
                &mut head_scratch.probs,
            )?;
            out.push(token);
        }
        Ok(out)
    }
}

impl ResidualBlock {
    fn forward(
        &self,
        ctx: &DeviceContext,
        input: &HiddenStates,
        scratch: &mut MedusaHeadScratch,
        num_hidden_layers: usize,
    ) -> Result<bool> {
        if num_hidden_layers == 0 {
            bail!("Medusa residual block requires at least one hidden layer");
        }

        let mut current_is_input = true;
        let mut current_is_a = false;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let out_is_a = layer_idx % 2 == 0;
            if current_is_input {
                Self::forward_one_layer(
                    ctx,
                    layer,
                    input,
                    &mut scratch.linear_out,
                    &scratch.ones,
                    &mut scratch.activated,
                    &mut scratch.residual_a,
                )?;
            } else if current_is_a {
                debug_assert!(!out_is_a);
                Self::forward_one_layer(
                    ctx,
                    layer,
                    &scratch.residual_a,
                    &mut scratch.linear_out,
                    &scratch.ones,
                    &mut scratch.activated,
                    &mut scratch.residual_b,
                )?;
            } else {
                debug_assert!(out_is_a);
                Self::forward_one_layer(
                    ctx,
                    layer,
                    &scratch.residual_b,
                    &mut scratch.linear_out,
                    &scratch.ones,
                    &mut scratch.activated,
                    &mut scratch.residual_a,
                )?;
            }
            current_is_input = false;
            current_is_a = out_is_a;
        }

        Ok(current_is_a)
    }

    fn forward_one_layer(
        ctx: &DeviceContext,
        layer: &DeviceMatrix,
        current: &HiddenStates,
        linear_out: &mut HiddenStates,
        ones: &HiddenStates,
        activated: &mut HiddenStates,
        residual_out: &mut HiddenStates,
    ) -> Result<()> {
        crate::ops::gemm_into(ctx, layer, current, linear_out);
        crate::ops::silu_mul_batch_into(ctx, linear_out, ones, activated)?;
        crate::ops::add_batch_into(ctx, current, activated, residual_out)
    }
}

pub struct MedusaScratch {
    input: HiddenStates,
    heads: Vec<MedusaHeadScratch>,
}

// SAFETY: Scratch is bound to one CUDA stream and used from the scheduler
// inference thread through a mutex in the draft wrapper.
unsafe impl Send for MedusaScratch {}

impl MedusaScratch {
    fn new(ctx: &DeviceContext, config: &MedusaConfig) -> Result<Self> {
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(config.hidden_size)
            .map_err(|e| anyhow!("alloc Medusa input dummy: {e}"))?;
        let input = HiddenStates {
            data: dummy,
            hidden_dim: config.hidden_size,
            seq_len: 1,
        };
        let mut heads = Vec::with_capacity(config.num_heads);
        for _ in 0..config.num_heads {
            heads.push(MedusaHeadScratch::new(ctx, config)?);
        }
        Ok(Self { input, heads })
    }
}

pub struct MedusaHeadScratch {
    linear_out: HiddenStates,
    activated: HiddenStates,
    residual_a: HiddenStates,
    residual_b: HiddenStates,
    ones: HiddenStates,
    logits: HiddenStates,
    logits_vec: DeviceVec,
    probs: CudaSlice<f32>,
    argmax_out: CudaSlice<i32>,
}

impl MedusaHeadScratch {
    fn new(ctx: &DeviceContext, config: &MedusaConfig) -> Result<Self> {
        let ones_host = vec![bf16::ONE; config.hidden_size];
        let ones = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&ones_host)
                .map_err(|e| anyhow!("H2D Medusa SiLU ones: {e}"))?,
            hidden_dim: config.hidden_size,
            seq_len: 1,
        };
        Ok(Self {
            linear_out: HiddenStates::zeros(ctx, config.hidden_size, 1)?,
            activated: HiddenStates::zeros(ctx, config.hidden_size, 1)?,
            residual_a: HiddenStates::zeros(ctx, config.hidden_size, 1)?,
            residual_b: HiddenStates::zeros(ctx, config.hidden_size, 1)?,
            ones,
            logits: HiddenStates::zeros(ctx, config.vocab_size, 1)?,
            logits_vec: DeviceVec::zeros(ctx, config.vocab_size)?.with_label("medusa_logits"),
            probs: ctx
                .stream
                .alloc_zeros::<f32>(config.vocab_size)
                .map_err(|e| anyhow!("alloc Medusa probs: {e}"))?,
            argmax_out: ctx
                .stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| anyhow!("alloc Medusa argmax_out: {e}"))?,
        })
    }
}
