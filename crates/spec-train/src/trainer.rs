//! The offline DSpark training loop.
//!
//! One optimizer step is `grad_accum` samples; each sample contributes up to
//! `num_anchors` blocks from a single trunk forward. Defaults are the
//! reference's `config/dspark/dspark_qwen3_8b.py`.

use anyhow::{Result, ensure};
use autograd::{
    Optimizer, Tape, TensorId, TensorStore,
    grad_clip::finite_optimizer_step,
    lr_schedule::{CosineWithWarmup, LrSchedule},
    ops,
    optim::AdamW,
};
use std::collections::HashSet;

use qwen35_spec::DsparkConfig;

use crate::{
    backbone::{Draft, Input},
    block::{self, Block},
    loss::{self, Batch},
};

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub lr: f32,
    pub warmup_ratio: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub num_anchors: usize,
    /// Blocks per backward. Everything wide lives here: attention scores are
    /// `[heads, chunk·block, ctx + chunk·block]` and logits `[chunk·block,
    /// vocab]`, so this is the knob that keeps a step inside VRAM. All 512
    /// anchors at once is ~70 GiB of score tensors alone.
    pub blocks_per_backward: usize,
    pub loss_decay_gamma: f32,
    pub weights: loss::Weights,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            lr: 6e-4,
            warmup_ratio: 0.04,
            weight_decay: 0.0,
            max_grad_norm: 1.0,
            num_anchors: 512,
            blocks_per_backward: 32,
            loss_decay_gamma: 4.0,
            weights: loss::Weights::default(),
            seed: 42,
        }
    }
}

pub struct Sample {
    pub input_ids: Vec<u32>,
    /// Positions the draft is trained on — the assistant turns.
    pub loss_mask: Vec<bool>,
}

/// What the trunk contributes per sample. The draft conditions on `taps` and
/// distills toward logits read off `last_hidden`; neither is cached to disk
/// (recomputing costs GPU hours, caching costs tens of terabytes).
pub trait Target {
    /// `([seq, taps·hidden], [seq, hidden])` for one tokenized sample.
    fn forward(&mut self, input_ids: &[u32]) -> Result<(Vec<f32>, Vec<f32>)>;
}

pub struct Trainer {
    pub draft: Draft,
    pub cfg: Config,
    opt: AdamW,
    schedule: CosineWithWarmup,
    params: Vec<TensorId>,
    step: usize,
}

impl Trainer {
    pub fn new(draft: Draft, cfg: Config, total_steps: usize) -> Self {
        let params = draft.parameters();
        Self {
            opt: AdamW::new(cfg.lr, (0.9, 0.999), 1e-8, cfg.weight_decay),
            schedule: CosineWithWarmup {
                base_lr: cfg.lr,
                min_lr: 0.0,
                warmup_steps: (total_steps as f32 * cfg.warmup_ratio) as u64,
                total_steps: total_steps as u64,
            },
            params,
            step: 0,
            draft,
            cfg,
        }
    }

    /// Accumulate over `samples`, clip, and step. Returns the mean loss over
    /// the samples that produced any supervision.
    pub fn train_step(
        &mut self,
        samples: &[Sample],
        target: &mut dyn Target,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<f32> {
        ensure!(!samples.is_empty(), "empty batch");
        AdamW::zero_grad(&mut self.opt, &self.params, store);

        let mut total = 0.0;
        let mut counted = 0usize;
        for (i, s) in samples.iter().enumerate() {
            let seed = self
                .cfg
                .seed
                .wrapping_add((self.step * samples.len() + i) as u64);
            match self.accumulate(s, seed, samples.len(), target, store, tape)? {
                Some(l) => {
                    total += l;
                    counted += 1;
                }
                None => continue,
            }
            tape.entries.clear();
            tape.set_enabled(true);
            store.retain_ids(&self.keep_set(store));
        }
        ensure!(counted > 0, "no sample in the batch carried supervision");

        let loss = total / counted as f32;
        Optimizer::set_lr(&mut self.opt, self.schedule.lr(self.step as u64));
        // One finite transaction: a NaN loss or grad norm clears the pending
        // grads and advances no parameter, moment or schedule step. Clipping a
        // NaN norm would otherwise scale every gradient by NaN and contaminate
        // the AdamW moments permanently.
        finite_optimizer_step(
            loss,
            &self.params,
            self.cfg.max_grad_norm,
            &mut self.opt,
            store,
        )?;
        self.step += 1;
        store.retain_ids(&self.keep_set(store));
        Ok(loss)
    }

    fn keep_set(&self, store: &TensorStore) -> HashSet<TensorId> {
        let mut keep: HashSet<TensorId> = [self.draft.embed_tokens, self.draft.lm_head]
            .into_iter()
            .collect();
        for &p in &self.params {
            keep.insert(p);
            if let Some(g) = store.get(p).and_then(|t| t.grad) {
                keep.insert(g);
            }
        }
        keep
    }

    /// One sample's backward, accumulated into the params' grads. `None` when
    /// the sample has no anchorable position.
    fn accumulate(
        &self,
        sample: &Sample,
        seed: u64,
        batch: usize,
        target: &mut dyn Target,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<Option<f32>> {
        ensure!(
            sample.input_ids.len() == sample.loss_mask.len(),
            "input_ids {} != loss_mask {}",
            sample.input_ids.len(),
            sample.loss_mask.len()
        );
        let block_size = self.draft.cfg.block_size;
        let anchors = block::sample_anchors(&sample.loss_mask, self.cfg.num_anchors, seed);
        if anchors.is_empty() {
            return Ok(None);
        }
        let blocks: Vec<Block> = anchors
            .iter()
            .map(|&a| block::build_block(&sample.input_ids, &sample.loss_mask, a, block_size))
            .collect::<Result<_>>()?;

        let weights: Vec<f32> = blocks
            .iter()
            .flat_map(|b| block::row_weights(b, Some(self.cfg.loss_decay_gamma)))
            .collect();
        if weights.iter().sum::<f32>() <= 0.0 {
            return Ok(None);
        }

        let seq = sample.input_ids.len();
        let hidden = self.draft.cfg.hidden_size;
        let (taps_host, last_hidden) = target.forward(&sample.input_ids)?;
        ensure!(
            taps_host.len() == seq * self.draft.cfg.target_layer_ids.len() * hidden
                && last_hidden.len() == seq * hidden,
            "target returned {} tap and {} hidden values for {seq} tokens",
            taps_host.len(),
            last_hidden.len()
        );

        let chunks = blocks.len().div_ceil(self.cfg.blocks_per_backward);
        let scale = 1.0 / (batch * chunks) as f32;
        let mut total = 0.0;
        for chunk in blocks.chunks(self.cfg.blocks_per_backward) {
            total +=
                self.chunk_backward(chunk, &taps_host, &last_hidden, seq, scale, store, tape)?;
            tape.entries.clear();
            tape.set_enabled(true);
            store.retain_ids(&self.keep_set(store));
        }
        Ok(Some(total / chunks as f32))
    }

    /// Forward + loss + backward for one chunk of blocks. The chunk is the unit
    /// because the two widest tensors — attention scores and logits — both
    /// scale with its row count, and a block only ever attends the context and
    /// its own rows, so chunking changes no result.
    #[allow(clippy::too_many_arguments)]
    fn chunk_backward(
        &self,
        blocks: &[Block],
        taps_host: &[f32],
        last_hidden: &[f32],
        seq: usize,
        scale: f32,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<f32> {
        let hidden = self.draft.cfg.hidden_size;
        let block_size = self.draft.cfg.block_size;
        let taps = store.from_slice(
            taps_host,
            &[seq, self.draft.cfg.target_layer_ids.len() * hidden],
        )?;

        let out = self.draft.forward(
            &Input {
                blocks,
                taps,
                ctx_len: seq,
            },
            store,
            tape,
        )?;

        // The trunk's own prediction for the same positions: row `t` of block
        // `j` predicts `anchor+1+t`, so the trunk hidden state to read is the
        // one at `anchor+t` — the draft row's own RoPE position.
        let aligned: Vec<f32> = block::draft_positions(blocks)
            .into_iter()
            .flat_map(|p| last_hidden[p.min(seq - 1) * hidden..][..hidden].to_vec())
            .collect();
        let rows = blocks.len() * block_size;
        let aligned = store.from_slice(&aligned, &[rows, hidden])?;
        let mut frozen = Tape::new();
        frozen.set_enabled(false);
        let target_logits = ops::matmul_bt(aligned, self.draft.lm_head, store, &mut frozen)?;

        let targets: Vec<usize> = blocks
            .iter()
            .flat_map(|b| b.targets.iter().map(|&t| t as usize))
            .collect();
        let weights: Vec<f32> = blocks
            .iter()
            .flat_map(|b| block::row_weights(b, Some(self.cfg.loss_decay_gamma)))
            .collect();
        if weights.iter().sum::<f32>() <= 0.0 {
            return Ok(0.0);
        }
        let loss = loss::dspark_loss(
            &Batch {
                draft_logits: out.logits,
                target_logits,
                targets: &targets,
                weights: &weights,
                conf_logits: out.confidence,
            },
            self.cfg.weights,
            store,
            tape,
        )?;
        let value = store.to_host(loss)?[0];
        // Scale by the fixed sample × chunk count, as the reference's fixed
        // global_batch_size does — not by whatever happened to carry
        // supervision, which would make the step size data-dependent.
        let scaled = ops::mul_scalar(loss, scale, store, tape)?;
        tape.backward_accumulate_only(scaled, store)?;
        Ok(value)
    }
}

/// Bytes of f32 activation the widest tensors hold for one backward.
///
/// The forward keeps four score-shaped tensors per layer (matmul, scale, mask,
/// softmax) live until backward, and the loss keeps five logit-shaped ones.
/// Both scale with `blocks_per_backward`, which is the only knob between the
/// reference recipe and an OOM: all 512 anchors at once is ~70 GiB of scores.
#[must_use]
pub fn peak_activation_bytes(
    draft: &DsparkConfig,
    blocks_per_backward: usize,
    ctx_len: usize,
    vocab: usize,
) -> usize {
    let rows = blocks_per_backward * draft.block_size;
    let scores = draft.num_attention_heads * rows * (ctx_len + rows);
    4 * (4 * scores * draft.num_hidden_layers + 5 * rows * vocab)
}

#[cfg(test)]
mod tests {
    use super::*;
    use autograd::{Backend, CpuBackend, Tensor};
    use qwen35_spec::{DsparkConfig, DsparkLayerType};
    use std::sync::Arc;

    const VOCAB: usize = 24;
    const HIDDEN: usize = 8;
    const SEQ: usize = 24;

    /// A fixed pseudo-trunk. Its taps and hidden states are an arbitrary but
    /// deterministic function of the token, so the draft has something
    /// learnable to fit.
    struct FakeTarget;

    impl Target for FakeTarget {
        fn forward(&mut self, ids: &[u32]) -> Result<(Vec<f32>, Vec<f32>)> {
            let f = |t: u32, i: usize| (t as f32 * 0.37 + i as f32 * 0.11).sin();
            let taps = ids
                .iter()
                .flat_map(|&t| (0..2 * HIDDEN).map(move |i| f(t, i)))
                .collect();
            let hidden = ids
                .iter()
                .flat_map(|&t| (0..HIDDEN).map(move |i| f(t.wrapping_add(1), i)))
                .collect();
            Ok((taps, hidden))
        }
    }

    fn cfg() -> DsparkConfig {
        DsparkConfig {
            hidden_size: HIDDEN,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            sliding_window: 4096,
            layer_types: vec![DsparkLayerType::Full; 2],
            block_size: 3,
            mask_token_id: 23,
            target_layer_ids: vec![-1, 0],
            next_token_heads: true,
        }
    }

    /// The gate the tiny-shape tests cannot be: the reference recipe at its
    /// real dimensions must fit beside a resident 27B trunk. Without chunking
    /// the same arithmetic gives ~70 GiB of score tensors alone.
    #[test]
    fn the_reference_recipe_fits_in_vram() {
        let mut c = cfg();
        c.hidden_size = 5120;
        c.num_hidden_layers = 5;
        c.num_attention_heads = 32;
        c.head_dim = 128;
        c.block_size = 7;
        let gib = |b: usize| b as f64 / (1 << 30) as f64;

        let chunked = peak_activation_bytes(&c, 32, 4096, 248_320);
        assert!(gib(chunked) < 8.0, "chunked peak {:.1} GiB", gib(chunked));

        let whole = peak_activation_bytes(&c, 512, 4096, 248_320);
        assert!(
            gib(whole) > 60.0,
            "unchunked peak {:.1} GiB — the chunking would not be earning its \
             keep and this gate is measuring the wrong thing",
            gib(whole)
        );
    }

    #[test]
    fn the_loop_reduces_the_loss() {
        let mut store = TensorStore::with_backend(Arc::new(CpuBackend) as Arc<dyn Backend>);
        let table = |salt: u64, store: &mut TensorStore| {
            let data: Vec<f32> = (0..VOCAB * HIDDEN)
                .map(|i| ((i as f32 + salt as f32) * 0.29).sin() * 0.1)
                .collect();
            store.alloc(Tensor::new(data, vec![VOCAB, HIDDEN], false).unwrap())
        };
        let embed = table(1, &mut store);
        let lm_head = table(2, &mut store);
        let draft = crate::backbone::init(cfg(), VOCAB, 4, embed, lm_head, &mut store).unwrap();

        let samples: Vec<Sample> = (0..2)
            .map(|s| Sample {
                input_ids: (0..SEQ as u32).map(|i| (i * 5 + s) % 20).collect(),
                loss_mask: (0..SEQ).map(|i| i >= 4).collect(),
            })
            .collect();

        let mut cfg = Config {
            lr: 5e-2,
            warmup_ratio: 0.0,
            num_anchors: 8,
            ..Config::default()
        };
        cfg.weights.confidence = 0.0;
        let mut trainer = Trainer::new(draft, cfg, 20);
        let mut tape = Tape::new();

        let first = trainer
            .train_step(&samples, &mut FakeTarget, &mut store, &mut tape)
            .unwrap();
        let mut last = first;
        for _ in 0..9 {
            last = trainer
                .train_step(&samples, &mut FakeTarget, &mut store, &mut tape)
                .unwrap();
        }
        assert!(first.is_finite() && last.is_finite(), "{first} -> {last}");
        assert!(last < first * 0.95, "loss did not fall: {first} -> {last}");
    }
}
