//! The offline DSpark training loop.
//!
//! One optimizer step is `grad_accum` samples; each sample contributes up to
//! `num_anchors` blocks from a single trunk forward. Defaults are the
//! reference's `config/dspark/dspark_qwen3_8b.py`.

use anyhow::{Result, ensure};
use autograd::{
    Backend, Optimizer, Tape, TensorId, TensorStore,
    grad_clip::finite_optimizer_step,
    lr_schedule::{CosineWithWarmup, LrSchedule},
    ops,
    optim::AdamW,
};
use std::{collections::HashSet, sync::Arc};

use qwen35_spec::DsparkConfig;

use crate::{
    backbone::{Draft, Input},
    block::{self, Block},
    loss::{self, Batch, Terms},
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

/// One optimizer step, as measured.
#[derive(Debug, Clone, Copy)]
pub struct StepStats {
    pub loss: f32,
    pub ce: f32,
    pub tv: f32,
    pub conf: f32,
    pub mean_accept: f32,
    pub grad_norm: f32,
    pub lr: f32,
    /// Samples in the batch that carried supervision.
    pub counted: usize,
    /// Backwards run, summed over those samples.
    pub chunks: usize,
}

#[derive(Debug)]
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

/// Everything a chunk needs from the sample it belongs to.
struct SampleCtx<'a> {
    /// `[seq, taps·hidden]`, the trunk residual stream at the tapped layers.
    taps: &'a [f32],
    /// `[seq, hidden]`, the trunk's final hidden state.
    last_hidden: &'a [f32],
    seq: usize,
    /// `Σ w` over the WHOLE sample. Every chunk divides by this one number, so
    /// no row's weight depends on which chunk it landed in.
    denom: f32,
}

/// One sample's contribution to the step.
struct SampleOut {
    loss: f32,
    terms: Terms,
    chunks: usize,
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
    /// `backend` keeps the AdamW moments device-resident; the host constructor
    /// would round-trip ~127M markov parameters through host memory per step.
    pub fn new(
        draft: Draft,
        cfg: Config,
        total_steps: usize,
        backend: Arc<dyn Backend + Send + Sync>,
    ) -> Self {
        let params = draft.parameters();
        Self {
            opt: AdamW::new_with_device(cfg.lr, (0.9, 0.999), 1e-8, cfg.weight_decay, backend),
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

    /// Accumulate over `samples`, clip, and step. Every reported quantity is a
    /// mean over the samples that produced any supervision.
    pub fn train_step(
        &mut self,
        samples: &[Sample],
        target: &mut dyn Target,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<StepStats> {
        ensure!(!samples.is_empty(), "empty batch");
        AdamW::zero_grad(&mut self.opt, &self.params, store);

        let mut total = 0.0;
        let mut terms = Terms::default();
        let mut counted = 0usize;
        let mut chunks = 0usize;
        for (i, s) in samples.iter().enumerate() {
            let seed = self
                .cfg
                .seed
                .wrapping_add((self.step * samples.len() + i) as u64);
            let Some(out) = self.accumulate(s, seed, samples.len(), target, store, tape)? else {
                continue;
            };
            total += out.loss;
            terms = add_terms(terms, out.terms);
            chunks += out.chunks;
            counted += 1;
            tape.entries.clear();
            tape.set_enabled(true);
            store.retain_ids(&self.keep_set(store));
        }
        ensure!(counted > 0, "no sample in the batch carried supervision");

        let n = counted as f32;
        let loss = total / n;
        let lr = self.schedule.lr(self.step as u64);
        Optimizer::set_lr(&mut self.opt, lr);
        // One finite transaction: a NaN loss or grad norm clears the pending
        // grads and advances no parameter, moment or schedule step. Clipping a
        // NaN norm would otherwise scale every gradient by NaN and contaminate
        // the AdamW moments permanently.
        let grad_norm = finite_optimizer_step(
            loss,
            &self.params,
            self.cfg.max_grad_norm,
            &mut self.opt,
            store,
        )?;
        self.step += 1;
        store.retain_ids(&self.keep_set(store));
        Ok(StepStats {
            loss,
            ce: terms.ce / n,
            tv: terms.tv / n,
            conf: terms.conf / n,
            mean_accept: terms.mean_accept / n,
            grad_norm: grad_norm as f32,
            lr,
            counted,
            chunks,
        })
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
    ) -> Result<Option<SampleOut>> {
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
        let denom: f32 = weights.iter().sum();
        if denom <= 0.0 {
            return Ok(None);
        }

        let seq = sample.input_ids.len();
        let hidden = self.draft.cfg.hidden_size;
        let (taps, last_hidden) = target.forward(&sample.input_ids)?;
        ensure!(
            taps.len() == seq * self.draft.cfg.target_layer_ids.len() * hidden
                && last_hidden.len() == seq * hidden,
            "target returned {} tap and {} hidden values for {seq} tokens",
            taps.len(),
            last_hidden.len()
        );
        let ctx = SampleCtx {
            taps: &taps,
            last_hidden: &last_hidden,
            seq,
            denom,
        };

        let scale = 1.0 / batch as f32;
        let mut out = SampleOut {
            loss: 0.0,
            terms: Terms::default(),
            chunks: 0,
        };
        let chunk_rows = self.cfg.blocks_per_backward * block_size;
        for (chunk, w) in blocks
            .chunks(self.cfg.blocks_per_backward)
            .zip(weights.chunks(chunk_rows))
        {
            let (l, t) = self.chunk_backward(chunk, w, &ctx, scale, store, tape)?;
            out.loss += l;
            out.terms = add_terms(out.terms, t);
            out.chunks += 1;
            tape.entries.clear();
            tape.set_enabled(true);
            store.retain_ids(&self.keep_set(store));
        }
        Ok(Some(out))
    }

    /// Forward + loss + backward for one chunk of blocks. The chunk is the unit
    /// because the two widest tensors — attention scores and logits — both
    /// scale with its row count.
    ///
    /// The split is a VRAM knob and nothing else: a block attends only the
    /// context and its own rows, and every chunk divides by the sample-wide
    /// `ctx.denom`, so the chunk losses sum to the sample's loss and the chunk
    /// gradients sum to the sample's gradient however the blocks are cut.
    fn chunk_backward(
        &self,
        blocks: &[Block],
        weights: &[f32],
        ctx: &SampleCtx<'_>,
        scale: f32,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<(f32, Terms)> {
        if weights.iter().sum::<f32>() <= 0.0 {
            return Ok((0.0, Terms::default()));
        }
        let (loss, terms) = self.chunk_loss(blocks, weights, ctx, store, tape)?;
        let value = store.to_host(loss)?[0];
        // Scale by the fixed sample count, as the reference's fixed
        // global_batch_size does — not by whatever happened to carry
        // supervision, which would make the step size data-dependent.
        let scaled = ops::mul_scalar(loss, scale, store, tape)?;
        tape.backward_accumulate_only(scaled, store)?;
        Ok((value, terms))
    }

    /// The chunk's graph up to the loss scalar — the point
    /// [`peak_activation_bytes`] models.
    fn chunk_loss(
        &self,
        blocks: &[Block],
        weights: &[f32],
        ctx: &SampleCtx<'_>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<(TensorId, Terms)> {
        let hidden = self.draft.cfg.hidden_size;
        let block_size = self.draft.cfg.block_size;
        let taps = store.from_slice(
            ctx.taps,
            &[ctx.seq, self.draft.cfg.target_layer_ids.len() * hidden],
        )?;

        let out = self.draft.forward(
            &Input {
                blocks,
                taps,
                ctx_len: ctx.seq,
            },
            store,
            tape,
        )?;

        // Hidden `p` predicts token `p+1`, and row `t` at RoPE `anchor+t` targets
        // `anchor+1+t`, so the supervising hidden is the row's own position.
        let aligned: Vec<f32> = block::draft_positions(blocks)
            .into_iter()
            .flat_map(|p| ctx.last_hidden[p.min(ctx.seq - 1) * hidden..][..hidden].to_vec())
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
        loss::dspark_loss(
            &Batch {
                draft_logits: out.logits,
                target_logits,
                targets: &targets,
                weights,
                denom: ctx.denom,
                conf_logits: out.confidence,
            },
            self.cfg.weights,
            store,
            tape,
        )
    }
}

fn add_terms(a: Terms, b: Terms) -> Terms {
    Terms {
        ce: a.ce + b.ce,
        tv: a.tv + b.tv,
        conf: a.conf + b.conf,
        mean_accept: a.mean_accept + b.mean_accept,
    }
}

/// Bytes of f32 activation live when the chunk's loss scalar completes — the
/// peak, because nothing between the chunk's first upload and its backward is
/// freed, and the backward frees each intermediate grad at its last consumer.
///
/// Counted off `backbone::forward` and `loss::dspark_loss` op by op; every
/// forward op allocates exactly one output. Terms narrower than a row vector
/// are dropped (the Markov rank, the confidence scalars, the per-row loss
/// vectors) — under 2% at training shapes. `blocks_per_backward` is the only
/// knob between the reference recipe and an OOM: all 512 anchors at once is
/// ~110 GiB.
#[must_use]
pub fn peak_activation_bytes(
    draft: &DsparkConfig,
    blocks_per_backward: usize,
    ctx_len: usize,
    vocab: usize,
) -> usize {
    let rows = blocks_per_backward * draft.block_size;
    let kv = ctx_len + rows;
    let (h, hd) = (draft.hidden_size, draft.head_dim);
    let q_dim = draft.num_attention_heads * hd;
    let kv_dim = draft.num_key_value_heads * hd;

    // Per layer: 4 score-shaped (matmul, scale, mask, softmax); 10 query-shaped
    // (project, the four head_norm steps, rope, the `q3` reshape, and the three
    // attention outputs); 7 at full head width (`repeat_kv` expands then
    // re-reshapes, for keys and values, plus `k3`, `v3` and the key transpose);
    // 13 at kv width (the context and draft projections and their cat, twice,
    // then head_norm and rope on the keys, reshape and transpose on the values,
    // and both `repeat_kv` inputs); 6 residual-shaped and 4 MLP-shaped.
    let per_layer = 4 * draft.num_attention_heads * rows * kv
        + 10 * rows * q_dim
        + 7 * kv * q_dim
        + 13 * kv * kv_dim
        + 6 * rows * h
        + 4 * rows * draft.intermediate_size;

    // The full-context path, re-run once per chunk: the tap upload, the `fc`
    // GEMM and its norm, the rope tables (cos and sin at half head width, for
    // the whole kv sequence and again for the queries), the dense mask.
    let context =
        ctx_len * draft.target_layer_ids.len() * h + 2 * ctx_len * h + (kv + rows) * hd + rows * kv;

    // 10 logit-shaped: the draft logits, the Markov bias and its add, the
    // trunk's logits, log_softmax, both softmaxes, the negation, the difference
    // and its abs. 5 hidden-shaped: the embedding and its reshape, the final
    // norm, the aligned trunk hidden, the confidence input.
    let head = 10 * rows * vocab + 5 * rows * h;

    4 * (per_layer * draft.num_hidden_layers + context + head)
}

#[cfg(test)]
mod tests {
    use super::*;
    use autograd::{CpuBackend, Tensor};
    use qwen35_spec::DsparkLayerType;

    const VOCAB: usize = 24;
    const HIDDEN: usize = 8;
    const SEQ: usize = 24;
    const MARKOV_RANK: usize = 4;

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

    fn table(salt: u64, store: &mut TensorStore) -> TensorId {
        let data: Vec<f32> = (0..VOCAB * HIDDEN)
            .map(|i| ((i as f32 + salt as f32) * 0.29).sin() * 0.1)
            .collect();
        store.alloc(Tensor::new(data, vec![VOCAB, HIDDEN], false).unwrap())
    }

    fn setup(config: Config) -> (TensorStore, Trainer) {
        let backend: Arc<dyn Backend + Send + Sync> = Arc::new(CpuBackend);
        let mut store = TensorStore::with_backend(backend.clone());
        let embed = table(1, &mut store);
        let lm_head = table(2, &mut store);
        let draft =
            crate::backbone::init(cfg(), VOCAB, MARKOV_RANK, embed, lm_head, &mut store).unwrap();
        let trainer = Trainer::new(draft, config, 20, backend);
        (store, trainer)
    }

    fn sample(salt: u32) -> Sample {
        Sample {
            input_ids: (0..SEQ as u32).map(|i| (i * 5 + salt) % 20).collect(),
            loss_mask: (0..SEQ).map(|i| i >= 4).collect(),
        }
    }

    /// The gate the tiny-shape tests cannot be: the reference recipe at its
    /// real dimensions must fit beside a resident 27B trunk. Without chunking
    /// the same arithmetic gives ~110 GiB.
    ///
    /// The hand count is what makes it a gate. A GiB bound loose enough to be
    /// stable is far wider than a whole dropped term, so the byte total is
    /// re-derived here from the op inventory instead.
    #[test]
    fn the_reference_recipe_fits_in_vram() {
        let mut c = cfg();
        c.hidden_size = 5120;
        c.num_hidden_layers = 5;
        c.num_attention_heads = 32;
        c.head_dim = 128;
        c.block_size = 7;
        let gib = |b: usize| b as f64 / (1 << 30) as f64;

        // Counted off `backbone::forward` and `loss::dspark_loss`, one output
        // per op, at rows = 32·7 = 224, kv = 4096 + 224 = 4320, head width
        // 32·128 = 4096 and kv width 1·128 = 128 (`cfg()`'s single kv head).
        let per_layer = 4 * 32 * 224 * 4320    // scores: matmul, scale, mask, softmax
            + 10 * 224 * 4096                  // q: proj, 4× head_norm, rope, q3, 3 attn outputs
            + 7 * 4320 * 4096                  // post-`repeat_kv`: expand + reshape ×2, k3, v3, kᵀ
            + 13 * 4320 * 128                  // pre-`repeat_kv` k and v, projections through its input
            + 6 * 224 * 5120                   // residual: 2 norms, 2 projections, 2 adds
            + 4 * 224 * 16; // MLP at `cfg()`'s intermediate_size
        let context = 4096 * 2 * 5120          // tap upload, 2 tapped layers
            + 2 * 4096 * 5120                  // the `fc` GEMM and its norm
            + (4320 + 224) * 128               // rope cos/sin over the kv sequence, then the queries
            + 224 * 4320; // dense additive mask
        let head = 10 * 224 * 248_320          // draft logits, markov bias and add, trunk logits,
            //   log_softmax, both softmaxes, negate, diff, abs
            + 5 * 224 * 5120; // embedding, its reshape, final norm, aligned trunk hidden, conf input
        let derived = 4 * (per_layer * 5 + context + head);
        assert_eq!(
            derived, 8_009_330_688,
            "the term-by-term count changed — re-derive it before trusting the formula"
        );

        let chunked = peak_activation_bytes(&c, 32, 4096, 248_320);
        assert_eq!(
            chunked,
            derived,
            "the formula and the op inventory disagree by {} B",
            chunked as i64 - derived as i64
        );
        assert!(gib(chunked) < 8.0, "chunked peak {:.1} GiB", gib(chunked));

        let whole = peak_activation_bytes(&c, 512, 4096, 248_320);
        assert!(
            gib(whole) > 60.0,
            "unchunked peak {:.1} GiB — the chunking would not be earning its \
             keep and this gate is measuring the wrong thing",
            gib(whole)
        );
    }

    /// The formula is the VRAM budget the CLI prints before a run, so it has to
    /// track what the graph actually allocates. Measured against a real chunk
    /// forward; the residual is the sub-row-sized tensors it deliberately drops.
    #[test]
    fn the_peak_formula_tracks_a_measured_forward() {
        let bpb = 5;
        let (mut store, trainer) = setup(Config {
            num_anchors: bpb,
            blocks_per_backward: bpb,
            ..Config::default()
        });

        let ids: Vec<u32> = (0..SEQ as u32).map(|i| i % 20).collect();
        let mask = vec![true; SEQ];
        let anchors = block::sample_anchors(&mask, bpb, 5);
        assert_eq!(anchors.len(), bpb, "need a full chunk to measure");
        let blocks: Vec<Block> = anchors
            .iter()
            .map(|&a| block::build_block(&ids, &mask, a, cfg().block_size).unwrap())
            .collect();
        let weights: Vec<f32> = blocks
            .iter()
            .flat_map(|b| block::row_weights(b, Some(4.0)))
            .collect();
        let (taps, last_hidden) = FakeTarget.forward(&ids).unwrap();
        let ctx = SampleCtx {
            taps: &taps,
            last_hidden: &last_hidden,
            seq: SEQ,
            denom: weights.iter().sum(),
        };

        // Element counts, not `live_host_bytes`: an op that took the device
        // path leaves `data` empty even on the CPU backend.
        let live = |s: &TensorStore| -> usize {
            s.live_ids()
                .iter()
                .filter_map(|&id| s.get(id))
                .map(|t| t.size * size_of::<f32>())
                .sum()
        };
        let mut tape = Tape::new();
        let before = live(&store);
        trainer
            .chunk_loss(&blocks, &weights, &ctx, &mut store, &mut tape)
            .unwrap();
        let measured = live(&store) - before;

        // The residual is the Markov rank, the confidence input's rank columns
        // and the per-row loss vectors — all of which vanish beside `vocab` and
        // `ctx_len` at training shapes but are ~1.7% at these.
        let predicted = peak_activation_bytes(&cfg(), bpb, SEQ, VOCAB);
        let ratio = predicted as f64 / measured as f64;
        assert!(
            (0.96..=1.02).contains(&ratio),
            "formula {predicted} B vs measured {measured} B (×{ratio:.3})"
        );
    }

    /// `blocks_per_backward` is a VRAM knob, so the gradient must not move when
    /// it does. A ragged tail (9 blocks at 4) is where a per-chunk denominator
    /// re-weights a row by which chunk it happened to land in.
    #[test]
    fn chunking_does_not_change_the_gradient() {
        let (mut store, mut trainer) = setup(Config {
            num_anchors: 9,
            blocks_per_backward: 9,
            ..Config::default()
        });
        let mut tape = Tape::new();
        let s = sample(0);

        let mut grads = |trainer: &mut Trainer, bpb: usize, store: &mut TensorStore| {
            trainer.cfg.blocks_per_backward = bpb;
            AdamW::zero_grad(&mut trainer.opt, &trainer.params, store);
            let out = trainer
                .accumulate(&s, 7, 1, &mut FakeTarget, store, &mut tape)
                .unwrap()
                .expect("sample carries supervision");
            assert_eq!(out.chunks, 9usize.div_ceil(bpb));
            let g = trainer
                .params
                .iter()
                .map(|&p| {
                    let g = store.get(p).and_then(|t| t.grad).expect("no gradient");
                    store.to_host(g).unwrap()
                })
                .collect::<Vec<_>>();
            (out.loss, g)
        };
        let (whole_loss, whole) = grads(&mut trainer, 9, &mut store);
        let (split_loss, split) = grads(&mut trainer, 4, &mut store);

        assert!(
            whole.iter().flatten().any(|g| g.abs() > 1e-6),
            "no gradient reached the parameters — the gate is vacuous"
        );
        assert!(
            (whole_loss - split_loss).abs() <= 1e-5 * whole_loss.abs(),
            "the reported loss moved with the split: {whole_loss} vs {split_loss}"
        );
        for (a, b) in whole.iter().zip(&split) {
            for (x, y) in a.iter().zip(b) {
                let tol = 1e-4 * (1.0 + x.abs().max(y.abs()));
                assert!((x - y).abs() <= tol, "grad {x} != {y} across the split");
            }
        }
    }

    #[test]
    fn the_loop_reduces_the_loss() {
        let (mut store, mut trainer) = setup(Config {
            lr: 5e-2,
            warmup_ratio: 0.0,
            num_anchors: 8,
            ..Config::default()
        });
        let samples: Vec<Sample> = (0..2).map(sample).collect();
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
        assert!(
            first.loss.is_finite() && last.loss.is_finite(),
            "{} -> {}",
            first.loss,
            last.loss
        );
        assert!(
            last.loss < first.loss * 0.95,
            "loss did not fall: {} -> {}",
            first.loss,
            last.loss
        );
        assert!(first.conf > 0.0, "the confidence term is not in the loop");
        assert!(
            (0.0..=1.0).contains(&first.mean_accept),
            "mean_accept {}",
            first.mean_accept
        );
        assert!(first.grad_norm > 0.0 && first.counted == 2 && first.chunks == 2);
    }

    /// The control [`the_loop_reduces_the_loss`] needs: a falling loss says
    /// nothing until the same loop with nothing learned is known to be flat.
    ///
    /// Anchors are resampled per step from `seed + step·batch + i`, so
    /// `num_anchors` is set to the fixture's whole candidate set —
    /// `sample_anchors` then returns all of it whatever the seed, and a moving
    /// parameter is the only thing left that could move the loss.
    #[test]
    fn the_loss_does_not_move_at_lr_zero() {
        let samples: Vec<Sample> = (0..2).map(sample).collect();
        let (mut store, mut trainer) = setup(Config {
            lr: 0.0,
            weight_decay: 0.0,
            warmup_ratio: 0.0,
            num_anchors: block::anchor_candidates(&samples[0].loss_mask).len(),
            ..Config::default()
        });
        let mut tape = Tape::new();
        let snapshot = |t: &Trainer, s: &mut TensorStore| -> Vec<Vec<f32>> {
            t.params.iter().map(|&p| s.to_host(p).unwrap()).collect()
        };

        let before = snapshot(&trainer, &mut store);
        let first = trainer
            .train_step(&samples, &mut FakeTarget, &mut store, &mut tape)
            .unwrap();
        let mut last = first;
        for _ in 0..9 {
            last = trainer
                .train_step(&samples, &mut FakeTarget, &mut store, &mut tape)
                .unwrap();
        }

        assert_eq!(first.lr, 0.0, "the schedule handed the optimizer an lr");
        assert!(
            first.grad_norm > 0.0,
            "no gradient reached the optimizer — nothing was held back and the \
             control is vacuous"
        );
        assert!(
            (last.loss - first.loss).abs() <= f32::EPSILON * first.loss.abs(),
            "the loss moved with nothing learned: {} -> {}",
            first.loss,
            last.loss
        );
        for (i, (a, b)) in before
            .iter()
            .zip(&snapshot(&trainer, &mut store))
            .enumerate()
        {
            assert_eq!(a, b, "parameter {i} moved after 10 steps at lr = 0");
        }
    }
}
