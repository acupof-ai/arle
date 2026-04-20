//! D9 — model convergence smoke: tiny model must learn (loss[0] - loss[9] > 0.5).
//!
//! Catches any broken backward op that synthetic-loss tests miss. Every
//! other Trainer test feeds a `mean(p * p)` loss that does not exercise
//! `embedding` / `matmul` / `log_softmax` / `gather_last_dim` backward paths
//! end-to-end. This one does, on a 4-token copy task with deterministic
//! parameter init (no RNG).
//!
//! Task shape:
//!   tokens  = [0, 1, 2, 3]
//!   targets = [0, 1, 2, 3]   (identity / copy task)
//!   model   = embedding(vocab=4, dim=8) -> reshape -> linear(8 -> vocab=4)
//!   loss    = cross_entropy(logits, targets)
//!
//! With AdamW lr=1e-2 over 10 steps the loss must drop by > 0.5. The margin
//! was chosen empirically; lower bound is 0.3 if the concrete task ever
//! becomes thermally noisy in CI.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use autograd::ops::{embedding, matmul, mul_scalar, reshape, transpose};
use autograd::{
    AdamW, ConstantLr, Tape, Tensor, TensorId, TensorStore,
    ops::{gather_last_dim, log_softmax, mean},
};
use train::grad_clip::NoClip;
use train::metrics::{MetricSample, MetricSink};
use train::{StepCtx, StepOutcome, Trainer, TrainerConfig};

// ---------------------------------------------------------------------------
// Collecting MetricSink — minimal reuse of the VecSink pattern in
// `test_trainer_loop.rs`. Kept private to this file so no cross-test coupling.
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct OwnedSample {
    #[allow(dead_code)]
    step: u64,
    fields: Vec<(String, f64)>,
}

struct VecSink {
    buf: Arc<Mutex<Vec<OwnedSample>>>,
}

impl MetricSink for VecSink {
    fn emit(&mut self, sample: &MetricSample<'_>) {
        let fields = sample
            .fields
            .iter()
            .map(|(k, v)| ((*k).to_string(), *v))
            .collect();
        self.buf.lock().unwrap().push(OwnedSample {
            step: sample.step,
            fields,
        });
    }
}

// ---------------------------------------------------------------------------
// Deterministic init helpers. No RNG: small distinct values so initial logits
// are non-degenerate (not all equal) but also not saturated.
// ---------------------------------------------------------------------------

const VOCAB: usize = 4;
const DIM: usize = 8;

fn init_embedding_table() -> Vec<f32> {
    // Shape [VOCAB, DIM]. Each row has a distinct high-magnitude signature
    // so `linear(embed[i])` projects to well-separated logits once the
    // linear layer rotates into place. Magnitudes chosen to dominate the
    // linear-weight ramp below → large initial gradients through both
    // backward paths.
    let mut data = vec![0.0_f32; VOCAB * DIM];
    for row in 0..VOCAB {
        for col in 0..DIM {
            let sign = if (row + col) % 2 == 0 { 1.0 } else { -1.0 };
            let mag = 1.5 + 0.15 * ((row * DIM + col) as f32);
            data[row * DIM + col] = sign * mag;
        }
    }
    data
}

fn init_linear_weight() -> Vec<f32> {
    // Linear is stored as [out_features, in_features] = [VOCAB, DIM] and
    // multiplied as x @ Wᵀ internally. Start with small, near-uniform
    // deterministic values so initial logits are close to uniform
    // (loss ≈ ln(VOCAB) ≈ 1.386). This maximizes the cross-entropy gradient
    // magnitude on every step.
    let mut data = vec![0.0_f32; VOCAB * DIM];
    for out in 0..VOCAB {
        for inp in 0..DIM {
            let sign = if (out * 3 + inp) % 2 == 0 { 1.0 } else { -1.0 };
            let mag = 0.0001 + 0.00005 * ((out * DIM + inp) as f32);
            data[out * DIM + inp] = sign * mag;
        }
    }
    data
}

// ---------------------------------------------------------------------------
// Forward: embedding -> reshape -> matmul(with transposed weight) -> CE loss.
// We hand-roll the linear projection (instead of using `module::Linear`) to
// keep the two trainable params explicit and the init deterministic.
// ---------------------------------------------------------------------------

fn forward_loss(
    embed_id: TensorId,
    weight_id: TensorId,
    tokens: &[usize],
    targets: &[usize],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> autograd::Result<TensorId> {
    // embedding -> [1, seq, DIM]
    let embedded = embedding(embed_id, tokens, store, tape)?;
    // reshape to [seq, DIM] so the matmul is rank-2 × rank-2.
    let flat = reshape(embedded, &[tokens.len(), DIM], store, tape)?;
    // weight is stored [VOCAB, DIM]; transpose to [DIM, VOCAB] for x @ Wᵀ.
    let weight_t = transpose(weight_id, 0, 1, store, tape)?;
    // logits [seq, VOCAB]
    let logits = matmul(flat, weight_t, store, tape)?;
    // cross_entropy = -mean(gather_last_dim(log_softmax(logits), targets))
    let log_probs = log_softmax(logits, store, tape)?;
    let picked = gather_last_dim(log_probs, targets, store, tape)?;
    let mean_lp = mean(picked, store, tape)?;
    mul_scalar(mean_lp, -1.0, store, tape)
}

// ---------------------------------------------------------------------------
// The test.
// ---------------------------------------------------------------------------

#[test]
fn convergence_smoke_tiny_model_learns() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();

    // Deterministic init. `requires_grad=true` → AdamW will update both.
    let embed_id = store.alloc(
        Tensor::new(init_embedding_table(), vec![VOCAB, DIM], true).expect("embedding tensor init"),
    );
    let weight_id = store.alloc(
        Tensor::new(init_linear_weight(), vec![VOCAB, DIM], true)
            .expect("linear weight tensor init"),
    );

    let tokens: Vec<usize> = (0..VOCAB).collect();
    let targets: Vec<usize> = (0..VOCAB).collect();

    let buf: Arc<Mutex<Vec<OwnedSample>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = VecSink {
        buf: Arc::clone(&buf),
    };

    let optim = AdamW::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
    let cfg = TrainerConfig {
        total_steps: 10,
        grad_accum_steps: 1,
        log_every: 1,
        eval_every: None,
        save_every: None,
        save_dir: None,
        resume_from: None,
        rng_seed: 0,
    };
    let mut trainer = Trainer::new(optim, NoClip, ConstantLr(1e-2), Box::new(sink), cfg);

    let params = vec![embed_id, weight_id];
    let names = vec![
        (embed_id, "embed".to_string()),
        (weight_id, "weight".to_string()),
    ];
    let tokens_for_closure = tokens.clone();
    let targets_for_closure = targets.clone();

    trainer
        .run(
            &mut store,
            &mut tape,
            params,
            names,
            HashSet::new(),
            |ctx: &mut StepCtx<'_>| {
                let loss_id = forward_loss(
                    embed_id,
                    weight_id,
                    &tokens_for_closure,
                    &targets_for_closure,
                    ctx.store,
                    ctx.tape,
                )?;
                Ok(StepOutcome {
                    loss_id,
                    token_count: tokens_for_closure.len() as u64,
                })
            },
        )
        .expect("trainer.run");

    assert_eq!(trainer.step(), 10, "trainer should have run 10 steps");

    let samples = buf.lock().unwrap();
    assert_eq!(
        samples.len(),
        10,
        "log_every=1 + total_steps=10 must emit exactly 10 samples, got {}",
        samples.len()
    );
    let losses: Vec<f64> = samples
        .iter()
        .map(|s| {
            s.fields
                .iter()
                .find(|(k, _)| k == "loss")
                .map(|(_, v)| *v)
                .expect("every sample carries a `loss` field")
        })
        .collect();

    let delta = losses[0] - losses[9];
    // Margin: the spec requires > 0.5; documented in the file header that 0.3
    // is the hard floor if this ever becomes noisy. We keep the stricter
    // 0.5 here and surface the concrete delta in the assertion message so a
    // regression is obvious. On the reference init this run observes
    // loss[0]≈1.356, loss[9]≈0.681, delta≈0.675.
    assert!(
        delta > 0.5,
        "expected loss[0] - loss[9] > 0.5, got {delta:.4} (losses: {losses:?})"
    );
}
