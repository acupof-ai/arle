//! DSpark RL sidecar trainer smoke test.
//!
//! Verifies:
//! 1. Trainer constructs without error
//! 2. `train_step` runs on synthetic experiences without panicking
//! 3. Loss is finite
//! 4. Weights can be extracted and have the right shape
//! 5. Multiple steps run cleanly (no resource leaks / tape issues)

use train::dspark_rl::{DsparkExperience, DsparkRlConfig, DsparkRlTrainer};

const VOCAB: usize = 100;
const BLOCK: usize = 4;
const RANK: usize = 8;

fn make_experience(accepted: usize) -> DsparkExperience {
    // Draft logits: random but with a peak at the "correct" token.
    let mut draft_logits = vec![0.0f32; BLOCK * VOCAB];
    let mut tokens = Vec::with_capacity(BLOCK);
    for b in 0..BLOCK {
        let tok = (b * 7 + 3) % VOCAB;
        tokens.push(tok as u32);
        draft_logits[b * VOCAB + tok] = 5.0; // strong peak
    }
    DsparkExperience {
        draft_tokens: tokens,
        draft_logits,
        target_logits: vec![0.0; BLOCK * VOCAB],
        accepted,
        block_size: BLOCK,
        vocab_size: VOCAB,
    }
}

#[test]
fn dspark_rl_trainer_smoke() {
    let config = DsparkRlConfig {
        vocab_size: VOCAB,
        markov_rank: RANK,
        learning_rate: 1e-3,
        batch_size: 4,
        baseline_ema_alpha: 0.1,
    };
    let mut trainer = DsparkRlTrainer::new(config).expect("trainer should construct");

    // Mix of high and low acceptance experiences.
    let experiences: Vec<DsparkExperience> = (0..8)
        .map(|i| make_experience(if i % 2 == 0 { BLOCK } else { 0 }))
        .collect();

    let loss = trainer
        .train_step(&experiences)
        .expect("train_step should succeed");
    assert!(loss.is_finite(), "loss should be finite, got {loss}");

    // Run several more steps to check stability.
    for _ in 0..5 {
        let loss = trainer
            .train_step(&experiences)
            .expect("subsequent train_step should succeed");
        assert!(loss.is_finite(), "loss should stay finite");
    }

    // Extract weights — shapes must match the config.
    let (w1, w2) = trainer.get_weights().expect("get_weights should succeed");
    assert_eq!(
        w1.len(),
        VOCAB * RANK,
        "w1 should be [vocab * rank], got {}",
        w1.len()
    );
    assert_eq!(
        w2.len(),
        RANK * VOCAB,
        "w2 should be [rank * vocab], got {}",
        w2.len()
    );
}

#[test]
fn dspark_rl_trainer_empty_batch() {
    let config = DsparkRlConfig {
        vocab_size: VOCAB,
        markov_rank: RANK,
        ..Default::default()
    };
    let mut trainer = DsparkRlTrainer::new(config).unwrap();
    let loss = trainer.train_step(&[]).unwrap();
    assert_eq!(loss, 0.0, "empty batch should return 0 loss");
}

#[test]
fn dspark_rl_trainer_converges() {
    // If we always give the same experience with full acceptance, the trainer
    // should increase the log-prob of those tokens (loss decreases).
    let config = DsparkRlConfig {
        vocab_size: VOCAB,
        markov_rank: RANK,
        learning_rate: 0.1,
        batch_size: 4,
        baseline_ema_alpha: 0.5,
    };
    let mut trainer = DsparkRlTrainer::new(config).unwrap();

    let exp = make_experience(BLOCK); // full acceptance
    let batch = vec![exp.clone(); 4];

    let first_loss = trainer.train_step(&batch).unwrap();
    let mut prev_loss = first_loss;
    for _ in 0..20 {
        let loss = trainer.train_step(&batch).unwrap();
        // Loss should generally decrease (or stay low) with consistent reward.
        assert!(loss.is_finite(), "loss should stay finite");
        prev_loss = loss;
    }
    // After 20 steps of full-acceptance signal, loss should be lower than start.
    assert!(
        prev_loss < first_loss,
        "loss should decrease with consistent positive reward: start={first_loss} end={prev_loss}"
    );
}
