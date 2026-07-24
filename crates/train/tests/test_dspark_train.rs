//! DSpark trainer smoke test.
//!
//! Verifies:
//! 1. Trainer constructs without error
//! 2. `train_step` runs on synthetic experiences without panicking
//! 3. Loss is finite
//! 4. Weights can be extracted and have the right shape
//! 5. Multiple steps run cleanly (no resource leaks / tape issues)

use train::dspark_train::{DsparkExperience, DsparkTrainConfig, DsparkTrainer};

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
        next_token_heads: false,
    }
}

#[test]
fn dspark_trainer_smoke() {
    let config = DsparkTrainConfig {
        markov_rank: RANK,
        learning_rate: 1e-3,
        batch_size: 4,
        baseline_ema_alpha: 0.1,
        ..Default::default()
    };
    let mut trainer = DsparkTrainer::new(
        config,
        None,
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    )
    .expect("trainer should construct");

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
fn dspark_trainer_empty_batch() {
    let config = DsparkTrainConfig {
        markov_rank: RANK,
        ..Default::default()
    };
    let mut trainer = DsparkTrainer::new(
        config,
        None,
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    )
    .unwrap();
    let loss = trainer.train_step(&[]).unwrap();
    assert_eq!(loss, 0.0, "empty batch should return 0 loss");
}

/// D1 + D3: the trainer's first-step loss must equal a reference computed in
/// the SERVE frame (w2 `[vocab, rank]` row-major, `bias[v] = Σ_r
/// w2[v*rank+r]·w1[cond*rank+r]`) with same-position row alignment (draft row
/// j = t+1 drafted chain[t+1] conditioned on chain[t], paired with target row
/// t; draft row 0 and the last target row excluded — both are NaN canaries
/// here, so any misalignment yields a non-finite loss).
#[test]
fn dspark_trainer_serve_frame_and_alignment() {
    const V: usize = 4;
    const R: usize = 2;
    let chain: Vec<u32> = vec![0, 2, 1];
    let block = chain.len(); // same-position: chain.len() == draft rows

    // Asymmetric weights so the [vocab, rank] and [rank, vocab] frames disagree.
    let w1: Vec<f32> = (0..V * R).map(|i| 0.1 * (i as f32) - 0.3).collect();
    let w2: Vec<f32> = (0..V * R)
        .map(|i| 0.07 * (i as f32 * i as f32) - 0.2)
        .collect();

    let mut draft_logits = vec![f32::NAN; V]; // row 0: dead (never drafts)
    draft_logits.extend((0..2 * V).map(|i| 0.3 * (i as f32) - 0.8));
    let mut target_logits: Vec<f32> = (0..2 * V).map(|i| -0.2 * (i as f32) + 0.5).collect();
    target_logits.extend(vec![f32::NAN; V]); // last row: beyond trained pairs

    let accepted = 2usize;
    let exp = DsparkExperience {
        draft_tokens: chain.clone(),
        draft_logits: draft_logits.clone(),
        target_logits: target_logits.clone(),
        accepted,
        block_size: block,
        vocab_size: V,
        next_token_heads: false,
    };

    let ema_alpha = 0.5f32;
    let baseline_init = 0.5f32;
    let alpha = 0.5f32; // prob_match_alpha default
    let gamma = 4.0f32;
    let config = DsparkTrainConfig {
        markov_rank: R,
        learning_rate: 1e-4,
        batch_size: 1,
        baseline_ema_alpha: ema_alpha,
        baseline_init,
        prob_match_alpha: alpha,
        loss_decay_gamma: Some(gamma),
        max_grad_norm: Some(1.0),
    };
    let mut trainer = DsparkTrainer::new(
        config,
        Some((w1.clone(), w2.clone(), R)),
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    )
    .unwrap();
    let loss = trainer.train_step(std::slice::from_ref(&exp)).unwrap();
    assert!(loss.is_finite(), "misaligned rows pulled in a NaN canary");

    // Reference in the serve frame.
    let softmax = |x: &[f32]| {
        let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = x.iter().map(|&v| (v - m).exp()).collect();
        let s: f32 = e.iter().sum();
        e.iter().map(|&v| v / s).collect::<Vec<f32>>()
    };
    let reward = accepted as f32 / block as f32;
    let baseline = (1.0 - ema_alpha) * baseline_init + ema_alpha * reward;
    let adv = reward - baseline;
    let (mut pg, mut pm, mut wsum) = (0.0f32, 0.0f32, 0.0f32);
    for t in 0..block - 1 {
        let j = t + 1; // same-position draft row
        let cond = chain[t] as usize;
        let tok = chain[t + 1] as usize;
        let w = (-(t as f32) / gamma).exp();
        wsum += w;
        let corrected: Vec<f32> = (0..V)
            .map(|v| {
                let bias: f32 = (0..R).map(|r| w2[v * R + r] * w1[cond * R + r]).sum();
                draft_logits[j * V + v] + bias
            })
            .collect();
        let p = softmax(&corrected);
        pg += -(p[tok].ln()) * adv * w;
        let q = softmax(&target_logits[t * V..(t + 1) * V]);
        pm += w * p
            .iter()
            .zip(&q)
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>();
    }
    let expected = (1.0 - alpha) * (pg / wsum) + alpha * (pm / (wsum * V as f32));
    assert!(
        (loss - expected).abs() < 1e-4,
        "trainer loss {loss} != serve-frame reference {expected}"
    );
}

#[test]
fn dspark_trainer_converges() {
    // If we always give the same experience with full acceptance, the trainer
    // should increase the log-prob of those tokens (loss decreases).
    let config = DsparkTrainConfig {
        markov_rank: RANK,
        learning_rate: 0.1,
        batch_size: 4,
        baseline_ema_alpha: 0.5,
        ..Default::default()
    };
    let mut trainer = DsparkTrainer::new(
        config,
        None,
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    )
    .unwrap();

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
