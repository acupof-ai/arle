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

    let accepted = 1usize;
    let exp = DsparkExperience {
        draft_tokens: chain.clone(),
        draft_logits: draft_logits.clone(),
        target_logits: target_logits.clone(),
        accepted,
        block_size: block,
        vocab_size: V,
        next_token_heads: false,
    };

    let alpha = 0.9f32; // α_tv
    let gamma = 4.0f32;
    let config = DsparkTrainConfig {
        markov_rank: R,
        learning_rate: 1e-4,
        batch_size: 1,
        prob_match_alpha: alpha,
        loss_decay_gamma: Some(gamma),
        max_grad_norm: Some(1.0),
        ..Default::default()
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
    let argmax = |x: &[f32]| {
        x.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    let (mut ce, mut tv, mut wsum) = (0.0f32, 0.0f32, 0.0f32);
    for t in 0..block - 1 {
        let j = t + 1; // same-position draft row
        let cond = chain[t] as usize;
        let w = (-(t as f32) / gamma).exp();
        wsum += w;
        let corrected: Vec<f32> = (0..V)
            .map(|v| {
                let bias: f32 = (0..R).map(|r| w2[v * R + r] * w1[cond * R + r]).sum();
                draft_logits[j * V + v] + bias
            })
            .collect();
        let p = softmax(&corrected);
        let q = softmax(&target_logits[t * V..(t + 1) * V]);
        // CE label is the TRUNK's token, not the drafted one.
        ce += -(p[argmax(&target_logits[t * V..(t + 1) * V])].ln()) * w;
        // TV is an L1 SUM over vocab; only the token axis is averaged.
        tv += w * p.iter().zip(&q).map(|(&a, &b)| (a - b).abs()).sum::<f32>();
    }
    let expected = (1.0 - alpha) * (ce / wsum) + alpha * (tv / wsum);
    assert!(
        (loss - expected).abs() < 1e-4,
        "trainer loss {loss} != serve-frame reference {expected}"
    );
}

#[test]
fn dspark_trainer_converges() {
    // Repeating one experience, the head should fit the trunk's distribution
    // on it: both CE and TV fall.
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

/// The saved head must land in the DRAFT LOADER's frame — those exact tensor
/// names, bf16, `[vocab, rank]` — or the checkpoint is unloadable and a whole
/// training run is wasted.
#[test]
fn dspark_trainer_saves_loadable_markov_head() {
    let mut trainer = DsparkTrainer::new(
        DsparkTrainConfig {
            markov_rank: RANK,
            ..Default::default()
        },
        None,
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    )
    .unwrap();
    trainer
        .train_step(&vec![make_experience(BLOCK); 4])
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested/markov_head.safetensors");
    trainer.save_weights(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let (w1, _) = trainer.get_weights().unwrap();
    for name in [
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
    ] {
        let t = st.tensor(name).unwrap();
        assert_eq!(t.dtype(), safetensors::Dtype::BF16, "{name} dtype");
        assert_eq!(t.shape(), [VOCAB, RANK], "{name} shape");
    }
    // Values must survive the f32 -> bf16 round trip, not just the shape — and
    // the reader that feeds `--dspark-markov-init` must agree with the writer.
    let (round1, _) = train::dspark_train::load_markov_head(&path).unwrap();
    assert_eq!(round1.len(), w1.len());
    for (i, (&a, &b)) in w1.iter().zip(&round1).enumerate() {
        assert!((a - b).abs() <= a.abs() * 0.01 + 1e-6, "w1[{i}] {a} vs {b}");
    }

    // A failed save must not eat the previous good checkpoint (write + rename).
    assert!(trainer.save_weights(&path).is_ok());
    assert!(!path.with_extension("safetensors.tmp").exists());
}

/// ISO composed with the real trainer: after N acceptance-weighted steps the
/// head's singular spectrum must still be the base checkpoint's, and the head
/// must still have moved. Pinning the spectrum by accident freezing the whole
/// head would pass a spectrum check and learn nothing.
#[test]
fn iso_fixed_spectrum_pins_the_spectrum_without_freezing_the_head() {
    let seed_w1: Vec<f32> = (0..VOCAB * RANK)
        .map(|i| ((i * 31 % 97) as f32 / 97.0 - 0.5) * (1.0 + (i % RANK) as f32 / 4.0))
        .collect();
    let seed_w2: Vec<f32> = (0..VOCAB * RANK)
        .map(|i| (i * 17 % 89) as f32 / 89.0 - 0.5)
        .collect();

    let run = |iso: bool| -> (Vec<f32>, f32) {
        let mut trainer = DsparkTrainer::new(
            DsparkTrainConfig {
                markov_rank: RANK,
                learning_rate: 1e-2, // large enough that an unconstrained step moves Σ
                iso_fixed_spectrum: iso,
                // Retraction shares the publish cadence; 10 steps at 2 lands on
                // a boundary, so the strict on-manifold invariant must hold at
                // the end AND the cadence path is the one under test.
                swap_every: 2,
                ..Default::default()
            },
            Some((seed_w1.clone(), seed_w2.clone(), RANK)),
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        )
        .unwrap();
        let batch = vec![make_experience(BLOCK); 4];
        for _ in 0..10 {
            trainer.train_step(&batch).unwrap();
        }
        let (w1, _) = trainer.get_weights().unwrap();
        // Spectrum drift = relative L2 between the SORTED eigenvalues of w1ᵀw1
        // (= σ(W)² multiset). Paper ISO rotates the frames freely, so the Gram
        // MATRIX moves; only its eigenvalue multiset is fixed — compare that.
        let spectrum = |w: &[f32]| -> Vec<f64> {
            let mut g = vec![0.0f64; RANK * RANK];
            for row in w.chunks_exact(RANK) {
                for (i, &a) in row.iter().enumerate() {
                    for (j, &b) in row.iter().enumerate() {
                        g[i * RANK + j] += f64::from(a) * f64::from(b);
                    }
                }
            }
            let mut eig = train::iso_spectrum::jacobi_eig(&g, RANK).1;
            eig.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eig
        };
        let (g0, g1) = (spectrum(&seed_w1), spectrum(&w1));
        let num: f64 = g0.iter().zip(&g1).map(|(a, b)| (a - b).powi(2)).sum();
        let den: f64 = g0.iter().map(|a| a * a).sum();
        (w1, (num / den).sqrt() as f32)
    };

    let (w1_iso, drift_iso) = run(true);
    let (_, drift_free) = run(false);
    assert!(
        drift_iso < 1e-3,
        "ISO must hold the base singular values: relative spectrum drift {drift_iso}"
    );
    assert!(
        drift_free > drift_iso * 10.0,
        "the unconstrained arm must move the spectrum, else the test proves nothing: \
         free={drift_free} iso={drift_iso}"
    );
    let moved = w1_iso
        .iter()
        .zip(&seed_w1)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        moved > 1e-5,
        "ISO must still rotate the frames, not freeze the head: max |Δw1| = {moved}"
    );
}

/// Exit/resume must preserve the original Σ₀. ISO retracts before every publish,
/// so a saved head is on ℱ(W₀) and its singular values ARE Σ₀ — a resumed trainer
/// re-captures the same spectrum from the reloaded weight, no separate Σ₀ file.
#[test]
fn iso_resume_recovers_the_same_spectrum() {
    let seed_w1: Vec<f32> = (0..VOCAB * RANK)
        .map(|i| ((i * 29 % 83) as f32 / 83.0 - 0.5) * (1.0 + (i % RANK) as f32 / 3.0))
        .collect();
    let seed_w2: Vec<f32> = (0..VOCAB * RANK)
        .map(|i| (i * 19 % 71) as f32 / 71.0 - 0.5)
        .collect();
    let cfg = || DsparkTrainConfig {
        markov_rank: RANK,
        learning_rate: 1e-2,
        iso_fixed_spectrum: true,
        // Retract every step so both the saved and resumed heads are read
        // on-manifold — the invariant under test is capture↔materialize, not the
        // between-cadence drift the other ISO test already covers.
        swap_every: 1,
        ..Default::default()
    };
    let flag = || std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Train, then save a retracted head.
    let mut a = DsparkTrainer::new(
        cfg(),
        Some((seed_w1.clone(), seed_w2.clone(), RANK)),
        flag(),
    )
    .unwrap();
    let batch = vec![make_experience(BLOCK); 4];
    for _ in 0..4 {
        a.train_step(&batch).unwrap();
    }
    let (saved_w1, _) = a.get_weights().unwrap(); // materialized (retract-then-read)

    // Resume: a fresh trainer seeded from the saved head re-captures Σ₀.
    let mut b = DsparkTrainer::new(
        cfg(),
        Some((saved_w1.clone(), seed_w2.clone(), RANK)),
        flag(),
    )
    .unwrap();
    b.train_step(&batch).unwrap();
    let (resumed_w1, _) = b.get_weights().unwrap();

    let sv = |w: &[f32]| {
        let mut g = vec![0.0f64; RANK * RANK];
        for row in w.chunks_exact(RANK) {
            for (i, &x) in row.iter().enumerate() {
                for (j, &y) in row.iter().enumerate() {
                    g[i * RANK + j] += f64::from(x) * f64::from(y);
                }
            }
        }
        let mut e = train::iso_spectrum::jacobi_eig(&g, RANK).1;
        e.sort_by(|x, y| x.partial_cmp(y).unwrap());
        e
    };
    let (s_saved, s_resumed) = (sv(&saved_w1), sv(&resumed_w1));
    let num: f64 = s_saved
        .iter()
        .zip(&s_resumed)
        .map(|(x, y)| (x - y).powi(2))
        .sum();
    let den: f64 = s_saved.iter().map(|x| x * x).sum();
    assert!(
        (num / den).sqrt() < 1e-3,
        "resumed spectrum must equal the saved Σ₀: {s_resumed:?} vs {s_saved:?}"
    );
}

/// ISO on + no seeded head (cold w2 = 0) must be rejected at construction, not
/// fail every train step on the background thread.
#[test]
fn iso_without_seed_is_rejected_at_construction() {
    let r = DsparkTrainer::new(
        DsparkTrainConfig {
            markov_rank: RANK,
            iso_fixed_spectrum: true,
            ..Default::default()
        },
        None,
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    );
    assert!(r.is_err(), "ISO with no seeded head must fail fast");
}

/// Pure prob-match (α=1) at a large vocab must produce a live loss and move the
/// head. Guards the 2026-07-28 fix: the PM term was `Σ(Δsoftmax)²/(wsum·V)`, which
/// at V≈248k underflowed f32 to 0.0000 — a null objective that trained nothing.
/// A small-V α=0.5 oracle cannot catch that; this reproduces the regime. Seeds a
/// nonzero head so w1's gradient isn't gated by cold w2≈0.
#[test]
fn prob_match_alpha_one_trains_at_large_vocab() {
    const V: usize = 8192; // large enough that a /V normalization underflows the term
    const R: usize = 8;
    let seed_w1: Vec<f32> = (0..V * R)
        .map(|i| ((i % 97) as f32 / 97.0 - 0.5) * 0.1)
        .collect();
    let seed_w2: Vec<f32> = (0..V * R)
        .map(|i| ((i % 89) as f32 / 89.0 - 0.5) * 0.1)
        .collect();

    // Draft and target deliberately diverge, so the prob-match distance is real.
    let mut draft_logits = vec![0.0f32; BLOCK * V];
    let mut target_logits = vec![0.0f32; BLOCK * V];
    let mut tokens = Vec::with_capacity(BLOCK);
    for b in 0..BLOCK {
        let d = (b * 7 + 3) % V;
        let t = (b * 13 + 500) % V;
        tokens.push(d as u32);
        draft_logits[b * V + d] = 6.0;
        target_logits[b * V + t] = 6.0;
    }
    let exp = DsparkExperience {
        draft_tokens: tokens,
        draft_logits,
        target_logits,
        accepted: BLOCK,
        block_size: BLOCK,
        vocab_size: V,
        next_token_heads: false,
    };

    let mut trainer = DsparkTrainer::new(
        DsparkTrainConfig {
            markov_rank: R,
            learning_rate: 1e-2,
            batch_size: 4,
            prob_match_alpha: 1.0, // pure prob-match — PG weight is zero
            ..Default::default()
        },
        Some((seed_w1.clone(), seed_w2, R)),
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    )
    .unwrap();

    let loss = trainer.train_step(&vec![exp; 4]).unwrap();
    assert!(
        loss > 1e-3,
        "α=1 PM loss must be a live gradient at V={V}, not underflowed to ~0: got {loss}"
    );
    let (w1, _) = trainer.get_weights().unwrap();
    let moved = w1
        .iter()
        .zip(&seed_w1)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        moved > 1e-6,
        "pure PM must move the head: max |Δw1| = {moved}"
    );
}
