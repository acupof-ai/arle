use train::curriculum::Difficulty;
use train::dataset::LcgRng;
use train::task_gen::{TaskGenerator, TierSpec};
use train::verifier::VerifierKind;

fn easy_medium_hard() -> Vec<TierSpec> {
    vec![
        TierSpec {
            difficulty: Difficulty::Easy,
            weight: 3.0,
            target_range: (2, 4),
            prompt_len: (4, 8),
            verifiers: vec![VerifierKind::Copy],
        },
        TierSpec {
            difficulty: Difficulty::Medium,
            weight: 1.0,
            target_range: (4, 8),
            prompt_len: (8, 16),
            verifiers: vec![VerifierKind::Copy, VerifierKind::ReverseCopy],
        },
        TierSpec {
            difficulty: Difficulty::Hard,
            weight: 1.0,
            target_range: (8, 16),
            prompt_len: (16, 24),
            verifiers: vec![VerifierKind::Palette {
                allowed_tokens: vec![1, 2, 3],
            }],
        },
    ]
}

fn tier_of(difficulty: Difficulty, tiers: &[TierSpec]) -> &TierSpec {
    tiers
        .iter()
        .find(|t| t.difficulty == difficulty)
        .expect("tier present")
}

#[test]
fn every_generated_task_carries_verifier_and_fits_tier_bounds() {
    let tiers = easy_medium_hard();
    let mut generator = TaskGenerator::new(tiers.clone());
    let mut rng = LcgRng::seed(7);

    for _ in 0..256 {
        let generated = generator.generate(&mut rng);
        let tier = tier_of(generated.task.difficulty, &tiers);
        assert!(
            (tier.target_range.0..=tier.target_range.1).contains(&generated.task.target_range),
            "target_range {} out of tier bounds {:?}",
            generated.task.target_range,
            tier.target_range,
        );
        assert!(
            (tier.prompt_len.0..=tier.prompt_len.1).contains(&generated.task.prompt_len),
            "prompt_len {} out of tier bounds {:?}",
            generated.task.prompt_len,
            tier.prompt_len,
        );
        let verifier_allowed = tier.verifiers.iter().any(|kind| {
            std::mem::discriminant(kind) == std::mem::discriminant(&generated.verifier)
        });
        assert!(verifier_allowed, "verifier kind escaped tier allowlist");
    }
}

#[test]
fn task_ids_are_monotonic_across_calls() {
    let mut generator = TaskGenerator::new(easy_medium_hard());
    let mut rng = LcgRng::seed(11);
    let mut prev = None;
    for _ in 0..16 {
        let generated = generator.generate(&mut rng);
        if let Some(prev_id) = prev {
            assert!(
                generated.task.id > prev_id,
                "ids must strictly increase: {prev_id} -> {}",
                generated.task.id,
            );
        }
        prev = Some(generated.task.id);
    }
}

#[test]
fn difficulty_distribution_approximates_declared_weights() {
    let tiers = easy_medium_hard();
    let mut generator = TaskGenerator::new(tiers);
    let mut rng = LcgRng::seed(2026);
    let samples = 4096;

    let mut easy = 0usize;
    let mut medium = 0usize;
    let mut hard = 0usize;
    for _ in 0..samples {
        match generator.generate(&mut rng).task.difficulty {
            Difficulty::Easy => easy += 1,
            Difficulty::Medium => medium += 1,
            Difficulty::Hard => hard += 1,
        }
    }

    // Declared weights: 3/5, 1/5, 1/5. Allow ±5% absolute slack.
    let total = samples as f32;
    let easy_frac = easy as f32 / total;
    let medium_frac = medium as f32 / total;
    let hard_frac = hard as f32 / total;
    assert!((easy_frac - 0.6).abs() < 0.05, "easy {easy_frac}");
    assert!((medium_frac - 0.2).abs() < 0.05, "medium {medium_frac}");
    assert!((hard_frac - 0.2).abs() < 0.05, "hard {hard_frac}");
}

#[test]
#[should_panic(expected = "verifier")]
fn rejects_tier_without_verifier() {
    let _ = TaskGenerator::new(vec![TierSpec {
        difficulty: Difficulty::Easy,
        weight: 1.0,
        target_range: (2, 4),
        prompt_len: (4, 8),
        verifiers: vec![],
    }]);
}

#[test]
#[should_panic(expected = "generator needs")]
fn rejects_empty_tier_list() {
    let _ = TaskGenerator::new(Vec::new());
}
