use train::curriculum::{Difficulty, Task, TaskPool};
use train::dataset::LcgRng;

fn make_task(id: usize, difficulty: Difficulty) -> Task {
    Task {
        id,
        difficulty,
        target_range: 4,
        prompt_len: 8,
    }
}

#[test]
fn add_returns_monotonic_indices_and_tracks_len() {
    let mut pool = TaskPool::new(4, 0.8);
    assert!(pool.is_empty());
    assert_eq!(pool.add(make_task(0, Difficulty::Easy)), 0);
    assert_eq!(pool.add(make_task(1, Difficulty::Medium)), 1);
    assert_eq!(pool.len(), 2);
    assert!(!pool.is_empty());
    assert_eq!(pool.task(0).id, 0);
    assert_eq!(pool.task(1).difficulty, Difficulty::Medium);
}

#[test]
fn sample_returns_some_valid_index_while_pool_nonempty() {
    let mut pool = TaskPool::new(4, 0.8);
    pool.add(make_task(0, Difficulty::Easy));
    pool.add(make_task(1, Difficulty::Hard));

    let mut rng = LcgRng::seed(42);
    for _ in 0..32 {
        let index = pool.sample(&mut rng).expect("sample");
        assert!(index < pool.len());
    }
}

#[test]
fn sample_returns_none_when_all_retired() {
    let mut pool = TaskPool::with_min_samples(4, 0.5, 2);
    pool.add(make_task(0, Difficulty::Easy));
    pool.record(0, true);
    pool.record(0, true);
    assert_eq!(pool.maybe_retire(), 1);

    let mut rng = LcgRng::seed(1);
    assert!(pool.sample(&mut rng).is_none());
    assert!(pool.is_retired(0));
}

#[test]
fn record_updates_rolling_pass_at_1() {
    let mut pool = TaskPool::new(4, 0.9);
    pool.add(make_task(0, Difficulty::Easy));

    assert_eq!(pool.pass_at_1(0), None);
    pool.record(0, true);
    pool.record(0, false);
    pool.record(0, true);
    pool.record(0, true);
    let rate = pool.pass_at_1(0).expect("rate");
    assert!((rate - 0.75).abs() < 1e-6, "got {rate}");

    // Sliding window drops the oldest sample.
    pool.record(0, false);
    let rate = pool.pass_at_1(0).expect("rate");
    assert!((rate - 0.5).abs() < 1e-6, "got {rate}");
}

#[test]
fn retire_requires_min_samples_and_threshold() {
    let mut pool = TaskPool::with_min_samples(4, 0.75, 3);
    pool.add(make_task(0, Difficulty::Medium));

    // Only two samples — ineligible even though pass@1 == 1.0.
    pool.record(0, true);
    pool.record(0, true);
    assert_eq!(pool.maybe_retire(), 0);
    assert!(!pool.is_retired(0));

    // Third sample pushes pass@1 over threshold.
    pool.record(0, true);
    assert_eq!(pool.maybe_retire(), 1);
    assert!(pool.is_retired(0));
    // Re-calling is idempotent.
    assert_eq!(pool.maybe_retire(), 0);
}

#[test]
fn retire_skips_tasks_below_threshold() {
    let mut pool = TaskPool::with_min_samples(4, 0.9, 2);
    pool.add(make_task(0, Difficulty::Hard));
    pool.record(0, true);
    pool.record(0, false);
    assert_eq!(pool.maybe_retire(), 0);
    assert!(!pool.is_retired(0));
}

#[test]
fn active_distribution_decrements_on_retire() {
    let mut pool = TaskPool::with_min_samples(2, 0.5, 2);
    pool.add(make_task(0, Difficulty::Easy));
    pool.add(make_task(1, Difficulty::Easy));
    pool.add(make_task(2, Difficulty::Medium));
    pool.add(make_task(3, Difficulty::Hard));

    let before = pool.active_distribution();
    assert_eq!(before[0], (Difficulty::Easy, 2));
    assert_eq!(before[1], (Difficulty::Medium, 1));
    assert_eq!(before[2], (Difficulty::Hard, 1));

    // Retire one easy task.
    pool.record(0, true);
    pool.record(0, true);
    assert_eq!(pool.maybe_retire(), 1);

    let after = pool.active_distribution();
    assert_eq!(after[0], (Difficulty::Easy, 1));
    assert_eq!(after[1], (Difficulty::Medium, 1));
    assert_eq!(after[2], (Difficulty::Hard, 1));
}

#[test]
fn sample_excludes_retired_tasks() {
    let mut pool = TaskPool::with_min_samples(2, 0.5, 2);
    pool.add(make_task(0, Difficulty::Easy));
    pool.add(make_task(1, Difficulty::Hard));
    pool.record(0, true);
    pool.record(0, true);
    assert_eq!(pool.maybe_retire(), 1);

    let mut rng = LcgRng::seed(99);
    for _ in 0..64 {
        let index = pool.sample(&mut rng).expect("sample");
        assert_eq!(index, 1, "retired task 0 must never be sampled");
    }
}
