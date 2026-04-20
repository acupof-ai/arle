use autograd::{ConstantLr, CosineWithWarmup, LinearWarmup, LrSchedule, parse_lr_schedule};

const EPS: f32 = 1e-6;

#[test]
fn constant_lr_is_flat() {
    let sched = ConstantLr(3.5e-4);
    assert!((sched.lr(0) - 3.5e-4).abs() < EPS);
    assert!((sched.lr(1) - 3.5e-4).abs() < EPS);
    assert!((sched.lr(999_999) - 3.5e-4).abs() < EPS);
    assert_eq!(sched.lr(0), sched.lr(999_999));
}

#[test]
fn linear_warmup_ramps_then_plateaus() {
    let sched = LinearWarmup {
        base_lr: 1.0e-3,
        warmup_steps: 100,
    };
    assert!(
        sched.lr(0) < 1e-6,
        "lr(0) should be ~0, got {}",
        sched.lr(0)
    );
    assert!(
        (sched.lr(100) - 1.0e-3).abs() < 1e-6,
        "lr(warmup) should equal base_lr, got {}",
        sched.lr(100)
    );
    // Past warmup -> constant base_lr exactly.
    assert_eq!(sched.lr(200), 1.0e-3);
    // Mid-ramp: lr(50) ~= 0.5 * base_lr.
    let mid = sched.lr(50);
    assert!((mid - 5.0e-4).abs() < 1e-6, "mid-ramp off: {}", mid);
}

#[test]
fn linear_warmup_zero_steps_is_constant() {
    let sched = LinearWarmup {
        base_lr: 2.5e-4,
        warmup_steps: 0,
    };
    assert_eq!(sched.lr(0), 2.5e-4);
    assert_eq!(sched.lr(1), 2.5e-4);
    assert_eq!(sched.lr(1_000_000), 2.5e-4);
}

#[test]
fn cosine_with_warmup_traces_full_curve() {
    let sched = CosineWithWarmup {
        base_lr: 1.0e-3,
        min_lr: 1.0e-5,
        warmup_steps: 100,
        total_steps: 1_000,
    };

    // Warmup endpoints.
    assert!(
        sched.lr(0) < 1e-6,
        "lr(0) should be ~0, got {}",
        sched.lr(0)
    );
    assert!(
        (sched.lr(100) - 1.0e-3).abs() < 1e-6,
        "lr(warmup) should equal base_lr, got {}",
        sched.lr(100)
    );

    // End of schedule lands exactly at min_lr (cosine floor).
    let end = sched.lr(1_000);
    assert!(
        (end - 1.0e-5).abs() < 1e-6,
        "lr(total) should equal min_lr, got {}",
        end
    );

    // Somewhere in the decay window: strictly between min_lr and base_lr.
    let mid = sched.lr(550);
    assert!(
        mid > 1.0e-5 && mid < 1.0e-3,
        "mid-decay lr should lie in (min_lr, base_lr), got {}",
        mid
    );

    // Past total_steps -> clamp to min_lr exactly.
    assert_eq!(sched.lr(2_000), 1.0e-5);
    assert_eq!(sched.lr(1_000 + 1_000), 1.0e-5);
}

#[test]
fn parse_lr_schedule_happy_paths() {
    let constant =
        parse_lr_schedule("constant", 1.0e-3, 100, 1_000, 1.0e-5).expect("constant parses");
    assert_eq!(constant.lr(0), 1.0e-3);
    assert_eq!(constant.lr(5_000), 1.0e-3);

    let linear = parse_lr_schedule("linear-warmup", 1.0e-3, 100, 1_000, 1.0e-5)
        .expect("linear-warmup parses");
    assert!(linear.lr(0) < 1e-6);
    assert!((linear.lr(100) - 1.0e-3).abs() < 1e-6);
    assert_eq!(linear.lr(500), 1.0e-3);

    let cosine = parse_lr_schedule("cosine-with-warmup", 1.0e-3, 100, 1_000, 1.0e-5)
        .expect("cosine-with-warmup parses");
    assert!(cosine.lr(0) < 1e-6);
    assert!((cosine.lr(100) - 1.0e-3).abs() < 1e-6);
    assert!((cosine.lr(1_000) - 1.0e-5).abs() < 1e-6);

    // describe() is non-empty for every schedule.
    assert!(!constant.describe().is_empty());
    assert!(!linear.describe().is_empty());
    assert!(!cosine.describe().is_empty());
}

#[test]
fn parse_lr_schedule_rejects_unknown() {
    let err = match parse_lr_schedule("warm-restart", 1.0e-3, 100, 1_000, 1.0e-5) {
        Ok(_) => panic!("unknown schedule must error"),
        Err(err) => err,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("warm-restart"),
        "error should name the bad spec: {msg}"
    );
    assert!(
        msg.contains("constant"),
        "error should list allowed values: {msg}"
    );
    assert!(
        msg.contains("linear-warmup"),
        "error should list allowed values: {msg}"
    );
    assert!(
        msg.contains("cosine-with-warmup"),
        "error should list allowed values: {msg}"
    );
}
