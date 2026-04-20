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

// LR-3 — guard against MLX #2617: cosine decay must not modulate the warmup
// phase; step < warmup_steps is pure linear warmup.
#[test]
fn cosine_with_warmup_no_decay_during_warmup() {
    let sched = CosineWithWarmup {
        base_lr: 1.0,
        min_lr: 0.1,
        warmup_steps: 10,
        total_steps: 100,
    };

    // Concrete probe: step 5 must be exactly 0.5 * base_lr (linear halfway
    // through warmup). A cosine term would knock this off by a measurable
    // amount because `cos(PI * (5 - 10) / 90) != 1` when mixed in naively.
    assert!(
        (sched.lr(5) - 0.5).abs() < EPS,
        "lr(5) should be exactly 0.5 * base_lr during warmup, got {}",
        sched.lr(5)
    );

    // Step 0 is exactly 0.0 — no min_lr floor during the ramp.
    assert_eq!(
        sched.lr(0),
        0.0,
        "lr(0) must be 0.0 with no cosine/min_lr bleed-through"
    );

    // Every step in [0, warmup_steps) must equal linear warmup exactly.
    for step in 0..sched.warmup_steps {
        let expected = sched.base_lr * (step as f32 / sched.warmup_steps as f32);
        let actual = sched.lr(step);
        assert!(
            (actual - expected).abs() < EPS,
            "warmup step {step}: expected linear {expected}, got {actual} \
             (cosine must not be applied during warmup)"
        );
    }
}

// LR-6 — guard NaN from cosine `cos(PI * (step-total) / decay)` when step
// overshoots; schedules must clamp to their terminal LR rather than extrapolate.
#[test]
fn step_past_total_clamps_to_final_lr() {
    let cosine = CosineWithWarmup {
        base_lr: 1.0e-3,
        min_lr: 1.0e-5,
        warmup_steps: 100,
        total_steps: 1_000,
    };
    let past = cosine.lr(1_000 + 1_000);
    assert!(
        past.is_finite(),
        "lr past total_steps must be finite, got {past}"
    );
    assert_eq!(past, 1.0e-5, "cosine lr past total must clamp to min_lr");
    let way_past = cosine.lr(u64::MAX / 2);
    assert!(
        way_past.is_finite(),
        "lr at extreme step must be finite, got {way_past}"
    );
    assert_eq!(way_past, 1.0e-5);

    // LinearWarmup has no decay phase: past warmup, lr is permanently at
    // base_lr (and never NaN even at extreme steps).
    let linear = LinearWarmup {
        base_lr: 2.5e-4,
        warmup_steps: 100,
    };
    let linear_past = linear.lr(u64::MAX / 2);
    assert!(
        linear_past.is_finite(),
        "linear-warmup lr at extreme step must be finite, got {linear_past}"
    );
    assert_eq!(linear_past, 2.5e-4);

    // ConstantLr is always flat; extreme step must still return base_lr.
    let constant = ConstantLr(3.5e-4);
    let constant_past = constant.lr(u64::MAX);
    assert!(constant_past.is_finite());
    assert_eq!(constant_past, 3.5e-4);
}
