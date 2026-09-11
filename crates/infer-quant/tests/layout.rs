//! Known-vector gate for the weight-layout planner. Each case pins one
//! decision the loader used to make inline, so a gate change that silently
//! demotes a weight to its fallback arm goes red here.

use infer_quant::{
    DeviceCaps, LayoutPolicy, RepackRequirement, RepackTransform, WeightFormatKind,
    WeightLayoutQuery, plan_weight_layout,
};

fn caps(major: i32, minor: i32) -> DeviceCaps {
    DeviceCaps {
        compute_capability: (major, minor),
    }
}

/// Everything on: Hopper, native bridge present, prefill batched, envelope
/// wide enough for the 512 floor.
fn policy_full() -> LayoutPolicy {
    LayoutPolicy {
        prefill_batched: true,
        deepgemm_native_available: true,
        deepgemm_row_envelope: Some((0, 512)),
        dsv4_decode_enabled: true,
    }
}

fn nvfp4() -> WeightLayoutQuery {
    WeightLayoutQuery {
        format: WeightFormatKind::Fp4E2M1Group,
        n: 5120,
        k: 17408,
        group_size: 16,
        quant_block_m: 0,
        quant_block_k: 0,
        scale_rows: 5120,
        scale_cols: 17408 / 16,
    }
}

#[test]
fn nvfp4_hopper_gets_marlin_required_and_sfb() {
    let plan = plan_weight_layout(&nvfp4(), &caps(9, 0), &policy_full()).unwrap();
    assert_eq!(plan.transform, RepackTransform::MarlinFp4);
    assert_eq!(plan.requirement, RepackRequirement::Required);
    assert!(plan.fp4_deepgemm_sfb);
    assert!(!plan.fp8_deepgemm_prefill);
}

#[test]
fn nvfp4_below_sm80_is_a_hard_error() {
    let err = plan_weight_layout(&nvfp4(), &caps(7, 5), &policy_full()).unwrap_err();
    assert!(err.to_string().contains("sm_80"), "got {err}");
}

#[test]
fn nvfp4_wrong_group_size_is_a_hard_error() {
    let mut q = nvfp4();
    q.group_size = 32;
    let err = plan_weight_layout(&q, &caps(9, 0), &policy_full()).unwrap_err();
    assert!(err.to_string().contains("group_size 16"), "got {err}");
}

#[test]
fn nvfp4_sfb_needs_hopper_and_native_bridge() {
    // Ampere: Marlin still Required, but the DeepGEMM sfb is off.
    let plan = plan_weight_layout(&nvfp4(), &caps(8, 0), &policy_full()).unwrap();
    assert_eq!(plan.transform, RepackTransform::MarlinFp4);
    assert!(!plan.fp4_deepgemm_sfb);
    // Hopper but the native bridge probe failed: sfb off, Marlin unchanged.
    let mut policy = policy_full();
    policy.deepgemm_native_available = false;
    let plan = plan_weight_layout(&nvfp4(), &caps(9, 0), &policy).unwrap();
    assert!(!plan.fp4_deepgemm_sfb);
    assert_eq!(plan.transform, RepackTransform::MarlinFp4);
}

#[test]
fn w8a16_fit_is_optional_and_demotes_below_sm80() {
    let q = WeightLayoutQuery {
        format: WeightFormatKind::W8A16,
        n: 5120,
        k: 5120,
        group_size: 128,
        quant_block_m: 0,
        quant_block_k: 0,
        scale_rows: 0,
        scale_cols: 0,
    };
    let plan = plan_weight_layout(&q, &caps(9, 0), &policy_full()).unwrap();
    assert_eq!(plan.transform, RepackTransform::MarlinW8a16);
    assert_eq!(plan.requirement, RepackRequirement::Optional);
    // Below sm_80 the same weight keeps its checkpoint layout (scalar arm).
    let plan = plan_weight_layout(&q, &caps(7, 5), &policy_full()).unwrap();
    assert_eq!(plan.transform, RepackTransform::None);
    // A group size the Marlin kernel is not instantiated for demotes too.
    let mut q = q;
    q.group_size = 33;
    let plan = plan_weight_layout(&q, &caps(9, 0), &policy_full()).unwrap();
    assert_eq!(plan.transform, RepackTransform::None);
    // K%16 alone is not enough: K must also divide the group size (K=48,
    // gs=32 passes K%16 but not K%32).
    let q = WeightLayoutQuery {
        format: WeightFormatKind::W8A16,
        n: 5120,
        k: 48,
        group_size: 32,
        quant_block_m: 0,
        quant_block_k: 0,
        scale_rows: 0,
        scale_cols: 0,
    };
    let plan = plan_weight_layout(&q, &caps(9, 0), &policy_full()).unwrap();
    assert_eq!(plan.transform, RepackTransform::None);
}

#[test]
fn fp8_per_channel_gets_marlin_and_prefill_flag() {
    let q = WeightLayoutQuery {
        format: WeightFormatKind::Fp8BlockScaled,
        n: 17408,
        k: 5120,
        group_size: 0,
        quant_block_m: 1,
        quant_block_k: 5120,
        scale_rows: 0,
        scale_cols: 0,
    };
    let plan = plan_weight_layout(&q, &caps(9, 0), &policy_full()).unwrap();
    assert_eq!(plan.transform, RepackTransform::MarlinFp8PerChannel);
    assert!(plan.fp8_deepgemm_prefill);
    // Block-scaled (block_m = 128) belongs to DeepGEMM's blocked arm, not the
    // per-channel Marlin repack.
    let mut q = q;
    q.quant_block_m = 128;
    let plan = plan_weight_layout(&q, &caps(9, 0), &policy_full()).unwrap();
    assert_eq!(plan.transform, RepackTransform::None);
    // Ampere keeps the Marlin repack but the Hopper-only prefill arm is off.
    let q = WeightLayoutQuery {
        quant_block_m: 1,
        ..q
    };
    let plan = plan_weight_layout(&q, &caps(8, 0), &policy_full()).unwrap();
    assert_eq!(plan.transform, RepackTransform::MarlinFp8PerChannel);
    assert!(!plan.fp8_deepgemm_prefill);
}

#[test]
fn dsv4_decode_cache_follows_the_flag() {
    let q = WeightLayoutQuery {
        format: WeightFormatKind::Dsv4Fp8BlockScaled,
        n: 512,
        k: 512,
        group_size: 0,
        quant_block_m: 0,
        quant_block_k: 0,
        scale_rows: 0,
        scale_cols: 0,
    };
    assert!(
        plan_weight_layout(&q, &caps(9, 0), &policy_full())
            .unwrap()
            .dsv4_decode_proj_cache
    );
    let mut policy = policy_full();
    policy.dsv4_decode_enabled = false;
    assert!(
        !plan_weight_layout(&q, &caps(9, 0), &policy)
            .unwrap()
            .dsv4_decode_proj_cache
    );
}

#[test]
fn formats_without_a_layout_decision_keep_their_checkpoint_layout() {
    for kind in [
        WeightFormatKind::W4A16,
        WeightFormatKind::Fp8PerShard,
        WeightFormatKind::Other,
    ] {
        let q = WeightLayoutQuery {
            format: kind,
            n: 512,
            k: 512,
            group_size: 128,
            quant_block_m: 1,
            quant_block_k: 512,
            scale_rows: 0,
            scale_cols: 0,
        };
        let plan = plan_weight_layout(&q, &caps(9, 0), &policy_full()).unwrap();
        assert_eq!(plan.transform, RepackTransform::None, "{kind:?}");
        assert!(!plan.fp4_deepgemm_sfb);
        assert!(!plan.fp8_deepgemm_prefill);
    }
}

// ── Dispatch-matrix table ──────────────────────────────────────────────────
//
// The format × shape × SM-tier repack decision lives in
// `plan_weight_layout`, and a boundary band (N=96, the N%64 arm) once moved
// with no CPU gate: the only coverage was a pod-only CUDA example.
// These tables pin every conjunct of every repack arm on the host, including
// the exact off-by-one band at each boundary, so a planner edit that demotes
// one shape goes red without a GPU.

/// One shape-conjunct mutation: moves a single field off its satisfying band.
type Mut = &'static dyn Fn(&mut WeightLayoutQuery);
type FailRow = (&'static str, Mut);

/// A query with every repack conjunct satisfied for `format`. Each case below
/// mutates exactly one field off its satisfying value and asserts the arm
/// demotes (or hard-errors for NVFP4).
fn satisfied(format: WeightFormatKind) -> WeightLayoutQuery {
    match format {
        WeightFormatKind::W8A16 => WeightLayoutQuery {
            format,
            n: 5120,
            k: 5120,
            // 5120 is a multiple of 32/64/128 and 16.
            group_size: 128,
            quant_block_m: 0,
            quant_block_k: 0,
            scale_rows: 0,
            scale_cols: 0,
        },
        WeightFormatKind::Fp8BlockScaled => WeightLayoutQuery {
            format,
            n: 17408,
            k: 5120,
            group_size: 0,
            quant_block_m: 1,
            quant_block_k: 5120,
            scale_rows: 0,
            scale_cols: 0,
        },
        WeightFormatKind::Fp4E2M1Group => WeightLayoutQuery {
            format,
            n: 5120,
            k: 17408,
            group_size: 16,
            quant_block_m: 0,
            quant_block_k: 0,
            scale_rows: 5120,
            scale_cols: 17408 / 16,
        },
        other => WeightLayoutQuery {
            format: other,
            n: 512,
            k: 512,
            group_size: 16,
            quant_block_m: 1,
            quant_block_k: 512,
            scale_rows: 512,
            scale_cols: 32,
        },
    }
}

#[test]
fn w8a16_repack_matrix_each_sm_tier() {
    // W8A16 takes Marlin iff sm>=80 AND N%64 AND K%16 AND K%gs AND gs in
    // {32,64,128}. It is Optional, so every failed conjunct demotes to None.
    for (sm, expect) in [
        ((7, 0), RepackTransform::None), // sm_70: too old even when shape fits
        ((7, 5), RepackTransform::None),
        ((8, 0), RepackTransform::MarlinW8a16), // sm_80 Ampere
        ((8, 9), RepackTransform::MarlinW8a16),
        ((9, 0), RepackTransform::MarlinW8a16), // sm_90 Hopper
        ((12, 0), RepackTransform::MarlinW8a16), // sm_120 Blackwell
    ] {
        let plan = plan_weight_layout(
            &satisfied(WeightFormatKind::W8A16),
            &caps(sm.0, sm.1),
            &policy_full(),
        )
        .unwrap();
        assert_eq!(plan.transform, expect, "sm_{} baseline", sm.0);
        if expect == RepackTransform::MarlinW8a16 {
            assert_eq!(plan.requirement, RepackRequirement::Optional);
        }
    }
}

#[test]
fn w8a16_repack_matrix_each_shape_conjunct_off_by_one() {
    let base = satisfied(WeightFormatKind::W8A16);
    // Each row: (label, mutate) moves exactly one conjunct to its failing band.
    let fail: &[FailRow] = &[
        // N%64: 5120 fits; 5119 and 5121 both fail, 5056 (=64*79) still fits.
        ("N=5119 fails N%64", &|q| q.n = 5119),
        ("N=5121 fails N%64", &|q| q.n = 5121),
        // K%16: 5120 fits; 5128 fails K%16? 5128%16=8. (K%gs checked separately.)
        ("K=5128 fails K%16", &|q| q.k = 5128),
        // K%gs with gs=128: K=5120 (=40*128) fits; K=5184 (=40*128+64) fails K%128
        // but passes K%16, isolating the K%gs conjunct.
        ("K=5184 fails K%gs128 only", &|q| q.k = 5184),
        // gs not instantiated: 33 fails the {32,64,128} set.
        ("gs=33 not instantiated", &|q| q.group_size = 33),
        // gs=16 passes K%16/K%gs but is not in the W8A16 instantiated set.
        ("gs=16 not instantiated", &|q| q.group_size = 16),
    ];
    for (label, mutate) in fail {
        let mut q = base;
        mutate(&mut q);
        let plan = plan_weight_layout(&q, &caps(9, 0), &policy_full()).unwrap();
        assert_eq!(plan.transform, RepackTransform::None, "{label}");
    }
    // Positive controls at the group sizes the kernel IS instantiated for.
    for gs in [32usize, 64, 128] {
        let mut q = base;
        q.group_size = gs;
        q.k = gs * 40; // K%16 and K%gs both satisfied
        let plan = plan_weight_layout(&q, &caps(9, 0), &policy_full()).unwrap();
        assert_eq!(plan.transform, RepackTransform::MarlinW8a16, "gs={gs}");
    }
}

#[test]
fn fp8_per_channel_repack_matrix_sm_tiers_and_flags() {
    // Marlin per-channel iff sm>=80 AND block_m==1 AND block_k>=k AND
    // K%64 AND N%64. Optional.
    for (sm, want) in [
        ((7, 5), RepackTransform::None),
        ((8, 0), RepackTransform::MarlinFp8PerChannel),
        ((9, 0), RepackTransform::MarlinFp8PerChannel),
        ((12, 0), RepackTransform::MarlinFp8PerChannel),
    ] {
        let plan = plan_weight_layout(
            &satisfied(WeightFormatKind::Fp8BlockScaled),
            &caps(sm.0, sm.1),
            &policy_full(),
        )
        .unwrap();
        assert_eq!(plan.transform, want, "sm_{} baseline", sm.0);
    }
    let base = satisfied(WeightFormatKind::Fp8BlockScaled);
    let fail: &[FailRow] = &[
        ("block_m=128 is blocked not per-channel", &|q| {
            q.quant_block_m = 128
        }),
        ("block_k<k fails per-channel", &|q| {
            q.quant_block_k = q.k / 2
        }),
        // block_k>k still qualifies (TP shard reading a slice of K).
        ("K=5119 fails K%64", &|q| q.k = 5119),
        ("N=17407 fails N%64", &|q| q.n = 17407),
    ];
    for (label, mutate) in fail {
        let mut q = base;
        mutate(&mut q);
        let plan = plan_weight_layout(&q, &caps(9, 0), &policy_full()).unwrap();
        assert_eq!(plan.transform, RepackTransform::None, "{label}");
    }
    // block_k > k (TP-shard case) still repacks.
    let mut q = base;
    q.quant_block_k = q.k * 2;
    let plan = plan_weight_layout(&q, &caps(9, 0), &policy_full()).unwrap();
    assert_eq!(plan.transform, RepackTransform::MarlinFp8PerChannel);
}

#[test]
fn nvfp4_repack_matrix_required_and_hard_errors() {
    // NVFP4 Marlin is Required (no fallback): sm>=80 AND gs==16 AND K%64 AND
    // N%64 AND scale shape [N, K/16]. Below sm_80 or any shape miss is a hard
    // error, not a demotion.
    let ok = plan_weight_layout(
        &satisfied(WeightFormatKind::Fp4E2M1Group),
        &caps(9, 0),
        &policy_full(),
    )
    .unwrap();
    assert_eq!(ok.transform, RepackTransform::MarlinFp4);
    assert_eq!(ok.requirement, RepackRequirement::Required);
    for (sm, name) in [((7, 5), "sm_75"), ((8, 0), "sm_80"), ((12, 0), "sm_120")] {
        let res = plan_weight_layout(
            &satisfied(WeightFormatKind::Fp4E2M1Group),
            &caps(sm.0, sm.1),
            &policy_full(),
        );
        // sm_80 and sm_120 build Marlin; sm_75 hard-errors.
        if sm.0 >= 8 {
            assert_eq!(res.unwrap().transform, RepackTransform::MarlinFp4, "{name}");
        } else {
            assert!(res.is_err(), "{name} must hard-error below sm_80");
        }
    }
    let base = satisfied(WeightFormatKind::Fp4E2M1Group);
    let fail: &[FailRow] = &[
        ("gs=32 is not the instantiated group_blocks=1", &|q| {
            q.group_size = 32
        }),
        ("K=17407 fails K%64", &|q| q.k = 17407),
        ("N=5119 fails N%64", &|q| q.n = 5119),
        ("scale_rows mismatch", &|q| q.scale_rows = q.n / 2),
        ("scale_cols mismatch (not K/16)", &|q| {
            q.scale_cols = q.k / 32
        }),
    ];
    for (label, mutate) in fail {
        let mut q = base;
        mutate(&mut q);
        assert!(
            plan_weight_layout(&q, &caps(9, 0), &policy_full()).is_err(),
            "{label} must be a hard NVFP4 error"
        );
    }
}

#[test]
fn deepgemm_arms_require_hopper_native_and_the_512_floor() {
    // The DeepGEMM flags are independent of the Marlin repack and stricter:
    // Hopper-only (wgmma), native bridge, and a prefill M envelope >= 512
    // (strictly above the decode row count).
    let q4 = satisfied(WeightFormatKind::Fp4E2M1Group);
    let q8 = satisfied(WeightFormatKind::Fp8BlockScaled);

    // Ampere: Marlin yes, DeepGEMM flags no even with everything else on.
    let plan = plan_weight_layout(&q4, &caps(8, 0), &policy_full()).unwrap();
    assert!(!plan.fp4_deepgemm_sfb);
    // K%128 is tighter than Marlin's K%64: K=4160 (=65*64) passes K%64 but
    // fails K%128, so Marlin still fits while sfb does not.
    let mut q = q4;
    q.k = 64 * 65; // 4160: %64 yes, %128 no
    q.scale_cols = q.k / 16;
    let plan = plan_weight_layout(&q, &caps(9, 0), &policy_full()).unwrap();
    assert_eq!(plan.transform, RepackTransform::MarlinFp4); // Marlin still fits
    assert!(!plan.fp4_deepgemm_sfb); // but sfb needs K%128

    // Native bridge probe off → sfb off.
    let mut p = policy_full();
    p.deepgemm_native_available = false;
    assert!(
        !plan_weight_layout(&q4, &caps(9, 0), &p)
            .unwrap()
            .fp4_deepgemm_sfb
    );

    // Prefill not batched → off.
    let mut p = policy_full();
    p.prefill_batched = false;
    assert!(
        !plan_weight_layout(&q4, &caps(9, 0), &p)
            .unwrap()
            .fp4_deepgemm_sfb
    );

    // Envelope under the 512 floor → off even on Hopper with the bridge.
    for envelope in [Some((0, 256)), Some((0, 511)), Some((512, 600)), None] {
        let mut p = policy_full();
        p.deepgemm_row_envelope = envelope;
        // decode_rows+1 must be <= prefill_rows; (512,600) gives floor 512 → on.
        let want = matches!(envelope, Some((d, pre)) if pre >= 512.max(d + 1));
        assert_eq!(
            plan_weight_layout(&q4, &caps(9, 0), &p)
                .unwrap()
                .fp4_deepgemm_sfb,
            want,
            "envelope {envelope:?}"
        );
        assert_eq!(
            plan_weight_layout(&q8, &caps(9, 0), &p)
                .unwrap()
                .fp8_deepgemm_prefill,
            want,
            "fp8 envelope {envelope:?}"
        );
    }
    // fp8 prefill also needs N%8 (looser than Marlin's N%64) and K%128.
    let mut q = q8;
    q.k = 64 * 65; // %64 yes, %128 no
    assert!(
        !plan_weight_layout(&q, &caps(9, 0), &policy_full())
            .unwrap()
            .fp8_deepgemm_prefill
    );
}
