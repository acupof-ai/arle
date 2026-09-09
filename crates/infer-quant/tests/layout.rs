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
