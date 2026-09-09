//! Host-only weight-layout planner: given a checkpoint format, a shape, and
//! static device properties, decide which kernel-layout repack a weight needs
//! before it becomes resident. The planner never touches a device — every
//! device-side fact it reads arrives as a host field the caller fills
//! (`DeviceCaps`, `LayoutPolicy`). Execution of the planned repacks stays in
//! `infer-cuda` / `cuda-kernels`.

use anyhow::bail;

/// Device properties a layout decision is allowed to read. No device handles.
#[derive(Clone, Copy, Debug)]
pub struct DeviceCaps {
    /// `(major, minor)` from `compute_capability()` — carried as the full
    /// tuple because some gates compare the whole pair (sm_90 == Hopper).
    pub compute_capability: (i32, i32),
}

/// Host-side knobs and device-probe results the planner reads. Filled by the
/// caller, which is allowed to touch the device and the environment.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayoutPolicy {
    /// Whether a prefill chunk may reach this weight's batched DeepGEMM arm.
    pub prefill_batched: bool,
    /// Result of the native DeepGEMM bridge preflight (device probe).
    pub deepgemm_native_available: bool,
    /// `(decode_rows, prefill_rows)` row envelope the engine presents.
    pub deepgemm_row_envelope: Option<(usize, usize)>,
    /// Whether the DSv4 fused decode WQKV path is enabled.
    pub dsv4_decode_enabled: bool,
}

/// Host mirror of the runtime weight formats that have a layout decision.
/// Formats without one map to `Other`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFormatKind {
    Fp8BlockScaled,
    Fp8PerShard,
    Fp4E2M1Group,
    W8A16,
    W4A16,
    Dsv4Fp8BlockScaled,
    Other,
}

/// Shape and scale metadata the layout decisions read.
#[derive(Clone, Copy, Debug)]
pub struct WeightLayoutQuery {
    pub format: WeightFormatKind,
    /// Output dimension.
    pub n: usize,
    /// Input dimension.
    pub k: usize,
    pub group_size: usize,
    pub quant_block_m: usize,
    pub quant_block_k: usize,
    pub scale_rows: usize,
    pub scale_cols: usize,
}

/// Whether the kernel layout is the weight's only serving path.
///
/// `Required` exists because NVFP4's one serving arm is Marlin: a device that
/// cannot build it has nowhere to go, so the load fails. W8A16 and per-channel
/// FP8 have a scalar/dequant fallback, so their repack is `Optional` and a
/// failed gate demotes silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepackRequirement {
    Required,
    Optional,
}

/// The repack transform to execute, or `None` when the weight keeps its
/// checkpoint layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepackTransform {
    None,
    MarlinW8a16,
    MarlinFp8PerChannel,
    MarlinFp4,
}

/// The layout decision for one weight. `requirement` is meaningful only when
/// `transform != None`.
#[derive(Clone, Copy, Debug)]
pub struct WeightLayoutPlan {
    pub transform: RepackTransform,
    pub requirement: RepackRequirement,
    /// Build the NVFP4 DeepGEMM `sfb` after the repack.
    pub fp4_deepgemm_sfb: bool,
    /// Allow the per-channel FP8 DeepGEMM prefill arm.
    pub fp8_deepgemm_prefill: bool,
    /// Build the DSv4 decode-projection DeepGEMM weight cache.
    pub dsv4_decode_proj_cache: bool,
}

/// Both DeepGEMM dense arms refuse an M below this.
const DEEPGEMM_PREFILL_MIN_M: usize = 512;

/// Decide the resident layout for one weight. Errors only when a `Required`
/// transform is impossible on this device (NVFP4 below sm_80, or a shape the
/// Marlin kernel is not instantiated for).
pub fn plan_weight_layout(
    q: &WeightLayoutQuery,
    caps: &DeviceCaps,
    policy: &LayoutPolicy,
) -> anyhow::Result<WeightLayoutPlan> {
    let (cc_major, cc_minor) = caps.compute_capability;

    let (transform, requirement) = match q.format {
        WeightFormatKind::W8A16 => {
            if matches!(q.group_size, 32 | 64 | 128)
                && q.k.is_multiple_of(16)
                && q.k.is_multiple_of(q.group_size)
                && q.n.is_multiple_of(64)
                && cc_major >= 8
            {
                (RepackTransform::MarlinW8a16, RepackRequirement::Optional)
            } else {
                (RepackTransform::None, RepackRequirement::Optional)
            }
        }
        WeightFormatKind::Fp8BlockScaled => {
            // Per-channel only: one scale per output row, spanning all of K.
            // `block_k >= k` rather than `== k` so a TP shard, whose `cols` is
            // a slice of the K the scale was defined over, still qualifies.
            // K%64, not the repack's K%16: `min_thread_k = 64` and every thread
            // config has thread_k in {64,128}, so a K the GEMM's
            // `is_valid_config` rejects would repack cleanly here and then throw
            // on every call. N%64 matches min_thread_n and the repack's tile_n.
            if q.quant_block_m == 1
                && q.quant_block_k >= q.k
                && q.k.is_multiple_of(64)
                && q.n.is_multiple_of(64)
                && cc_major >= 8
            {
                (
                    RepackTransform::MarlinFp8PerChannel,
                    RepackRequirement::Optional,
                )
            } else {
                (RepackTransform::None, RepackRequirement::Optional)
            }
        }
        WeightFormatKind::Fp4E2M1Group => {
            if cc_major < 8 {
                bail!(
                    "NVFP4 requires sm_80 or newer for the Marlin tensor-core path; this device \
                     is sm_{cc_major}{cc_minor}. Serve an FP8 or W4A16 checkpoint instead."
                );
            }
            // kFE2M1f is instantiated only at group_blocks == 1 (group_size
            // 16); the tile grid needs N % 64 and K % 64.
            if !(q.group_size == 16
                && q.k.is_multiple_of(64)
                && q.n.is_multiple_of(64)
                && q.scale_rows == q.n
                && q.scale_cols == q.k / 16)
            {
                bail!(
                    "NVFP4 weight [{}x{}] gs={} scales=[{}x{}] cannot take the Marlin layout \
                     (kFE2M1f needs group_size 16, and the tile grid needs K%64 and N%64). \
                     NVFP4 has no other serving path.",
                    q.n,
                    q.k,
                    q.group_size,
                    q.scale_rows,
                    q.scale_cols
                );
            }
            (RepackTransform::MarlinFp4, RepackRequirement::Required)
        }
        _ => (RepackTransform::None, RepackRequirement::Optional),
    };

    // The smallest M this engine will present to a batched prefill arm.
    let deepgemm_floor = |base: usize| -> Option<usize> {
        let (decode_rows, prefill_rows) = policy.deepgemm_row_envelope?;
        let floor = base.max(decode_rows + 1);
        (prefill_rows >= floor).then_some(floor)
    };
    // DeepGEMM's dense arms are Hopper-only (wgmma); the native bridge refuses
    // every other major.
    let sm_hopper = cc_major == 9;

    let fp4_deepgemm_sfb = policy.prefill_batched
        && q.format == WeightFormatKind::Fp4E2M1Group
        && q.group_size == 16
        && q.n.is_multiple_of(64)
        && q.k.is_multiple_of(128)
        && sm_hopper
        && policy.deepgemm_native_available
        && deepgemm_floor(DEEPGEMM_PREFILL_MIN_M).is_some();

    let fp8_deepgemm_prefill = policy.prefill_batched
        && q.format == WeightFormatKind::Fp8BlockScaled
        && q.quant_block_m == 1
        && q.quant_block_k >= q.k
        && q.n.is_multiple_of(8)
        && q.k.is_multiple_of(128)
        && sm_hopper
        && policy.deepgemm_native_available
        && deepgemm_floor(DEEPGEMM_PREFILL_MIN_M).is_some();

    let dsv4_decode_proj_cache =
        policy.dsv4_decode_enabled && q.format == WeightFormatKind::Dsv4Fp8BlockScaled;

    Ok(WeightLayoutPlan {
        transform,
        requirement,
        fp4_deepgemm_sfb,
        fp8_deepgemm_prefill,
        dsv4_decode_proj_cache,
    })
}
