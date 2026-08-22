use anyhow::{Result, anyhow, bail};
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::quant_linear as cuda_ql;
use cuda_kernels::tensor::WeightFormat;
use cudarc::driver::{CudaSlice, sys::CUevent_flags};
use half::bf16;
use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[path = "quant_linear_fp4.rs"]
mod fp4;
#[path = "quant_linear_fp8.rs"]
mod fp8;
#[path = "quant_linear_int.rs"]
mod int;

pub(crate) use fp4::{fp4_deepgemm_available, warm_fp4_deepgemm_dense};
pub(crate) use fp8::{fp8_deepgemm_per_channel_available, warm_fp8_deepgemm_dense};

// Only POLICY_ID is read here (stats hash); the route policy lives in fp8.
mod qwen_fp8_dense_policy {
    #![allow(dead_code)]
    include!("generated/qwen_fp8_dense_projection.rs");
}

/// Dequant→BF16→WMMA GEMM floor. M=1 (decode) stays on the batched GEMV;
/// M>=2 uses the WMMA GEMM which avoids the 2× memory blowup of cuBLAS for
/// small batches (DSpark verify, MoE expert routing).
///
/// Safe for FP4/W8A16 only because a Marlin arm claims every M first, and only
/// for weights the repack accepted. A W8A16 weight Marlin could
/// not repack still reaches its dequant arm at M=2 and pays a full-weight dequant
/// per decode step; unmeasured, no such checkpoint on hand. Weights with no
/// Marlin layout must use [`QWEN_DEQUANT_GEMM_PREFILL_MIN_M`] — see its note.
pub(super) const QWEN_FP8_DEQUANT_GEMM_MIN_M: usize = 2;

/// M floor for a dequant→BF16 GEMM on a weight with no Marlin layout. A
/// full-weight dequant per call costs ~5× the weight bytes a batched GEMV reads,
/// so it is a prefill trade and must never fire on a decode step. This is a
/// routing invariant, not a tuned crossover: above any decode batch, below
/// `SchedulerConfig::chunked_prefill_size` (2048).
///
/// Marlin `kFE4M3fn` did NOT retire this. It claims per-channel FP8 only. A
/// 128×128 block-scaled FP8 weight fails the `quant_block_m != 1` guard in
/// `repack_for_marlin_fp8`, so it has no Marlin layout — and DeepGEMM refuses it
/// off Hopper (`qwen_fp8_dense_sm_supports_deepgemm` requires `major == 9`).
/// On sm_80 / sm_100 / sm_120 those weights reach the dequant arm at every M,
/// which is exactly the decode-shadowing defect this floor exists to stop.
pub(super) const QWEN_DEQUANT_GEMM_PREFILL_MIN_M: usize = 512;

/// Lower bound on the M floor for sending a per-channel FP8 weight to DeepGEMM
/// instead of Marlin. The floor in force is
/// [`fp8::dense_deepgemm_prefill_floor`], which raises this above the engine's
/// decode row count.
///
/// Marlin's `kFE4M3fn` arm dequantizes E4M3 to BF16 and runs a BF16 MMA (148
/// TFLOPS on H20); DeepGEMM contracts the E4M3 bytes natively (274 TFLOPS, 93%
/// of this card's FP8 peak). Measured on `gate_up [34816, 5120]`: Marlin 0.060
/// ms at M=1 and 0.082 ms at M=16 against DeepGEMM's 0.182 ms floor, then
/// Marlin 8.457 ms against DeepGEMM 2.664 ms at M=2048 (`down [5120, 17408]`
/// is 4.288 vs 1.418). So the crossover lies in (16, 512] and nothing measures
/// between those two points; 512 is the upper end of that interval. Lowering it
/// needs a measured M sweep, not an interpolation of the two endpoints.
///
/// The two small-M DeepGEMM figures predate `examples/marlin_fp4_probe` moving
/// to the dense NT entry, so they time a 128-row launch and bound the real
/// M=1 / M=16 cost from above. The floor does not rest on them.
pub(super) const QWEN_FP8_DEEPGEMM_PER_CHANNEL_MIN_M: usize = 512;

/// Lower bound on the M floor for widening an NVFP4 weight to E4M3 and sending
/// it to DeepGEMM instead of Marlin. Raised above the engine's decode row count
/// by [`fp8::dense_deepgemm_prefill_floor`], same as its FP8 twin.
///
/// sm_90 has no FP4 tensor core, so both arms widen the nibbles; the question
/// is only what they widen to. Marlin makes BF16 and runs a BF16 MMA — measured
/// 84 TFLOPS on `gate_up [34816, 5120]` at M=2048, against a 148 peak. Widening
/// to E4M3 instead lets DeepGEMM contract at 274, and costs one dequant pass:
/// 278 MB of traffic against a 2.664 ms GEMM, ~3.4%. Net 265 against 84.
///
/// The dequant is per call — the E4M3 copy lives in scratch, never resident —
/// so its cost is fixed while the GEMM's shrinks with M. At M=512 (0.707 ms)
/// it is 13% and the arm still wins 2.9x; below that it stops being obviously
/// right, and nothing measures between M=16 and M=512.
pub(super) const QWEN_FP4_DEEPGEMM_MIN_M: usize = 512;

/// Reusable Marlin W8A16 GEMM scratch: fp32-reduce `c_tmp` + int lock
/// `workspace`. Both sizes depend only on the device SM count (constant), so
/// they are allocated ONCE and never grow — the Qwen decode loop is CUDA-graph
/// captured, and a per-call `cudaMalloc` would break capture (the FP8 paths use
/// the same pre-alloc discipline). `workspace` is zeroed once at allocation
/// (Marlin leaves the locks at 0 after each GEMM, so reuse is safe).
#[derive(Default)]
pub(super) struct MarlinScratch {
    pub(super) c_tmp: Option<CudaSlice<f32>>,
    pub(super) workspace: Option<CudaSlice<i32>>,
}

thread_local! {
    static MARLIN_SCRATCH: RefCell<MarlinScratch> = RefCell::new(MarlinScratch::default());
}

/// Shared by the W8A16, FP8, and NVFP4 Marlin arms.
fn marlin_scratch_init(ctx: &DeviceContext, scratch: &mut MarlinScratch) -> Result<()> {
    if scratch.c_tmp.is_some() {
        return Ok(());
    }
    let sms = ctx.sm_count();
    let c_tmp_floats = cuda_ql::marlin_c_tmp_floats(64, sms)?;
    let ws_ints = cuda_ql::marlin_workspace_ints(sms)?;
    scratch.c_tmp = Some(
        ctx.stream
            .alloc_zeros::<f32>(c_tmp_floats)
            .map_err(|e| anyhow!("Marlin c_tmp alloc failed: {e}"))?,
    );
    scratch.workspace = Some(
        ctx.stream
            .alloc_zeros::<i32>(ws_ints)
            .map_err(|e| anyhow!("Marlin workspace alloc failed: {e}"))?,
    );
    Ok(())
}

/// The three quant families share it — one weight at a time per thread.
pub(super) fn with_marlin_scratch<T>(
    ctx: &DeviceContext,
    f: impl FnOnce(&mut MarlinScratch) -> Result<T>,
) -> Result<T> {
    MARLIN_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        marlin_scratch_init(ctx, &mut scratch)?;
        f(&mut scratch)
    })
}

/// Marlin tensor-core GEMM is Ampere+ (`mma.sync.m16n8k16` + `cp.async`, gated
/// `#if __CUDA_ARCH__ < 800` to no-op stubs in the vendored kernels). One binary
/// runs sm_80..sm_120. Below sm_80 the shim returns NOT_SUPPORTED; cache the gate
/// ONCE so decode dispatch avoids a per-step `cuDeviceGetAttribute`. When off,
/// W8A16/NVFP4 keep the dequant→BF16 GEMM (large M) / scalar GEMV (small M).
pub(super) fn marlin_sm_supported(ctx: &DeviceContext) -> bool {
    static SUPPORTS: OnceLock<bool> = OnceLock::new();
    *SUPPORTS.get_or_init(|| {
        let (major, _minor) = ctx.compute_capability();
        let supports = major >= 8;
        if !supports {
            log::info!(
                "Marlin SM-gated OFF on sm_{major}x (Ampere sm_80+ required for \
                 mma.sync tensor cores); using dequant→BF16 GEMM / scalar GEMV fallback"
            );
        }
        supports
    })
}

fn qwen_quant_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("ARLE_QWEN35_PROFILE").is_some()
            || std::env::var_os("ARLE_QWEN35_QUANT_PROFILE").is_some()
    })
}

pub(super) fn qwen_quant_profile<T>(
    ctx: &DeviceContext,
    label: &'static str,
    seq_len: usize,
    rows: usize,
    cols: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if !qwen_quant_profile_enabled() {
        return f();
    }
    let start = ctx.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let stop = ctx.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(&ctx.stream)?;
    let host_t0 = Instant::now();
    let result = f();
    let host_ms = host_t0.elapsed().as_secs_f64() * 1000.0;
    stop.record(&ctx.stream)?;
    stop.synchronize()?;
    let cuda_ms = start.elapsed_ms(&stop)? as f64;
    if std::env::var("INFER_TP_RANK")
        .map(|rank| rank == "0")
        .unwrap_or(true)
    {
        eprintln!(
            "[qwen-quant-profile] {label} seq={seq_len} rows={rows} cols={cols} cuda_ms={cuda_ms:.3} host_ms={host_ms:.3}"
        );
    }
    result
}

// Qwen quant dispatch counters. Per-impl split so the stats surface can tell
// which quant format took which path (deepgemm / marlin / dequant+gemm / gemv).
// fallback_count is derived (all dequant+gemm + all gemv), not an independent
// atomic. Counters live with their family; the table and the derived count stay
// here so the stats surface has one aggregation point.
static FP8_IMPLEMENTATION_IDS: &[(&AtomicU64, &str)] = &[
    (&fp8::DEEPGEMM_HITS, "cuda.qwen.fp8_pack_deepgemm"),
    (
        &fp8::DEEPGEMM_PER_CHANNEL_HITS,
        "cuda.qwen.fp8_per_channel_deepgemm",
    ),
    (
        &fp8::FP8_DEQUANT_GEMM_HITS,
        "cuda.qwen.fp8_dequant_bf16_gemm",
    ),
    (
        &int::W8A16_DEQUANT_GEMM_HITS,
        "cuda.w8a16.dequant_bf16_gemm",
    ),
    (&int::MARLIN_W8A16_HITS, "cuda.w8a16.marlin_tensorcore"),
    (&fp4::MARLIN_FP4_HITS, "cuda.fp4.marlin_tensorcore"),
    (&fp4::FP4_DEEPGEMM_HITS, "cuda.fp4.widen_fp8_deepgemm"),
    (&fp8::MARLIN_FP8_HITS, "cuda.qwen.fp8_marlin_tensorcore"),
    (&fp8::FP8_GEMV_HITS, "cuda.qwen.fp8_gemv"),
    (&int::W8A16_GEMV_HITS, "cuda.w8a16.gemv"),
    (&int::W4A16_GEMV_HITS, "cuda.w4a16.gemv"),
];

/// Cumulative operator dispatch stats for Qwen FP8 dense projection.
///
/// Materialized only at an explicit stats request boundary. Dispatch itself only
/// increments atomics; no request or engine-tick path allocates telemetry data.
pub(crate) fn qwen_fp8_dense_operator_stats() -> infer_seam::OperatorDispatchStats {
    use infer_seam::OperatorImplementationHits;

    let implementation_hits: Vec<_> = FP8_IMPLEMENTATION_IDS
        .iter()
        .filter_map(|(counter, id)| {
            let hits = counter.load(Ordering::Relaxed);
            (hits > 0).then(|| OperatorImplementationHits {
                implementation_id: (*id).into(),
                hits,
            })
        })
        .collect();
    let fallback_count = fp8::FP8_DEQUANT_GEMM_HITS.load(Ordering::Relaxed)
        + int::W8A16_DEQUANT_GEMM_HITS.load(Ordering::Relaxed)
        + fp8::FP8_GEMV_HITS.load(Ordering::Relaxed)
        + int::W8A16_GEMV_HITS.load(Ordering::Relaxed)
        + int::W4A16_GEMV_HITS.load(Ordering::Relaxed);

    infer_seam::OperatorDispatchStats {
        policy_hash: qwen_fp8_dense_policy::POLICY_ID.into(),
        implementation_hits,
        fallback_count,
    }
}

/// Load-time storage gate: after final repack and source release, every M this
/// dispatcher can be handed (gemv M=1, gemm_batch M>1) must have a resident
/// consumer in the weight's route owner. The route predicates are M-independent
/// once a representation is complete, so one check covers both lanes. Fails the
/// load with the tensor context — never defers a missing buffer to serve time
/// (the W8A16 lm_head defect class).
pub(crate) fn validate_storage(
    ctx: &DeviceContext,
    name: &str,
    weight: &DeviceMatrix,
) -> Result<()> {
    let sm_marlin = marlin_sm_supported(ctx);
    let missing = match weight.weight_format {
        WeightFormat::Fp8BlockScaled | WeightFormat::Fp8PerShard => {
            fp8::fp8_missing_representation(fp8::Fp8Storage::of(weight), sm_marlin)
        }
        WeightFormat::Fp4E2M1Group => {
            fp4::fp4_missing_representation(fp4::Fp4Storage::of(weight), sm_marlin)
        }
        WeightFormat::W8A16 | WeightFormat::W4A16 => {
            int::int_missing_representation(int::IntStorage::of(weight), sm_marlin)
        }
        WeightFormat::Dsv4Fp8BlockScaled | WeightFormat::Dsv4Fp4BlockScaled => {
            (weight.qweight.is_none() || weight.dsv4_scales.is_none())
                .then_some("the qweight + dsv4_scales source pair")
        }
        // DenseBf16 and formats this dispatcher does not serve keep their
        // existing validation.
        _ => None,
    };
    if let Some(missing) = missing {
        bail!(
            "{name}: {} [{}x{}] gs={} has no consumable quant-linear storage for every M; \
             missing {missing}",
            weight.weight_format,
            weight.rows,
            weight.cols,
            weight.group_size,
        );
    }
    Ok(())
}

/// The one quantized dispatch entry: one match on the stored format, one route
/// owner per family. `m` is the row count — `x.seq_len` for `gemm_batch`, 1 for
/// `gemv` — so both lanes run the identical order with identical gates.
pub(super) fn run(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<()> {
    match weight.weight_format {
        WeightFormat::Fp8BlockScaled | WeightFormat::Fp8PerShard => {
            fp8::run(ctx, weight, input, output, m)
        }
        WeightFormat::Fp4E2M1Group => fp4::run(ctx, weight, input, output, m),
        WeightFormat::W8A16 | WeightFormat::W4A16 => int::run(ctx, weight, input, output, m),
        WeightFormat::Dsv4Fp8BlockScaled | WeightFormat::Dsv4Fp4BlockScaled => {
            let qw = weight
                .qweight
                .as_ref()
                .ok_or_else(|| anyhow!("{} missing qweight", weight.weight_format))?;
            let scales = weight
                .dsv4_scales
                .as_ref()
                .ok_or_else(|| anyhow!("{} missing dsv4_scales", weight.weight_format))?;
            match weight.weight_format {
                WeightFormat::Dsv4Fp8BlockScaled => cuda_ql::dsv4_fp8_gemv_batch(
                    ctx,
                    qw,
                    scales,
                    input,
                    output,
                    m,
                    weight.rows,
                    weight.cols,
                    weight.dsv4_scale_rows,
                    weight.dsv4_scale_cols,
                )?,
                WeightFormat::Dsv4Fp4BlockScaled => cuda_ql::dsv4_fp4_gemv_batch(
                    ctx,
                    qw,
                    scales,
                    input,
                    output,
                    m,
                    weight.rows,
                    weight.cols,
                    weight.dsv4_scale_rows,
                    weight.dsv4_scale_cols,
                )?,
                _ => unreachable!(),
            }
            Ok(())
        }
        other => bail!("quant_linear unsupported resident quant weight format {other}"),
    }
}

pub(super) fn gemm_batch(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    run(ctx, weight, &x.data, &mut out.data, x.seq_len)
}

/// The same arms `gemm_batch` runs, at m=1. `output_projection` reaches this
/// with lm_head every single-row step (`qwen35_forward.rs`), which is exactly a
/// repacked weight. A routing miss is loud for NVFP4 — its pre-repack bytes are
/// freed at load, so the scalar arm has nothing to read — and silent for a
/// per-channel FP8 weight whose source DeepGEMM kept alive, so prove engagement
/// from `cuda.qwen.fp8_marlin_tensorcore`.
pub(super) fn gemv(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &DeviceVec,
    out: &mut DeviceVec,
) -> Result<()> {
    run(ctx, weight, &x.data, &mut out.data, 1)
}

#[cfg(test)]
mod tests {
    use super::fp4::{self, Fp4Query, Fp4Route, Fp4Storage};
    use super::fp8::{self, Fp8Query, Fp8Route, Fp8Storage};
    use super::int::{self, IntQuery, IntRoute, IntStorage};

    #[test]
    fn fp8_routes() {
        for &m in &[1usize, 16, 511, 512, 4096] {
            // Repacked per-channel: Marlin owns every M on sm80+.
            assert_eq!(
                fp8::fp8_route(
                    Fp8Query {
                        repacked: true,
                        source: false,
                        per_channel_shape: true,
                        deepgemm_prefill: true,
                    },
                    m,
                    true,
                    true,
                ),
                Fp8Route::Marlin
            );
        }
        // Source-only (repack declined): small M stays on GEMV, prefill M on dequant.
        let source_only = Fp8Query {
            repacked: false,
            source: true,
            per_channel_shape: false,
            deepgemm_prefill: false,
        };
        assert_eq!(fp8::fp8_route(source_only, 1, true, true), Fp8Route::Gemv);
        assert_eq!(
            fp8::fp8_route(source_only, 512, true, true),
            Fp8Route::DequantGemm
        );
        // Pre-sm80 changes nothing for a source-only weight: Marlin was never an option.
        assert_eq!(
            fp8::fp8_route(source_only, 512, false, true),
            Fp8Route::DequantGemm
        );
        // Neither layout: route says GEMV; run() turns the missing source into the terminal error.
        assert_eq!(
            fp8::fp8_route(
                Fp8Query {
                    repacked: false,
                    source: false,
                    per_channel_shape: false,
                    deepgemm_prefill: false,
                },
                512,
                true,
                true,
            ),
            Fp8Route::Gemv
        );
        // Per-channel prefill shape the DeepGEMM arm can claim at/above its floor.
        let prefill = Fp8Query {
            repacked: false,
            source: true,
            per_channel_shape: true,
            deepgemm_prefill: true,
        };
        assert_eq!(fp8::fp8_route(prefill, 512, true, true), Fp8Route::DeepGemm);
        assert_eq!(
            fp8::fp8_route(prefill, 511, true, true),
            Fp8Route::DequantGemm
        );
        // DeepGEMM disabled: the same weight falls back to the dequant order.
        assert_eq!(
            fp8::fp8_route(prefill, 512, true, false),
            Fp8Route::DequantGemm
        );
    }

    #[test]
    fn fp4_routes() {
        // Marlin-ready weight without sfb: Marlin at every M.
        assert_eq!(
            fp4::fp4_route(
                Fp4Query {
                    marlin_ready: true,
                    sfb: false,
                    prefill_shape: false,
                },
                1,
            ),
            Fp4Route::Marlin
        );
        // sfb + prefill shape: DeepGEMM at/above its floor, Marlin below.
        let prefill = Fp4Query {
            marlin_ready: true,
            sfb: true,
            prefill_shape: true,
        };
        assert_eq!(fp4::fp4_route(prefill, 512), Fp4Route::DeepGemm);
        assert_eq!(fp4::fp4_route(prefill, 511), Fp4Route::Marlin);
    }

    /// The load-time storage validator, over the states the loader can produce
    /// (and the invalid ones it must reject). `true` = valid (no missing
    /// representation).
    #[test]
    fn storage_states() {
        let f8 = |marlin: bool, src_w: bool, src_s: bool, per_shard: bool| Fp8Storage {
            marlin_packed: marlin,
            marlin_scales: marlin,
            source_weight: src_w,
            source_scale: src_s,
            per_shard,
        };
        let fp8_cases: &[(Fp8Storage, bool, bool)] = &[
            // Post-repack per-channel: Marlin pair, source freed, scale kept.
            (f8(true, false, true, false), true, true),
            // Repack-declined 128x128 block weight: source pair only.
            (f8(false, true, true, false), true, true),
            // The W8A16-lm_head defect class: everything released, no consumer.
            (f8(false, false, false, false), true, false),
            // Marlin pair resident but device below sm_80: no route can read it.
            (f8(true, false, false, false), false, false),
            // qweight_u8 without its scale: GEMV/dequant both need scale_f32.
            (f8(false, true, false, false), true, false),
            // Per-shard has one route, its source pair.
            (f8(false, true, true, true), true, true),
            (f8(false, false, true, true), true, false),
        ];
        for &(s, sm, valid) in fp8_cases {
            assert_eq!(fp8::fp8_missing_representation(s, sm).is_none(), valid);
        }
        // Half Marlin pair is always rejected.
        assert!(
            fp8::fp8_missing_representation(
                Fp8Storage {
                    marlin_packed: true,
                    marlin_scales: false,
                    source_weight: true,
                    source_scale: true,
                    per_shard: false,
                },
                true,
            )
            .is_some()
        );

        let f4 = |marlin: bool, sfb: bool, global: bool| Fp4Storage {
            marlin_packed: marlin,
            marlin_scales: marlin,
            sfb,
            global_scale: global,
        };
        let fp4_cases: &[(Fp4Storage, bool, bool)] = &[
            // Normal post-repack: Marlin only (decode) or Marlin + sfb (prefill arm).
            (f4(true, false, false), true, true),
            (f4(true, true, true), true, true),
            // sfb without the global scale errors inside the widen arm.
            (f4(true, true, false), true, false),
            // The repack's silent format-gate no-op: nothing resident.
            (f4(false, false, false), true, false),
            // Marlin pair on a pre-sm80 device: unreadable.
            (f4(true, false, false), false, false),
        ];
        for &(s, sm, valid) in fp4_cases {
            assert_eq!(fp4::fp4_missing_representation(s, sm).is_none(), valid);
        }

        let i8s = |marlin: bool, source: bool, w8: bool, grouped: bool| IntStorage {
            marlin_packed: marlin,
            marlin_scales: marlin,
            source_weight: source,
            source_scales: source,
            is_w8a16: w8,
            group_aligned: grouped,
        };
        let int_cases: &[(IntStorage, bool, bool)] = &[
            // Post-repack W8A16: Marlin pair only (source freed by the repack).
            (i8s(true, false, true, false), true, true),
            // Repack-declined W8A16: group-aligned source pair.
            (i8s(false, true, true, true), true, true),
            // Source retained but group size does not divide cols.
            (i8s(false, true, true, false), true, false),
            // The lm_head defect: repack freed the source, Marlin unreadable.
            (i8s(true, false, true, false), false, false),
            // Nothing resident.
            (i8s(false, false, true, false), true, false),
            // W4A16: source pair is the only route.
            (i8s(false, true, false, true), true, true),
            (i8s(false, false, false, true), true, false),
        ];
        for &(s, sm, valid) in int_cases {
            assert_eq!(int::int_missing_representation(s, sm).is_none(), valid);
        }
        // Half pairs are rejected regardless of everything else.
        assert!(
            int::int_missing_representation(
                IntStorage {
                    marlin_packed: true,
                    marlin_scales: false,
                    source_weight: true,
                    source_scales: true,
                    is_w8a16: true,
                    group_aligned: true,
                },
                true,
            )
            .is_some()
        );
        assert!(
            int::int_missing_representation(
                IntStorage {
                    marlin_packed: true,
                    marlin_scales: true,
                    source_weight: true,
                    source_scales: false,
                    is_w8a16: true,
                    group_aligned: true,
                },
                true,
            )
            .is_some()
        );
    }

    #[test]
    fn int_routes() {
        let repacked = IntQuery {
            repacked: true,
            source: false,
            is_w8a16: true,
        };
        for &m in &[1usize, 16, 512] {
            assert_eq!(int::int_route(repacked, m, true), IntRoute::Marlin);
        }
        let source_only = IntQuery {
            repacked: false,
            source: true,
            is_w8a16: true,
        };
        assert_eq!(int::int_route(source_only, 1, true), IntRoute::Gemv);
        assert_eq!(
            int::int_route(source_only, 512, true),
            IntRoute::DequantGemm
        );
        // W4A16 has no Marlin or dequant arm at any M.
        let w4 = IntQuery {
            repacked: false,
            source: true,
            is_w8a16: false,
        };
        for &m in &[1usize, 512] {
            assert_eq!(int::int_route(w4, m, true), IntRoute::Gemv);
        }
    }
}
