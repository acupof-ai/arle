//! NVFP4 routed-expert GEMV correctness, on REAL checkpoint bytes.
//!
//! Qwen3.8-Flash-Next stores each expert matrix as FOUR tensors — a U8 plane of
//! packed E2M1 nibbles, an FP8-E4M3 `weight_scale` (one per 16 values), an F32
//! per-tensor `weight_scale_2`, and an F32 `input_scale` for the W4A4
//! *activation* quantizer this f32-activation lane does not use. The vendored
//! `DATA_A_NVFP4` shaders read ggml `block_nvfp4` (`d[4] || qs[32]` per 64
//! values), which is neither of those layouts, so the device bytes are BUILT by
//! [`repack_nvfp4_planes`] rather than copied.
//!
//! Three claims are under test, each with its own oracle:
//!
//! 1. **The repack is a faithful relabelling.** A plane-direct dequantizer
//!    written here — it never constructs a block — must reproduce, bit for bit,
//!    what `infer_gguf::dequant::dequantize_row_nvfp4` reads back out of the
//!    repacked blocks. Two independently written decoders agreeing on 3.3M real
//!    weights per projection is a much stronger statement than "it round-trips".
//! 2. **The nibble ORDER is the producer's, not ggml's.** That convention is a
//!    fact about the checkpoint, so it is pinned against something in the
//!    checkpoint that is not quantized at all — see
//!    `nvfp4_nibble_order_matches_the_layers_own_bf16_channel_profile`.
//! 3. **The device computes the same product the host does.** Both the plain
//!    (`GemvNvfp4`) and the fused MoE (`GemvIdNvfp4`) pipelines are run against
//!    an f64 CPU dot product over the dequantized weights, with the per-tensor
//!    `weight_scale_2` riding the fused path's SCALE0 output fusion — the slot
//!    it has to use, since `block_nvfp4` has nowhere to keep it.
//!
//! Skips cleanly with no device or no checkpoint.
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};

use infer_gguf::dequant::dequantize_row_nvfp4;
use infer_gguf::safetensors::SafeTensorsDir;
use vulkan_kernels::{
    BLOCK_NVFP4_BYTES, Kernel, KernelCache, MAT_VEC_FUSION_SCALE0, QK_NVFP4, QK_NVFP4_SUB,
    gemv_dispatch, gemv_id_dispatch, gemv_id_params_fused, gemv_params_f32_b, launch_cached,
    nvfp4_row_bytes, repack_nvfp4_planes,
};

use vulkan_sys::{DeviceBuffer, VulkanContext};

/// The only difference between the device and the f64 reference should be the
/// order of an f32 summation: `dequantize` and `ue4m3_to_fp32` produce exactly
/// the same weight values on both sides (every E2M1 magnitude and every UE4M3
/// scale is an exact binary fraction), so nothing else can round.
///
/// Worst measured on the 8060S over the four cases below is **4.3e-7** — f32
/// epsilon territory. 1e-5 keeps ~20x of headroom for a different driver's
/// reduction order while staying five decades away from what a real defect
/// costs: a wrong nibble order, a dropped scale, or a slipped expert offset all
/// produce O(1) relative error, never 1e-5.
const TOL: f32 = 1e-5;

/// Measured on this box, 2026-08. Override with `INFER_SAFETENSORS_TEST_DIR`,
/// matching `infer_gguf::safetensors`'s own on-box tests.
const CHECKPOINT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";
/// Layer 0's first expert bank; expert 0 lives here.
const EXPERT_SHARD: &str = "layer-00000-experts-0000-0127.safetensors";
const EXPERT_PREFIX: &str = "model.language_model.layers.0.mlp.experts";

fn checkpoint_dir() -> Option<PathBuf> {
    let dir = std::env::var("INFER_SAFETENSORS_TEST_DIR").unwrap_or_else(|_| CHECKPOINT.into());
    let path = PathBuf::from(dir);
    if path.is_dir() {
        return Some(path);
    }
    eprintln!(
        "skip: {} not present (set INFER_SAFETENSORS_TEST_DIR)",
        path.display()
    );
    None
}

fn open_shards(paths: Vec<PathBuf>) -> Option<SafeTensorsDir> {
    for p in &paths {
        if !p.is_file() {
            eprintln!("skip: {} not present", p.display());
            return None;
        }
    }
    Some(SafeTensorsDir::open_files(&paths).expect("open safetensors shards"))
}

/// Deterministic xorshift64 so a failure reproduces exactly.
struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
    /// Uniform in [-1, 1); the activation scale is irrelevant to a linear op,
    /// only its sign pattern and dynamic range matter for the error bound.
    fn next_unit_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// One NVFP4 expert projection, as four tensors plus its derived shape.
struct Projection<'a> {
    /// Rows = output features; a row is `ncols` input values.
    nrows: usize,
    ncols: usize,
    qs_plane: &'a [u8],
    scale_plane: &'a [u8],
    weight_scale_2: f32,
}

impl<'a> Projection<'a> {
    fn load(st: &'a SafeTensorsDir, label: &'static str, expert: usize) -> Self {
        let base = format!("{EXPERT_PREFIX}.{expert}.{label}");
        let info = st
            .tensor(&format!("{base}.weight"))
            .unwrap_or_else(|| panic!("{base}.weight missing"));
        assert_eq!(info.dtype, "U8", "{base}.weight dtype");
        // GGUF `ne` order: dims[0] is the contiguous dim, i.e. bytes per row.
        let nrows = info.dims[1] as usize;
        let ncols = info.dims[0] as usize * 2;

        let scale_info = st
            .tensor(&format!("{base}.weight_scale"))
            .unwrap_or_else(|| panic!("{base}.weight_scale missing"));
        assert_eq!(scale_info.dtype, "F8_E4M3", "{base}.weight_scale dtype");
        assert_eq!(
            scale_info.dims,
            vec![(ncols / QK_NVFP4_SUB) as u64, nrows as u64],
            "{base}.weight_scale must be one FP8 scale per {QK_NVFP4_SUB} values"
        );

        let s2 = st
            .tensor_data(&format!("{base}.weight_scale_2"))
            .unwrap_or_else(|_| panic!("{base}.weight_scale_2 missing"));
        assert_eq!(s2.len(), 4, "weight_scale_2 is a rank-0 f32");

        Self {
            nrows,
            ncols,
            qs_plane: st.tensor_data(&format!("{base}.weight")).expect("weight"),
            scale_plane: st
                .tensor_data(&format!("{base}.weight_scale"))
                .expect("weight_scale"),
            weight_scale_2: f32::from_le_bytes([s2[0], s2[1], s2[2], s2[3]]),
        }
    }

    fn repack(&self) -> Vec<u8> {
        let row_bytes = nvfp4_row_bytes(self.ncols).expect("ncols is a block multiple");
        let mut out = vec![0u8; self.nrows * row_bytes];
        repack_nvfp4_planes(
            self.qs_plane,
            self.scale_plane,
            self.nrows,
            self.ncols,
            &mut out,
        )
        .expect("repack expert planes");
        out
    }
}

/// UE4M3 sub-block scale -> f32, written straight from the bit layout
/// (`x eeee mmm`, bias 7, `exp == 0` subnormal, `0x7F` flushed to zero — the
/// policy `types.glsl:1775` and `ggml-cuda/common.cuh:843` share). Deliberately
/// a second implementation: this file must not import the decoder whose output
/// it is checking.
fn ue4m3(byte: u8) -> f32 {
    let u = u32::from(byte & 0x7F);
    if u == 0 || u == 0x7F {
        return 0.0;
    }
    let (exp, man) = (u >> 3, u & 7);
    if exp == 0 {
        man as f32 / 512.0
    } else {
        f32::from_bits(((exp + 120) << 23) | (man << 20))
    }
}

/// The 16 OCP E2M1 values, indexed by nibble `s ee m`.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Dequantize one row DIRECTLY from the producer's two planes — no block is
/// ever built. Element `k` is the `k & 1` nibble of plane byte `k / 2` (ModelOpt
/// packs consecutive pairs), scaled by `weight_scale[k / 16]`. `weight_scale_2`
/// is deliberately NOT applied: it is the out-of-band factor the GEMV folds
/// into its output, and mixing it in here would hide a missing fold.
fn dequantize_row_from_planes(qs: &[u8], scales: &[u8], ncols: usize) -> Vec<f32> {
    (0..ncols)
        .map(|k| {
            let byte = qs[k / 2];
            let nibble = if k % 2 == 0 { byte & 0xF } else { byte >> 4 };
            E2M1[nibble as usize] * ue4m3(scales[k / QK_NVFP4_SUB])
        })
        .collect()
}

/// f64 dot product of a dequantized weight row with the activation, times the
/// per-tensor scale. f64 so the reference's own rounding is ~5 orders below the
/// f32 accumulation being measured.
fn reference_row(weights: &[f32], x: &[f32], weight_scale_2: f32) -> f32 {
    let sum: f64 = weights
        .iter()
        .zip(x)
        .map(|(&w, &xi)| f64::from(w) * f64::from(xi))
        .sum();
    (sum * f64::from(weight_scale_2)) as f32
}

fn upload<'a>(ctx: &'a VulkanContext, bytes: &[u8]) -> DeviceBuffer<'a> {
    let mut buf = DeviceBuffer::alloc_uma(ctx, bytes.len()).expect("alloc device buffer");
    buf.copy_from_host(bytes).expect("upload");
    buf
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_f32(buf: &DeviceBuffer<'_>, count: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; count * 4];
    buf.copy_to_host(&mut bytes).expect("read back dst");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Worst `|got - want| / max(|want|, rms(want))` over the vector.
///
/// The RMS floor is the point: a matvec output crosses zero, and dividing a
/// bounded absolute error by a near-zero row would report a huge "relative"
/// error for a perfectly good result. Measuring against the output's own
/// magnitude is the honest question — does this row carry the same information
/// as the reference.
fn max_relative_error(got: &[f32], want: &[f32]) -> (f32, usize) {
    let rms = (want
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / want.len().max(1) as f64)
        .sqrt() as f32;
    let floor = if rms > 0.0 { rms } else { 1e-6 };
    let mut worst = 0.0f32;
    let mut worst_row = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let rel = (g - w).abs() / w.abs().max(floor);
        if rel > worst {
            worst = rel;
            worst_row = i;
        }
    }
    (worst, worst_row)
}

/// Dequantize every row two ways — the repacked `block_nvfp4` stream vs the
/// plane-direct reader — assert bit-identity, and return the dequantized
/// weights for the caller's oracle.
fn dequant_and_check_repack(
    label: &str,
    experts: &[Projection<'_>],
    packed: &[Vec<u8>],
) -> Vec<Vec<Vec<f32>>> {
    let (nrows, ncols) = (experts[0].nrows, experts[0].ncols);
    let row_bytes = nvfp4_row_bytes(ncols).expect("row bytes");
    let mut weights: Vec<Vec<Vec<f32>>> = Vec::with_capacity(experts.len());
    for (e, proj) in experts.iter().enumerate() {
        let mut rows = Vec::with_capacity(nrows);
        for r in 0..nrows {
            let from_blocks =
                dequantize_row_nvfp4(&packed[e][r * row_bytes..(r + 1) * row_bytes], ncols)
                    .expect("dequantize repacked row");
            let from_planes = dequantize_row_from_planes(
                &proj.qs_plane[r * (ncols / 2)..(r + 1) * (ncols / 2)],
                &proj.scale_plane[r * (ncols / QK_NVFP4_SUB)..(r + 1) * (ncols / QK_NVFP4_SUB)],
                ncols,
            );
            assert_eq!(
                from_blocks, from_planes,
                "{label} expert {e} row {r}: repacked block_nvfp4 disagrees with the                  plane-direct dequantizer"
            );
            rows.push(from_blocks);
        }
        weights.push(rows);
    }
    eprintln!(
        "[{label}] repack verified: {} weights, 2 independent decoders agree bit for bit",
        experts.len() * nrows * ncols
    );
    weights
}

/// Claim 1 standalone, on real checkpoint bytes, with NO `VulkanContext`: a
/// box without a Vulkan device verifies the repack instead of skipping it
/// together with the GPU claims.
#[test]
fn repack_matches_plane_dequant_on_real_expert_bytes() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(st) = open_shards(vec![dir.join(EXPERT_SHARD)]) else {
        return;
    };
    for label in ["gate_proj", "down_proj"] {
        let experts: Vec<Projection<'_>> =
            (0..2).map(|e| Projection::load(&st, label, e)).collect();
        let packed: Vec<Vec<u8>> = experts.iter().map(Projection::repack).collect();
        dequant_and_check_repack(label, &experts, &packed);
    }
}

/// Both NVFP4 pipelines against an f64 CPU dot product over real expert bytes.
#[test]
fn nvfp4_gemv_matches_cpu_oracle_on_real_expert_bytes() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(st) = open_shards(vec![dir.join(EXPERT_SHARD)]) else {
        return;
    };
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping NVFP4 GEMV test");
            return;
        }
    };
    eprintln!(
        "ARLE Vulkan NVFP4 expert GEMV proof on: {}",
        ctx.device_name()
    );

    // Two experts so the fused path's id indirection is exercised with a real
    // (non-identity) permutation, and two projections so both real widths run:
    // gate_proj is [640, 2560] and down_proj is [2560, 640].
    for label in ["gate_proj", "down_proj"] {
        let experts: Vec<Projection<'_>> =
            (0..2).map(|e| Projection::load(&st, label, e)).collect();
        let (nrows, ncols) = (experts[0].nrows, experts[0].ncols);
        assert!(
            nvfp4_row_bytes(ncols).is_some(),
            "{label}: ncols {ncols} must be a multiple of {QK_NVFP4}"
        );

        // --- Claim 1: the repack is a faithful relabelling of the planes.
        // (Also runs GPU-free as `repack_matches_plane_dequant_on_real_expert_bytes`.)
        let packed: Vec<Vec<u8>> = experts.iter().map(Projection::repack).collect();
        let weights = dequant_and_check_repack(label, &experts, &packed);

        // Shared activation for the token.
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let x: Vec<f32> = (0..ncols).map(|_| rng.next_unit_f32()).collect();
        let x_bytes = f32_bytes(&x);

        // --- Claim 3a: the plain GEMV (Kernel::GemvNvfp4) on expert 0. ---
        // No SCALE fusion on the non-MUL_MAT_ID branch, so `weight_scale_2` is
        // applied by the caller here; the fused case below folds it in-shader.
        let mut cache = KernelCache::new();
        let buf_w = upload(&ctx, &packed[0]);
        let buf_x = upload(&ctx, &x_bytes);
        let buf_d = DeviceBuffer::alloc_host_cached(&ctx, nrows * 4).expect("alloc dst");
        let dummy = upload(&ctx, &[0u8; 4]);
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::GemvNvfp4,
            &[&buf_w, &buf_x, &buf_d, &dummy, &dummy],
            gemv_dispatch(nrows as u32),
            &gemv_params_f32_b(ncols as u32, nrows as u32).to_le_bytes(),
            Kernel::GemvNvfp4.specialization_u32(),
        )
        .expect("plain NVFP4 GEMV dispatch");
        let plain: Vec<f32> = read_f32(&buf_d, nrows)
            .into_iter()
            .map(|v| v * experts[0].weight_scale_2)
            .collect();

        let want0: Vec<f32> = (0..nrows)
            .map(|r| reference_row(&weights[0][r], &x, experts[0].weight_scale_2))
            .collect();
        let (rel, row) = max_relative_error(&plain, &want0);
        eprintln!(
            "[{label}] GemvNvfp4  [{nrows}x{ncols}] max rel err = {rel:.3e} (row {row}, \
             got {} want {})",
            plain[row], want0[row]
        );
        assert!(
            rel < TOL,
            "{label}: plain NVFP4 GEMV rel err {rel} >= {TOL}"
        );

        // --- Claim 3b: the fused expert GEMV (Kernel::GemvIdNvfp4). ---
        // ids are out of order so a slot->id mixup cannot pass, and
        // `weight_scale_2` rides binding 3 under SCALE0, indexed by SLOT.
        let ids: [i32; 2] = [1, 0];
        let stacked: Vec<u8> = packed.concat();
        let scales_by_slot: Vec<f32> = ids
            .iter()
            .map(|&id| experts[id as usize].weight_scale_2)
            .collect();

        let buf_stacked = upload(&ctx, &stacked);
        let buf_ids = upload(&ctx, &i32_bytes(&ids));
        let buf_scale = upload(&ctx, &f32_bytes(&scales_by_slot));
        let buf_fused =
            DeviceBuffer::alloc_host_cached(&ctx, ids.len() * nrows * 4).expect("alloc fused dst");
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::GemvIdNvfp4,
            &[
                &buf_stacked,
                &buf_x,
                &buf_fused,
                &buf_scale,
                &dummy,
                &buf_ids,
            ],
            gemv_id_dispatch(nrows as u32, ids.len() as u32),
            &gemv_id_params_fused(
                ncols as u32,
                nrows as u32,
                ids.len() as u32,
                MAT_VEC_FUSION_SCALE0,
            )
            .to_le_bytes(),
            Kernel::GemvIdNvfp4.specialization_u32(),
        )
        .expect("fused NVFP4 expert GEMV dispatch");
        let fused = read_f32(&buf_fused, ids.len() * nrows);

        for (slot, &id) in ids.iter().enumerate() {
            let e = id as usize;
            let want: Vec<f32> = (0..nrows)
                .map(|r| reference_row(&weights[e][r], &x, experts[e].weight_scale_2))
                .collect();
            let got = &fused[slot * nrows..(slot + 1) * nrows];
            let (rel, row) = max_relative_error(got, &want);
            eprintln!(
                "[{label}] GemvIdNvfp4 slot {slot} (expert {e}) max rel err = {rel:.3e} \
                 (row {row}, got {} want {})",
                got[row], want[row]
            );
            assert!(
                rel < TOL,
                "{label}: fused NVFP4 GEMV slot {slot} rel err {rel} >= {TOL}"
            );
        }
    }
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Pin the nibble ORDER against a tensor in the same checkpoint that was never
/// quantized.
///
/// The producer packs element `2i` in the low nibble of byte `i` and `2i+1` in
/// the high nibble; ggml's `block_nvfp4` puts elements 0..7 in the low nibbles
/// of a 16-value sub-block's 8 bytes and 8..15 in the high nibbles. Both
/// hypotheses yield the same MULTISET of weights per group of 16 and the same
/// per-group amax, so no self-consistency check on the expert bytes alone can
/// tell them apart — and getting it wrong permutes every group of 16 weights
/// against its activations while still producing finite, plausible output.
///
/// The discriminator: NVFP4 normalizes each group of 16 by its own amax, so the
/// *within-group* profile of `mean |w| / group_amax` per input channel survives
/// quantization and is a property of the residual stream, not of this matrix.
/// Layer 0's `shared_expert.gate_proj` reads the same residual stream and is
/// excluded from quantization (`hf_quant_config.json`: `*.mlp.shared_expert.*`),
/// so it supplies that profile in BF16. Measured on this box, 4 experts x 640
/// rows, group means removed:
///
/// ```text
///   hypothesis        vs shared_expert.gate_proj   vs shared_expert.up_proj
///   consecutive pairs            +0.536                     +0.497
///   ggml split halves            +0.039                     +0.029
/// ```
///
/// (Control: the same profile against the MTP layer's BF16 routed experts is
/// -0.05, so the +0.54 is layer-specific structure and not an artifact of the
/// statistic. Ceiling: gate vs up, both BF16, is +0.89.)
#[test]
fn nvfp4_nibble_order_matches_the_layers_own_bf16_channel_profile() {
    let Some(dir) = checkpoint_dir() else { return };
    let mut paths = vec![dir.join(EXPERT_SHARD)];
    let mut bf16: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read checkpoint dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_bf16_shard(p))
        .collect();
    bf16.sort();
    paths.extend(bf16);
    let Some(st) = open_shards(paths) else { return };

    const REFS: [&str; 2] = ["gate_proj", "up_proj"];
    const N_EXPERTS: usize = 4;
    let Some(reference_shape) = st.tensor(&bf16_name(REFS[0])).map(|t| t.dims.clone()) else {
        eprintln!("skip: {} not in the opened shards", bf16_name(REFS[0]));
        return;
    };
    let ncols = reference_shape[0] as usize;

    // The hypothesis under test is the one `repack_nvfp4_planes` implements;
    // the alternative is a straight `d || qs` copy, i.e. treating the producer's
    // plane bytes as if they were already in ggml nibble order.
    let mine = nvfp4_channel_profile(&st, ncols, N_EXPERTS, Hypothesis::Repack);
    let naive = nvfp4_channel_profile(&st, ncols, N_EXPERTS, Hypothesis::CopyPlaneBytes);

    for name in REFS {
        let reference = bf16_channel_profile(&st, &bf16_name(name), ncols);
        let good = correlation(&within_group(&mine), &within_group(&reference));
        let bad = correlation(&within_group(&naive), &within_group(&reference));
        eprintln!(
            "[nibble order] vs BF16 shared_expert.{name}: repack {good:+.3}, \
             plane-byte copy {bad:+.3}"
        );
        assert!(
            good > 0.35,
            "the repack's channel profile should track the layer's own BF16 \
             {name} (measured +0.54); got {good:+.3}"
        );
        assert!(
            bad < 0.15,
            "a straight plane-byte copy should NOT track it (measured +0.05); \
             got {bad:+.3} — the two nibble orders are no longer distinguishable, \
             so this test has stopped discriminating"
        );
    }
}

fn is_bf16_shard(p: &Path) -> bool {
    p.extension().is_some_and(|e| e == "safetensors")
        && p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("model-bf16-"))
}

fn bf16_name(proj: &str) -> String {
    format!("model.language_model.layers.0.mlp.shared_expert.{proj}.weight")
}

enum Hypothesis {
    /// Element `2i` / `2i+1` are the low / high nibbles of plane byte `i`.
    Repack,
    /// The plane bytes are already ggml's split-halves `qs`.
    CopyPlaneBytes,
}

/// Mean `|w| / group_amax` per input channel, over `n_experts` routed experts.
///
/// Dividing by the group amax is what makes this comparable to a BF16 matrix:
/// NVFP4 stores exactly that ratio (as one of 8 magnitudes), so the statistic
/// is invariant to the block scales and to `weight_scale_2`.
fn nvfp4_channel_profile(
    st: &SafeTensorsDir,
    ncols: usize,
    n_experts: usize,
    hypothesis: Hypothesis,
) -> Vec<f64> {
    let mut acc = vec![0.0f64; ncols];
    let mut rows_seen = 0usize;
    let row_bytes = nvfp4_row_bytes(ncols).expect("row bytes");
    for e in 0..n_experts {
        let proj = Projection::load(st, "gate_proj", e);
        assert_eq!(proj.ncols, ncols);
        let packed = match hypothesis {
            Hypothesis::Repack => proj.repack(),
            Hypothesis::CopyPlaneBytes => copy_plane_bytes(&proj),
        };
        for r in 0..proj.nrows {
            let row = dequantize_row_nvfp4(&packed[r * row_bytes..(r + 1) * row_bytes], ncols)
                .expect("dequantize row");
            for group in 0..ncols / QK_NVFP4_SUB {
                let slice = &row[group * QK_NVFP4_SUB..(group + 1) * QK_NVFP4_SUB];
                let amax = slice.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let inv = if amax > 0.0 {
                    1.0 / f64::from(amax)
                } else {
                    0.0
                };
                for (j, v) in slice.iter().enumerate() {
                    acc[group * QK_NVFP4_SUB + j] += f64::from(v.abs()) * inv;
                }
            }
            rows_seen += 1;
        }
    }
    let denom = rows_seen as f64;
    acc.iter().map(|v| v / denom).collect()
}

/// The rejected hypothesis, built the same way the repack is so the two differ
/// in exactly one thing: `d || qs` with the plane bytes copied verbatim.
fn copy_plane_bytes(proj: &Projection<'_>) -> Vec<u8> {
    let row_bytes = nvfp4_row_bytes(proj.ncols).expect("row bytes");
    let subs = QK_NVFP4 / QK_NVFP4_SUB;
    let mut out = vec![0u8; proj.nrows * row_bytes];
    for r in 0..proj.nrows {
        let qs = &proj.qs_plane[r * (proj.ncols / 2)..(r + 1) * (proj.ncols / 2)];
        let sc = &proj.scale_plane
            [r * (proj.ncols / QK_NVFP4_SUB)..(r + 1) * (proj.ncols / QK_NVFP4_SUB)];
        let dst = &mut out[r * row_bytes..(r + 1) * row_bytes];
        for (b, block) in dst.chunks_exact_mut(BLOCK_NVFP4_BYTES).enumerate() {
            block[..subs].copy_from_slice(&sc[b * subs..(b + 1) * subs]);
            block[subs..].copy_from_slice(&qs[b * (QK_NVFP4 / 2)..(b + 1) * (QK_NVFP4 / 2)]);
        }
    }
    out
}

/// Same statistic on an unquantized BF16 matrix: `mean |w| / group_amax` per
/// column, groups being the same contiguous 16 input channels.
fn bf16_channel_profile(st: &SafeTensorsDir, name: &str, ncols: usize) -> Vec<f64> {
    let info = st.tensor(name).unwrap_or_else(|| panic!("{name} missing"));
    assert_eq!(info.dtype, "BF16", "{name} dtype");
    assert_eq!(info.dims[0] as usize, ncols, "{name} input width");
    let nrows = info.dims[1] as usize;
    let bytes = st.tensor_data(name).expect("bf16 tensor bytes");

    let mut acc = vec![0.0f64; ncols];
    for r in 0..nrows {
        let row: Vec<f32> = bytes[r * ncols * 2..(r + 1) * ncols * 2]
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect();
        for group in 0..ncols / QK_NVFP4_SUB {
            let slice = &row[group * QK_NVFP4_SUB..(group + 1) * QK_NVFP4_SUB];
            let amax = slice.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let inv = if amax > 0.0 {
                1.0 / f64::from(amax)
            } else {
                0.0
            };
            for (j, v) in slice.iter().enumerate() {
                acc[group * QK_NVFP4_SUB + j] += f64::from(v.abs()) * inv;
            }
        }
    }
    acc.iter().map(|v| v / nrows as f64).collect()
}

/// Remove each group-of-16's mean, leaving only the WITHIN-group channel
/// structure — the only part the two nibble hypotheses disagree about.
fn within_group(profile: &[f64]) -> Vec<f64> {
    let mut out = profile.to_vec();
    for group in out.chunks_exact_mut(QK_NVFP4_SUB) {
        let mean = group.iter().sum::<f64>() / QK_NVFP4_SUB as f64;
        for v in group.iter_mut() {
            *v -= mean;
        }
    }
    out
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        cov += (x - ma) * (y - mb);
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
    }
    if va == 0.0 || vb == 0.0 {
        return 0.0;
    }
    cov / (va * vb).sqrt()
}
