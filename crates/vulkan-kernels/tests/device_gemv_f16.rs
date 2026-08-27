//! The DENSE tier's GEMV — `Kernel::GemvF16` / `Kernel::GemvBf16` — on REAL
//! checkpoint bytes, plus what it actually achieves against this part's memory
//! ceiling.
//!
//! Qwen3.8-Flash-Next quantizes its 512 routed experts to NVFP4 and leaves
//! everything else alone: the 12 full-attention layers' q/k/v/o projections and
//! the 248320-row `lm_head` are BF16 on disk. The q/k/v/o of the 12 full layers
//! plus `lm_head` alone are 2.47 GB read per token, and until `Kernel::GemvF16`
//! / `Kernel::GemvBf16` existed no device kernel could read any of it — so it
//! ran on the host at ~55 GB/s (~45 ms/token) instead of the ~230 GB/s measured
//! below (~9.9 ms/token).
//!
//! Three claims, each with its own oracle:
//!
//! 1. **Exactly what the f16 tier costs this checkpoint.** Every weight is
//!    re-encoded by `bf16_to_f16` and decoded again by a decoder written against
//!    the f16 *bit layout* rather than as the encoder's inverse. The result is
//!    not "nothing moved" — 0.02% of these weights do — but a sharp bound on
//!    which ones and by how much: only values already below f16's smallest
//!    normal, and only by half a step of its 2^-24 subnormal grid.
//!
//!    That measurement inverts the premise this file was written under. F16 has
//!    3 more mantissa bits than BF16, so an f16 tier fed from f32 would be the
//!    more precise one — but fed from a BF16 checkpoint it can only ever tie
//!    (normal band, where the widening is exact) or lose (subnormal band). For
//!    reading THESE bytes `Kernel::GemvBf16` is strictly the better default:
//!    identical arithmetic, no convert pass, no tail loss, and the binding can
//!    point straight at the mmap.
//! 2. **Both pipelines compute the same product the CPU does.** f64 dot
//!    products over the weights each kernel actually reads — decoded from the
//!    f16 bytes for `GemvF16`, from the bf16 bytes for `GemvBf16` — at every
//!    real width in the model.
//! 3. **It is worth moving the tier for.** Achieved GB/s per width, because a
//!    GEMV that is correct at 60 GB/s does not buy the lever the residency
//!    replan is paying for. Read the two regimes separately: `lm_head` and
//!    `q_proj` are past the cache and report the sustained link (202-234 GB/s,
//!    79-91% of the 256 GB/s LPDDR5X spec), while the 2.5-30 MiB projections
//!    report 378-630 GB/s because this bench re-reads ONE buffer and it stays
//!    resident. A real decode walks 48 layers before returning, so treat the
//!    small-tensor rows as a cache-resident ceiling, not a forecast.
//!
//! Claim 3 and the `lm_head` half of claims 1-2 allocate ~2.5 GB and take a few
//! seconds, so they are opt-in:
//!
//! ```text
//! ARLE_DENSE_GEMV_BENCH=1 cargo test -p vulkan-kernels --features vulkan \
//!     --test device_gemv_f16 --release -- --nocapture --test-threads=1
//! ```
//!
//! Skips cleanly with no device or no checkpoint.
#![cfg(feature = "vulkan")]

use std::path::PathBuf;
use std::time::Instant;

use vulkan_kernels::{
    Dispatch, GemvDenseSpec, Kernel, KernelCache, bf16_to_f16, gemv_dense_dispatch,
    gemv_params_f32_b, launch_cached, record_dispatch,
};
use vulkan_sys::{CommandRecorder, DescriptorSet, DeviceBuffer, VulkanContext};

/// The device accumulates in f32 through a `subgroupAdd` tree; the reference
/// sums the same terms in f64, in order. Nothing else differs — both sides see
/// the identical set of weight values, since decoding f16/bf16 to f32 is exact
/// on both. So the tolerance is purely a summation-order bound, and the widest
/// row here is 6144 terms (`sqrt(6144) * 2^-24` ~ 5e-6).
///
/// Worst measured on the 8060S across all five tensors and both kernels is
/// **7.8e-7** max-rel (1.9e-7 vector-rel). 1e-4 leaves ~128x of headroom for a
/// different reduction order while staying decades away from what a real defect
/// costs — 2.9e0 for a wrong row stride, 8.9e-2 for a dropped tail, both
/// measured in `dense_gemv_oracle_rejects_a_wrong_row_stride_and_a_dropped_tail`
/// rather than assumed.
const TOL: f32 = 1e-4;

/// Measured on this box, 2026-08. Override with `INFER_SAFETENSORS_TEST_DIR`,
/// matching `infer_gguf::safetensors`'s own on-box tests.
const CHECKPOINT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";

/// Layer 3 is the first `full_attention` layer (they sit at zero-indexed
/// 3, 7, ... 47), so its projections are the real dense attention shapes.
const ATTN_SHARD: &str = "model-bf16-00011.safetensors";
const LM_HEAD_SHARD: &str = "model-bf16-00012.safetensors";
const ATTN_PREFIX: &str = "model.language_model.layers.3.self_attn";
const MLP_PREFIX: &str = "model.language_model.layers.3.mlp";

/// Cap on how many output rows the f64 oracle covers. Only `lm_head` exceeds it;
/// the GEMV still runs over all 248320 rows, and [`sampled_rows`] strides so the
/// checked set spans the full matrix and always includes the last row — a
/// first-N check would miss exactly the row-addressing bugs that grow with the
/// row index.
const MAX_ORACLE_ROWS: usize = 8192;

/// The sharp bf16 -> f16 exactness threshold, 2^-17.
///
/// Above f16's smallest normal (2^-14) the widening is trivially exact — 7
/// mantissa bits into 10. It stays exact for three binades BELOW it too, and
/// that is arithmetic rather than luck: a bf16 in the f16 subnormal band is
/// re-encoded as `((m << 3) | 0x400) >> (113 - e)`, and the low 3 bits of that
/// numerator are always zero, so nothing is discarded until the shift exceeds 3
/// — i.e. until `e < 110`, i.e. below 2^-17. So EVERY bf16 weight at or above
/// this value survives verbatim, and only the three decades under it can move.
const F16_EXACT_FROM_BF16: f32 = 7.629_394_5e-6;

/// Half a step of f16's subnormal grid, 2^-25. A correctly-rounded value in the
/// subnormal band cannot move further than this, so it is the bound a defective
/// encoder breaks.
const F16_SUBNORMAL_HALF_ULP: f32 = 2.980_232_2e-8;

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

fn bench_enabled() -> bool {
    std::env::var("ARLE_DENSE_GEMV_BENCH").is_ok()
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
    /// Uniform in [-1, 1); a linear op does not care about the activation's
    /// scale, only its sign pattern and dynamic range.
    fn next_unit_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// bf16 -> f32: bf16 IS the top half of the f32 bit pattern.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// f16 -> f32, written from the IEEE binary16 layout rather than as the inverse
/// of `bf16_to_f16`'s bit surgery. That independence is the point of claim 1:
/// two decoders that share an author's mistake agree on the mistake.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = (bits >> 10) & 0x1F;
    let man = f32::from(bits & 0x3FF);
    match exp {
        // Subnormals sit on the fixed 2^-24 grid; zero falls out of the same arm.
        0 => sign * man * 2.0f32.powi(-24),
        0x1F if man == 0.0 => sign * f32::INFINITY,
        0x1F => f32::NAN,
        e => sign * (1.0 + man / 1024.0) * 2.0f32.powi(i32::from(e) - 15),
    }
}

/// One dense `[nrows, ncols]` BF16 weight matrix, straight off the mmap.
struct Dense<'a> {
    label: &'static str,
    nrows: usize,
    ncols: usize,
    bf16: &'a [u8],
}

impl<'a> Dense<'a> {
    fn load(
        st: &'a infer_gguf::safetensors::SafeTensorsDir,
        label: &'static str,
        name: &str,
    ) -> Self {
        let info = st
            .tensor(name)
            .unwrap_or_else(|| panic!("{name} missing from the opened shards"));
        assert_eq!(info.dtype, "BF16", "{name} dtype");
        // `SafeTensorsDir` reverses the header shape into GGUF `ne` order, so
        // dims[0] is the contiguous input width and dims[1] the output rows.
        let (ncols, nrows) = (info.dims[0] as usize, info.dims[1] as usize);
        let bf16 = st.tensor_data(name).expect("tensor bytes");
        assert_eq!(bf16.len(), nrows * ncols * 2, "{name} byte count");
        Self {
            label,
            nrows,
            ncols,
            bf16,
        }
    }

    fn f16(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.bf16.len()];
        let report = bf16_to_f16(self.bf16, &mut out).expect("bf16 -> f16");
        assert_eq!(
            report.overflowed, 0,
            "{}: a dense weight above 65504 would saturate, and a saturated \
             weight is a distortion of unbounded size — this tensor must not use \
             Kernel::GemvF16",
            self.label
        );
        out
    }

    /// Decode one row to f32 from whichever 16-bit encoding `decode` names.
    fn row(&self, bytes: &[u8], r: usize, decode: fn(u16) -> f32) -> Vec<f32> {
        bytes[r * self.ncols * 2..(r + 1) * self.ncols * 2]
            .chunks_exact(2)
            .map(|c| decode(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    }
}

fn sampled_rows(nrows: usize) -> Vec<usize> {
    if nrows <= MAX_ORACLE_ROWS {
        return (0..nrows).collect();
    }
    let stride = nrows.div_ceil(MAX_ORACLE_ROWS);
    let mut rows: Vec<usize> = (0..nrows).step_by(stride).collect();
    if rows.last() != Some(&(nrows - 1)) {
        rows.push(nrows - 1);
    }
    rows
}

/// f64 dot product of one dequantized weight row with the activation.
fn reference_row(weights: &[f32], x: &[f32]) -> f32 {
    weights
        .iter()
        .zip(x)
        .map(|(&w, &xi)| f64::from(w) * f64::from(xi))
        .sum::<f64>() as f32
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

/// Two error numbers, because they answer different questions.
///
/// `max_rel` is the worst single output row, floored by the output's own RMS —
/// a matvec crosses zero, and dividing a bounded absolute error by a near-zero
/// row would report a huge "relative" error for a perfectly good result.
/// `vector_rel` is `||got - want|| / ||want||`, which is what a downstream
/// softmax or residual add actually sees and which a single unlucky row cannot
/// inflate.
struct ErrorProfile {
    max_rel: f32,
    max_rel_row: usize,
    vector_rel: f32,
}

fn error_profile(got: &[f32], want: &[f32], rows: &[usize]) -> ErrorProfile {
    let rms = (want
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / want.len().max(1) as f64)
        .sqrt() as f32;
    let floor = if rms > 0.0 { rms } else { 1e-6 };
    let mut max_rel = 0.0f32;
    let mut max_rel_row = rows[0];
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let rel = (g - w).abs() / w.abs().max(floor);
        if rel > max_rel {
            max_rel = rel;
            max_rel_row = rows[i];
        }
        num += f64::from(g - w) * f64::from(g - w);
        den += f64::from(w) * f64::from(w);
    }
    ErrorProfile {
        max_rel,
        max_rel_row,
        vector_rel: if den > 0.0 {
            (num.sqrt() / den.sqrt()) as f32
        } else {
            0.0
        },
    }
}

/// Run one dense GEMV over the whole matrix and return the full output vector.
fn run_gemv(
    ctx: &VulkanContext,
    kernel: Kernel,
    weights: &[u8],
    x_bytes: &[u8],
    nrows: usize,
    ncols: usize,
    spec: &GemvDenseSpec,
) -> Vec<f32> {
    let mut cache = KernelCache::new();
    let buf_w = upload(ctx, weights);
    let buf_x = upload(ctx, x_bytes);
    // HOST_CACHED, not `alloc_uma`: this is read back by the CPU, and a
    // write-combined read of even 1 MB costs ~10 ms on this part.
    let buf_d = DeviceBuffer::alloc_host_cached(ctx, nrows * 4).expect("alloc dst");
    let dummy = upload(ctx, &[0u8; 4]);
    launch_cached(
        &mut cache,
        ctx,
        kernel,
        &[&buf_w, &buf_x, &buf_d, &dummy, &dummy],
        gemv_dense_dispatch(nrows as u32, spec),
        &gemv_params_f32_b(ncols as u32, nrows as u32).to_le_bytes(),
        spec.specialization_u32(),
    )
    .expect("dense GEMV dispatch");
    read_f32(&buf_d, nrows)
}

/// The dense tensors of this model, at every width the forward pass uses.
/// `attn` and `lm_head` live in different shards, and the `lm_head` one is
/// 3.7 GB, so it is opened only when the caller wants it.
fn open_dense(
    dir: &std::path::Path,
    with_lm_head: bool,
) -> Option<infer_gguf::safetensors::SafeTensorsDir> {
    let mut paths = vec![dir.join(ATTN_SHARD)];
    if with_lm_head {
        paths.push(dir.join(LM_HEAD_SHARD));
    }
    for p in &paths {
        if !p.is_file() {
            eprintln!("skip: {} not present", p.display());
            return None;
        }
    }
    Some(infer_gguf::safetensors::SafeTensorsDir::open_files(&paths).expect("open shards"))
}

fn dense_tensors<'a>(
    st: &'a infer_gguf::safetensors::SafeTensorsDir,
    with_lm_head: bool,
) -> Vec<Dense<'a>> {
    let mut out = vec![
        // Gated attention: q_proj emits 2x the 24*256 head width, so its 12288
        // rows are the model's real q shape, not the 6144 that o_proj's input
        // width would suggest.
        Dense::load(st, "q_proj", &format!("{ATTN_PREFIX}.q_proj.weight")),
        Dense::load(st, "k_proj", &format!("{ATTN_PREFIX}.k_proj.weight")),
        // The only tensor here whose ncols is 6144 rather than 2560.
        Dense::load(st, "o_proj", &format!("{ATTN_PREFIX}.o_proj.weight")),
        // 640 wide, and deliberately so: it is the one dense width in this model
        // that does NOT divide `K_PER_ITER * BLOCK_SIZE` (4 * 128 = 512), which
        // is where `mul_mat_vec.comp`'s per-thread `num_iters` bump either
        // covers the row tail exactly or reads past it. Every other tensor here
        // divides evenly and would let that path go untested.
        Dense::load(
            st,
            "shared.down",
            &format!("{MLP_PREFIX}.shared_expert.down_proj.weight"),
        ),
    ];
    if with_lm_head {
        out.push(Dense::load(st, "lm_head", "lm_head.weight"));
    }
    out
}

/// Claim 1 standalone, GPU-free: a box without a Vulkan device still gets the
/// bound on what the f16 tier costs this checkpoint.
#[test]
fn bf16_to_f16_loses_only_the_subnormal_tail_of_this_checkpoint() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(st) = open_dense(&dir, bench_enabled()) else {
        return;
    };
    eprintln!(
        "{:<10} {:>12} {:>8} {:>10} {:>12} {:>12}",
        "tensor", "weights", "flushed", "inexact", "max |dw|", "max |w| hit"
    );
    for dense in dense_tensors(&st, bench_enabled()) {
        let f16 = dense.f16();
        // Decode BOTH encodings of every weight with independently written
        // decoders and measure where they disagree. `f16()` already refused a
        // saturating overflow; this finds everything else the encoder's own
        // report would miss if that report were wrong.
        let (mut inexact, mut flushed) = (0usize, 0usize);
        let (mut max_delta, mut max_hit) = (0.0f32, 0.0f32);
        for (b, h) in dense.bf16.chunks_exact(2).zip(f16.chunks_exact(2)) {
            let want = bf16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            let got = f16_to_f32(u16::from_le_bytes([h[0], h[1]]));
            if got.to_bits() == want.to_bits() {
                continue;
            }
            inexact += 1;
            if got == 0.0 {
                flushed += 1;
            }
            max_delta = max_delta.max((got - want).abs());
            max_hit = max_hit.max(want.abs());
        }
        let total = dense.nrows * dense.ncols;
        eprintln!(
            "{:<10} {total:>12} {flushed:>8} {inexact:>10} {max_delta:>12.3e} {max_hit:>12.3e}",
            dense.label
        );
        // The claim is NOT that nothing moved — 0.025% of these weights do.
        // It is that everything that moved was already below f16's smallest
        // normal, and moved by at most half a step of f16's 2^-24 subnormal
        // grid. Both bounds are sharp: a normal-range value that changed, or a
        // delta above half an ULP, is an encoder defect rather than the tail.
        assert!(
            max_hit < F16_EXACT_FROM_BF16,
            "{}: a weight of magnitude {max_hit:.3e} changed under the f16              re-encode, but the encoding is exact for everything at or above              {:.3e} — the loss has escaped the band it is allowed to touch",
            dense.label,
            F16_EXACT_FROM_BF16
        );
        assert!(
            max_delta <= F16_SUBNORMAL_HALF_ULP,
            "{}: f16 re-encode moved a weight by {max_delta:.3e}, more than half              a step of the 2^-24 subnormal grid ({:.3e})",
            dense.label,
            F16_SUBNORMAL_HALF_ULP
        );
    }
}

/// Claim 2: both dense pipelines against an f64 CPU dot product, at every real
/// width. `lm_head` joins under `ARLE_DENSE_GEMV_BENCH` (2.5 GB of uploads).
#[test]
fn dense_gemv_matches_cpu_oracle_on_real_dense_bytes() {
    let Some(dir) = checkpoint_dir() else { return };
    let with_lm_head = bench_enabled();
    let Some(st) = open_dense(&dir, with_lm_head) else {
        return;
    };
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping dense GEMV test");
            return;
        }
    };
    eprintln!("ARLE Vulkan dense GEMV proof on: {}", ctx.device_name());
    let spec = GemvDenseSpec::DEFAULT;

    for dense in dense_tensors(&st, with_lm_head) {
        let (nrows, ncols) = (dense.nrows, dense.ncols);
        let f16 = dense.f16();

        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let x: Vec<f32> = (0..ncols).map(|_| rng.next_unit_f32()).collect();
        let x_bytes = f32_bytes(&x);
        let rows = sampled_rows(nrows);

        for (kernel, bytes, decode) in [
            (
                Kernel::GemvF16,
                f16.as_slice(),
                f16_to_f32 as fn(u16) -> f32,
            ),
            (Kernel::GemvBf16, dense.bf16, bf16_to_f32 as fn(u16) -> f32),
        ] {
            let got_all = run_gemv(&ctx, kernel, bytes, &x_bytes, nrows, ncols, &spec);
            // The oracle decodes the SAME bytes the kernel was handed, so a
            // difference is the kernel's arithmetic, never the encoding.
            let want: Vec<f32> = rows
                .iter()
                .map(|&r| reference_row(&dense.row(bytes, r, decode), &x))
                .collect();
            let got: Vec<f32> = rows.iter().map(|&r| got_all[r]).collect();
            let e = error_profile(&got, &want, &rows);
            eprintln!(
                "[{}] {kernel:?} [{nrows}x{ncols}] max rel {:.3e} (row {}, got {} want {}), \
                 vector rel {:.3e}, {} rows checked",
                dense.label,
                e.max_rel,
                e.max_rel_row,
                got[rows.iter().position(|&r| r == e.max_rel_row).unwrap_or(0)],
                want[rows.iter().position(|&r| r == e.max_rel_row).unwrap_or(0)],
                e.vector_rel,
                rows.len()
            );
            assert!(
                e.max_rel < TOL,
                "{}: {kernel:?} max rel err {} >= {TOL}",
                dense.label,
                e.max_rel
            );
            assert!(
                e.vector_rel < TOL,
                "{}: {kernel:?} vector rel err {} >= {TOL}",
                dense.label,
                e.vector_rel
            );
        }
    }
}

/// The two asserts above have to be able to fail, and at f32-epsilon tolerances
/// that is not obvious — so the discriminating power is measured here rather
/// than assumed.
///
/// Both mutations are the ones a dense GEMV actually gets wrong, applied to the
/// SAME metric and the SAME tolerance the real test uses:
///
/// - **wrong row stride.** The weights are uploaded with rows `ncols + 4`
///   elements apart while the push block still says `ncols`, so every row past
///   the first reads bytes shifted out from under it. This is the live hazard
///   for a caller binding a padded or sliced tensor, and it is worth pinning
///   that `mul_mat_vec.comp` gives it no defence: the shader's row stride is
///   `p.ncols` (`ibi += p.ncols`) and it never reads `p.stride_a` at all, so
///   [`gemv_params_f32_b`]'s `stride_a` word cannot express a padded row.
/// - **dropped final accumulation.** The oracle omits the last term of each dot
///   product — exactly the residue a kernel leaves when its tail iteration
///   never runs, which on this shader is the per-thread `num_iters` bump that
///   the 640-wide tensor depends on.
///
/// Measured on the 8060S (k_proj, 512x2560): the clean path reports 2.820e-7,
/// the wrong stride 2.864e0, the dropped tail 8.869e-2 — a 1e7x and a 887x
/// margin over the 1e-4 tolerance. `MIN_DEFECT_SIGNAL` asserts that gap survives; if
/// a future change ever narrows it, this test says so instead of the real test
/// quietly ceasing to discriminate.
#[test]
fn dense_gemv_oracle_rejects_a_wrong_row_stride_and_a_dropped_tail() {
    /// A defect must be at least this far above [`TOL`] to count as detected.
    const MIN_DEFECT_SIGNAL: f32 = 100.0 * TOL;
    /// Elements of padding per row. A multiple of 4 so the padded layout still
    /// satisfies the shader's only real shape constraint — the mutation under
    /// test is the stride, not the alignment.
    const PAD: usize = 4;

    let Some(dir) = checkpoint_dir() else { return };
    let Some(st) = open_dense(&dir, false) else {
        return;
    };
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping mutation control");
            return;
        }
    };
    let dense = Dense::load(&st, "k_proj", &format!("{ATTN_PREFIX}.k_proj.weight"));
    let (nrows, ncols) = (dense.nrows, dense.ncols);
    let spec = GemvDenseSpec::DEFAULT;
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let x: Vec<f32> = (0..ncols).map(|_| rng.next_unit_f32()).collect();
    let x_bytes = f32_bytes(&x);
    let rows: Vec<usize> = (0..nrows).collect();

    let want: Vec<f32> = rows
        .iter()
        .map(|&r| reference_row(&dense.row(dense.bf16, r, bf16_to_f32), &x))
        .collect();

    let clean = run_gemv(
        &ctx,
        Kernel::GemvBf16,
        dense.bf16,
        &x_bytes,
        nrows,
        ncols,
        &spec,
    );
    let baseline = error_profile(&clean, &want, &rows).max_rel;

    // Mutation A: rows `ncols + PAD` apart, push block unchanged.
    let padded_stride = (ncols + PAD) * 2;
    let mut padded = vec![0u8; nrows * padded_stride];
    for r in 0..nrows {
        padded[r * padded_stride..r * padded_stride + ncols * 2]
            .copy_from_slice(&dense.bf16[r * ncols * 2..(r + 1) * ncols * 2]);
    }
    let strided = run_gemv(
        &ctx,
        Kernel::GemvBf16,
        &padded,
        &x_bytes,
        nrows,
        ncols,
        &spec,
    );
    let wrong_stride = error_profile(&strided, &want, &rows).max_rel;

    // Mutation B: the oracle stops one term short of the row.
    let dropped_want: Vec<f32> = rows
        .iter()
        .map(|&r| reference_row(&dense.row(dense.bf16, r, bf16_to_f32)[..ncols - 1], &x))
        .collect();
    let dropped_tail = error_profile(&clean, &dropped_want, &rows).max_rel;

    eprintln!(
        "[mutation control] k_proj [{nrows}x{ncols}] max rel: clean {baseline:.3e},          row stride +{PAD} {wrong_stride:.3e}, dropped final term {dropped_tail:.3e}          (tolerance {TOL:.0e})"
    );
    assert!(baseline < TOL, "clean path regressed: {baseline} >= {TOL}");
    for (label, signal) in [
        ("a row stride of ncols+4", wrong_stride),
        ("a dropped final accumulation", dropped_tail),
    ] {
        assert!(
            signal > MIN_DEFECT_SIGNAL,
            "the oracle no longer discriminates {label}: it scores {signal:.3e},              which is not clear of {MIN_DEFECT_SIGNAL:.0e} — the correctness              asserts above have stopped being able to fail"
        );
    }
}

/// Repeats per timed run, chosen so a small (cache-resident) matrix is not
/// timed against submit latency and a 1.27 GB one does not run for a second.
fn passes_for(bytes: usize) -> usize {
    const TARGET_BYTES: usize = 2 << 30;
    (TARGET_BYTES / bytes.max(1)).clamp(8, 512)
}

/// Achieved GB/s for one (kernel, geometry, matrix), reading the weights
/// `passes` times back-to-back in ONE submit with a barrier between — the
/// serial shape the decode path issues, not an overlapped best case.
#[allow(clippy::too_many_arguments)]
fn bandwidth<'a>(
    ctx: &'a VulkanContext,
    cache: &mut KernelCache<'a>,
    kernel: Kernel,
    buf_w: &DeviceBuffer<'a>,
    buf_x: &DeviceBuffer<'a>,
    buf_d: &DeviceBuffer<'a>,
    dummy: &DeviceBuffer<'a>,
    nrows: usize,
    ncols: usize,
    spec: &GemvDenseSpec,
) -> (f64, f64) {
    let push = gemv_params_f32_b(ncols as u32, nrows as u32).to_le_bytes();
    let (pipeline, layout) = cache
        .get(ctx, kernel, spec.specialization_u32(), push.len() as u32, 5)
        .expect("dense GEMV pipeline");
    let set = DescriptorSet::storage_buffers(ctx, layout, &[buf_w, buf_x, buf_d, dummy, dummy])
        .expect("bind");
    let d: Dispatch = gemv_dense_dispatch(nrows as u32, spec);
    let weight_bytes = nrows * ncols * 2;
    let passes = passes_for(weight_bytes);

    let run = || {
        let mut rec = CommandRecorder::new(ctx).expect("recorder");
        rec.begin().expect("begin");
        for _ in 0..passes {
            record_dispatch(&mut rec, pipeline, &set, &push, [d.x, d.y, d.z]);
            rec.barrier();
        }
        let t0 = Instant::now();
        rec.submit_and_wait().expect("submit");
        t0.elapsed().as_secs_f64()
    };
    run(); // warm: first pass pays page-in
    let secs = run();
    let gbps = (weight_bytes as f64 * passes as f64) / secs / 1e9;
    (gbps, secs / passes as f64 * 1e6)
}

/// Claim 3: what the dense tier actually gets, per width, against the 204.9
/// GB/s streaming-read ceiling measured on this part.
#[test]
fn report_dense_gemv_bandwidth_by_width() {
    if !bench_enabled() {
        eprintln!("set ARLE_DENSE_GEMV_BENCH=1 to run the dense GEMV bandwidth sweep; skipping");
        return;
    }
    let Some(dir) = checkpoint_dir() else { return };
    let Some(st) = open_dense(&dir, true) else {
        return;
    };
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping dense GEMV bench");
            return;
        }
    };
    eprintln!("ARLE dense GEMV bandwidth on: {}", ctx.device_name());
    let mut cache = KernelCache::new();
    let dummy = upload(&ctx, &[0u8; 4]);
    let spec = GemvDenseSpec::DEFAULT;

    eprintln!(
        "{:<10} {:>6} {:>6} {:>9} {:>12} {:>9} {:>12} {:>9}",
        "tensor", "rows", "cols", "MiB", "f16 GB/s", "f16 us", "bf16 GB/s", "bf16 us"
    );
    for dense in dense_tensors(&st, true) {
        let (nrows, ncols) = (dense.nrows, dense.ncols);
        let f16 = dense.f16();
        let x_bytes = f32_bytes(&vec![0.5f32; ncols]);
        let buf_x = upload(&ctx, &x_bytes);
        // Never read back, so `alloc_uma` keeps the destination on the device
        // heap and out of the measurement.
        let buf_d = DeviceBuffer::alloc_uma(&ctx, nrows * 4).expect("alloc dst");

        let mut row = Vec::new();
        for (kernel, bytes) in [
            (Kernel::GemvF16, f16.as_slice()),
            (Kernel::GemvBf16, dense.bf16),
        ] {
            let buf_w = upload(&ctx, bytes);
            row.push(bandwidth(
                &ctx, &mut cache, kernel, &buf_w, &buf_x, &buf_d, &dummy, nrows, ncols, &spec,
            ));
        }
        eprintln!(
            "{:<10} {nrows:>6} {ncols:>6} {:>9.1} {:>12.1} {:>9.1} {:>12.1} {:>9.1}",
            dense.label,
            (nrows * ncols * 2) as f64 / (1 << 20) as f64,
            row[0].0,
            row[0].1,
            row[1].0,
            row[1].1
        );
    }
}

/// Where the `SPEC_GEMV_DENSE` geometry came from — and the evidence that it is
/// not a lever. `lm_head` only: at 1.27 GB it is far past the 16 MiB cache
/// cliff, so the number is the sustained link and not a cache artifact.
///
/// Over six sittings the whole 12-cell grid lands in 205.8-236.6 GB/s against a
/// 256 GB/s spec — the kernel is bandwidth-saturated everywhere, so read this
/// table for the absence of a knee, not for a winner. Run it repeatedly before
/// believing any single cell: the run-to-run spread is ~+-2%, comparable to the
/// gap between the best and second-best geometry.
#[test]
fn report_dense_gemv_geometry_sweep() {
    if !bench_enabled() {
        eprintln!("set ARLE_DENSE_GEMV_BENCH=1 to run the dense GEMV geometry sweep; skipping");
        return;
    }
    let Some(dir) = checkpoint_dir() else { return };
    let Some(st) = open_dense(&dir, true) else {
        return;
    };
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping geometry sweep");
            return;
        }
    };
    eprintln!("ARLE dense GEMV geometry sweep on: {}", ctx.device_name());
    let dense = Dense::load(&st, "lm_head", "lm_head.weight");
    let (nrows, ncols) = (dense.nrows, dense.ncols);
    let mut cache = KernelCache::new();
    let dummy = upload(&ctx, &[0u8; 4]);
    let buf_w = upload(&ctx, dense.bf16);
    let buf_x = upload(&ctx, &f32_bytes(&vec![0.5f32; ncols]));
    let buf_d = DeviceBuffer::alloc_uma(&ctx, nrows * 4).expect("alloc dst");

    eprintln!("lm_head [{nrows}x{ncols}] bf16, GB/s by BLOCK_SIZE x NUM_ROWS:");
    eprintln!(
        "{:>10} {:>10} {:>10} {:>10} {:>10}",
        "block", "rows=1", "rows=2", "rows=4", "rows=8"
    );
    for block in [32u32, 64, 128] {
        let mut cells = Vec::new();
        for num_rows in [1u32, 2, 4, 8] {
            let spec = GemvDenseSpec::new(block, num_rows);
            let (gbps, _) = bandwidth(
                &ctx,
                &mut cache,
                Kernel::GemvBf16,
                &buf_w,
                &buf_x,
                &buf_d,
                &dummy,
                nrows,
                ncols,
                &spec,
            );
            cells.push(gbps);
        }
        eprintln!(
            "{block:>10} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
            cells[0], cells[1], cells[2], cells[3]
        );
    }
}
