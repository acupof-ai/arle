//! The dense-tier Q8_0 flip, end to end on REAL checkpoint bytes:
//! `quantize_q8_0_from_bf16` (load-time CPU quantizer) -> `Kernel::GemvQ8_0`
//! (vendored `mul_mat_vecq` int-dot GEMV) -> what it achieves against this
//! part's memory ceiling. This file is the machinery proof the
//! `model_qwen4_exp.rs` integration copies; nothing here touches the model
//! file.
//!
//! Qwen3.8-Flash-Next's per-token GPU stream is ~10 GB, and the single
//! largest tier is dense BF16 (~7.1 GB: linattn in/out projections 4.16 GB,
//! full-attn 1.2, shared expert 0.47, lm_head 1.27). Q8_0 re-encodes 2 B/elem
//! as 1.0625 B/elem, so a flipped tensor moves 1.88x fewer bytes through the
//! ~207 GB/s link. The three tensors here are the real shapes of that tier:
//! linattn `in_proj_qkv` [10240x2560], linattn `out_proj` [2560x6144], and
//! `lm_head` [248320x2560] — both activation widths the model has (2560 and
//! 6144), and one tensor big enough to report the sustained link.
//!
//! Three claims, each with its own oracle:
//!
//! 1. **The device GEMV computes the product of the bytes it was handed.**
//!    f64 dot products over `infer_gguf`'s independent `block_q8_0`
//!    dequantizer x the q8_1_x4 activation bytes READ BACK from the device —
//!    both operands exactly what the kernel sees, so this column isolates the
//!    GEMV arithmetic from the quantization. Bound: 1e-4.
//! 2. **What the flip costs in accuracy.** The same device output against the
//!    f64 (bf16 weights x f32 activation) reference — the pipeline being
//!    replaced. This column carries BOTH quantizations (Q8_0 weights + q8_1
//!    activations); it is the number the default-flip decision hangs on.
//! 3. **What the flip buys.** Achieved GB/s per width vs the same-sitting
//!    BF16 GEMV on the same tensor (the back-to-back ratio is the
//!    throttle-proof number), plus the projected per-token saving at the
//!    tier sizes above and the measured `QuantizeQ8_1` dispatch cost at both
//!    activation widths — this box is fence-sensitive, so the integration
//!    needs the price of each extra dispatch, isolated AND interleaved.
//!
//! The activation recipe claim 1 exercises is the one the integration copies:
//! ONE `QuantizeQ8_1` dispatch per activation width per token, shared by every
//! Q8_0 GEMV of that width — here the single width-2560 slot feeds both
//! `in_proj_qkv` and `lm_head`.
//!
//! `lm_head` and the bandwidth/dispatch-cost benches allocate ~2.6 GB of
//! device memory and are opt-in:
//!
//! ```text
//! ARLE_Q8_DENSE_BENCH=1 cargo test -p vulkan-kernels --features vulkan \
//!     --test device_gemv_q8_dense --release -- --nocapture --test-threads=1
//! ```
//!
//! Mutation runs (each applied to the live code, watched fail, reverted;
//! numbers recorded in the campaign report):
//! - quantizer `.round()` -> truncation: lib fixture
//!   `q8_0_rounding_is_half_away_from_zero_not_rne` fails (2.5 -> 2, want 3)
//!   and `q8_0_roundtrip_stays_within_half_an_lsb_of_the_scale` fails;
//!   here `q8_0_dense_gemv_mutations_fail_loudly` measures the same mutation
//!   live on every run — the byte diff and the quality-error growth.
//! - swapped 16-byte qs halves and a plain-f32 B operand: also run live
//!   inside `q8_0_dense_gemv_mutations_fail_loudly` — the clean/mutated error
//!   gap is asserted, not assumed.
//!
//! Skips cleanly with no device or no checkpoint.
#![cfg(feature = "vulkan")]

use std::path::PathBuf;
use std::time::Instant;

use infer_gguf::dequant::dequantize_row_q8_0;
use vulkan_kernels::{
    BLOCK_Q8_0_BYTES, BLOCK_Q8_1_BYTES, Dispatch, GemvDenseSpec, Kernel, KernelCache,
    gemv_dense_dispatch, gemv_dispatch, gemv_params, gemv_params_f32_b, q8_0_gemv_with_params,
    q8_0_row_bytes, q8_1_quantize, q8_1_quantize_dispatch, q8_1_quantize_params,
    quantize_q8_0_from_bf16, record_dispatch,
};
use vulkan_sys::{CommandRecorder, DescriptorSet, DeviceBuffer, VulkanContext};

/// Claim-1 bound. The device runs exact int8 dot products per 32-block and
/// combines the two f16 scales in f32; the oracle repeats the same terms in
/// f64, so the residue is summation order. Worst measured on the 8060S across
/// all three tensors is 3.7e-7 max-rel; 1e-4 leaves ~270x headroom while
/// staying decades under what the mutation controls below measure for real
/// defects (4.4e0 for swapped block halves, 2.5e6 for a plain-f32 B).
const TOL_GEMV: f32 = 1e-4;

/// Claim-2 sanity ceiling. The measured quality cost of the flip is the
/// REPORTED number (~1e-2-class vector rel for both quantizations combined);
/// this assert only refuses a silent order-of-magnitude regression.
const TOL_QUALITY: f32 = 5e-2;

/// Measured on this box, 2026-08. Override with `INFER_SAFETENSORS_TEST_DIR`,
/// matching `infer_gguf::safetensors`'s own on-box tests.
const CHECKPOINT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";

/// Layer 12 is a `linear_attention` layer (full attention sits at zero-indexed
/// 3, 7, ... 47), so its projections are the real linattn shapes — the 4.16
/// GB/token tier this flip targets first.
const LINATTN_SHARD: &str = "model-bf16-00011.safetensors";
const LM_HEAD_SHARD: &str = "model-bf16-00012.safetensors";
const LINATTN_PREFIX: &str = "model.language_model.layers.12.linear_attn";

/// Cap on how many output rows the f64 oracle covers. Only `lm_head` exceeds
/// it; the GEMV still runs over all 248320 rows, and [`sampled_rows`] strides
/// so the checked set spans the whole matrix including the last row.
const MAX_ORACLE_ROWS: usize = 8192;

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
    std::env::var("ARLE_Q8_DENSE_BENCH").is_ok()
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
    fn next_unit_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// bf16 -> f32: bf16 IS the top half of the f32 bit pattern.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Minimal IEEE binary16 decode for the q8_1_x4 `ds` fields.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = (bits >> 10) & 0x1F;
    let man = f32::from(bits & 0x3FF);
    match exp {
        0 => sign * man * 2.0f32.powi(-24),
        0x1F if man == 0.0 => sign * f32::INFINITY,
        0x1F => f32::NAN,
        e => sign * (1.0 + man / 1024.0) * 2.0f32.powi(i32::from(e) - 15),
    }
}

fn to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn upload<'a>(ctx: &'a VulkanContext, bytes: &[u8]) -> DeviceBuffer<'a> {
    let mut buf = DeviceBuffer::alloc_uma(ctx, bytes.len()).expect("alloc device buffer");
    buf.copy_from_host(bytes).expect("upload");
    buf
}

fn read_f32(buf: &DeviceBuffer<'_>, count: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; count * 4];
    buf.copy_to_host(&mut bytes).expect("read back dst");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One dense `[nrows, ncols]` BF16 weight matrix off the mmap, plus its Q8_0
/// re-encode — the pair every claim below compares.
struct Dense<'a> {
    label: &'static str,
    nrows: usize,
    ncols: usize,
    bf16: &'a [u8],
    q8_0: Vec<u8>,
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
        let row_bytes = q8_0_row_bytes(ncols)
            .unwrap_or_else(|| panic!("{name}: ncols {ncols} is not a whole number of blocks"));
        let mut q8_0 = vec![0u8; nrows * row_bytes];
        let t0 = Instant::now();
        quantize_q8_0_from_bf16(bf16, nrows, ncols, &mut q8_0)
            .unwrap_or_else(|e| panic!("{name}: q8_0 quantize failed: {e}"));
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "[{label}] CPU quantize {nrows}x{ncols}: {:.0} ms ({:.2} GB/s of bf16 in)",
            secs * 1e3,
            bf16.len() as f64 / secs / 1e9
        );
        Self {
            label,
            nrows,
            ncols,
            bf16,
            q8_0,
        }
    }

    fn q8_row_bytes(&self) -> usize {
        q8_0_row_bytes(self.ncols).expect("validated at load")
    }

    fn bf16_row(&self, r: usize) -> Vec<f32> {
        self.bf16[r * self.ncols * 2..(r + 1) * self.ncols * 2]
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    }

    /// One row of quantized bytes through `infer_gguf`'s independent
    /// `block_q8_0` reader — the oracle side of claim 1.
    fn q8_row(&self, bytes: &[u8], r: usize) -> Vec<f32> {
        let rb = self.q8_row_bytes();
        dequantize_row_q8_0(&bytes[r * rb..(r + 1) * rb], self.ncols).expect("dequant q8_0 row")
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

/// f64 dot product of one weight row with the activation.
fn reference_row(weights: &[f32], x: &[f32]) -> f32 {
    weights
        .iter()
        .zip(x)
        .map(|(&w, &xi)| f64::from(w) * f64::from(xi))
        .sum::<f64>() as f32
}

/// Two error numbers, because they answer different questions. `max_rel` is
/// the worst single output row, floored by the output's own RMS (a matvec
/// crosses zero); `vector_rel` is `||got - want|| / ||want||`, what a
/// downstream softmax or residual add actually sees.
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

/// THE RECIPE, step 1: quantize one activation vector into a `block_q8_1_x4`
/// device slot with a single `Kernel::QuantizeQ8_1` dispatch. Returns the
/// device buffer (shared by every same-width GEMV afterwards) AND the decoded
/// f32 values read back from it, so the claim-1 oracle multiplies exactly the
/// activation bytes the kernel reads.
fn quantize_activation_slot<'a>(ctx: &'a VulkanContext, x: &[f32]) -> (DeviceBuffer<'a>, Vec<f32>) {
    let ne = x.len();
    let out_len = ne.div_ceil(128) * 4 * BLOCK_Q8_1_BYTES;
    let buf_in = upload(ctx, &to_bytes(x));
    let mut buf_out = DeviceBuffer::alloc(ctx, out_len).expect("alloc q8_1 slot");
    buf_out
        .copy_from_host(&vec![0u8; out_len])
        .expect("zero q8_1 slot");
    q8_1_quantize(
        ctx,
        &[&buf_in, &buf_out],
        q8_1_quantize_dispatch(ne as u32),
        &q8_1_quantize_params(ne as u32),
    )
    .expect("QuantizeQ8_1 dispatch");

    let mut bytes = vec![0u8; out_len];
    buf_out.copy_to_host(&mut bytes).expect("read back q8_1");
    (buf_out, decode_q8_1_x4(&bytes, ne))
}

/// Decode `block_q8_1_x4` bytes ({f16vec2 ds[4]; int32 qs[32]} per 144 B, the
/// int8s sequential) back to f32 — the activation the int-dot GEMV actually
/// multiplies.
fn decode_q8_1_x4(bytes: &[u8], ne: usize) -> Vec<f32> {
    (0..ne)
        .map(|k| {
            let base = k / 128 * 144;
            let inner = (k % 128) / 32;
            let d = f16_to_f32(u16::from_le_bytes([
                bytes[base + inner * 4],
                bytes[base + inner * 4 + 1],
            ]));
            f32::from(bytes[base + 16 + k % 128] as i8) * d
        })
        .collect()
}

/// THE RECIPE, step 2: one `Kernel::GemvQ8_0` over a Q8_0 weight matrix and a
/// shared q8_1_x4 activation slot. 5 bindings `[A, B, D, Fuse0, Fuse1]`,
/// [`gemv_params`] push block (stride_b in q8_1 BLOCKS), one workgroup per
/// output row.
fn run_q8_gemv(
    ctx: &VulkanContext,
    weights: &[u8],
    buf_b: &DeviceBuffer<'_>,
    nrows: usize,
    ncols: usize,
) -> Vec<f32> {
    let buf_w = upload(ctx, weights);
    // HOST_CACHED: read back by the CPU; a write-combined read is the ~100x
    // trap this repo keeps re-measuring.
    let buf_d = DeviceBuffer::alloc_host_cached(ctx, nrows * 4).expect("alloc dst");
    let dummy = upload(ctx, &[0u8; 4]);
    q8_0_gemv_with_params(
        ctx,
        &[&buf_w, buf_b, &buf_d, &dummy, &dummy],
        gemv_dispatch(nrows as u32),
        &gemv_params(ncols as u32, nrows as u32),
    )
    .expect("GemvQ8_0 dispatch");
    read_f32(&buf_d, nrows)
}

fn open_shards(
    dir: &std::path::Path,
    with_lm_head: bool,
) -> Option<infer_gguf::safetensors::SafeTensorsDir> {
    let mut paths = vec![dir.join(LINATTN_SHARD)];
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
        Dense::load(
            st,
            "in_proj_qkv",
            &format!("{LINATTN_PREFIX}.in_proj_qkv.weight"),
        ),
        Dense::load(st, "out_proj", &format!("{LINATTN_PREFIX}.out_proj.weight")),
    ];
    if with_lm_head {
        out.push(Dense::load(st, "lm_head", "lm_head.weight"));
    }
    out
}

/// Claims 1 and 2: the device GEMV against BOTH oracles, per tensor, with the
/// width-2560 activation slot quantized ONCE and shared by `in_proj_qkv` and
/// `lm_head` — the exact dispatch structure the integration copies.
#[test]
fn q8_0_dense_gemv_matches_oracles_on_real_bytes() {
    let Some(dir) = checkpoint_dir() else { return };
    let with_lm_head = bench_enabled();
    let Some(st) = open_shards(&dir, with_lm_head) else {
        return;
    };
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping q8_0 dense GEMV test");
            return;
        }
    };
    eprintln!("ARLE q8_0 dense GEMV proof on: {}", ctx.device_name());

    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let x2560: Vec<f32> = (0..2560).map(|_| rng.next_unit_f32()).collect();
    let x6144: Vec<f32> = (0..6144).map(|_| rng.next_unit_f32()).collect();
    // ONE QuantizeQ8_1 dispatch per width per token. The 2560 slot feeds both
    // in_proj_qkv and lm_head below; only out_proj needs the 6144 slot.
    let (slot2560, xq2560) = quantize_activation_slot(&ctx, &x2560);
    let (slot6144, xq6144) = quantize_activation_slot(&ctx, &x6144);

    eprintln!(
        "{:<12} {:>7}x{:<5} | {:>13} {:>13} | {:>13} {:>13}",
        "tensor", "rows", "cols", "gemv max_rel", "gemv vec_rel", "qual max_rel", "qual vec_rel"
    );
    for dense in dense_tensors(&st, with_lm_head) {
        let (nrows, ncols) = (dense.nrows, dense.ncols);
        let (x, xq, slot) = if ncols == 2560 {
            (&x2560, &xq2560, &slot2560)
        } else {
            (&x6144, &xq6144, &slot6144)
        };
        let got_all = run_q8_gemv(&ctx, &dense.q8_0, slot, nrows, ncols);
        let rows = sampled_rows(nrows);
        let got: Vec<f32> = rows.iter().map(|&r| got_all[r]).collect();

        // Claim 1: the GEMV in isolation — quantized weights x the q8_1
        // activation values read back from the device.
        let want_gemv: Vec<f32> = rows
            .iter()
            .map(|&r| reference_row(&dense.q8_row(&dense.q8_0, r), xq))
            .collect();
        // Claim 2: the pipeline being replaced — bf16 weights x f32 x.
        let want_qual: Vec<f32> = rows
            .iter()
            .map(|&r| reference_row(&dense.bf16_row(r), x))
            .collect();
        let e_gemv = error_profile(&got, &want_gemv, &rows);
        let e_qual = error_profile(&got, &want_qual, &rows);
        eprintln!(
            "{:<12} {:>7}x{:<5} | {:>13.3e} {:>13.3e} | {:>13.3e} {:>13.3e}  ({} rows)",
            dense.label,
            nrows,
            ncols,
            e_gemv.max_rel,
            e_gemv.vector_rel,
            e_qual.max_rel,
            e_qual.vector_rel,
            rows.len()
        );
        assert!(
            e_gemv.max_rel < TOL_GEMV,
            "{}: GemvQ8_0 disagrees with the f64 oracle over its own bytes: \
             max rel {} (row {}) >= {TOL_GEMV}",
            dense.label,
            e_gemv.max_rel,
            e_gemv.max_rel_row
        );
        assert!(
            e_qual.vector_rel < TOL_QUALITY,
            "{}: the flip's quality cost exploded: vector rel {} >= {TOL_QUALITY}",
            dense.label,
            e_qual.vector_rel
        );
    }
}

/// The truncation mutant of `quantize_q8_0_from_bf16`, kept test-local: the
/// real quantizer supplies the scale bytes so the mutation is ONLY the value
/// rounding — `trunc` where ggml has `roundf`.
fn quantize_q8_0_truncating(dense: &Dense<'_>) -> Vec<u8> {
    let mut out = vec![0u8; dense.q8_0.len()];
    for (src, dst) in dense
        .bf16
        .chunks_exact(64)
        .zip(out.chunks_exact_mut(BLOCK_Q8_0_BYTES))
    {
        let vals: Vec<f32> = src
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut clean = [0u8; BLOCK_Q8_0_BYTES];
        quantize_q8_0_from_bf16(src, 1, 32, &mut clean).expect("scale bytes");
        dst[..2].copy_from_slice(&clean[..2]);
        for (q, v) in dst[2..].iter_mut().zip(&vals) {
            *q = (v * id).trunc() as i8 as u8;
        }
    }
    out
}

/// The asserts above must be able to fail, and the failure must be loud. The
/// three mutations are the real ways this machinery goes wrong, applied live
/// against the SAME oracles and metric:
///
/// - **Swapped 16-byte qs halves** in every `block_q8_0` — the layout bug a
///   wrong repack ships. Finite, plausible-looking output.
/// - **A plain f32 activation as the B operand** — `mul_mat_vecq` gives it no
///   defence; the quantizer doc calls this silent garbage and this test is
///   the measurement behind that sentence.
/// - **Truncating value rounding** in the quantizer — the byte diff against
///   the round-half-away bytes plus the measured quality-error growth. (The
///   exact-.5 fixture that separates half-away from nearest-even lives in the
///   lib tests, where the bytes are hand-computable.)
#[test]
fn q8_0_dense_gemv_mutations_fail_loudly() {
    /// A defect signal must clear the tolerance by this factor to count.
    const MIN_DEFECT_SIGNAL: f32 = 100.0 * TOL_GEMV;

    let Some(dir) = checkpoint_dir() else { return };
    let Some(st) = open_shards(&dir, false) else {
        return;
    };
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping q8_0 mutation controls");
            return;
        }
    };
    let dense = Dense::load(
        &st,
        "in_proj_qkv",
        &format!("{LINATTN_PREFIX}.in_proj_qkv.weight"),
    );
    let (nrows, ncols) = (dense.nrows, dense.ncols);
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let x: Vec<f32> = (0..ncols).map(|_| rng.next_unit_f32()).collect();
    let (slot, xq) = quantize_activation_slot(&ctx, &x);
    let rows = sampled_rows(nrows);

    // The clean oracle: quantized weights x the device's own q8_1 activation.
    let want: Vec<f32> = rows
        .iter()
        .map(|&r| reference_row(&dense.q8_row(&dense.q8_0, r), &xq))
        .collect();
    let clean_all = run_q8_gemv(&ctx, &dense.q8_0, &slot, nrows, ncols);
    let clean: Vec<f32> = rows.iter().map(|&r| clean_all[r]).collect();
    let baseline = error_profile(&clean, &want, &rows).max_rel;

    // Mutation A: swap the two 16-byte halves of every block's qs.
    let mut swapped = dense.q8_0.clone();
    for block in swapped.chunks_exact_mut(BLOCK_Q8_0_BYTES) {
        let (a, b) = block[2..].split_at_mut(16);
        a.swap_with_slice(b);
    }
    let got_all = run_q8_gemv(&ctx, &swapped, &slot, nrows, ncols);
    let got: Vec<f32> = rows.iter().map(|&r| got_all[r]).collect();
    let swapped_halves = error_profile(&got, &want, &rows).max_rel;

    // Mutation B: bind the plain f32 activation vector as B, through the same
    // call path. The buffer is even a plausible size — nothing in the API
    // stops this, which is exactly why the recipe documents it as the trap.
    let f32_b = upload(&ctx, &to_bytes(&x));
    let got_all = run_q8_gemv(&ctx, &dense.q8_0, &f32_b, nrows, ncols);
    let got: Vec<f32> = rows.iter().map(|&r| got_all[r]).collect();
    let plain_f32_b = error_profile(&got, &want, &rows).max_rel;

    // Mutation C: truncating value rounding. Byte diff first (deterministic
    // and loud), then the measured quality growth on the device.
    let truncated = quantize_q8_0_truncating(&dense);
    let differing = truncated
        .iter()
        .zip(&dense.q8_0)
        .filter(|(a, b)| a != b)
        .count();
    let diff_frac = differing as f64 / truncated.len() as f64;
    let want_qual: Vec<f32> = rows
        .iter()
        .map(|&r| reference_row(&dense.bf16_row(r), &x))
        .collect();
    let clean_qual = error_profile(&clean, &want_qual, &rows).vector_rel;
    let got_all = run_q8_gemv(&ctx, &truncated, &slot, nrows, ncols);
    let got: Vec<f32> = rows.iter().map(|&r| got_all[r]).collect();
    let trunc_qual = error_profile(&got, &want_qual, &rows).vector_rel;

    eprintln!(
        "[mutation control] in_proj_qkv [{nrows}x{ncols}] max rel vs the q8 oracle: \
         clean {baseline:.3e}, swapped qs halves {swapped_halves:.3e}, plain-f32 B \
         {plain_f32_b:.3e} (tolerance {TOL_GEMV:.0e}); truncating rounding: {:.1}% of \
         bytes differ, quality vector rel {clean_qual:.3e} -> {trunc_qual:.3e}",
        diff_frac * 100.0
    );
    assert!(
        baseline < TOL_GEMV,
        "clean path regressed: {baseline} >= {TOL_GEMV}"
    );
    for (label, signal) in [
        ("swapped 16-byte qs halves", swapped_halves),
        ("a plain f32 B operand", plain_f32_b),
    ] {
        assert!(
            signal > MIN_DEFECT_SIGNAL,
            "the oracle no longer discriminates {label}: it scores {signal:.3e}, \
             not clear of {MIN_DEFECT_SIGNAL:.0e} — the correctness assert above \
             has stopped being able to fail"
        );
    }
    assert!(
        diff_frac > 0.25,
        "truncating rounding changed only {:.1}% of bytes — the rounding \
         semantics have stopped being observable",
        diff_frac * 100.0
    );
    assert!(
        trunc_qual > 1.3 * clean_qual,
        "truncating rounding must worsen the quality column measurably: \
         {trunc_qual:.3e} vs clean {clean_qual:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Claim 3: bandwidth and dispatch cost (opt-in, ~2.6 GB of device memory).
// ---------------------------------------------------------------------------

/// Repeats per timed run: enough passes that submit latency amortizes, capped
/// so a 675 MB matrix does not run for seconds.
fn passes_for(bytes: usize) -> usize {
    const TARGET_BYTES: usize = 2 << 30;
    (TARGET_BYTES / bytes.max(1)).clamp(8, 512)
}

/// Achieved GB/s for one (kernel, matrix): `passes` barrier-separated
/// dispatches in ONE submit — the serial shape the decode path issues.
/// Returns (GB/s over `bytes_read`, µs per dispatch).
#[allow(clippy::too_many_arguments)]
fn gemv_bandwidth<'a>(
    ctx: &'a VulkanContext,
    cache: &mut KernelCache<'a>,
    kernel: Kernel,
    spec: &[(u32, u32)],
    push: &[u8],
    buffers: &[&DeviceBuffer<'_>],
    dispatch: Dispatch,
    bytes_read: usize,
) -> (f64, f64) {
    let (pipeline, layout) = cache
        .get(ctx, kernel, spec, push.len() as u32, buffers.len())
        .expect("pipeline");
    let set = DescriptorSet::storage_buffers(ctx, layout, buffers).expect("bind");
    let passes = passes_for(bytes_read);
    let run = || {
        let mut rec = CommandRecorder::new(ctx).expect("recorder");
        rec.begin().expect("begin");
        for _ in 0..passes {
            record_dispatch(
                &mut rec,
                pipeline,
                &set,
                push,
                [dispatch.x, dispatch.y, dispatch.z],
            );
            rec.barrier();
        }
        let t0 = Instant::now();
        rec.submit_and_wait().expect("submit");
        t0.elapsed().as_secs_f64()
    };
    run(); // warm: page-in + pipeline first-use
    let secs = run();
    (
        bytes_read as f64 * passes as f64 / secs / 1e9,
        secs / passes as f64 * 1e6,
    )
}

/// The tier sizes the projection prices, in GB read per token (from the
/// residency measurements this round builds on).
const TIER_GB: [(&str, f64); 4] = [
    ("linattn in/out", 4.16),
    ("full-attn qkvo", 1.20),
    ("shared expert", 0.47),
    ("lm_head", 1.27),
];

#[test]
fn report_q8_0_dense_bandwidth_and_dispatch_cost() {
    if !bench_enabled() {
        eprintln!("set ARLE_Q8_DENSE_BENCH=1 to run the q8_0 dense bench; skipping");
        return;
    }
    let Some(dir) = checkpoint_dir() else { return };
    let Some(st) = open_shards(&dir, true) else {
        return;
    };
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping q8_0 dense bench");
            return;
        }
    };
    eprintln!(
        "ARLE q8_0 dense bandwidth on: {} — one sitting; the bf16 column is the \
         same-sitting baseline (>=200 GB/s on lm_head means a Performance-mode \
         sitting; trust the ratios regardless)",
        ctx.device_name()
    );
    let mut cache = KernelCache::new();
    let dummy = upload(&ctx, &[0u8; 4]);
    let dense_spec = GemvDenseSpec::DEFAULT;

    // -- per-tensor bandwidth, q8_0 vs bf16 back-to-back ---------------------
    eprintln!(
        "{:<12} {:>7}x{:<5} {:>8} {:>10} {:>9} {:>10} {:>9} {:>7}",
        "tensor", "rows", "cols", "q8 MiB", "q8 GB/s", "q8 us", "bf16 GB/s", "bf16 us", "ratio"
    );
    let mut q8_sustained = 0.0f64;
    let mut bf16_sustained = 0.0f64;
    for dense in dense_tensors(&st, true) {
        let (nrows, ncols) = (dense.nrows, dense.ncols);
        let x = vec![0.5f32; ncols];
        let (slot, _xq) = quantize_activation_slot(&ctx, &x);
        let buf_x = upload(&ctx, &to_bytes(&x));
        let buf_d = DeviceBuffer::alloc_uma(&ctx, nrows * 4).expect("alloc dst");

        let buf_wq = upload(&ctx, &dense.q8_0);
        let push_q = gemv_params(ncols as u32, nrows as u32).to_le_bytes();
        let (q8_gbps, q8_us) = gemv_bandwidth(
            &ctx,
            &mut cache,
            Kernel::GemvQ8_0,
            Kernel::GemvQ8_0.specialization_u32(),
            &push_q,
            &[&buf_wq, &slot, &buf_d, &dummy, &dummy],
            gemv_dispatch(nrows as u32),
            dense.q8_0.len(),
        );
        let buf_wb = upload(&ctx, dense.bf16);
        let push_b = gemv_params_f32_b(ncols as u32, nrows as u32).to_le_bytes();
        let (bf_gbps, bf_us) = gemv_bandwidth(
            &ctx,
            &mut cache,
            Kernel::GemvBf16,
            dense_spec.specialization_u32(),
            &push_b,
            &[&buf_wb, &buf_x, &buf_d, &dummy, &dummy],
            gemv_dense_dispatch(nrows as u32, &dense_spec),
            dense.bf16.len(),
        );
        eprintln!(
            "{:<12} {:>7}x{:<5} {:>8.1} {:>10.1} {:>9.1} {:>10.1} {:>9.1} {:>6.2}x",
            dense.label,
            nrows,
            ncols,
            dense.q8_0.len() as f64 / f64::from(1u32 << 20),
            q8_gbps,
            q8_us,
            bf_gbps,
            bf_us,
            bf_us / q8_us
        );
        if dense.label == "lm_head" {
            q8_sustained = q8_gbps;
            bf16_sustained = bf_gbps;
        }
    }
    // Two regimes in that table, as in the f16 bench: this bench re-reads ONE
    // buffer, so a matrix under the 8060S's 32 MiB MALL reports the cache
    // (~500 GB/s and a flattering ratio — in_proj_qkv's q8 side fits, its
    // bf16 side does not), while lm_head is far past it and reports the
    // sustained link. A real decode walks 48 layers between re-reads, so
    // lm_head's row is the forecast and its ratio is the honest one.
    eprintln!("  (rows under ~32 MiB are MALL-resident ceilings; lm_head is the sustained link)");

    // -- QuantizeQ8_1 dispatch cost at both model widths ---------------------
    // GPU timestamps, not wall clock. Wall-clock protocols measured this three
    // ways and produced three answers (a 256-iter average that read 2.3 µs and
    // 6.6 µs in consecutive sittings, negative slopes, and a quantize+GEMV
    // pair stream reproducibly ~1 ms per submit FASTER than its own GEMV-only
    // subset at two different sizes): there is a ms-scale per-submit host
    // effect on this box that no A/B of whole submits controls for. The
    // ARLE_GPU_TIMESTAMPS machinery is immune to it: a BOTTOM_OF_PIPE
    // timestamp after every dispatch, so each dispatch's completion-to-
    // completion delta inside ONE submit — for a barrier-separated stream,
    // exactly execution + fence drain, the cost the integration adds per
    // extra dispatch.
    //
    // SAFETY: device tests run single-threaded (--test-threads=1 is required
    // for this suite), so no other thread touches the environment.
    unsafe { std::env::set_var("ARLE_GPU_TIMESTAMPS", "1") };
    const STREAM: usize = 512;
    for ne in [2560usize, 6144] {
        let x = vec![0.5f32; ne];
        let buf_in = upload(&ctx, &to_bytes(&x));
        let out_len = ne.div_ceil(128) * 4 * BLOCK_Q8_1_BYTES;
        let buf_out = DeviceBuffer::alloc_uma(&ctx, out_len).expect("alloc q8_1 out");
        let push = q8_1_quantize_params(ne as u32).to_le_bytes();
        let dispatch = q8_1_quantize_dispatch(ne as u32);
        let (pipeline, layout) = cache
            .get(
                &ctx,
                Kernel::QuantizeQ8_1,
                Kernel::QuantizeQ8_1.specialization_u32(),
                push.len() as u32,
                2,
            )
            .expect("quantize pipeline");
        let set = DescriptorSet::storage_buffers(&ctx, layout, &[&buf_in, &buf_out]).expect("bind");
        let mut rec = CommandRecorder::new(&ctx).expect("recorder");
        let mut per_us = f64::NAN;
        for _round in 0..2 {
            // round 0 warms; round 1's profile is kept.
            rec.begin().expect("begin");
            for _ in 0..STREAM {
                rec.label_next("quantize");
                record_dispatch(
                    &mut rec,
                    pipeline,
                    &set,
                    &push,
                    [dispatch.x, dispatch.y, dispatch.z],
                );
                rec.barrier();
            }
            rec.submit_and_wait().expect("submit");
            for (label, count, total_ms) in rec.take_gpu_profile() {
                if label == "quantize" && count > 0 {
                    per_us = total_ms * 1e3 / count as f64;
                }
            }
        }
        eprintln!(
            "QuantizeQ8_1 ne={ne}: {per_us:.2} us GPU per (dispatch+barrier), \
             homogeneous {STREAM}-dispatch stream"
        );
    }

    // -- the marginal cost of interleaving a quantize into a GEMV stream -----
    // What the integration adds per token is not an isolated dispatch but one
    // more (dispatch + barrier) inside an existing fenced stream, so measure
    // it there: quantize dispatches labeled inside a real GEMV stream, per-
    // dispatch GPU time from the same timestamp machinery as above. Two
    // caches, because `KernelCache::get` borrows the cache for the entry's
    // lifetime and this recording needs both pipelines live at once.
    {
        let dense = Dense::load(
            &st,
            "out_proj",
            &format!("{LINATTN_PREFIX}.out_proj.weight"),
        );
        let (nrows, ncols) = (dense.nrows, dense.ncols);
        let x = vec![0.5f32; ncols];
        let (slot, _xq) = quantize_activation_slot(&ctx, &x);
        let buf_in = upload(&ctx, &to_bytes(&x));
        let buf_w = upload(&ctx, &dense.q8_0);
        let buf_d = DeviceBuffer::alloc_uma(&ctx, nrows * 4).expect("alloc dst");
        let push_g = gemv_params(ncols as u32, nrows as u32).to_le_bytes();
        let push_q = q8_1_quantize_params(ncols as u32).to_le_bytes();

        let mut cache_q = KernelCache::new();
        let (pipe_g, layout_g) = cache
            .get(
                &ctx,
                Kernel::GemvQ8_0,
                Kernel::GemvQ8_0.specialization_u32(),
                push_g.len() as u32,
                5,
            )
            .expect("gemv pipeline");
        let (pipe_q, layout_q) = cache_q
            .get(
                &ctx,
                Kernel::QuantizeQ8_1,
                Kernel::QuantizeQ8_1.specialization_u32(),
                push_q.len() as u32,
                2,
            )
            .expect("quantize pipeline");
        let set_g = DescriptorSet::storage_buffers(
            &ctx,
            layout_g,
            &[&buf_w, &slot, &buf_d, &dummy, &dummy],
        )
        .expect("bind gemv");
        let set_q =
            DescriptorSet::storage_buffers(&ctx, layout_q, &[&buf_in, &slot]).expect("bind q8_1");

        // 512 pairs = 1024 dispatches + the batch-start slot, inside the 8192
        // timestamp capacity. The quantize label's per-dispatch GPU time IS
        // the marginal cost of interleaving it (completion-to-completion in a
        // barrier-separated stream); the gemv label in the same stream against
        // the gemv-only stream shows any interference on the GEMV itself.
        const PAIRS: usize = 512;
        let d_g = gemv_dispatch(nrows as u32);
        let d_q = q8_1_quantize_dispatch(ncols as u32);
        let mut rec = CommandRecorder::new(&ctx).expect("recorder");
        let per_us = |profile: &[(&'static str, u64, f64)], want: &str| {
            profile
                .iter()
                .find(|(l, c, _)| *l == want && *c > 0)
                .map_or(f64::NAN, |(_, c, ms)| ms * 1e3 / *c as f64)
        };
        let (mut gemv_alone_us, mut gemv_pair_us, mut quant_pair_us) =
            (f64::NAN, f64::NAN, f64::NAN);
        for _round in 0..2 {
            // round 0 warms both streams; round 1's profiles are kept.
            rec.begin().expect("begin");
            for _ in 0..PAIRS {
                rec.label_next("gemv");
                record_dispatch(&mut rec, pipe_g, &set_g, &push_g, [d_g.x, d_g.y, d_g.z]);
                rec.barrier();
            }
            rec.submit_and_wait().expect("submit");
            gemv_alone_us = per_us(&rec.take_gpu_profile(), "gemv");

            rec.begin().expect("begin");
            for _ in 0..PAIRS {
                rec.label_next("quantize");
                record_dispatch(&mut rec, pipe_q, &set_q, &push_q, [d_q.x, d_q.y, d_q.z]);
                rec.barrier();
                rec.label_next("gemv");
                record_dispatch(&mut rec, pipe_g, &set_g, &push_g, [d_g.x, d_g.y, d_g.z]);
                rec.barrier();
            }
            rec.submit_and_wait().expect("submit");
            let profile = rec.take_gpu_profile();
            quant_pair_us = per_us(&profile, "quantize");
            gemv_pair_us = per_us(&profile, "gemv");
        }
        eprintln!(
            "interleaved QuantizeQ8_1 ne={ncols} in a GEMV stream: {quant_pair_us:.2} us \
             GPU marginal per (dispatch+barrier)  [gemv itself: {gemv_alone_us:.2} us \
             alone vs {gemv_pair_us:.2} us interleaved, {PAIRS} pairs]"
        );
    }

    // -- the projection the residency replan prices --------------------------
    let total_bf16: f64 = TIER_GB.iter().map(|(_, gb)| gb).sum();
    eprintln!(
        "projection at measured sustained rates (q8_0 {q8_sustained:.1} GB/s, bf16 \
         {bf16_sustained:.1} GB/s), Q8_0 = 17/32 of the bytes:"
    );
    let mut total_saved_ms = 0.0;
    for (label, gb) in TIER_GB {
        let bf16_ms = gb / bf16_sustained * 1e3;
        let q8_ms = gb * 17.0 / 32.0 / q8_sustained * 1e3;
        total_saved_ms += bf16_ms - q8_ms;
        eprintln!(
            "  {label:<15} {gb:>5.2} GB: {bf16_ms:>6.2} ms -> {q8_ms:>6.2} ms  \
             (saves {:>5.2} ms/tok)",
            bf16_ms - q8_ms
        );
    }
    eprintln!(
        "  {:<15} {total_bf16:>5.2} GB: total saving {total_saved_ms:.2} ms/token \
         against the 84.9 ms/token decode",
        "dense tier"
    );
}
