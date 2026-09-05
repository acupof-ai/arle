//! The **W4A16** Q4_K arm of the dense-tier flip, end to end on REAL
//! checkpoint bytes — plus the free **W8A16** Q8_0 rung — through the
//! float-B `mul_mat_vec` family:
//!
//! `quantize_q4_k_from_bf16` / `quantize_q8_0_from_bf16` (load-time CPU
//! quantizers) -> `Kernel::GemvQ4KDense` / `Kernel::GemvQ8_0Dense`
//! (quantized weights x a **plain f32 activation vector**) -> what each flip
//! costs in accuracy and buys in bytes, PER TENSOR FAMILY and SIDE BY SIDE.
//! The deliverable is the quality table this file prints: the integration
//! chooses Q4_K vs Q8_0 vs BF16 per family from it.
//!
//! **W4A16, not W4A8.** These kernels are NOT `Kernel::GemvQ4K` /
//! `Kernel::GemvQ8_0` (`mul_mat_vecq`), which requantize the ACTIVATIONS to
//! 8 bits through `Kernel::QuantizeQ8_1` — a different quality contract from
//! the plain-f32 activations this model's forward computes. No QuantizeQ8_1
//! dispatch appears anywhere in this file's clean paths; only the weights
//! are quantized. The mutation test measures the reversed trap — a genuine
//! `block_q8_1_x4` buffer bound as B — because float-B kernels give it no
//! defence.
//!
//! ## Width qualification (the 256 constraint, checked against the model)
//!
//! `block_q4_K` is a 256-value super-block; `ncols` must be a multiple of
//! 256 (and the specialized shader walks whole super-blocks, so the GEMV has
//! the same gate as the quantizer). Qwen3.8-Flash-Next's dense tier:
//!
//! - 2560 wide (10 super-blocks): linattn `in_proj_qkv` [10240x2560],
//!   full-attn `q_proj` [12288x2560] / `k_proj` / `v_proj` [512x2560],
//!   shared-expert `gate_proj`/`up_proj` [640x2560], PLE `key_proj`
//!   [10240x2560] / `value_proj` [2560x2560], `lm_head` [248320x2560] — ALL
//!   qualify;
//! - 6144 wide (24 super-blocks): linattn `out_proj` and full-attn `o_proj`
//!   [2560x6144] — qualify;
//! - 640 wide (2.5 super-blocks): shared-expert `down_proj` [2560x640] —
//!   FAILS. That family stays Q8_0 (640 = 20 q8_0 blocks, and the W8A16
//!   `GemvQ8_0Dense` handles it: its only gate is ncols % 8) or BF16; its
//!   row in the table carries Q8_0 columns only.
//!
//! ## The claims, each with its own oracle
//!
//! 1. **GEMV isolation**: the device output vs f64 dot products over
//!    `infer_gguf`'s independent `block_q4_K` / `block_q8_0` dequantizers x
//!    the SAME f32 activation the kernel reads (B is plain f32, so unlike
//!    the W4A8 harness there is no activation decode step — the uploaded
//!    vector IS what the kernel multiplies). Isolates the GEMV arithmetic
//!    from the quantization. Bound: 1e-4.
//! 2. **Quality vs BF16**: the same device output vs the f64 (bf16 weights x
//!    f32 activation) reference — the pipeline being replaced. Because the
//!    activations are exact, this column carries ONLY the weight
//!    quantization — the number the per-family flip decision hangs on,
//!    reported side by side for Q4_K and Q8_0 from the same activations and
//!    the same sampled rows.
//! 3. **Bytes**: achieved GB/s per width vs the same-sitting Q8_0 and BF16
//!    GEMVs, and the projected per-token saving for the 7.1 GB dense tier.
//!    **GPU timestamps only.** The Q8_0 harness (`device_gemv_q8_dense.rs`)
//!    documents the measurement three ways wall-clock A/B failed on this
//!    box: a per-submit host effect of ~1 ms that no A/B of whole submits
//!    controls for (consecutive sittings read 2.3 vs 6.6 us for the same
//!    dispatch, and a superset stream timed FASTER than its own subset).
//!    Completion-to-completion timestamp deltas inside one barrier-separated
//!    submit are immune to it, so every bandwidth number here comes from
//!    `ARLE_GPU_TIMESTAMPS` profiles, not `Instant`.
//!
//! `lm_head` (1.27 GB of BF16 to quantize twice + multi-GB uploads) and the
//! bandwidth bench are opt-in:
//!
//! ```text
//! ARLE_Q4K_DENSE_BENCH=1 cargo test -p vulkan-kernels --features vulkan \
//!     --test device_gemv_q4k_dense --release -- --nocapture --test-threads=1
//! ```
//!
//! Mutation runs (`q4_k_dense_gemv_mutations_fail_loudly` applies each live
//! on every run and asserts the clean/mutated gap; numbers in the campaign
//! report):
//! - **sub-scale packing off by one bit-position**: the high 2 bits of
//!   sub-scales 4-7 written with `<< 5` instead of `<< 6` — the packing bug
//!   a wrong writer ships. Finite, plausible-looking output.
//! - **super-scale swap**: `d` and `dmin` exchanged in every block —
//!   scale/min roles mixed at the super-block level.
//! - **a `block_q8_1_x4` buffer as the B operand** — the REVERSED trap:
//!   these are float-B kernels, and the q8_1 slot another code path would
//!   hand a `mul_mat_vecq` kernel is silent garbage here, in both new
//!   kernels. (The forward direction — plain f32 into `mul_mat_vecq` — is
//!   measured in the Q8_0 harness.)
//!
//! Skips cleanly with no device or no checkpoint.
#![cfg(feature = "vulkan")]

use std::path::PathBuf;
use std::time::Instant;

use infer_gguf::dequant::{dequantize_row_q4_k, dequantize_row_q8_0};
use vulkan_kernels::{
    BLOCK_Q4_K_BYTES, Dispatch, GemvDenseSpec, Kernel, KernelCache, KernelParams,
    gemv_dense_dispatch, gemv_dispatch, gemv_params_f32_b, q4_k_dense_gemv_with_params,
    q4_k_row_bytes, q8_0_dense_gemv_with_params, q8_0_row_bytes, quantize_q4_k_from_bf16,
    quantize_q8_0_from_bf16, record_dispatch,
};
use vulkan_sys::{CommandRecorder, DescriptorSet, DeviceBuffer, VulkanContext};

/// Claim-1 bound, shared by both W-A16 GEMVs. The kernels run f32 fma chains
/// over dequantized weights x the exact f32 activation; the oracles repeat
/// the same real numbers in f64, so the residue is rounding and summation
/// order. Worst measured on the 8060S across every family below is 1.4e-6
/// max-rel (Q4_K) / 4.7e-7 (Q8_0); 1e-4 leaves ~70x headroom while staying
/// decades under the mutation signals (4.6e0 for the scale mis-pack, 9.7e1
/// for the d/dmin swap, and a q8_1_x4 B NaNs every row — scored as inf).
const TOL_GEMV: f32 = 1e-4;

/// Claim-2 sanity ceilings. The measured quality cost is the REPORTED number
/// (weight-only — the activations are exact): Q4_K reads 7.2-7.9e-2 vector
/// rel across the families, which is the format's honest 4-bit price on
/// ~uniform-ish weights (15 levels over a sub-block range puts the
/// per-weight relative RMS at ~7%), and Q8_0 reads 5.2-6.2e-3. These asserts
/// only refuse a regression past the measured band, not judge the flip —
/// judging it against perplexity is the integration's call, from this table.
const TOL_QUALITY_Q4K: f32 = 1e-1;
const TOL_QUALITY_Q8: f32 = 2e-2;

/// Measured on this box, 2026-08. Override with `INFER_SAFETENSORS_TEST_DIR`.
const CHECKPOINT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";

/// Shard 11 holds layer 12 (`linear_attention` + its shared expert) AND
/// layer 15 (`self_attn` — full attention sits at zero-indexed 3, 7, ...,
/// 47); shards 1 and 10 hold layer 1's PLE `key_proj` / `value_proj`; shard
/// 12 holds `lm_head`. Four files cover every family.
const PLE_KEY_SHARD: &str = "model-bf16-00001.safetensors";
const PLE_VALUE_SHARD: &str = "model-bf16-00010.safetensors";
const DENSE_SHARD: &str = "model-bf16-00011.safetensors";
const LM_HEAD_SHARD: &str = "model-bf16-00012.safetensors";
const LINATTN_PREFIX: &str = "model.language_model.layers.12.linear_attn";
const FULLATTN_PREFIX: &str = "model.language_model.layers.15.self_attn";
const SHARED_PREFIX: &str = "model.language_model.layers.12.mlp.shared_expert";
const PLE_PREFIX: &str = "model.language_model.layers.1.ple";

/// Cap on how many output rows the f64 oracles cover. The GEMV still runs
/// over all rows; [`sampled_rows`] strides so the checked set spans the
/// whole matrix including the last row.
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
    std::env::var("ARLE_Q4K_DENSE_BENCH").is_ok()
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

/// Quantize the whole matrix through the row-sliced API on every core: rows
/// are self-contained in both formats, so this is the same computation as
/// one call — and the parallel slicing the real model load will use.
fn quantize_rows_parallel<F>(bf16: &[u8], ncols: usize, row_bytes: usize, quantize: F) -> Vec<u8>
where
    F: Fn(&[u8], usize, usize, &mut [u8]) -> vulkan_kernels::Result<()> + Sync,
{
    let nrows = bf16.len() / (ncols * 2);
    let mut out = vec![0u8; nrows * row_bytes];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk_rows = nrows.div_ceil(threads).max(1);
    let quantize = &quantize;
    std::thread::scope(|s| {
        for (src_chunk, dst_chunk) in bf16
            .chunks(chunk_rows * ncols * 2)
            .zip(out.chunks_mut(chunk_rows * row_bytes))
        {
            s.spawn(move || {
                let rows = src_chunk.len() / (ncols * 2);
                quantize(src_chunk, rows, ncols, dst_chunk).expect("quantize row slice");
            });
        }
    });
    out
}

/// One dense `[nrows, ncols]` BF16 weight matrix off the mmap, plus BOTH of
/// its quantized re-encodes. `q4_k` is `None` exactly when `ncols` fails the
/// 256 constraint — the table prints that family as staying Q8_0/BF16.
struct Dense<'a> {
    label: &'static str,
    nrows: usize,
    ncols: usize,
    bf16: &'a [u8],
    q8_0: Vec<u8>,
    q4_k: Option<Vec<u8>>,
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

        let q8_rb = q8_0_row_bytes(ncols)
            .unwrap_or_else(|| panic!("{name}: ncols {ncols} is not a whole number of q8 blocks"));
        let t0 = Instant::now();
        let q8_0 = quantize_rows_parallel(bf16, ncols, q8_rb, quantize_q8_0_from_bf16);
        let q8_secs = t0.elapsed().as_secs_f64();

        let q4_k = q4_k_row_bytes(ncols).map(|rb| {
            let t0 = Instant::now();
            let q = quantize_rows_parallel(bf16, ncols, rb, quantize_q4_k_from_bf16);
            let secs = t0.elapsed().as_secs_f64();
            eprintln!(
                "[{label}] CPU quantize {nrows}x{ncols}: q4_k {:.0} ms ({:.2} GB/s of bf16 in), \
                 q8_0 {:.0} ms",
                secs * 1e3,
                bf16.len() as f64 / secs / 1e9,
                q8_secs * 1e3,
            );
            q
        });
        if q4_k.is_none() {
            eprintln!(
                "[{label}] ncols {ncols} = {} super-blocks -> Q4_K refused; family stays \
                 Q8_0/BF16 (q8_0 quantize {:.0} ms)",
                ncols as f64 / 256.0,
                q8_secs * 1e3,
            );
        }
        Self {
            label,
            nrows,
            ncols,
            bf16,
            q8_0,
            q4_k,
        }
    }

    fn bf16_row(&self, r: usize) -> Vec<f32> {
        self.bf16[r * self.ncols * 2..(r + 1) * self.ncols * 2]
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    }

    /// One row of Q8_0 bytes through `infer_gguf`'s independent reader.
    fn q8_row(&self, r: usize) -> Vec<f32> {
        let rb = q8_0_row_bytes(self.ncols).expect("validated at load");
        dequantize_row_q8_0(&self.q8_0[r * rb..(r + 1) * rb], self.ncols).expect("dequant q8_0 row")
    }

    /// One row of Q4_K bytes through `infer_gguf`'s independent reader.
    fn q4_row(&self, bytes: &[u8], r: usize) -> Vec<f32> {
        let rb = q4_k_row_bytes(self.ncols).expect("q4_k rows exist only when validated");
        dequantize_row_q4_k(&bytes[r * rb..(r + 1) * rb], self.ncols).expect("dequant q4_k row")
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
        // A non-finite output is infinitely wrong — and comparison-blind:
        // NaN fails every `>`, so without this arm a poisoned run scores
        // max_rel 0.0 and the "loud" asserts silently lose their teeth.
        // Measured, not hypothetical: the q8_1_x4-as-B trap NaNs every row
        // (f16 scale bytes reinterpreted as f32 exponents) and scored
        // 0.000e0 until this arm existed.
        let rel = if g.is_finite() {
            (g - w).abs() / w.abs().max(floor)
        } else {
            f32::INFINITY
        };
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

/// The launcher type both W-A16 GEMVs share: 5 bindings
/// `[A, B(plain f32), D, Fuse0, Fuse1]`, [`gemv_params_f32_b`] push block
/// (stride in ELEMENTS), one workgroup per output row.
type DenseQuantGemv =
    fn(&VulkanContext, &[&DeviceBuffer<'_>], Dispatch, &KernelParams) -> vulkan_kernels::Result<()>;

/// One W-A16 GEMV over a quantized weight matrix and the plain f32
/// activation. No activation quantization exists on this path — that is the
/// contract under test.
fn run_dense_quant_gemv(
    ctx: &VulkanContext,
    launcher: DenseQuantGemv,
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
    launcher(
        ctx,
        &[&buf_w, buf_b, &buf_d, &dummy, &dummy],
        gemv_dispatch(nrows as u32),
        &gemv_params_f32_b(ncols as u32, nrows as u32),
    )
    .expect("W-A16 GEMV dispatch");
    read_f32(&buf_d, nrows)
}

fn open_shards(
    dir: &std::path::Path,
    shards: &[&str],
) -> Option<infer_gguf::safetensors::SafeTensorsDir> {
    let paths: Vec<PathBuf> = shards.iter().map(|s| dir.join(s)).collect();
    for p in &paths {
        if !p.is_file() {
            eprintln!("skip: {} not present", p.display());
            return None;
        }
    }
    Some(infer_gguf::safetensors::SafeTensorsDir::open_files(&paths).expect("open shards"))
}

/// Every family the flip decision needs, in tier order. The three activation
/// widths (2560, 6144, 640) each get ONE f32 activation vector below, shared
/// across every tensor of that width. `lm_head` is 1.27 GB of BF16 to
/// quantize twice, so it joins only in the opt-in bench run.
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
        Dense::load(st, "q_proj", &format!("{FULLATTN_PREFIX}.q_proj.weight")),
        Dense::load(st, "o_proj", &format!("{FULLATTN_PREFIX}.o_proj.weight")),
        Dense::load(st, "ple_key", &format!("{PLE_PREFIX}.key_proj.weight")),
        Dense::load(st, "ple_value", &format!("{PLE_PREFIX}.value_proj.weight")),
        Dense::load(st, "se_gate", &format!("{SHARED_PREFIX}.gate_proj.weight")),
        Dense::load(st, "se_up", &format!("{SHARED_PREFIX}.up_proj.weight")),
        Dense::load(st, "se_down", &format!("{SHARED_PREFIX}.down_proj.weight")),
    ];
    if with_lm_head {
        out.push(Dense::load(st, "lm_head", "lm_head.weight"));
    }
    out
}

/// Claims 1 and 2: both W-A16 GEMVs against both oracles, per family — THE
/// TABLE. Q4_K and Q8_0 columns share the same activation vector and the
/// same sampled output rows, so they differ only by the weight format.
#[test]
fn q4_k_dense_gemv_matches_oracles_on_real_bytes() {
    let Some(dir) = checkpoint_dir() else { return };
    let with_lm_head = bench_enabled();
    let mut shards = vec![PLE_KEY_SHARD, PLE_VALUE_SHARD, DENSE_SHARD];
    if with_lm_head {
        shards.push(LM_HEAD_SHARD);
    }
    let Some(st) = open_shards(&dir, &shards) else {
        return;
    };
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping q4_k dense GEMV test");
            return;
        }
    };
    eprintln!("ARLE W4A16 q4_k dense GEMV proof on: {}", ctx.device_name());

    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let x2560: Vec<f32> = (0..2560).map(|_| rng.next_unit_f32()).collect();
    let x6144: Vec<f32> = (0..6144).map(|_| rng.next_unit_f32()).collect();
    let x640: Vec<f32> = (0..640).map(|_| rng.next_unit_f32()).collect();
    // One plain-f32 upload per width, shared across formats and tensors —
    // exactly what the forward already has in registers. Nothing quantizes it.
    let (buf2560, buf6144, buf640) = (
        upload(&ctx, &to_bytes(&x2560)),
        upload(&ctx, &to_bytes(&x6144)),
        upload(&ctx, &to_bytes(&x640)),
    );

    eprintln!(
        "{:<12} {:>7}x{:<5} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10}",
        "tensor",
        "rows",
        "cols",
        "q4k qual",
        "q4k worst",
        "q4k gemv",
        "q8 qual",
        "q8 worst",
        "q8 gemv"
    );
    for dense in dense_tensors(&st, with_lm_head) {
        let (nrows, ncols) = (dense.nrows, dense.ncols);
        let (x, buf_x) = match ncols {
            2560 => (&x2560, &buf2560),
            6144 => (&x6144, &buf6144),
            640 => (&x640, &buf640),
            other => panic!("unexpected width {other}"),
        };
        let rows = sampled_rows(nrows);
        // Claim 2 oracle: the pipeline being replaced — bf16 weights x f32 x.
        let want_qual: Vec<f32> = rows
            .iter()
            .map(|&r| reference_row(&dense.bf16_row(r), x))
            .collect();

        // Q8_0 columns: the W8A16 rung, every family qualifies.
        let got_all = run_dense_quant_gemv(
            &ctx,
            q8_0_dense_gemv_with_params,
            &dense.q8_0,
            buf_x,
            nrows,
            ncols,
        );
        let got8: Vec<f32> = rows.iter().map(|&r| got_all[r]).collect();
        let want8: Vec<f32> = rows
            .iter()
            .map(|&r| reference_row(&dense.q8_row(r), x))
            .collect();
        let e8_gemv = error_profile(&got8, &want8, &rows);
        let e8_qual = error_profile(&got8, &want_qual, &rows);

        // Q4_K columns, when the width qualifies.
        let q4 = dense.q4_k.as_ref().map(|q4_bytes| {
            let got_all = run_dense_quant_gemv(
                &ctx,
                q4_k_dense_gemv_with_params,
                q4_bytes,
                buf_x,
                nrows,
                ncols,
            );
            let got4: Vec<f32> = rows.iter().map(|&r| got_all[r]).collect();
            let want4: Vec<f32> = rows
                .iter()
                .map(|&r| reference_row(&dense.q4_row(q4_bytes, r), x))
                .collect();
            (
                error_profile(&got4, &want4, &rows),
                error_profile(&got4, &want_qual, &rows),
            )
        });

        match &q4 {
            Some((e4_gemv, e4_qual)) => eprintln!(
                "{:<12} {:>7}x{:<5} | {:>10.3e} {:>10.3e} {:>10.3e} | {:>10.3e} {:>10.3e} {:>10.3e}  \
                 (worst rows q4 {} q8 {}, {} sampled)",
                dense.label,
                nrows,
                ncols,
                e4_qual.vector_rel,
                e4_qual.max_rel,
                e4_gemv.max_rel,
                e8_qual.vector_rel,
                e8_qual.max_rel,
                e8_gemv.max_rel,
                e4_qual.max_rel_row,
                e8_qual.max_rel_row,
                rows.len()
            ),
            None => eprintln!(
                "{:<12} {:>7}x{:<5} | {:>10} {:>10} {:>10} | {:>10.3e} {:>10.3e} {:>10.3e}  \
                 (stays Q8_0/BF16: {} super-blocks; worst row q8 {})",
                dense.label,
                nrows,
                ncols,
                "--",
                "--",
                "--",
                e8_qual.vector_rel,
                e8_qual.max_rel,
                e8_gemv.max_rel,
                ncols as f64 / 256.0,
                e8_qual.max_rel_row
            ),
        }

        assert!(
            e8_gemv.max_rel < TOL_GEMV,
            "{}: GemvQ8_0Dense disagrees with the f64 oracle over its own bytes: \
             max rel {} (row {}) >= {TOL_GEMV}",
            dense.label,
            e8_gemv.max_rel,
            e8_gemv.max_rel_row
        );
        assert!(
            e8_qual.vector_rel < TOL_QUALITY_Q8,
            "{}: the Q8_0 flip's quality cost exploded: vector rel {} >= {TOL_QUALITY_Q8}",
            dense.label,
            e8_qual.vector_rel
        );
        if let Some((e4_gemv, e4_qual)) = q4 {
            assert!(
                e4_gemv.max_rel < TOL_GEMV,
                "{}: GemvQ4KDense disagrees with the f64 oracle over its own bytes: \
                 max rel {} (row {}) >= {TOL_GEMV}",
                dense.label,
                e4_gemv.max_rel,
                e4_gemv.max_rel_row
            );
            assert!(
                e4_qual.vector_rel < TOL_QUALITY_Q4K,
                "{}: the Q4_K flip's quality cost exploded: vector rel {} >= {TOL_QUALITY_Q4K}",
                dense.label,
                e4_qual.vector_rel
            );
        }
    }
}

/// Local copy of ggml's `get_scale_min_k4` so the mutation below can decode
/// the clean packing before re-encoding it wrongly.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// f32 -> IEEE f16 bits, round-to-nearest-even, tiny values flushed to zero.
/// Only the reversed-trap encoder below uses it; the trap's loudness does not
/// depend on the last mantissa bit, but the encoding should still be a real
/// `block_q8_1` a `mul_mat_vecq` kernel would accept.
fn f16_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7F_FFFF;
    if exp >= 143 {
        return sign | 0x7C00; // overflow -> inf; unreachable for |x| <= 1 inputs
    }
    if exp <= 112 {
        return sign; // subnormal band flushed; irrelevant at these magnitudes
    }
    let mut half = (((exp - 112) as u32) << 10) | (man >> 13);
    let round = (man >> 12) & 1;
    let sticky = u32::from(man & 0xFFF != 0);
    half += round & (sticky | (half & 1));
    sign | half as u16
}

/// CPU encoder for `block_q8_1_x4` (the 144-byte `{f16vec2 ds[4]; int8
/// qs[128]}` blocks `Kernel::QuantizeQ8_1` produces on device): per 32-value
/// sub-block, `d = amax/127`, `q = round(x/d)`, `s = d * sum(q)`. This file
/// deliberately runs NO QuantizeQ8_1 dispatch — the encoder exists only to
/// build the reversed-trap operand, and building it on the host keeps the
/// statement "no activation quantization anywhere in these paths" literally
/// checkable. Padded with zeros up to `pad_to` bytes so the float-B kernel's
/// full `ncols * 4`-byte read window stays in bounds: the trap being measured
/// is misinterpretation, not out-of-bounds behavior — and a real q8_1 slot
/// allocation has exactly this kind of slack.
fn encode_q8_1_x4_padded(x: &[f32], pad_to: usize) -> Vec<u8> {
    let mut out = vec![0u8; (x.len().div_ceil(128) * 144).max(pad_to)];
    for (g, chunk) in x.chunks(128).enumerate() {
        let base = g * 144;
        for (b, sub) in chunk.chunks(32).enumerate() {
            let amax = sub.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let mut sum = 0i32;
            for (i, &v) in sub.iter().enumerate() {
                let q = (v * id).round() as i32;
                sum += q;
                out[base + 16 + 32 * b + i] = q as i8 as u8;
            }
            let s = d * sum as f32;
            out[base + 4 * b..base + 4 * b + 2].copy_from_slice(&f16_from_f32(d).to_le_bytes());
            out[base + 4 * b + 2..base + 4 * b + 4].copy_from_slice(&f16_from_f32(s).to_le_bytes());
        }
    }
    out
}

/// The asserts above must be able to fail, and the failure must be loud. The
/// three mutations are the real ways THIS machinery goes wrong, applied live
/// against the same oracle and metric:
///
/// - **Sub-scale packing off by one bit-position**: sub-scales/mins decoded
///   from the clean blocks, re-packed with the high 2 bits of scales 4-7
///   written `<< 5` instead of `<< 6` (`quantize_row_q4_K_ref`'s :1442 line
///   mis-typed). Every super-block whose upper sub-scales reach 16 decodes
///   differently — finite, plausible-looking output.
/// - **Super-scale swap**: `d` and `dmin` (the two f16s at offset 0)
///   exchanged in every block — mins scaled like scales and vice versa.
/// - **A `block_q8_1_x4` buffer as the B operand** — the REVERSED trap.
///   These are float-B kernels; the q8_1 slot the W4A8 path would share is
///   silent garbage here, and it is measured for BOTH new kernels.
#[test]
fn q4_k_dense_gemv_mutations_fail_loudly() {
    /// A defect signal must clear the tolerance by this factor to count.
    const MIN_DEFECT_SIGNAL: f32 = 100.0 * TOL_GEMV;

    let Some(dir) = checkpoint_dir() else { return };
    let Some(st) = open_shards(&dir, &[DENSE_SHARD]) else {
        return;
    };
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping q4_k mutation controls");
            return;
        }
    };
    let dense = Dense::load(
        &st,
        "in_proj_qkv",
        &format!("{LINATTN_PREFIX}.in_proj_qkv.weight"),
    );
    let (nrows, ncols) = (dense.nrows, dense.ncols);
    let q4_clean = dense.q4_k.as_ref().expect("2560 wide must quantize");
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let x: Vec<f32> = (0..ncols).map(|_| rng.next_unit_f32()).collect();
    let buf_x = upload(&ctx, &to_bytes(&x));
    let rows = sampled_rows(nrows);

    // The clean oracle: independently dequantized weights x the same f32 x.
    let want: Vec<f32> = rows
        .iter()
        .map(|&r| reference_row(&dense.q4_row(q4_clean, r), &x))
        .collect();
    let clean_all = run_dense_quant_gemv(
        &ctx,
        q4_k_dense_gemv_with_params,
        q4_clean,
        &buf_x,
        nrows,
        ncols,
    );
    let clean: Vec<f32> = rows.iter().map(|&r| clean_all[r]).collect();
    let baseline = error_profile(&clean, &want, &rows).max_rel;

    // Mutation A: re-pack every block's 12 scale bytes with the ls high bits
    // one position low. Only blocks whose sub-scales 4-7 reach 16 change —
    // count them so the diff is part of the record.
    let mut mispacked = q4_clean.clone();
    let mut blocks_changed = 0usize;
    for block in mispacked.chunks_exact_mut(BLOCK_Q4_K_BYTES) {
        let mut decoded = [(0u8, 0u8); 8];
        for (j, d) in decoded.iter_mut().enumerate() {
            *d = scale_min_k4(j, &block[4..16]);
        }
        let scales = &mut block[4..16];
        scales.fill(0);
        for (j, &(ls, lm)) in decoded.iter().enumerate() {
            if j < 4 {
                scales[j] = ls;
                scales[j + 4] = lm;
            } else {
                scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                scales[j - 4] |= (ls >> 4) << 5; // BUG: << 6 in the reference
                scales[j] |= (lm >> 4) << 6;
            }
        }
        let mut reread = [(0u8, 0u8); 8];
        for (j, d) in reread.iter_mut().enumerate() {
            *d = scale_min_k4(j, &scales[..]);
        }
        if reread != decoded {
            blocks_changed += 1;
        }
    }
    let got_all = run_dense_quant_gemv(
        &ctx,
        q4_k_dense_gemv_with_params,
        &mispacked,
        &buf_x,
        nrows,
        ncols,
    );
    let got: Vec<f32> = rows.iter().map(|&r| got_all[r]).collect();
    let scale_mispack = error_profile(&got, &want, &rows).max_rel;
    let mispack_frac = blocks_changed as f64 / (q4_clean.len() / BLOCK_Q4_K_BYTES) as f64;

    // Mutation B: swap the two f16 super-scales in every block.
    let mut swapped = q4_clean.clone();
    for block in swapped.chunks_exact_mut(BLOCK_Q4_K_BYTES) {
        let (d, rest) = block.split_at_mut(2);
        d.swap_with_slice(&mut rest[..2]);
    }
    let got_all = run_dense_quant_gemv(
        &ctx,
        q4_k_dense_gemv_with_params,
        &swapped,
        &buf_x,
        nrows,
        ncols,
    );
    let got: Vec<f32> = rows.iter().map(|&r| got_all[r]).collect();
    let super_swap = error_profile(&got, &want, &rows).max_rel;

    // Mutation C, the REVERSED trap: bind a genuine `block_q8_1_x4` encoding
    // of the same activation as B, through the same call path, into BOTH
    // float-B kernels. Nothing in the API stops this — a caller flipping a
    // tensor from the W4A8 arm to this one and forgetting to swap the slot
    // compiles clean and dispatches clean.
    let q8_1_b = upload(&ctx, &encode_q8_1_x4_padded(&x, ncols * 4));
    let got_all = run_dense_quant_gemv(
        &ctx,
        q4_k_dense_gemv_with_params,
        q4_clean,
        &q8_1_b,
        nrows,
        ncols,
    );
    let got: Vec<f32> = rows.iter().map(|&r| got_all[r]).collect();
    let q4k_nonfinite = got.iter().filter(|v| !v.is_finite()).count();
    let q8_1_as_b_q4k = error_profile(&got, &want, &rows).max_rel;

    let want_q8: Vec<f32> = rows
        .iter()
        .map(|&r| reference_row(&dense.q8_row(r), &x))
        .collect();
    let got_all = run_dense_quant_gemv(
        &ctx,
        q8_0_dense_gemv_with_params,
        &dense.q8_0,
        &q8_1_b,
        nrows,
        ncols,
    );
    let got: Vec<f32> = rows.iter().map(|&r| got_all[r]).collect();
    let q8_nonfinite = got.iter().filter(|v| !v.is_finite()).count();
    let q8_1_as_b_q8 = error_profile(&got, &want_q8, &rows).max_rel;

    eprintln!(
        "[mutation control] in_proj_qkv [{nrows}x{ncols}] max rel vs the dequant oracle: \
         clean {baseline:.3e}, sub-scale mis-pack {scale_mispack:.3e} ({:.1}% of blocks \
         change), d/dmin swap {super_swap:.3e}, q8_1_x4-as-B q4k {q8_1_as_b_q4k:.3e} \
         ({q4k_nonfinite}/{} rows non-finite) / q8 {q8_1_as_b_q8:.3e} ({q8_nonfinite} \
         non-finite) (tolerance {TOL_GEMV:.0e})",
        mispack_frac * 100.0,
        rows.len(),
    );
    assert!(
        baseline < TOL_GEMV,
        "clean path regressed: {baseline} >= {TOL_GEMV}"
    );
    assert!(
        mispack_frac > 0.5,
        "the packing mutation changed only {:.1}% of blocks — sub-scales 4-7 no \
         longer reach 16 and the control has stopped observing the bit position",
        mispack_frac * 100.0
    );
    for (label, signal) in [
        ("a sub-scale packing bit-position bug", scale_mispack),
        ("a d/dmin super-scale swap", super_swap),
        ("a q8_1_x4 B operand in GemvQ4KDense", q8_1_as_b_q4k),
        ("a q8_1_x4 B operand in GemvQ8_0Dense", q8_1_as_b_q8),
    ] {
        assert!(
            signal > MIN_DEFECT_SIGNAL,
            "the oracle no longer discriminates {label}: it scores {signal:.3e}, \
             not clear of {MIN_DEFECT_SIGNAL:.0e} — the correctness assert above \
             has stopped being able to fail"
        );
    }
}

// ---------------------------------------------------------------------------
// Claim 3: bandwidth and the projection (opt-in, multi-GB device buffers).
// ---------------------------------------------------------------------------

/// Repeats per timed run: enough passes that the per-dispatch timestamp
/// deltas dominate their own granularity, capped inside the 8192-slot
/// timestamp pool and so a 357 MB matrix does not run for seconds.
fn passes_for(bytes: usize) -> usize {
    const TARGET_BYTES: usize = 2 << 30;
    (TARGET_BYTES / bytes.max(1)).clamp(8, 512)
}

/// Achieved GB/s for one (kernel, matrix), **GPU timestamps only**: `passes`
/// barrier-separated dispatches in ONE submit — the serial shape the decode
/// path issues — and the per-dispatch completion-to-completion delta from
/// the `ARLE_GPU_TIMESTAMPS` profile. The file header carries the measured
/// reasons wall-clock A/B is not trusted on this box. Two rounds; round 0
/// warms page-in and pipeline first-use, round 1's profile is kept.
/// Returns (GB/s over `bytes_read`, us per dispatch).
#[allow(clippy::too_many_arguments)]
fn gemv_bandwidth_gpu_ts<'a>(
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
    let mut rec = CommandRecorder::new(ctx).expect("recorder");
    let mut per_us = f64::NAN;
    for _round in 0..2 {
        rec.begin().expect("begin");
        for _ in 0..passes {
            rec.label_next("gemv");
            record_dispatch(
                &mut rec,
                pipeline,
                &set,
                push,
                [dispatch.x, dispatch.y, dispatch.z],
            );
            rec.barrier();
        }
        rec.submit_and_wait().expect("submit");
        for (label, count, total_ms) in rec.take_gpu_profile() {
            if label == "gemv" && count > 0 {
                per_us = total_ms * 1e3 / count as f64;
            }
        }
    }
    assert!(
        per_us.is_finite(),
        "no GPU timestamp profile came back for {kernel:?} — the bench's numbers \
         would silently be garbage (is ARLE_GPU_TIMESTAMPS reaching the recorder?)"
    );
    (bytes_read as f64 / (per_us / 1e6) / 1e9, per_us)
}

/// The tier sizes the projection prices, in BF16 GB read per token (from the
/// residency measurements the Q8_0 round established), with the fraction of
/// each family's bytes whose width passes the 256 constraint: everything is
/// fully flippable except the shared expert, where `down_proj` (one of three
/// equal-size tensors, ncols 640) stays on the Q8_0 rung.
const TIER: [(&str, f64, f64); 4] = [
    ("linattn in/out", 4.16, 1.0),
    ("full-attn qkvo", 1.20, 1.0),
    ("shared expert", 0.47, 2.0 / 3.0),
    ("lm_head", 1.27, 1.0),
];

#[test]
fn report_q4_k_dense_bandwidth_and_projection() {
    if !bench_enabled() {
        eprintln!("set ARLE_Q4K_DENSE_BENCH=1 to run the q4_k dense bench; skipping");
        return;
    }
    let Some(dir) = checkpoint_dir() else { return };
    let Some(st) = open_shards(&dir, &[DENSE_SHARD, LM_HEAD_SHARD]) else {
        return;
    };
    // GPU timestamps for every number in this bench — see the file header for
    // why wall clock is not trusted on this box.
    // SAFETY: device tests run single-threaded (--test-threads=1 is required
    // for this suite), so no other thread touches the environment.
    unsafe { std::env::set_var("ARLE_GPU_TIMESTAMPS", "1") };
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping q4_k dense bench");
            return;
        }
    };
    eprintln!(
        "ARLE W4A16 q4_k dense bandwidth on: {} — one sitting, all three formats \
         back-to-back per tensor, GPU timestamps only (trust the ratios; record \
         the power mode)",
        ctx.device_name()
    );
    let mut cache = KernelCache::new();
    let dummy = upload(&ctx, &[0u8; 4]);
    let dense_spec = GemvDenseSpec::DEFAULT;

    eprintln!(
        "{:<12} {:>7}x{:<5} {:>8} {:>9} {:>8} {:>8} {:>8} {:>9} {:>8} {:>8} {:>9}",
        "tensor",
        "rows",
        "cols",
        "q4k MiB",
        "q4k GB/s",
        "q4k us",
        "q8 GB/s",
        "q8 us",
        "bf16 GB/s",
        "bf16 us",
        "q8/q4k",
        "bf16/q4k"
    );
    let (mut q4_sustained, mut q8_sustained, mut bf16_sustained) = (0.0f64, 0.0f64, 0.0f64);
    for (label, name) in [
        (
            "in_proj_qkv",
            format!("{LINATTN_PREFIX}.in_proj_qkv.weight"),
        ),
        ("out_proj", format!("{LINATTN_PREFIX}.out_proj.weight")),
        ("lm_head", "lm_head.weight".to_string()),
    ] {
        let dense = Dense::load(&st, label, &name);
        let (nrows, ncols) = (dense.nrows, dense.ncols);
        let q4_bytes = dense.q4_k.as_ref().expect("bench tensors all qualify");
        let x = vec![0.5f32; ncols];
        let buf_x = upload(&ctx, &to_bytes(&x));
        let buf_d = DeviceBuffer::alloc_uma(&ctx, nrows * 4).expect("alloc dst");
        let push = gemv_params_f32_b(ncols as u32, nrows as u32).to_le_bytes();

        let buf_w4 = upload(&ctx, q4_bytes);
        let (q4_gbps, q4_us) = gemv_bandwidth_gpu_ts(
            &ctx,
            &mut cache,
            Kernel::GemvQ4KDense,
            Kernel::GemvQ4KDense.specialization_u32(),
            &push,
            &[&buf_w4, &buf_x, &buf_d, &dummy, &dummy],
            gemv_dispatch(nrows as u32),
            q4_bytes.len(),
        );
        drop(buf_w4);
        let buf_w8 = upload(&ctx, &dense.q8_0);
        let (q8_gbps, q8_us) = gemv_bandwidth_gpu_ts(
            &ctx,
            &mut cache,
            Kernel::GemvQ8_0Dense,
            Kernel::GemvQ8_0Dense.specialization_u32(),
            &push,
            &[&buf_w8, &buf_x, &buf_d, &dummy, &dummy],
            gemv_dispatch(nrows as u32),
            dense.q8_0.len(),
        );
        drop(buf_w8);
        let buf_wb = upload(&ctx, dense.bf16);
        let (bf_gbps, bf_us) = gemv_bandwidth_gpu_ts(
            &ctx,
            &mut cache,
            Kernel::GemvBf16,
            dense_spec.specialization_u32(),
            &push,
            &[&buf_wb, &buf_x, &buf_d, &dummy, &dummy],
            gemv_dense_dispatch(nrows as u32, &dense_spec),
            dense.bf16.len(),
        );
        eprintln!(
            "{:<12} {:>7}x{:<5} {:>8.1} {:>9.1} {:>8.1} {:>8.1} {:>8.1} {:>9.1} {:>8.1} {:>7.2}x {:>8.2}x",
            dense.label,
            nrows,
            ncols,
            q4_bytes.len() as f64 / f64::from(1u32 << 20),
            q4_gbps,
            q4_us,
            q8_gbps,
            q8_us,
            bf_gbps,
            bf_us,
            q8_us / q4_us,
            bf_us / q4_us
        );
        if dense.label == "lm_head" {
            q4_sustained = q4_gbps;
            q8_sustained = q8_gbps;
            bf16_sustained = bf_gbps;
        }
    }
    // Same two regimes as the f16/q8 benches: this bench re-reads ONE buffer,
    // so any matrix under the 8060S's 32 MiB MALL reports the cache (the q4k
    // side of BOTH small tensors fits; even in_proj_qkv's q8 side does), while
    // lm_head is far past it in every format and reports the sustained link.
    // A real decode walks 48 layers between re-reads: lm_head's row is the
    // forecast, its ratios are the honest ones.
    eprintln!("  (rows under ~32 MiB are MALL-resident ceilings; lm_head is the sustained link)");

    // -- the projection the residency replan prices --------------------------
    // Bytes per element: bf16 2, q8_0 17/16, q4_k 9/16. The activation side
    // is IDENTICAL across all three (the same plain f32 vector, zero extra
    // dispatches — W-A16 means no QuantizeQ8_1 exists on any of these paths),
    // so the per-format delta is purely weight bytes over measured rate.
    let total_bf16: f64 = TIER.iter().map(|(_, gb, _)| gb).sum();
    eprintln!(
        "projection at measured sustained rates (q4_k {q4_sustained:.1} GB/s, q8_0 \
         {q8_sustained:.1} GB/s, bf16 {bf16_sustained:.1} GB/s); q4_k covers the \
         width-qualified fraction of each family, the rest stays q8_0:"
    );
    let (mut total_q8_ms, mut total_q4_ms, mut total_bf16_ms) = (0.0, 0.0, 0.0);
    for (label, gb, q4_frac) in TIER {
        let bf16_ms = gb / bf16_sustained * 1e3;
        let q8_ms = gb * 17.0 / 32.0 / q8_sustained * 1e3;
        let q4_ms = gb * q4_frac * 9.0 / 32.0 / q4_sustained * 1e3
            + gb * (1.0 - q4_frac) * 17.0 / 32.0 / q8_sustained * 1e3;
        total_bf16_ms += bf16_ms;
        total_q8_ms += q8_ms;
        total_q4_ms += q4_ms;
        eprintln!(
            "  {label:<15} {gb:>5.2} GB: bf16 {bf16_ms:>6.2} ms -> q8 {q8_ms:>5.2} ms -> \
             q4k {q4_ms:>5.2} ms  (q4k saves {:>5.2} vs q8, {:>5.2} vs bf16 ms/tok)",
            q8_ms - q4_ms,
            bf16_ms - q4_ms
        );
    }
    eprintln!(
        "  {:<15} {total_bf16:>5.2} GB: bf16 {total_bf16_ms:.2} -> q8 {total_q8_ms:.2} -> \
         q4k {total_q4_ms:.2} ms; q4k saves {:.2} ms/token over q8 and {:.2} over bf16, \
         against the 84.9 ms/token decode",
        "dense tier",
        total_q8_ms - total_q4_ms,
        total_bf16_ms - total_q4_ms
    );
}
