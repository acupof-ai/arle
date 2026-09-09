//! On-device proof of the BF16 coopmat GEMM pair: `Kernel::F16KvPack` (f32 →
//! f16 rows) staging the activations for `Kernel::MmCmBf16` (bf16-bit A ×
//! plain-f16 B → f32 D on the matrix cores).
//!
//! The pairing is the contract under test, not just each kernel alone.
//! Upstream `mul_mm.comp` decodes BOTH operands through one `TO_FLOAT_TYPE`,
//! which would force bf16-bit B here — and bf16's 2^-8 activation rounding
//! measurably compounds across the 48-layer qwen4_exp prefill into an argmax
//! flip. The vendored `TO_FLOAT_TYPE_B` seam (ARLE patch in `mul_mm.comp` /
//! `mul_mm_funcs.glsl`) lets this build take plain f16 B (2^-11) like every
//! other coopmat variant; the CPU reference here multiplies bf16-rounded A by
//! f16-rounded B, so a re-vendor that clobbers the seam (B silently decoded
//! as bf16 bits ⇒ values ~2^112 too small) fails loudly.
//!
//! Skips cleanly without a device or without a usable f16 subgroup matrix
//! shape (the runtime falls back to per-token GEMVs there).
#![cfg(feature = "vulkan")]

use vulkan_kernels::{
    Kernel, KernelCache, MmSpec, f16_kv_pack_dispatch_rows, f16_kv_pack_params, launch_cached,
    mm_dispatch, mm_with_params_and_spec, mmq_params, record_dispatch,
};
use vulkan_sys::{CommandRecorder, DescriptorSet, DeviceBuffer, VulkanContext};

/// Per-element tolerance as a fraction of the sum of term MAGNITUDES. The A
/// operand rounds to bf16 (2^-8 relative per value) before the f16-precision
/// multiply, so ~1e-2 of the term magnitude is the honest bound for a
/// partially-cancelling dot; B's f16 rounding (2^-11) hides under it.
const MAGNITUDE_REL_TOL: f32 = 1.5e-2;

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

/// f32 -> bf16 bits (round to nearest even) — the host twin of the
/// checkpoint's storage format (the A operand's encoding).
fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7FFF + lsb)) >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// f32 -> IEEE-754 half (round-to-nearest-even, subnormals flushed — the
/// flush is below this test's magnitude tolerance).
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    let mut h = sign | ((exp as u16) << 10) | (mant >> 13) as u16;
    if (mant & 0x1000) != 0 && ((mant & 0x0fff) != 0 || (h & 1) != 0) {
        h += 1;
    }
    h
}

/// IEEE-754 half -> f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if (h >> 15) & 1 == 1 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    let mag = if exp == 0 {
        frac * 2f32.powi(-24)
    } else if exp == 0x1f {
        if frac == 0.0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + frac / 1024.0) * 2f32.powi(exp - 15)
    };
    sign * mag
}

/// Round-trip through f16 — the value the GEMM actually sees for `B`.
fn as_f16(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
}

fn upload<'a>(ctx: &'a VulkanContext, bytes: &[u8]) -> DeviceBuffer<'a> {
    let mut buf = DeviceBuffer::alloc(ctx, bytes.len().max(4)).expect("alloc");
    buf.copy_from_host(bytes).expect("upload");
    buf
}

#[test]
fn coopmat_mm_bf16_matches_reference_with_device_staged_b() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device ({e})");
            return;
        }
    };
    let Some(shape) = ctx.coopmat() else {
        eprintln!(
            "SKIP: no cooperative-matrix support on {}",
            ctx.device_name()
        );
        return;
    };
    let warp = ctx.subgroup_size().0;
    let shared = ctx.max_compute_shared_memory_size();
    eprintln!(
        "ARLE MmCmBf16 proof on: {} (shape {shape:?}, warp {warp})",
        ctx.device_name()
    );
    let mut cache = KernelCache::new();

    // (m, n, k): a q/k/v-like shape at a wide-tile n, and an odd n at the
    // model's widest contraction to exercise the unaligned bounds checks.
    for &(m, n, k) in &[(128usize, 64usize, 512usize), (320, 33, 2560)] {
        let Some(spec) = MmSpec::choose(shape, warp, n as u32, shared) else {
            eprintln!("SKIP: no warptile for n = {n}");
            return;
        };
        let mut rng = Rng(0xB16B_00B5 ^ ((m as u64) << 32) ^ k as u64);

        // A: bf16 weights, row-major [m][k].
        let a_vals: Vec<f32> = (0..m * k).map(|_| rng.next_unit_f32()).collect();
        let a_bits: Vec<u16> = a_vals.iter().map(|&v| f32_to_bf16_bits(v)).collect();
        let a_bytes: Vec<u8> = a_bits.iter().flat_map(|b| b.to_le_bytes()).collect();

        // B: f32 activations [n][k], staged to f16 ON DEVICE via F16KvPack —
        // exactly the qwen4_exp prefill's staging.
        let b_vals: Vec<f32> = (0..n * k).map(|_| rng.next_unit_f32() * 4.0).collect();
        let b_bytes: Vec<u8> = b_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        let buf_a = upload(&ctx, &a_bytes);
        let buf_b32 = upload(&ctx, &b_bytes);
        let mut buf_b16 = DeviceBuffer::alloc(&ctx, n * k * 2).expect("alloc b16");
        buf_b16
            .copy_from_host(&vec![0u8; n * k * 2])
            .expect("zero b16");
        let pack_push = f16_kv_pack_params((n * k) as u32).to_le_bytes();
        let pd = f16_kv_pack_dispatch_rows((n * k) as u32, 1);
        launch_cached(
            &mut cache,
            &ctx,
            Kernel::F16KvPack,
            &[&buf_b32, &buf_b16],
            pd,
            &pack_push,
            Kernel::F16KvPack.specialization_u32(),
        )
        .expect("f16 pack dispatch");

        let mut expected = vec![0f32; n * m];
        let mut magnitude = vec![0f32; n * m];
        for t in 0..n {
            for r in 0..m {
                let mut acc = 0f32;
                let mut mag = 0f32;
                for i in 0..k {
                    let a = bf16_to_f32(a_bits[r * k + i]);
                    let b = as_f16(b_vals[t * k + i]);
                    acc += a * b;
                    mag += (a * b).abs();
                }
                expected[t * m + r] = acc;
                magnitude[t * m + r] = mag;
            }
        }

        let out_len = n * m * 4;
        let mut buf_d = DeviceBuffer::alloc(&ctx, out_len).expect("alloc dst");
        buf_d.copy_from_host(&vec![0u8; out_len]).expect("zero dst");
        mm_with_params_and_spec(
            Kernel::MmCmBf16,
            &ctx,
            &[&buf_a, &buf_b16, &buf_d],
            mm_dispatch(m as u32, n as u32, &spec),
            &mmq_params(m as u32, n as u32, k as u32),
            &spec,
        )
        .expect("MmCmBf16 dispatch");

        let mut out = vec![0u8; out_len];
        buf_d.copy_to_host(&mut out).expect("read dst");
        let got: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut worst = 0f32;
        for t in 0..n {
            for r in 0..m {
                let idx = t * m + r;
                let rel = (got[idx] - expected[idx]).abs() / magnitude[idx].max(1e-3);
                worst = worst.max(rel);
                assert!(
                    rel < MAGNITUDE_REL_TOL,
                    "[{m}x{n}x{k}] token {t} row {r}: got {} vs {} (rel {rel:.3e})",
                    got[idx],
                    expected[idx]
                );
            }
        }
        eprintln!("[MmCmBf16 {m}x{n}x{k}] PASS (worst magnitude-rel {worst:.3e})");
    }
}

/// The same pack + GEMM pair recorded the way the qwen4_exp prefill records
/// it: sub-range descriptor bindings at NONZERO offsets inside larger buffers
/// (the weight slab / chunk arena pattern), one command buffer, a barrier
/// between the pack and the GEMM. Splits "the kernels are right" from "the
/// recording route is right".
#[test]
fn coopmat_mm_bf16_offset_bound_operands_match() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device ({e})");
            return;
        }
    };
    let Some(shape) = ctx.coopmat() else {
        eprintln!("SKIP: no cooperative-matrix support");
        return;
    };
    let warp = ctx.subgroup_size().0;
    // The model's own (m, k) shapes at the parity chunk width — including the
    // k = 10240 `mix_down` contraction no other test reaches — over
    // HOST-CACHED memory, the prefill arena's actual flavor.
    for (m, n, k) in [
        (320usize, 24usize, 10240usize),
        (12288, 24, 2560),
        (512, 24, 2560),
        (10240, 24, 2560),
        (6144, 24, 2560),
        (2560, 24, 6144),
        (640, 24, 2560),
        (2560, 24, 640),
    ] {
        run_offset_case(&ctx, shape, warp, m, n, k);
    }
}

fn run_offset_case(
    ctx: &VulkanContext,
    shape: vulkan_kernels::CoopmatShape,
    warp: u32,
    m: usize,
    n: usize,
    k: usize,
) {
    let ctx = &ctx;
    let spec = MmSpec::choose(shape, warp, n as u32, ctx.max_compute_shared_memory_size())
        .expect("warptile");
    let mut rng = Rng(0x0FF5_E7B1_6B00_57ED ^ ((m as u64) << 32) ^ k as u64);

    let a_vals: Vec<f32> = (0..m * k).map(|_| rng.next_unit_f32()).collect();
    let a_bits: Vec<u16> = a_vals.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    let b_vals: Vec<f32> = (0..n * k).map(|_| rng.next_unit_f32() * 4.0).collect();

    // Offsets: all 256-B aligned, all nonzero, inside deliberately larger
    // allocations. Arena layout: [pad | B32 | pad | B16 | pad | D].
    let a_off = 768u64;
    let b32_at = 512u64;
    let b16_at = b32_at + (n * k * 4) as u64 + 1024;
    let d_at = b16_at + (n * k * 2) as u64 + 1536;
    let mut slab = DeviceBuffer::alloc(ctx, a_off as usize + m * k * 2 + 256).expect("slab");
    let mut a_bytes = vec![0u8; a_off as usize];
    a_bytes.extend(a_bits.iter().flat_map(|b| b.to_le_bytes()));
    slab.copy_from_host(&a_bytes).expect("upload A");
    let mut arena =
        DeviceBuffer::alloc_host_cached(ctx, d_at as usize + n * m * 4 + 256).expect("arena");
    let mut arena_bytes = vec![0u8; b32_at as usize];
    arena_bytes.extend(b_vals.iter().flat_map(|v| v.to_le_bytes()));
    arena.copy_from_host(&arena_bytes).expect("upload B32");

    let mut cache = KernelCache::new();
    let mut recorder = CommandRecorder::new(ctx).expect("recorder");
    recorder.begin().expect("begin");

    // Pack: f32 B at b32_at -> f16 rows at b16_at.
    let pack_push = f16_kv_pack_params((n * k) as u32).to_le_bytes();
    let pd = f16_kv_pack_dispatch_rows((n * k) as u32, 1);
    let (pack_pipe, pack_layout) = cache
        .get(
            ctx,
            Kernel::F16KvPack,
            Kernel::F16KvPack.specialization_u32(),
            pack_push.len() as u32,
            2,
        )
        .expect("pack pipeline");
    let pack_set = DescriptorSet::storage_buffers_ranged(
        ctx,
        pack_layout,
        &[
            (&arena, b32_at, (n * k * 4) as u64),
            (&arena, b16_at, (n * k * 2) as u64),
        ],
    )
    .expect("pack set");
    record_dispatch(
        &mut recorder,
        pack_pipe,
        &pack_set,
        &pack_push,
        [pd.x, pd.y, pd.z],
    );
    recorder.barrier();

    let push = mmq_params(m as u32, n as u32, k as u32).to_le_bytes();
    let d = mm_dispatch(m as u32, n as u32, &spec);
    let (mm_pipe, mm_layout) = cache
        .get(
            ctx,
            Kernel::MmCmBf16,
            spec.specialization_u32(),
            push.len() as u32,
            3,
        )
        .expect("mm pipeline");
    let mm_set = DescriptorSet::storage_buffers_ranged(
        ctx,
        mm_layout,
        &[
            (&slab, a_off, (m * k * 2) as u64),
            (&arena, b16_at, (n * k * 2) as u64),
            (&arena, d_at, (n * m * 4) as u64),
        ],
    )
    .expect("mm set");
    record_dispatch(&mut recorder, mm_pipe, &mm_set, &push, [d.x, d.y, d.z]);
    recorder.submit_and_wait().expect("submit");
    drop(pack_set);
    drop(mm_set);

    let mut out = vec![0u8; n * m * 4];
    arena.copy_to_host_at(d_at, &mut out).expect("read D");
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut worst = 0f32;
    for t in 0..n {
        for r in 0..m {
            let mut acc = 0f32;
            let mut mag = 0f32;
            for i in 0..k {
                let a = bf16_to_f32(a_bits[r * k + i]);
                let b = as_f16(b_vals[t * k + i]);
                acc += a * b;
                mag += (a * b).abs();
            }
            let idx = t * m + r;
            let rel = (got[idx] - acc).abs() / mag.max(1e-3);
            worst = worst.max(rel);
            assert!(
                rel < MAGNITUDE_REL_TOL,
                "offset-bound token {t} row {r}: got {} vs {acc} (rel {rel:.3e})",
                got[idx]
            );
        }
    }
    eprintln!("[MmCmBf16 offset-bound {m}x{n}x{k}] PASS (worst {worst:.3e})");
}
