//! The residency plan for `qwen4_exp` (Qwen3.8-Flash-Next), measured — not
//! estimated — off the real on-box checkpoint.
//!
//! This is the seam test for the three pieces that just landed: it opens the
//! checkpoint with [`infer_gguf::safetensors::SafeTensorsDir`], classifies every
//! tensor name with [`infer_vulkan::qwen4_names::classify_qwen4_tensor`], and
//! sums the *declared* byte lengths per family per residency tier. The output is
//! the number the Vulkan lane is actually gated on: how much of the 74.43 GiB
//! device-local heap a text-only load of this model wants.
//!
//! Why declared file bytes are the right unit for the device tiers:
//! - `DevicePacked` (the NVFP4 experts) is uploaded byte-exact — the packed U8
//!   plane, the FP8 block scales and the F32 scalars all go up unchanged, and
//!   the GEMV decodes them in-shader. File bytes == device bytes.
//! - `DeviceDequant` is BF16 in the file and F16 on the device: 2 bytes either
//!   way, so this tier is byte-neutral too.
//!
//! So the device total below needs no conversion factor. What it does NOT
//! include is allocator padding; the `vulkan` build adds a `SlabPlan` dry-run at
//! the end, which is what answers "does it fit" for real.
//!
//! Runs in ~1s and needs no GPU, so it is not gated behind an opt-in flag; it
//! skips cleanly when the checkpoint is absent. Point `ARLE_QWEN4_CKPT` at the
//! directory to move it.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use infer_gguf::safetensors::SafeTensorsDir;
use infer_vulkan::qwen4_names::{
    HcSite, Nvfp4Part, Qwen4Residency, Qwen4Stream, Qwen4TensorKind, Qwen4TensorRole,
    classify_qwen4_tensor,
};

const CKPT_ENV: &str = "ARLE_QWEN4_CKPT";
const CKPT_DEFAULT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";

const GIB: f64 = (1u64 << 30) as f64;

/// Device-local heap on the Radeon 8060S (Strix Halo, 128 GB LPDDR5X UMA), as
/// reported by `VkPhysicalDeviceMemoryProperties` for the DEVICE_LOCAL heap.
const HEAP_GIB: f64 = 74.43;

/// `maxMemoryAllocationSize` on this device, queried via maintenance3. Every
/// binding must fit inside one slab of this size.
const SLAB_BYTES: u64 = 2048 << 20;

fn checkpoint_dir() -> PathBuf {
    std::env::var_os(CKPT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(CKPT_DEFAULT))
}

/// A readable family label for a classified tensor.
///
/// Coarser than [`Qwen4TensorKind`] on purpose: the point of the table is to
/// show where the bytes are, and 60 one-row families hide that. Splits are kept
/// exactly where they carry byte weight — the four NVFP4 planes stay apart
/// because their ratio is the thing to sanity-check, and the PLE n-gram table
/// stays apart from its index buffers because one is 47 GiB and the other is
/// 200 bytes.
fn family(kind: Qwen4TensorKind) -> &'static str {
    use Qwen4TensorKind as K;
    match kind {
        K::EmbedTokens => "embed_tokens",
        K::LmHead => "lm_head",

        K::HyperConnection {
            site: HcSite::Attn, ..
        } => "hyper_connection.attn",
        K::HyperConnection {
            site: HcSite::Mlp, ..
        } => "hyper_connection.mlp",
        K::HyperConnection {
            site: HcSite::Mixer,
            ..
        } => "hyper_connection.mixer",

        K::LinearAttnInProjQkv
        | K::LinearAttnInProjZ
        | K::LinearAttnInProjB
        | K::LinearAttnInProjA => "linear_attn.in_proj",
        K::LinearAttnOutProj => "linear_attn.out_proj",
        K::LinearAttnConv1d => "linear_attn.conv1d",
        K::LinearAttnALog | K::LinearAttnDtBias | K::LinearAttnNorm => "linear_attn.norm+params",

        K::AttnQProj | K::AttnKProj | K::AttnVProj | K::AttnOProj => "attn.qkvo_proj",
        K::AttnQNorm | K::AttnKNorm => "attn.qk_norm",
        K::IndexerQkProj => "indexer.qk_proj",
        K::IndexerQNorm | K::IndexerKNorm => "indexer.qk_norm",

        K::MoeRouter => "moe.router",
        K::SharedExpertGate => "moe.shared_expert_gate",
        K::SharedExpertGateProj | K::SharedExpertUpProj | K::SharedExpertDownProj => {
            "moe.shared_expert.mlp"
        }
        K::Expert {
            part: Nvfp4Part::Packed,
            ..
        } => "moe.experts.nvfp4 .weight (u8)",
        K::Expert {
            part: Nvfp4Part::BlockScale,
            ..
        } => "moe.experts.nvfp4 .weight_scale (fp8)",
        K::Expert {
            part: Nvfp4Part::GlobalScale,
            ..
        } => "moe.experts.nvfp4 .weight_scale_2 (f32)",
        K::Expert {
            part: Nvfp4Part::InputScale,
            ..
        } => "moe.experts.nvfp4 .input_scale (f32)",
        K::ExpertsStackedGateUp | K::ExpertsStackedDown => "moe.experts.stacked (bf16)",

        K::PleKeyProj | K::PleValueProj => "ple.kv_proj",
        K::PleNormKey | K::PleNormQuery | K::PleNormConv => "ple.norms",
        K::PleConv1d => "ple.conv1d",
        K::PleNgramShard => "ple.ngram_table (fp8, 128 shards)",
        K::PleNgramWeightScale
        | K::PleNgramLayerMultipliers
        | K::PleNgramHeadsOffsets
        | K::PleNgramHeadsVocabSizes => "ple.ngram_index_buffers",

        K::MtpFcEmbedding | K::MtpFcHidden => "fc",
        K::MtpPreFcNormEmbedding | K::MtpPreFcNormHidden => "pre_fc_norm",

        K::Vision(_) => "tower",
    }
}

/// Stream prefix, so an `mtp.*` copy of a text family cannot be read as the text
/// one. (Every non-text tensor is `Drop`, so they never share a section, but the
/// row label should still say which tree it came from.)
fn stream_prefix(stream: Qwen4Stream) -> &'static str {
    match stream {
        Qwen4Stream::Text => "",
        Qwen4Stream::Mtp => "mtp.",
        Qwen4Stream::Vision => "vision.",
    }
}

fn tier_name(r: Qwen4Residency) -> &'static str {
    match r {
        Qwen4Residency::DevicePacked => "DevicePacked",
        Qwen4Residency::DeviceDequant => "DeviceDequant",
        Qwen4Residency::HostGather => "HostGather",
        Qwen4Residency::Drop => "Drop",
    }
}

#[derive(Default, Clone, Copy)]
struct Row {
    count: u64,
    bytes: u64,
}

impl Row {
    fn add(&mut self, bytes: u64) {
        self.count += 1;
        self.bytes += bytes;
    }
}

fn print_section(title: &str, rows: &BTreeMap<String, Row>, denom: u64) {
    println!("  {title}");
    println!(
        "    {:<42} {:>9} {:>18} {:>10} {:>7}",
        "family", "tensors", "bytes", "GiB", "%"
    );
    let mut sorted: Vec<_> = rows.iter().collect();
    sorted.sort_by(|a, b| b.1.bytes.cmp(&a.1.bytes).then(a.0.cmp(b.0)));
    for (name, row) in sorted {
        println!(
            "    {:<42} {:>9} {:>18} {:>10.3} {:>6.2}%",
            name,
            row.count,
            row.bytes,
            row.bytes as f64 / GIB,
            if denom == 0 {
                0.0
            } else {
                100.0 * row.bytes as f64 / denom as f64
            },
        );
    }
    let sub: u64 = rows.values().map(|r| r.bytes).sum();
    let n: u64 = rows.values().map(|r| r.count).sum();
    println!(
        "    {:<42} {:>9} {:>18} {:>10.3}",
        "-- subtotal --",
        n,
        sub,
        sub as f64 / GIB
    );
    println!();
}

#[test]
fn real_checkpoint_residency_plan_fits_the_device_heap() {
    let dir = checkpoint_dir();
    if !dir.is_dir() {
        eprintln!(
            "SKIP: no qwen3.8-flash-next checkpoint at {} (set {CKPT_ENV})",
            dir.display()
        );
        return;
    }

    let t0 = Instant::now();
    let st = match SafeTensorsDir::open_dir(&dir) {
        Ok(st) => st,
        Err(e) => {
            eprintln!(
                "SKIP: cannot open {} as a safetensors dir ({e}); set {CKPT_ENV}",
                dir.display()
            );
            return;
        }
    };
    let open_ms = t0.elapsed().as_millis();

    // ---- classify every tensor, bucket by (tier, family) ------------------
    let mut per_tier: BTreeMap<&'static str, BTreeMap<String, Row>> = BTreeMap::new();
    let mut tier_bytes: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut tier_count: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut dtype_bytes: BTreeMap<String, Row> = BTreeMap::new();
    let mut unclassified: Vec<(String, String)> = Vec::new();
    let mut all_bytes: u64 = 0;

    // The single biggest binding on each device tier, because
    // maxMemoryAllocationSize (2 GiB) is a per-binding limit, not a total.
    let mut largest_device: (u64, String) = (0, String::new());

    for info in st.tensors() {
        all_bytes += info.len;
        let role: Qwen4TensorRole = match classify_qwen4_tensor(&info.name) {
            Ok(r) => r,
            Err(e) => {
                if unclassified.len() < 20 {
                    unclassified.push((info.name.clone(), e.to_string()));
                }
                continue;
            }
        };

        let tier = tier_name(role.residency);
        let label = format!("{}{}", stream_prefix(role.stream), family(role.kind));
        per_tier
            .entry(tier)
            .or_default()
            .entry(label)
            .or_default()
            .add(info.len);
        *tier_bytes.entry(tier).or_default() += info.len;
        *tier_count.entry(tier).or_default() += 1;
        dtype_bytes
            .entry(info.dtype.clone())
            .or_default()
            .add(info.len);

        if matches!(
            role.residency,
            Qwen4Residency::DevicePacked | Qwen4Residency::DeviceDequant
        ) && info.len > largest_device.0
        {
            largest_device = (info.len, info.name.clone());
        }
    }

    let packed = *tier_bytes.get("DevicePacked").unwrap_or(&0);
    let dequant = *tier_bytes.get("DeviceDequant").unwrap_or(&0);
    let host = *tier_bytes.get("HostGather").unwrap_or(&0);
    let dropped = *tier_bytes.get("Drop").unwrap_or(&0);
    let device = packed + dequant;

    // ---- report -----------------------------------------------------------
    println!();
    println!("=== qwen4_exp (Qwen3.8-Flash-Next) residency plan ===");
    println!("checkpoint : {}", dir.display());
    println!(
        "shards     : {}  tensors: {}  declared bytes: {} ({:.2} GiB)  [open {} ms]",
        st.shard_count(),
        st.tensors().len(),
        all_bytes,
        all_bytes as f64 / GIB,
        open_ms,
    );
    println!(
        "device bytes == file bytes: NVFP4 goes up packed, BF16 -> F16 is 2 bytes either way."
    );
    println!();

    for tier in ["DevicePacked", "DeviceDequant", "HostGather", "Drop"] {
        if let Some(rows) = per_tier.get(tier) {
            print_section(&format!("[{tier}]"), rows, all_bytes);
        }
    }

    println!("  [by dtype, all tiers]");
    let mut dts: Vec<_> = dtype_bytes.iter().collect();
    dts.sort_by(|a, b| b.1.bytes.cmp(&a.1.bytes));
    for (dt, row) in dts {
        println!(
            "    {:<12} {:>9} tensors {:>18} B  {:>10.3} GiB",
            dt,
            row.count,
            row.bytes,
            row.bytes as f64 / GIB
        );
    }
    println!();

    println!("  === totals ===");
    for (tier, bytes) in [
        ("DevicePacked ", packed),
        ("DeviceDequant", dequant),
        ("HostGather   ", host),
        ("Drop         ", dropped),
    ] {
        println!(
            "    {} {:>9} tensors {:>18} B  {:>10.3} GiB  {:>6.2}% of file",
            tier,
            tier_count.get(tier.trim()).copied().unwrap_or(0),
            bytes,
            bytes as f64 / GIB,
            100.0 * bytes as f64 / all_bytes as f64,
        );
    }
    println!(
        "    DEVICE TOTAL  {:>9}          {:>18} B  {:>10.3} GiB",
        tier_count.get("DevicePacked").copied().unwrap_or(0)
            + tier_count.get("DeviceDequant").copied().unwrap_or(0),
        device,
        device as f64 / GIB,
    );
    let heap = HEAP_GIB * GIB;
    println!(
        "    vs device-local heap {:.2} GiB: {:.2}% used, {:.2} GiB headroom",
        HEAP_GIB,
        100.0 * device as f64 / heap,
        (heap - device as f64) / GIB,
    );
    println!(
        "    largest single device binding: {:.3} GiB  {}  (maxMemoryAllocationSize = {:.0} GiB)",
        largest_device.0 as f64 / GIB,
        largest_device.1,
        SLAB_BYTES as f64 / GIB,
    );

    // The slab dry-run is what actually answers "does it fit": a 71 GiB plan
    // that pads to 76 GiB of committed slabs does not.
    #[cfg(feature = "vulkan")]
    {
        let mut lens: Vec<u64> = st
            .tensors()
            .iter()
            .filter(|i| {
                classify_qwen4_tensor(&i.name).is_ok_and(|r| {
                    matches!(
                        r.residency,
                        Qwen4Residency::DevicePacked | Qwen4Residency::DeviceDequant
                    )
                })
            })
            .map(|i| i.len)
            .collect();
        // Largest-first: measured to be the difference between the ceil() floor
        // and ~4 GiB of stranded slab tails.
        lens.sort_unstable_by(|a, b| b.cmp(a));
        let mut plan = vulkan_sys::SlabPlan::new(SLAB_BYTES, 16).expect("slab geometry");
        for len in &lens {
            plan.place(*len).expect("every device tensor places");
        }
        println!();
        println!(
            "    slab dry-run (largest-first, {} B slabs, 16 B align): {} slabs, \
             committed {:.3} GiB, used {:.3} GiB, wasted {:.3} GiB",
            SLAB_BYTES,
            plan.slab_count(),
            plan.committed_bytes() as f64 / GIB,
            plan.used_bytes() as f64 / GIB,
            plan.wasted_bytes() as f64 / GIB,
        );
        println!(
            "    committed vs heap: {:.2}% used, {:.2} GiB headroom",
            100.0 * plan.committed_bytes() as f64 / heap,
            (heap - plan.committed_bytes() as f64) / GIB,
        );
        assert!(
            (plan.committed_bytes() as f64) < heap,
            "committed slabs {:.3} GiB exceed the {HEAP_GIB} GiB device-local heap",
            plan.committed_bytes() as f64 / GIB,
        );
    }
    println!();

    // ---- assertions -------------------------------------------------------
    assert!(
        unclassified.is_empty(),
        "{} tensor name(s) did not classify, first few: {:?}",
        unclassified.len(),
        &unclassified[..unclassified.len().min(5)],
    );
    assert_eq!(
        packed + dequant + host + dropped,
        all_bytes,
        "tiers must partition the checkpoint's declared bytes"
    );
    assert_eq!(
        tier_count.values().sum::<u64>(),
        st.tensors().len() as u64,
        "every tensor lands in exactly one tier"
    );

    let low = 65.0 * GIB;
    let high = 75.0 * GIB;
    assert!(
        (device as f64) >= low && (device as f64) <= high,
        "device-resident plan is {:.3} GiB, outside the 65-75 GiB window \
         (packed {:.3} + dequant {:.3}); host {:.3}, dropped {:.3}, file total {:.3}",
        device as f64 / GIB,
        packed as f64 / GIB,
        dequant as f64 / GIB,
        host as f64 / GIB,
        dropped as f64 / GIB,
        all_bytes as f64 / GIB,
    );
    assert!(
        (device as f64) < heap,
        "device plan {:.3} GiB does not fit the {HEAP_GIB} GiB heap",
        device as f64 / GIB
    );
    assert!(
        largest_device.0 <= SLAB_BYTES,
        "device tensor {} is {} B, above maxMemoryAllocationSize {} B — it needs splitting",
        largest_device.1,
        largest_device.0,
        SLAB_BYTES,
    );
}
