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
//! So the family table below needs no conversion factor. What it does NOT
//! include is the F32 tier or allocator padding, and that gap is not small:
//! `qwen4_upload` uploads the hyper-connection weights, norms and routers at F32
//! because their shaders declare `float[]`, which is 1.33 GiB more than the BF16
//! the file holds. **The classifier tally is therefore a FLOOR, not the number
//! the load is gated on** — 71.314 GiB against the planner's 72.640. The fit
//! assertions run `plan_qwen4_upload` for exactly that reason; the table stays
//! because it answers a different question (where are the bytes).
//!
//! ## The heap size is not the budget
//!
//! Heap 1 reports 74.43 GiB of `VkMemoryHeap::size` and 70.71 GiB of
//! `heapBudget`. Planning against the size is what let a 72.64 GiB residency
//! look like it fit; over-committing this UMA part is not
//! `ERROR_OUT_OF_DEVICE_MEMORY` but silent page demotion, so the load appears to
//! work and is quietly several times slow.
//! `device_budget_is_the_planning_number` pins the live numbers against the
//! constants here, so a driver or BIOS change surfaces as a failing test rather
//! than as a mystery.
//!
//! Runs in ~1s and needs no GPU, so it is not gated behind an opt-in flag; it
//! skips cleanly when the checkpoint is absent. Point `ARLE_QWEN4_CKPT` at the
//! directory to move it.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use infer_gguf::safetensors::SafeTensorsDir;
use infer_vulkan::loader::{DeviceBudget, DeviceBudgetSource};
use infer_vulkan::qwen4_names::{
    HcSite, Nvfp4Part, Qwen4Residency, Qwen4Stream, Qwen4TensorKind, Qwen4TensorRole,
    classify_qwen4_tensor,
};
use infer_vulkan::qwen4_upload::{
    DEFAULT_RESERVE_BYTES, Qwen4UploadConfig, Qwen4UploadScope, plan_qwen4_upload,
};

const CKPT_ENV: &str = "ARLE_QWEN4_CKPT";
const CKPT_DEFAULT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";

const GIB: f64 = (1u64 << 30) as f64;

/// Device-local heap SIZE on the Radeon 8060S (Strix Halo, 128 GB LPDDR5X UMA),
/// as reported by `VkPhysicalDeviceMemoryProperties`: 74.4322 GiB. Informative
/// only — nothing is planned against it.
const HEAP_BYTES: u64 = 79_920_955_392;

/// What the DRIVER grants on that heap: `VK_EXT_memory_budget`'s `heapBudget`,
/// 70.7107 GiB, 3.72 GiB under the size. THIS is the number a residency plan
/// lives or dies by.
const BUDGET_BYTES: u64 = 75_924_905_984;

/// `maxMemoryAllocationSize` on this device, queried via maintenance3. Every
/// binding must fit inside one slab of this size.
const SLAB_BYTES: u64 = 2048 << 20;

/// The pinned budget, shaped for `Qwen4Plan::ensure_fits`.
fn pinned_budget() -> DeviceBudget {
    DeviceBudget {
        bytes: BUDGET_BYTES,
        source: DeviceBudgetSource::DriverBudget,
        heap_index: 1,
        heap_size: HEAP_BYTES,
    }
}

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

/// The family table, and the CLASSIFIER FLOOR it adds up to.
///
/// Deliberately not a fit check any more: this tally is 1.33 GiB under the
/// planner's because it does not model the F32 tier, so asserting a fit here
/// would assert the wrong number. `full_plan_fits_the_driver_budget_only_after_spilling`
/// owns that question.
#[test]
fn real_checkpoint_residency_table_and_classifier_floor() {
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
    let heap = HEAP_BYTES as f64;
    let budget = BUDGET_BYTES as f64;
    println!(
        "    vs device-local heap {:.2} GiB: {:.2}% used, {:.2} GiB headroom",
        heap / GIB,
        100.0 * device as f64 / heap,
        (heap - device as f64) / GIB,
    );
    println!(
        "    vs DRIVER BUDGET     {:.2} GiB: {:.2}% used, {:.2} GiB headroom  <- the real limit",
        budget / GIB,
        100.0 * device as f64 / budget,
        (budget - device as f64) / GIB,
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
            "    committed vs budget: {:.2}% used, {:.2} GiB headroom",
            100.0 * plan.committed_bytes() as f64 / budget,
            (budget - plan.committed_bytes() as f64) / GIB,
        );
        // The classifier floor already over-commits the budget before the F32
        // tier is even counted. Assert the direction rather than a fit: the
        // number that has to fit is `full_plan_fits_the_driver_budget`'s.
        assert!(
            plan.committed_bytes() > BUDGET_BYTES - DEFAULT_RESERVE_BYTES,
            "the classifier FLOOR ({:.3} GiB of slabs) is under budget minus reserve \
             ({:.3} GiB) — the spill tier and this whole guard may no longer be needed, \
             which is worth failing over",
            plan.committed_bytes() as f64 / GIB,
            (BUDGET_BYTES - DEFAULT_RESERVE_BYTES) as f64 / GIB,
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
        largest_device.0 <= SLAB_BYTES,
        "device tensor {} is {} B, above maxMemoryAllocationSize {} B — it needs splitting",
        largest_device.1,
        largest_device.0,
        SLAB_BYTES,
    );
}

/// The number every residency plan is gated on, read off the live driver.
///
/// The constants in this file are pinned so the other tests are not tautologies
/// (`assert!(x <= x)` passes on any box). This is the one test that compares
/// them to reality, so a driver update, a BIOS UMA carve-out change or a
/// different part surfaces here instead of as an over-committed load.
#[cfg(feature = "vulkan")]
#[test]
fn device_budget_is_the_planning_number() {
    use infer_vulkan::loader::device_local_budget_from;

    let ctx = match vulkan_sys::VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device ({e})");
            return;
        }
    };
    let heaps = ctx.memory_heaps();
    let budgets = ctx.memory_budgets();
    println!("device: {}", ctx.device_name());
    for (i, (size, device_local)) in heaps.iter().enumerate() {
        let b = budgets.as_ref().and_then(|b| b.get(i));
        println!(
            "  heap {i}: size {:.2} GiB device_local={device_local} budget {} usage {}",
            *size as f64 / GIB,
            b.map_or("-".into(), |&(x, _)| format!("{:.2} GiB", x as f64 / GIB)),
            b.map_or("-".into(), |&(_, u)| format!("{:.2} GiB", u as f64 / GIB)),
        );
    }

    let live = device_local_budget_from(&heaps, budgets.as_deref()).expect("a device-local heap");
    println!(
        "  planning budget: {:.4} GiB from heap {} ({}), heap size {:.4} GiB",
        live.bytes as f64 / GIB,
        live.heap_index,
        live.source.label(),
        live.heap_size as f64 / GIB,
    );

    assert_eq!(
        live.source,
        DeviceBudgetSource::DriverBudget,
        "VK_EXT_memory_budget is supported here; falling back to heap size would silently \
         hand the plan 3.72 GiB it does not have"
    );
    assert_eq!(live.heap_size, HEAP_BYTES, "heap size moved");
    // Usage is not zero on a busy box, so the budget is an upper bound on what
    // is grantable right now; the pinned constant is the idle figure.
    assert!(
        live.bytes <= BUDGET_BYTES,
        "live budget {} exceeds the pinned {BUDGET_BYTES}",
        live.bytes
    );
    assert!(
        live.bytes + (2 << 30) > BUDGET_BYTES,
        "live budget {:.2} GiB is more than 2 GiB under the pinned {:.2} GiB — either this \
         box is loaded or the constant is stale",
        live.bytes as f64 / GIB,
        BUDGET_BYTES as f64 / GIB,
    );
    assert!(
        live.bytes < live.heap_size,
        "budget must be under the heap size on this part; that gap is the whole point"
    );
}

/// The number the load is really gated on: `plan_qwen4_upload`'s, against the
/// DRIVER'S budget.
///
/// Distinct from the classifier tally above, which is 1.33 GiB lighter because
/// it does not model the F32 tier. Both are over budget; only this one is over
/// by the right amount.
#[test]
fn full_plan_fits_the_driver_budget_only_after_spilling() {
    let dir = checkpoint_dir();
    if !dir.is_dir() {
        eprintln!("SKIP: no checkpoint at {} (set {CKPT_ENV})", dir.display());
        return;
    }
    let Ok(st) = SafeTensorsDir::open_dir(&dir) else {
        eprintln!("SKIP: cannot open {} as a safetensors dir", dir.display());
        return;
    };
    let cfg = Qwen4UploadConfig::default();
    let plan =
        plan_qwen4_upload(&st, &cfg, &Qwen4UploadScope::full()).expect("plan the full model");
    let budget = pinned_budget();
    let usable = BUDGET_BYTES - DEFAULT_RESERVE_BYTES;
    println!(
        "planner: device {:.3} GiB vs budget {:.3} GiB - reserve {:.3} GiB = {:.3} GiB usable",
        plan.device_bytes as f64 / GIB,
        BUDGET_BYTES as f64 / GIB,
        DEFAULT_RESERVE_BYTES as f64 / GIB,
        usable as f64 / GIB,
    );

    // 1. It does NOT fit. This is the assertion that fails at HEAD's behaviour
    //    (sized against the 74.43 GiB heap) and passes now.
    let err = plan
        .ensure_fits(&budget, DEFAULT_RESERVE_BYTES)
        .expect_err("the full residency must not fit the driver's budget");
    println!("refused: {err}");
    assert!(
        plan.device_bytes > usable,
        "plan {:.3} GiB is within the {:.3} GiB the driver grants — the spill tier would be \
         dead code, which is worth failing over",
        plan.device_bytes as f64 / GIB,
        usable as f64 / GIB,
    );

    // 2. It fits once the coldest suballocations move to the host heap.
    let mut spilled = plan.clone();
    let moved = spilled
        .spill_to_fit(&budget, DEFAULT_RESERVE_BYTES, 35 << 30)
        .expect("the plan fits after spilling");
    println!(
        "spilled {} suballocation(s) / {:.3} GiB -> device {:.3} GiB, host {:.3} GiB",
        moved.items,
        moved.bytes as f64 / GIB,
        spilled.device_bytes as f64 / GIB,
        spilled.spill_bytes as f64 / GIB,
    );
    spilled
        .ensure_fits(&budget, DEFAULT_RESERVE_BYTES)
        .expect("the spilled plan fits");
    assert!(
        spilled.spill_bytes > 0 && spilled.spill_bytes < 8 * (1 << 30),
        "the spill should be single-digit GiB, not the whole model: {:.3} GiB",
        spilled.spill_bytes as f64 / GIB,
    );
    assert_eq!(
        spilled.device_bytes + spilled.spill_bytes,
        plan.device_bytes,
        "a spill relocates bytes, it does not create or destroy them"
    );

    // 3. And the SLABS fit too, which the plan's own bytes cannot answer.
    #[cfg(feature = "vulkan")]
    {
        let alignment = 16;
        use infer_vulkan::qwen4_upload::Qwen4Tier;
        let device_pack = spilled
            .choose_packing(Qwen4Tier::Device, SLAB_BYTES, alignment)
            .expect("pack the device tier");
        let spill_pack = spilled
            .choose_packing(Qwen4Tier::HostSpill, SLAB_BYTES, alignment)
            .expect("pack the spill tier");
        println!(
            "slabs: device {:.3} GiB / {} slabs of {:.0} MiB, spill {:.3} GiB / {} slabs",
            device_pack.committed_bytes as f64 / GIB,
            device_pack.slab_count,
            device_pack.slab_bytes as f64 / (1u64 << 20) as f64,
            spill_pack.committed_bytes as f64 / GIB,
            spill_pack.slab_count,
        );
        device_pack
            .ensure_fits(BUDGET_BYTES, DEFAULT_RESERVE_BYTES)
            .expect("the device tier's committed slabs fit the budget");
        // The unspilled plan's slabs do NOT, which is what the spill bought.
        assert!(
            plan.choose_packing(Qwen4Tier::Device, SLAB_BYTES, alignment)
                .expect("pack the unspilled plan")
                .ensure_fits(BUDGET_BYTES, DEFAULT_RESERVE_BYTES)
                .is_err(),
            "the unspilled residency's slabs must not fit the budget"
        );
    }
}
