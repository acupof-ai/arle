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
//! ## S7 prep: MTP + vision are back in the plan
//!
//! The classifier no longer has a `Drop` tier: the MTP head (4.86 GiB — the
//! speculative-decode lever, an explicit product keep) and the vision tower
//! (0.84 GiB) are `DeviceDequant` INTENT now. `plan_qwen4_upload` still skips
//! them — it gates on `stream != Text`, not on residency — so nothing about
//! the shipping 84.9 ms/token load changes until the integration flips that
//! gate. The two S7 tests at the bottom are the paper plan that integration
//! consumes: per-name (shape, dtype, tier) pins over the real checkpoint, and
//! the scenario table (dense BF16 vs dense Q8_0, each with MTP + vision,
//! against budget minus reserve, spill volume named).
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
//! the load is gated on** — 77.006 GiB of device INTENT since the S7
//! reclassification (71.314 of it the text stream, which the planner prices
//! at 71.452). The fit assertions run `plan_qwen4_upload` for exactly that
//! reason; the table stays because it answers a different question (where are
//! the bytes).
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
    HcPart, HcSite, Nvfp4Part, Qwen4Residency, Qwen4Stream, Qwen4TensorKind, Qwen4TensorRole,
    classify_qwen4_tensor,
};
use infer_vulkan::qwen4_upload::{
    DEFAULT_RESERVE_BYTES, Qwen4DeviceFormat, Qwen4UploadConfig, Qwen4UploadScope,
    plan_qwen4_upload,
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
/// Deliberately not a fit check: this tally does not model the F32 tier
/// (1.33 GiB on the text stream) or slab tails, and since the S7
/// reclassification it counts the MTP + vision intent the shipping planner
/// does not price yet, so asserting a fit here would assert the wrong number.
/// `full_plan_fits_the_driver_budget_only_after_spilling` owns the shipping
/// plan's fit; `s7_scenario_table_dense_bf16_vs_q8_0` owns the S7 one.
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
    // Every device tensor over that limit; each one is a split the upload
    // MUST perform, so they are pinned by name below.
    let mut oversized_device: Vec<String> = Vec::new();

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
        ) {
            if info.len > largest_device.0 {
                largest_device = (info.len, info.name.clone());
            }
            if info.len > SLAB_BYTES {
                oversized_device.push(info.name.clone());
            }
        }
    }

    let packed = *tier_bytes.get("DevicePacked").unwrap_or(&0);
    let dequant = *tier_bytes.get("DeviceDequant").unwrap_or(&0);
    let host = *tier_bytes.get("HostGather").unwrap_or(&0);
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

    for tier in ["DevicePacked", "DeviceDequant", "HostGather"] {
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
            // The MTP stacked gate_up (3.125 GiB) is over maxMemoryAllocationSize;
            // it is a 512-expert stack, so it splits legally at any expert
            // boundary. Halving models the coarsest such split — the batch_stride
            // contract survives as two 256-expert sub-stacks.
            .flat_map(|i| {
                if i.len > SLAB_BYTES {
                    vec![i.len / 2, i.len - i.len / 2]
                } else {
                    vec![i.len]
                }
            })
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
        packed + dequant + host,
        all_bytes,
        "tiers must partition the checkpoint's declared bytes"
    );
    assert_eq!(
        tier_count.values().sum::<u64>(),
        st.tensors().len() as u64,
        "every tensor lands in exactly one tier"
    );

    // With MTP + vision reclassified into device intent the floor moved from
    // ~71.3 to ~77.0 GiB — deliberately OVER the 70.71 GiB budget; the S7
    // scenario test below owns the fit-vs-spill arithmetic.
    let low = 74.0 * GIB;
    let high = 80.0 * GIB;
    assert!(
        (device as f64) >= low && (device as f64) <= high,
        "device-intent floor is {:.3} GiB, outside the 74-80 GiB window \
         (packed {:.3} + dequant {:.3}); host {:.3}, file total {:.3}",
        device as f64 / GIB,
        packed as f64 / GIB,
        dequant as f64 / GIB,
        host as f64 / GIB,
        all_bytes as f64 / GIB,
    );
    // Exactly one device tensor is over maxMemoryAllocationSize: the MTP
    // stacked gate_up (3.125 GiB). That is a split the integration MUST
    // perform — a 512-expert stack splits legally at any expert boundary — so
    // pin WHICH tensor it is rather than forbidding it, and fail if the set
    // ever grows (a new member would be a new mandatory split) or shrinks
    // (the note would be stale).
    assert_eq!(
        oversized_device,
        vec!["mtp.layers.0.mlp.experts.gate_up_proj".to_string()],
        "the set of device tensors above maxMemoryAllocationSize ({SLAB_BYTES} B) moved"
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

// ------------------------------------------------------------------ S7 prep
//
// The MTP head and the vision tower are back in the residency intent
// (`qwen4_names` has no `Drop` tier any more), and the dense tier is about to
// halve under Q8_0. The two tests below are the paper plan the integration
// consumes: per-name (shape, dtype, tier) pins over the real checkpoint, and
// the scenario table against the driver budget minus the reserve.

/// Q8_0 device bytes for a `[nrows, ncols]` weight: 34 bytes (one f16 scale +
/// 32 i8 quants) per 32-element block, the GGUF layout `GemvQ8_0` consumes.
/// `None` when a row is not a whole number of blocks — a ragged Q8_0 row would
/// make row `r > 0` read its neighbour's tail, the same failure class the
/// NVFP4 planner guards against.
fn q8_0_bytes(ncols: u64, nrows: u64) -> Option<u64> {
    (ncols != 0 && ncols.is_multiple_of(32)).then(|| nrows * (ncols / 32) * 34)
}

/// The families the planner prices at `dense_format` (the mirror of
/// `qwen4_upload::device_format`'s dense arm) plus the two MTP fusion fcs the
/// integration will put on the same tier. These are exactly the weights the
/// Q8_0 lever applies to: plain-GEMV consumers with whole-block rows.
///
/// Deliberately NOT included:
/// - the hyper-connection mixes — their kernels are the `qwen4_hc_*_bf16`
///   family, so under a Q8_0 dense tier they must STAY BF16 rather than take
///   `device_format`'s F32 fallback (which would cost +1.18 GiB);
/// - the MTP expert stacks — the batched expert GEMV path; quantizing them is
///   a further, separate lever (the table prints what it would buy).
fn takes_dense_format(kind: Qwen4TensorKind) -> bool {
    use Qwen4TensorKind as K;
    matches!(
        kind,
        K::LinearAttnInProjQkv
            | K::LinearAttnInProjZ
            | K::LinearAttnOutProj
            | K::AttnQProj
            | K::AttnKProj
            | K::AttnVProj
            | K::AttnOProj
            | K::IndexerQkProj
            | K::SharedExpertGateProj
            | K::SharedExpertUpProj
            | K::SharedExpertDownProj
            | K::PleKeyProj
            | K::PleValueProj
            | K::LmHead
            | K::MtpFcEmbedding
            | K::MtpFcHidden
    )
}

/// The small families `qwen4_upload::device_format` sends to F32 because their
/// consuming shaders declare `float[]`: on the device they cost 2x their BF16
/// file bytes. Mirrored here so the MTP additions are priced the way the
/// planner will price them (it does not price the MTP stream yet). The two
/// `pre_fc` norms follow by the same rule — they are `Qwen4ExpTextRMSNorm`
/// weights and will land in the same `rms_norm.comp` consumer.
fn f32_on_device(kind: Qwen4TensorKind) -> bool {
    use Qwen4TensorKind as K;
    matches!(
        kind,
        K::HyperConnection {
            part: HcPart::Norm,
            ..
        } | K::AttnQNorm
            | K::AttnKNorm
            | K::IndexerQNorm
            | K::IndexerKNorm
            | K::MoeRouter
            | K::SharedExpertGate
            | K::MtpPreFcNormEmbedding
            | K::MtpPreFcNormHidden
    )
}

/// One S7 residency pin: a real checkpoint tensor, the stream and tier it must
/// classify to, and the dtype + shape its shard header must declare.
struct S7Pin {
    name: &'static str,
    stream: Qwen4Stream,
    residency: Qwen4Residency,
    dtype: &'static str,
    /// Header (row-major) order, as `model.safetensors.index.json`'s shards
    /// spell it. `SafeTensorInfo::dims` is this REVERSED (GGUF ne order); the
    /// test reverses on compare so the pin stays in the checkpoint's words.
    shape: &'static [u64],
}

/// The reclassified families, pinned name -> (shape, dtype, tier) over the
/// real checkpoint, plus contrast rows whose tier did NOT move so a global
/// flip of the residency logic cannot pass. Roles verified against
/// `modeling_qwen4_exp.py` — see the `Qwen4Stream::Mtp` / `::Vision` docs in
/// `qwen4_names` for the class-by-class mapping and line numbers.
const S7_PINS: &[S7Pin] = &[
    // ---- MTP head: 31 tensors, every one device-wanted ----
    S7Pin {
        name: "mtp.fc_embedding.weight",
        stream: Qwen4Stream::Mtp,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[2560, 2560],
    },
    S7Pin {
        name: "mtp.fc_hidden.weight",
        stream: Qwen4Stream::Mtp,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[2560, 2560],
    },
    // [2560]: norms the NEXT token's embedding row.
    S7Pin {
        name: "mtp.pre_fc_norm_embedding.weight",
        stream: Qwen4Stream::Mtp,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[2560],
    },
    // [10240]: norms the PRE-mixer 4-stream hidden state — the width is the
    // proof the MTP taps the stream before `hyper_connection_mixer` collapses
    // it, which is what the integration must keep alive.
    S7Pin {
        name: "mtp.pre_fc_norm_hidden.weight",
        stream: Qwen4Stream::Mtp,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[10240],
    },
    S7Pin {
        name: "mtp.hyper_connection_mixer.hc_norm.weight",
        stream: Qwen4Stream::Mtp,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[10240],
    },
    // The 4.69 GiB that forced the old Drop tier: the quant-excluded stacked
    // expert layout (`Qwen4ExpTextExperts`, modeling_qwen4_exp.py:859).
    // gate_up is fused, hence 2 x 640; at 3.125 GiB it is the one tensor over
    // maxMemoryAllocationSize and must split at an expert boundary.
    S7Pin {
        name: "mtp.layers.0.mlp.experts.gate_up_proj",
        stream: Qwen4Stream::Mtp,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[512, 1280, 2560],
    },
    S7Pin {
        name: "mtp.layers.0.mlp.experts.down_proj",
        stream: Qwen4Stream::Mtp,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[512, 2560, 640],
    },
    // The MTP decoder layer is full-attention: q carries the sigmoid output
    // gate (x2), same as the text stream's 12 full layers.
    S7Pin {
        name: "mtp.layers.0.self_attn.q_proj.weight",
        stream: Qwen4Stream::Mtp,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[12288, 2560],
    },
    S7Pin {
        name: "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
        stream: Qwen4Stream::Mtp,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[640, 2560],
    },
    S7Pin {
        name: "mtp.layers.0.mlp.gate.weight",
        stream: Qwen4Stream::Mtp,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[512, 2560],
    },
    S7Pin {
        name: "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        stream: Qwen4Stream::Mtp,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[2560, 640],
    },
    S7Pin {
        name: "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
        stream: Qwen4Stream::Mtp,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[4, 10240],
    },
    // ---- vision tower: 333 tensors, prefill-only, device-wanted ----
    S7Pin {
        name: "model.visual.patch_embed.proj.weight",
        stream: Qwen4Stream::Vision,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[1152, 3, 2, 16, 16],
    },
    S7Pin {
        name: "model.visual.pos_embed.weight",
        stream: Qwen4Stream::Vision,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[2304, 1152],
    },
    S7Pin {
        name: "model.visual.blocks.0.attn.qkv.weight",
        stream: Qwen4Stream::Vision,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[3456, 1152],
    },
    // A bias, so the coarse Vision slots are proven to cover both planes.
    S7Pin {
        name: "model.visual.blocks.13.norm2.bias",
        stream: Qwen4Stream::Vision,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[1152],
    },
    S7Pin {
        name: "model.visual.blocks.26.mlp.linear_fc2.weight",
        stream: Qwen4Stream::Vision,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[1152, 4304],
    },
    // merger fc1 works on the 2x2-merged width (1152 x 4 = 4608); fc2 is the
    // one vision weight whose output is the TEXT hidden size.
    S7Pin {
        name: "model.visual.merger.linear_fc1.weight",
        stream: Qwen4Stream::Vision,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[4608, 4608],
    },
    S7Pin {
        name: "model.visual.merger.linear_fc2.weight",
        stream: Qwen4Stream::Vision,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[2560, 4608],
    },
    // ---- contrast rows: tiers that did NOT move in the S7 prep ----
    S7Pin {
        name: "model.language_model.embed_tokens.weight",
        stream: Qwen4Stream::Text,
        residency: Qwen4Residency::HostGather,
        dtype: "BF16",
        shape: &[248320, 2560],
    },
    S7Pin {
        name: "lm_head.weight",
        stream: Qwen4Stream::Text,
        residency: Qwen4Residency::DeviceDequant,
        dtype: "BF16",
        shape: &[248320, 2560],
    },
    S7Pin {
        name: "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
        stream: Qwen4Stream::Text,
        residency: Qwen4Residency::DevicePacked,
        dtype: "U8",
        shape: &[640, 1280],
    },
    S7Pin {
        name: "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
        stream: Qwen4Stream::Text,
        residency: Qwen4Residency::HostGather,
        dtype: "F8_E4M3",
        shape: &[2500012, 160],
    },
];

/// PIN: the S7 reclassification, name by name, over the real checkpoint.
///
/// `qwen4_names`' own tests prove the counts and shapes per FAMILY; what they
/// cannot prove is the tier, because they compute the expected role with the
/// same code that assigns it. The rows here spell stream + tier as literals,
/// so swapping two families' residencies (or quietly re-dropping the MTP) is a
/// failing diff, not a silent re-plan.
#[test]
fn s7_mtp_vision_pins_shape_dtype_and_tier() {
    let dir = checkpoint_dir();
    if !dir.is_dir() {
        eprintln!("SKIP: no checkpoint at {} (set {CKPT_ENV})", dir.display());
        return;
    }
    let Ok(st) = SafeTensorsDir::open_dir(&dir) else {
        eprintln!("SKIP: cannot open {} as a safetensors dir", dir.display());
        return;
    };
    let by_name: BTreeMap<&str, _> = st.tensors().iter().map(|i| (i.name.as_str(), i)).collect();

    for pin in S7_PINS {
        let role = classify_qwen4_tensor(pin.name)
            .unwrap_or_else(|e| panic!("classify `{}`: {e}", pin.name));
        assert_eq!(role.stream, pin.stream, "stream of `{}`", pin.name);
        assert_eq!(role.residency, pin.residency, "tier of `{}`", pin.name);

        let info = by_name
            .get(pin.name)
            .unwrap_or_else(|| panic!("`{}` is not in the checkpoint", pin.name));
        assert_eq!(info.dtype, pin.dtype, "dtype of `{}`", pin.name);
        let header_order: Vec<u64> = info.dims.iter().rev().copied().collect();
        assert_eq!(header_order, pin.shape, "header shape of `{}`", pin.name);
    }
    println!(
        "PASS: {} S7 pins hold (name -> stream, tier, dtype, shape)",
        S7_PINS.len()
    );
}

/// THE TABLE: the full S7 plan at (a) dense BF16 + MTP/vision and (b) dense
/// Q8_0 + MTP/vision, against the 70.71 GiB driver budget minus the 1.5 GiB
/// reserve, spill volume named. All floors — suballocation bytes, before slab
/// tails (~0.02..1.3 GiB depending on the packing, `choose_packing`'s job).
///
/// Pricing rules, mirrored from `qwen4_upload::device_format` because the
/// planner does not price the MTP/vision streams yet: NVFP4 stays packed;
/// dense-GEMV weights cost 2 B/elem at BF16 or 34 B per 32 elems at Q8_0; the
/// `float[]`-shader families cost 4 B/elem; everything else BF16-verbatim.
/// The vision tower is priced BF16-verbatim whole — its F32 norm/bias delta
/// is under 4 MiB, inside the table's rounding.
#[test]
fn s7_scenario_table_dense_bf16_vs_q8_0() {
    let dir = checkpoint_dir();
    if !dir.is_dir() {
        eprintln!("SKIP: no checkpoint at {} (set {CKPT_ENV})", dir.display());
        return;
    }
    let Ok(st) = SafeTensorsDir::open_dir(&dir) else {
        eprintln!("SKIP: cannot open {} as a safetensors dir", dir.display());
        return;
    };

    // The brief's reserve: 1.5 GiB held back for KV + recurrent state +
    // activations (the MTP layer's own KV rides in the same reserve).
    assert_eq!(DEFAULT_RESERVE_BYTES, 3 << 29, "the 1.5 GiB reserve moved");
    let usable = BUDGET_BYTES - DEFAULT_RESERVE_BYTES;

    // ---- text stream: the planner's own number, then the Q8_0 delta -------
    let cfg = Qwen4UploadConfig::default();
    let plan =
        plan_qwen4_upload(&st, &cfg, &Qwen4UploadScope::full()).expect("plan the text stream");
    let mut text_dense_bf16 = 0u64;
    let mut text_dense_q8 = 0u64;
    for item in &plan.items {
        if item.format == Qwen4DeviceFormat::Bf16 && takes_dense_format(item.role.kind) {
            assert_eq!(
                item.bytes,
                (item.ncols * item.nrows) as u64 * 2,
                "`{}` is priced as plain BF16 rows",
                item.name
            );
            let q8 = q8_0_bytes(item.ncols as u64, item.nrows as u64).unwrap_or_else(|| {
                panic!(
                    "`{}`: ncols {} is not whole Q8_0 blocks",
                    item.name, item.ncols
                )
            });
            text_dense_bf16 += item.bytes;
            text_dense_q8 += q8;
        }
    }
    let text_savings = text_dense_bf16 - text_dense_q8;
    // The lever the Q8_0 sibling is building, sized exactly: 6.702 GiB of
    // dense-GEMV BF16 becomes 3.560 GiB of Q8_0 (x 17/32). If the dense tier's
    // membership or default format moves, this is where it surfaces.
    assert_eq!(text_dense_bf16, 7_195_852_800, "dense GEMV tier moved");
    assert_eq!(
        text_savings, 3_373_056_000,
        "Q8_0 savings moved (3.141 GiB)"
    );

    // ---- MTP + vision additions, priced per the rules above ---------------
    let mut mtp_file = 0u64;
    let mut vis_file = 0u64;
    let mut mtp_a = 0u64; // scenario (a): dense BF16
    let mut mtp_b = 0u64; // scenario (b): dense Q8_0, stacks kept BF16
    let mut mtp_stacks_bf16 = 0u64;
    let mut mtp_stacks_q8 = 0u64; // informational further lever
    for info in st.tensors() {
        let role = classify_qwen4_tensor(&info.name)
            .unwrap_or_else(|e| panic!("classify `{}`: {e}", info.name));
        match role.stream {
            Qwen4Stream::Text => {}
            Qwen4Stream::Vision => vis_file += info.len,
            Qwen4Stream::Mtp => {
                mtp_file += info.len;
                // dims are reversed, so dims[0] is the contraction width.
                let ncols = info.dims.first().copied().unwrap_or(1);
                let nrows = info.element_count() / ncols;
                let stacked = matches!(
                    role.kind,
                    Qwen4TensorKind::ExpertsStackedGateUp | Qwen4TensorKind::ExpertsStackedDown
                );
                let (a, b) = if f32_on_device(role.kind) {
                    (info.len * 2, info.len * 2)
                } else if takes_dense_format(role.kind) {
                    let q8 = q8_0_bytes(ncols, nrows).unwrap_or_else(|| {
                        panic!("`{}`: ncols {ncols} is not whole Q8_0 blocks", info.name)
                    });
                    (info.len, q8)
                } else {
                    (info.len, info.len)
                };
                mtp_a += a;
                mtp_b += b;
                if stacked {
                    mtp_stacks_bf16 += info.len;
                    mtp_stacks_q8 += q8_0_bytes(ncols, nrows)
                        .unwrap_or_else(|| panic!("`{}`: ragged stack rows", info.name));
                }
            }
        }
    }

    // The shipping planner still books these very bytes as dropped; when the
    // integration flips its stream gate, this equality is what guarantees no
    // bytes appear or vanish in the handover.
    assert_eq!(
        plan.dropped_bytes,
        mtp_file + vis_file,
        "planner dropped_bytes must equal the measured MTP + vision file bytes"
    );
    // The Drop tier the user vetoed, measured piece by piece (5.692 GiB
    // total). These sums are deterministic functions of the checkpoint and
    // THIS file's pricing rules, so they pin exactly; a change in either is a
    // failing diff by design.
    assert_eq!(mtp_file, 5_214_301_696, "MTP file bytes moved (4.856 GiB)");
    assert_eq!(vis_file, 897_862_112, "vision file bytes moved (0.836 GiB)");
    assert_eq!(
        mtp_stacks_bf16, 5_033_164_800,
        "MTP expert stacks are 4.6875 GiB"
    );
    // mtp_a = file + 2.6 MiB of F32 doubling; mtp_b additionally converts the
    // 132.5 MiB of MTP dense-GEMV weights (q/k/v/o, indexer, shared expert,
    // the two fusion fcs) to Q8_0.
    assert_eq!(mtp_a, 5_217_016_832, "scenario (a) MTP pricing moved");
    assert_eq!(mtp_b, 5_151_890_432, "scenario (b) MTP pricing moved");

    let a_total = plan.device_bytes + mtp_a + vis_file;
    let b_total = plan.device_bytes - text_savings + mtp_b + vis_file;
    let b_prime = b_total - mtp_stacks_bf16 + mtp_stacks_q8;
    let over = |total: u64| total.saturating_sub(usable);

    println!();
    println!("=== S7 scenario table: qwen4_exp full plan incl. MTP + vision ===");
    println!(
        "budget {:.3} GiB - reserve {:.3} GiB = usable {:.3} GiB (suballocation floors, pre-slab-tails)",
        BUDGET_BYTES as f64 / GIB,
        DEFAULT_RESERVE_BYTES as f64 / GIB,
        usable as f64 / GIB,
    );
    println!(
        "  text stream (planner, dense BF16 + F32 tier)  : {:>7.3} GiB",
        plan.device_bytes as f64 / GIB
    );
    println!(
        "  Q8_0 over the dense GEMV tier ({:.3} GiB BF16) : {:>7.3} GiB saved",
        text_dense_bf16 as f64 / GIB,
        text_savings as f64 / GIB
    );
    println!(
        "  MTP additions   (a) {:>6.3} GiB / (b) {:>6.3} GiB  [stacks {:.3} GiB stay BF16 in (b)]",
        mtp_a as f64 / GIB,
        mtp_b as f64 / GIB,
        mtp_stacks_bf16 as f64 / GIB
    );
    println!(
        "  vision additions    {:>6.3} GiB (both)",
        vis_file as f64 / GIB
    );
    for (label, total) in [
        ("(a) dense BF16 + MTP + vision", a_total),
        ("(b) dense Q8_0 + MTP + vision", b_total),
    ] {
        println!(
            "  {label}: {:>7.3} GiB -> over usable by {:>6.3} GiB -> SPILL {:>6.3} GiB",
            total as f64 / GIB,
            over(total) as f64 / GIB,
            over(total) as f64 / GIB,
        );
    }
    println!(
        "  spill victims, coldest first: MTP expert stacks {:.3} GiB (10/512 slices, only when \
         speculating), vision tower {:.3} GiB (image prefill only), remainder {:.3} GiB from the \
         coldest NVFP4 stacks",
        mtp_stacks_bf16 as f64 / GIB,
        vis_file as f64 / GIB,
        over(b_total).saturating_sub(mtp_stacks_bf16 + vis_file) as f64 / GIB,
    );
    println!(
        "  further lever, not assumed: MTP stacks at Q8_0 too -> {:.3} GiB total, SPILL {:.3} GiB",
        b_prime as f64 / GIB,
        over(b_prime) as f64 / GIB,
    );
    println!();

    // ---- the verdicts the integration plans against -----------------------
    // Everything above `plan.device_bytes` is exact-pinned, so the only slack
    // these windows absorb is planner churn (its F32 tier, item pricing).
    // Measured this sitting (2026-08-28, checkpoint 206 shards, no GPU
    // involved): over_a 7.936 GiB, over_b 4.734 GiB.
    //
    // (a) does NOT fit device-resident — over by ~7.9 GiB, most of a full
    // heap-0 spill budget. This is the answer to "can we have dense BF16 and
    // the MTP back": only with 7.9 GiB streaming from the host heap.
    let over_a = over(a_total) as f64 / GIB;
    assert!(
        (7.4..8.5).contains(&over_a),
        "(a) must overshoot usable by ~7.9 GiB, got {over_a:.3} GiB over",
    );
    // (b) does not fit either — Q8_0 buys back 3.14 GiB but not the MTP mass —
    // and must land exactly the Q8_0 deltas under (a).
    let over_b = over(b_total) as f64 / GIB;
    assert!(
        (4.2..5.3).contains(&over_b),
        "(b) must overshoot usable by ~4.7 GiB, got {over_b:.3} GiB over",
    );
    assert_eq!(
        a_total - b_total,
        text_savings + (mtp_a - mtp_b),
        "the two scenarios differ by exactly the Q8_0 deltas"
    );
    // The invariant that makes (b) the plan worth shipping: the newly admitted
    // cold mass ALONE covers its spill — demote the MTP stacks and the vision
    // tower and every per-token-hot byte stays device-resident.
    assert!(
        mtp_stacks_bf16 + vis_file >= over(b_total),
        "the MTP stacks + vision ({:.3} GiB) no longer cover (b)'s {:.3} GiB spill — \
         something hot would have to move to the host heap",
        (mtp_stacks_bf16 + vis_file) as f64 / GIB,
        over_b,
    );
}
