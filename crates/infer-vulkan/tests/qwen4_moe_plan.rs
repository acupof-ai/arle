//! THE plan-equality gate for the fence-free grouped-MoE prefill: the device
//! planner (`Qwen4MoePlanCount` -> `Scan` -> `Emit`) must produce, ELEMENT
//! FOR ELEMENT, the same structures the host oracle `plan_moe_groups` builds
//! for the same ids — the scatter map, the per-class block counts, the live
//! expert-id list prefixes and the `VkDispatchIndirectCommand` triples.
//! Same plan + same record-time binds = bitwise identical GEMV work, which is
//! what lets the bit-exact prefill=decode gate keep holding 0.000e0 with no
//! ids fence.
//!
//! Two id sources:
//! - randomized rosters over the shapes that stress every planner branch
//!   (needs only a Vulkan device, no checkpoint);
//! - REAL router outputs, captured through the `moe_ids` diagnostic fence on
//!   the truncated 4-layer SubsetF32 fixture (needs `ARLE_QWEN4_CKPT`).
//!
//! Device tests — run with `--test-threads=1`.

#![cfg(feature = "vulkan")]

use std::path::PathBuf;

use infer_gguf::safetensors::SafeTensorsDir;
use infer_vulkan::model_qwen4_exp::{
    MoeGroupPlan, MoePlanLayout, Qwen4ExpDeviceMode, VulkanQwen4ExpModel, moe_ids, plan_moe_groups,
};
use infer_vulkan::qwen4_config::Qwen4ExpConfig;
use vulkan_kernels::{
    Kernel, KernelCache, launch_cached, qwen4_moe_plan_args_down_at, qwen4_moe_plan_args_gateup_at,
    qwen4_moe_plan_count_dispatch, qwen4_moe_plan_count_params, qwen4_moe_plan_emit_dispatch,
    qwen4_moe_plan_emit_params, qwen4_moe_plan_nblk_at, qwen4_moe_plan_scan_dispatch,
    qwen4_moe_plan_scan_params, qwen4_moe_plan_scratch_elems,
};
use vulkan_sys::{DeviceBuffer, VulkanContext};

const CKPT_ENV: &str = "ARLE_QWEN4_CKPT";
const CKPT_DEFAULT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";
/// Class cap — must match the model's `PF_MOE_COLS_CAP` and the shaders' CAP.
const COLS_CAP: usize = 8;

fn device() -> Option<VulkanContext> {
    match VulkanContext::create() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIP: no Vulkan device ({e})");
            None
        }
    }
}

/// Run the three planner kernels on `raw_ids` and read the products back.
/// Each `launch_cached` is its own submit-and-wait, so no barriers are needed
/// between the passes here (the real record path uses in-buffer barriers).
struct DevicePlan {
    scatter: Vec<u32>,
    ids: Vec<i32>,
    n_blocks: Vec<u32>,
    /// `(gate/up triple, down triple)` per class, widest-first.
    args: Vec<([u32; 3], [u32; 3])>,
}

fn run_device_planner(
    ctx: &VulkanContext,
    raw_ids: &[i32],
    t: usize,
    ids_stride: usize,
    top_k: usize,
    n_experts: usize,
    x_gateup: u32,
    x_down: u32,
) -> DevicePlan {
    let pairs = t * top_k;
    let layout = MoePlanLayout::new(pairs, n_experts);
    let e32 = u32::try_from(n_experts).expect("n_experts fits u32");
    let scratch = qwen4_moe_plan_scratch_elems(e32) as usize;

    let le = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut ids_buf = DeviceBuffer::alloc_host_cached(ctx, raw_ids.len() * 4).expect("ids buf");
    ids_buf.copy_from_host(&le(raw_ids)).expect("ids upload");
    // Poison the outputs so a lane the emit pass fails to write is caught as
    // an inequality, not mistaken for a correct zero.
    let mut plan_buf = DeviceBuffer::alloc_host_cached(ctx, scratch * 4).expect("plan buf");
    plan_buf
        .copy_from_host(&vec![0xABu8; scratch * 4])
        .expect("plan poison");
    let mut scat_buf = DeviceBuffer::alloc_host_cached(ctx, pairs * 4).expect("scatter buf");
    scat_buf
        .copy_from_host(&vec![0xCDu8; pairs * 4])
        .expect("scatter poison");
    let list_len = layout.list_capacity.max(1);
    let mut list_buf = DeviceBuffer::alloc_host_cached(ctx, list_len * 4).expect("list buf");
    list_buf
        .copy_from_host(&vec![0xEFu8; list_len * 4])
        .expect("list poison");

    let mut cache = KernelCache::new();
    let push = qwen4_moe_plan_count_params(pairs as u32, top_k as u32, ids_stride as u32, e32);
    launch_cached(
        &mut cache,
        ctx,
        Kernel::Qwen4MoePlanCount,
        &[&ids_buf, &plan_buf],
        qwen4_moe_plan_count_dispatch(),
        &push.to_le_bytes(),
        Kernel::Qwen4MoePlanCount.specialization_u32(),
    )
    .expect("count pass");
    let push = qwen4_moe_plan_scan_params(e32, x_gateup, x_down);
    launch_cached(
        &mut cache,
        ctx,
        Kernel::Qwen4MoePlanScan,
        &[&plan_buf],
        qwen4_moe_plan_scan_dispatch(),
        &push.to_le_bytes(),
        Kernel::Qwen4MoePlanScan.specialization_u32(),
    )
    .expect("scan pass");
    let mut pair_base = [0u32; 8];
    let mut list_base = [0u32; 8];
    for w in 1..=COLS_CAP {
        pair_base[w - 1] = u32::try_from(layout.pair_base(w)).expect("pair base");
        list_base[w - 1] = u32::try_from(layout.list_base(w)).expect("list base");
    }
    let push = qwen4_moe_plan_emit_params(
        pairs as u32,
        top_k as u32,
        ids_stride as u32,
        e32,
        pair_base,
        list_base,
    );
    launch_cached(
        &mut cache,
        ctx,
        Kernel::Qwen4MoePlanEmit,
        &[&ids_buf, &plan_buf, &scat_buf, &list_buf],
        qwen4_moe_plan_emit_dispatch(pairs as u32, e32),
        &push.to_le_bytes(),
        Kernel::Qwen4MoePlanEmit.specialization_u32(),
    )
    .expect("emit pass");

    let read_u32 = |buf: &DeviceBuffer<'_>, n: usize| -> Vec<u32> {
        let mut bytes = vec![0u8; n * 4];
        buf.copy_to_host(&mut bytes).expect("readback");
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let scratch_words = read_u32(&plan_buf, scratch);
    let nblk_at = qwen4_moe_plan_nblk_at(e32) as usize;
    let mut args = Vec::new();
    for w in (1..=COLS_CAP).rev() {
        let gu = qwen4_moe_plan_args_gateup_at(e32, w as u32) as usize;
        let dn = qwen4_moe_plan_args_down_at(e32, w as u32) as usize;
        args.push((
            [
                scratch_words[gu],
                scratch_words[gu + 1],
                scratch_words[gu + 2],
            ],
            [
                scratch_words[dn],
                scratch_words[dn + 1],
                scratch_words[dn + 2],
            ],
        ));
    }
    DevicePlan {
        scatter: read_u32(&scat_buf, pairs),
        ids: read_u32(&list_buf, list_len)
            .into_iter()
            .map(|v| v as i32)
            .collect(),
        n_blocks: scratch_words[nblk_at..nblk_at + COLS_CAP].to_vec(),
        args,
    }
}

/// Element-for-element comparison against the host oracle. Compared: the
/// whole scatter map, all 8 block counts, each class's LIVE id-list prefix
/// (entries past the count are dead by contract — no dispatch reads them),
/// and both indirect triples per class.
fn assert_plans_equal(
    dev: &DevicePlan,
    host: &MoeGroupPlan,
    x_gateup: u32,
    x_down: u32,
    what: &str,
) {
    assert_eq!(dev.scatter, host.scatter, "{what}: scatter maps diverge");
    for w in 1..=COLS_CAP {
        assert_eq!(
            dev.n_blocks[w - 1] as usize,
            host.n_blocks[w - 1],
            "{what}: class {w} block count"
        );
        let base = host.layout.list_base(w);
        let n = host.n_blocks[w - 1];
        assert_eq!(
            &dev.ids[base..base + n],
            &host.ids[base..base + n],
            "{what}: class {w} id list"
        );
        let (gu, dn) = dev.args[COLS_CAP - w];
        assert_eq!(
            gu,
            [x_gateup, host.n_blocks[w - 1] as u32, 1],
            "{what}: class {w} gate/up indirect args"
        );
        assert_eq!(
            dn,
            [x_down, host.n_blocks[w - 1] as u32, 1],
            "{what}: class {w} down indirect args"
        );
    }
}

/// Distinct-per-token router rosters over shapes that hit every planner
/// branch: chunk tails (t=1), hot experts past the cap, uniform spread, the
/// real chunk shape. Deterministic xorshift so failures reproduce.
fn synthetic_rosters() -> Vec<(Vec<i32>, usize, usize, usize)> {
    let mut out = Vec::new();
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    // (t, top_k, n_experts, hot): `hot` pins slot 0 to expert 5 for a
    // count >> cap ceil-split.
    let shapes = [
        (1usize, 10usize, 512usize, false),
        (3, 4, 32, true),
        (12, 4, 32, true),
        (100, 10, 512, false),
        (256, 10, 512, true),
    ];
    for (t, top_k, n_experts, hot) in shapes {
        let stride = top_k + 3; // a non-trivial stride, unlike the arena's 64
        let mut raw = vec![-7i32; t * stride];
        for tok in 0..t {
            let row = &mut raw[tok * stride..tok * stride + top_k];
            for slot in 0..top_k {
                let mut id = if hot && slot == 0 {
                    5
                } else {
                    (next() % n_experts as u64) as i32
                };
                while row[..slot].contains(&id) {
                    id = (id + 1).rem_euclid(n_experts as i32);
                }
                row[slot] = id;
            }
        }
        out.push((raw, t, top_k, n_experts));
    }
    out
}

/// Gate (a), randomized half: no checkpoint needed.
#[test]
fn device_plan_matches_host_plan_on_randomized_ids() {
    let Some(ctx) = device() else { return };
    let (x_gateup, x_down) = (640u32, 2560u32);
    for (raw, t, top_k, n_experts) in synthetic_rosters() {
        let stride = top_k + 3;
        let host = plan_moe_groups(&raw, t, stride, top_k, n_experts).expect("host plan");
        let dev = run_device_planner(&ctx, &raw, t, stride, top_k, n_experts, x_gateup, x_down);
        assert_plans_equal(
            &dev,
            &host,
            x_gateup,
            x_down,
            &format!("t={t} top_k={top_k} E={n_experts}"),
        );
    }
    eprintln!("device planner == host oracle on all randomized rosters");
}

/// Gate (a), real-router half: the truncated 4-layer SubsetF32 fixture runs a
/// chunked prefill with the `moe_ids` capture on (which routes record_moe
/// through the fenced host path — the capture needs the ids on host anyway),
/// then every captured (layer, chunk) roster goes through BOTH planners.
#[test]
fn device_plan_matches_host_plan_on_real_router_ids() {
    let dir = PathBuf::from(std::env::var(CKPT_ENV).unwrap_or_else(|_| CKPT_DEFAULT.into()));
    if !dir.is_dir() {
        eprintln!(
            "SKIP: checkpoint {} not present ({CKPT_ENV})",
            dir.display()
        );
        return;
    }
    let Some(ctx) = device() else { return };
    let st = SafeTensorsDir::open_dir(&dir).expect("open checkpoint");
    let mut cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("parse config");
    assert!(cfg.num_hidden_layers >= 4);
    cfg.num_hidden_layers = 4;
    cfg.layer_types.truncate(4);
    let mode = Qwen4ExpDeviceMode::SubsetF32(vec![0, 1, 2, 3]);
    let mut model = match VulkanQwen4ExpModel::load(Some(&ctx), &st, cfg.clone(), &mode) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP: subset load failed ({e:#}) — device memory likely contended");
            return;
        }
    };
    let toks: Vec<u32> = (0..24)
        .map(|i| {
            let id = (1009 + i * 37) % (cfg.vocab_size.min(30_000) as u32);
            if cfg.stop_token_ids.contains(&id) {
                id + 1
            } else {
                id
            }
        })
        .collect();
    moe_ids::set_enabled(true);
    let run = model.forward_prompt_chunked(0, &toks, 0, 7);
    moe_ids::set_enabled(false);
    run.expect("chunked prefill");
    let captured = moe_ids::take();
    assert!(!captured.is_empty(), "no MoE layers captured");

    let top_k = cfg.num_experts_per_tok;
    let (x_gateup, x_down) = (cfg.moe_intermediate_size as u32, cfg.hidden_size as u32);
    for (layer, ids) in &captured {
        assert!(ids.len() % top_k == 0);
        let t = ids.len() / top_k;
        // The capture is compact: stride == top_k.
        let host = plan_moe_groups(ids, t, top_k, top_k, cfg.num_experts).expect("host plan");
        let dev = run_device_planner(
            &ctx,
            ids,
            t,
            top_k,
            top_k,
            cfg.num_experts,
            x_gateup,
            x_down,
        );
        assert_plans_equal(
            &dev,
            &host,
            x_gateup,
            x_down,
            &format!("layer {layer} t={t}"),
        );
    }
    eprintln!(
        "device planner == host oracle on {} real router rosters",
        captured.len()
    );
}
