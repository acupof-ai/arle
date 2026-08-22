//! Resident DSv4-Flash A/B harness.
//!
//! Loads the TP=8/EP=8 DSv4 executor once, then runs multiple decode variants in
//! the same process. This removes the 149GB model reload from every fused-WQKV
//! on/off comparison and makes kernel A/B loops seconds-scale after load.
//!
//! Env:
//!   * `INFER_DSV4_MODEL_PATH`       required DSv4 FP8 safetensors dir.
//!   * `INFER_DSV4_PROMPT_IDS`       comma-separated DeepSeek ids.
//!   * `INFER_DSV4_AB_VARIANTS`      `baseline,fused_wqkv` by default.
//!   * `INFER_DSV4_AB_MAX_NEW`       generated-token count, default 128.
//!   * `INFER_DSV4_AB_WARMUP_NEW`    decode steps excluded from steady timing,
//!     default 16.
//!   * `INFER_DSV4_AB_REPEAT`        repeat the variant list after one load,
//!     default 1.
//!   * `INFER_DSV4_AB_PROFILE_VARIANT` optional variant name; starts CUDA
//!     profiler after warmup for ncu/nsys attach.

#![allow(clippy::print_stdout, clippy::print_stderr)]

fn main() -> anyhow::Result<()> {
    real::run()
}

#[cfg(not(feature = "cuda"))]
mod real {
    pub(super) fn run() -> anyhow::Result<()> {
        eprintln!(
            "dsv4_resident_ab is a CUDA harness; rebuild with \
             --features cuda (single-rank) or --features cuda,nccl (multi-rank \
             NCCL file-rendezvous via INFER_NCCL_ID_FILE)."
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Context, Result, bail};
    use infer_cuda::{
        CudaExecutor, CudaKvPool, print_dsv4_linear_profile, print_dsv4_stage_profile,
        reset_dsv4_linear_profile, reset_dsv4_stage_profile, set_dsv4_fused_wqkv_decode_override,
        set_dsv4_stage_profile_active,
    };
    use infer_plan::{ForwardMode, ForwardPlan, PrefillRow, SamplingParams};
    use infer_seam::{BackendExecutor, KvAllocator, KvPool, KvQuery, PollResult};
    use std::time::Instant;

    const DEFAULT_PROMPT_IDS: &str = "671,6102,294,8760,344";
    const DEFAULT_MAX_NEW: usize = 128;
    /// Design-max KV arena for this A/B harness — length-agnostic (handles any
    /// prompt up to this without per-test sizing), not a tunable knob.
    const AB_MAX_SEQ_LEN: usize = 32768;
    const DEFAULT_WARMUP_NEW: usize = 16;
    const ORACLE_16: [u32; 16] = [
        11111, 603, 671, 6102, 294, 8760, 344, 11111, 603, 671, 6102, 294, 8760, 344, 11111, 603,
    ];

    unsafe extern "C" {
        fn cudaProfilerStart() -> i32;
        fn cudaProfilerStop() -> i32;
    }

    #[derive(Clone, Copy, Debug)]
    struct Variant {
        name: &'static str,
        fused_wqkv: bool,
    }

    #[derive(Debug)]
    struct VariantResult {
        name: &'static str,
        fused_wqkv: bool,
        tokens: Vec<u32>,
        prefill_ms: f64,
        decode_ms: f64,
        timed_decode_ms: f64,
        decode_steps: usize,
        timed_decode_steps: usize,
        bail_at: Option<(usize, String)>,
    }

    pub(super) fn run() -> Result<()> {
        resident_ab()
    }

    #[cfg(feature = "nccl")]
    fn mint_nccl_id_hex() -> Result<String> {
        use cuda_kernels::ffi::nccl;

        let mut id = nccl::ncclUniqueId {
            internal: [0i8; 128],
        };
        // SAFETY: `id` is a valid NCCL handle destination.
        let res = unsafe { nccl::ncclGetUniqueId(&mut id) };
        nccl::check(res).context("ncclGetUniqueId failed")?;

        let mut hex = String::with_capacity(256);
        for &b in &id.internal {
            use std::fmt::Write;
            write!(hex, "{:02x}", b as u8).expect("write to String is infallible");
        }
        Ok(hex)
    }

    #[cfg(feature = "nccl")]
    fn export_nccl_id(hex: &str) {
        // SAFETY: called at process startup before the executor creates threads
        // or NCCL state.
        unsafe { std::env::set_var("INFER_NCCL_UNIQUE_ID", hex) };
    }

    #[cfg(feature = "nccl")]
    fn nccl_file_rendezvous(rank: usize, id_file: &std::path::Path) -> Result<()> {
        use std::time::{Duration, Instant};

        if rank == 0 {
            let hex = mint_nccl_id_hex()?;
            let tmp = id_file.with_extension("hex.tmp");
            std::fs::write(&tmp, &hex).with_context(|| format!("write NCCL id to {tmp:?}"))?;
            std::fs::rename(&tmp, id_file)
                .with_context(|| format!("rename NCCL id -> {id_file:?}"))?;
            export_nccl_id(&hex);
            eprintln!("[dsv4-ab rank=0] minted NCCL id -> {id_file:?}");
        } else {
            let deadline = Instant::now() + Duration::from_secs(120);
            loop {
                if let Ok(s) = std::fs::read_to_string(id_file) {
                    let s = s.trim();
                    if s.len() == 256 {
                        export_nccl_id(s);
                        break;
                    }
                }
                anyhow::ensure!(
                    Instant::now() < deadline,
                    "rank {rank} timed out waiting for NCCL id at {id_file:?}"
                );
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "nccl"))]
    fn nccl_file_rendezvous(_rank: usize, _id_file: &std::path::Path) -> Result<()> {
        bail!("INFER_NCCL_ID_FILE needs the nccl feature; rebuild with --features cuda,nccl")
    }

    fn resident_ab() -> Result<()> {
        let model_path = std::env::var("INFER_DSV4_MODEL_PATH")
            .context("INFER_DSV4_MODEL_PATH must point at the DSv4 FP8 safetensors directory")?;
        let prompt = parse_prompt_ids(
            &std::env::var("INFER_DSV4_PROMPT_IDS")
                .unwrap_or_else(|_| DEFAULT_PROMPT_IDS.to_string()),
        )?;
        anyhow::ensure!(!prompt.is_empty(), "INFER_DSV4_PROMPT_IDS resolved empty");

        let max_new = parse_usize_env("INFER_DSV4_AB_MAX_NEW", DEFAULT_MAX_NEW)?;
        let warmup_new = parse_usize_env("INFER_DSV4_AB_WARMUP_NEW", DEFAULT_WARMUP_NEW)?;
        let repeat = parse_usize_env("INFER_DSV4_AB_REPEAT", 1)?;
        anyhow::ensure!(max_new >= 1, "INFER_DSV4_AB_MAX_NEW must be >= 1");
        anyhow::ensure!(repeat >= 1, "INFER_DSV4_AB_REPEAT must be >= 1");
        let variants = parse_variants(
            &std::env::var("INFER_DSV4_AB_VARIANTS")
                .unwrap_or_else(|_| "baseline,fused_wqkv".to_string()),
        )?;
        let rank = parse_usize_env("INFER_TP_RANK", 0)?;
        let profile_variant = std::env::var("INFER_DSV4_AB_PROFILE_VARIANT").ok();

        if let Ok(id_file) = std::env::var("INFER_NCCL_ID_FILE") {
            nccl_file_rendezvous(rank, std::path::Path::new(&id_file))?;
        }

        let prompt_head: Vec<u32> = prompt.iter().copied().take(16).collect();
        eprintln!(
            "[dsv4-ab rank={rank}] model={model_path} prompt_len={} \
             prompt_head={prompt_head:?} max_new={max_new} warmup_new={warmup_new} \
             repeat={repeat} variants={}",
            prompt.len(),
            variants
                .iter()
                .map(|v| v.name)
                .collect::<Vec<_>>()
                .join(",")
        );

        let load_t0 = Instant::now();
        let mut exec = CudaExecutor::from_dsv4_fp8_safetensors(
            &model_path,
            1,
            AB_MAX_SEQ_LEN,
            None,
            None,
            None,
            0.5,
            0.0,
        )
        .context("from_dsv4_fp8_safetensors failed")?;
        let load_ms = load_t0.elapsed().as_secs_f64() * 1000.0;

        for rep in 0..repeat {
            let mut results = Vec::with_capacity(variants.len());
            for &variant in &variants {
                results.push(run_variant(
                    &mut exec,
                    &prompt,
                    variant,
                    max_new,
                    warmup_new,
                    profile_variant.as_deref(),
                )?);
            }
            let scalar_tokens = results
                .iter()
                .find(|result| result.name == "scalar")
                .map(|result| result.tokens.as_slice());
            if rank == 0 {
                for result in &results {
                    let oracle16 = compare_prefix(&result.tokens, &ORACLE_16);
                    let scalar_ref = scalar_ref_text(scalar_tokens, result);
                    print_rank0_result(rep, load_ms, warmup_new, result, oracle16, &scalar_ref);
                }
            }
        }
        set_dsv4_fused_wqkv_decode_override(None);
        Ok(())
    }

    fn run_variant(
        exec: &mut CudaExecutor,
        prompt: &[u32],
        variant: Variant,
        max_new: usize,
        warmup_new: usize,
        profile_variant: Option<&str>,
    ) -> Result<VariantResult> {
        set_dsv4_fused_wqkv_decode_override(Some(variant.fused_wqkv));
        set_env_var("INFER_DSV4_AB_CURRENT_VARIANT", variant.name);
        reset_dsv4_linear_profile();
        reset_dsv4_stage_profile();

        // DSv4 does not use the host page pool for real KV, but the seam's
        // KvBatchDescriptor check reads kv.seq_len(slot) (set by materialize_plan_kv
        // below), so size the host pool to the executor's DESIGN max context
        // (AB_MAX_SEQ_LEN) — length-agnostic, handles any prompt up to the max
        // without per-test sizing. Recreated per variant for clean bookkeeping.
        let page_size = 16usize;
        let kv_pages = AB_MAX_SEQ_LEN.div_ceil(page_size) + 1;
        let mut kv = CudaKvPool::new(1, kv_pages, page_size);

        let prefill_t0 = Instant::now();
        let first = forward_once(exec, &mut kv, prefill_plan(prompt, 0))?;
        let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1000.0;
        let mut tokens = vec![first];

        let profile_this = profile_variant == Some(variant.name);
        let stage_profile_this = std::env::var_os("ARLE_DSV4_STAGE_PROFILE").is_some()
            && (profile_variant.is_none() || profile_this);
        let warmup_decode_steps = warmup_new.min(max_new.saturating_sub(1));
        let mut bail_at = None;
        let mut profiler_on = false;
        let decode_t0 = Instant::now();
        let mut timed_t0: Option<Instant> = None;

        // Spec-aware: MTP emits 1-2 tokens/call (accepted base_next + bonus) and the
        // executor self-manages slot KV, so loop on emitted TOKEN count (not calls)
        // and feed the LAST emitted token (= executor `pending`) each step. For
        // non-spec this degenerates to 1 token/call == the original loop.
        let mut timed_token_start = 0usize;
        let mut calls = 0usize;
        while tokens.len() < max_new {
            if timed_t0.is_none() && calls >= warmup_decode_steps {
                if profile_this {
                    cuda_profiler_start()
                        .with_context(|| format!("cudaProfilerStart for {}", variant.name))?;
                    profiler_on = true;
                }
                set_dsv4_stage_profile_active(stage_profile_this);
                timed_t0 = Some(Instant::now());
                timed_token_start = tokens.len();
            }

            let kv_seq_len = prompt.len() + tokens.len() - 1;
            let last = *tokens.last().expect("tokens is non-empty");
            let new = match forward_multi(exec, &mut kv, decode_plan(last, kv_seq_len)) {
                Ok(new) => new,
                Err(e) => {
                    bail_at = Some((calls + 1, format!("{e:#}")));
                    break;
                }
            };
            // MTP emitted new.len() tokens but materialize_plan_kv alloc'd only 1
            // (the decode row); advance the host pool by the rest so the next
            // step's kv_seq_len matches the executor's actual KV.
            for _ in 1..new.len() {
                kv.alloc(0, 1).context("DSv4 MTP host-pool sync alloc")?;
            }
            tokens.extend(new);
            calls += 1;
        }

        let decode_ms = decode_t0.elapsed().as_secs_f64() * 1000.0;
        let timed_decode_ms = timed_t0
            .map(|t0| t0.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        if profiler_on {
            cuda_profiler_stop()
                .with_context(|| format!("cudaProfilerStop for {}", variant.name))?;
        }
        set_dsv4_stage_profile_active(false);
        print_dsv4_linear_profile(variant.name);
        // decode_steps / timed_decode_steps are EMITTED TOKEN counts so tok/s is
        // tokens/sec for both non-spec (1/call) and MTP (1-2/call).
        let decode_steps = tokens.len().saturating_sub(1);
        let timed_decode_steps = if timed_t0.is_some() {
            tokens.len().saturating_sub(timed_token_start)
        } else {
            0
        };
        print_dsv4_stage_profile(variant.name, timed_decode_steps, timed_decode_ms);

        Ok(VariantResult {
            name: variant.name,
            fused_wqkv: variant.fused_wqkv,
            tokens,
            prefill_ms,
            decode_ms,
            timed_decode_ms,
            decode_steps,
            timed_decode_steps,
            bail_at,
        })
    }

    fn print_rank0_result(
        rep: usize,
        load_ms: f64,
        warmup_new: usize,
        result: &VariantResult,
        oracle16: Option<usize>,
        scalar_ref_text: &str,
    ) {
        let decode_tok_s = if result.decode_ms > 0.0 {
            (result.decode_steps as f64) / (result.decode_ms / 1000.0)
        } else {
            0.0
        };
        let steady_tok_s = if result.timed_decode_ms > 0.0 {
            (result.timed_decode_steps as f64) / (result.timed_decode_ms / 1000.0)
        } else {
            0.0
        };
        let oracle16_text = match oracle16 {
            None => "PASS".to_string(),
            Some(idx) => format!("FAIL@{idx}"),
        };
        println!(
            "ab_variant={} rep={} fused_wqkv={}  tokens={:?} oracle16={} scalar_ref={} \
             load_ms={:.3} prefill_ms={:.3} decode_steps={} decode_ms={:.3} \
             decode_tok_s={:.3} warmup_decode_steps={} timed_decode_steps={} \
             timed_decode_ms={:.3} steady_tok_s={:.3}",
            result.name,
            rep,
            u8::from(result.fused_wqkv),
            result.tokens,
            oracle16_text,
            scalar_ref_text,
            load_ms,
            result.prefill_ms,
            result.decode_steps,
            result.decode_ms,
            decode_tok_s,
            warmup_new,
            result.timed_decode_steps,
            result.timed_decode_ms,
            steady_tok_s,
        );
        if let Some((step, msg)) = &result.bail_at {
            eprintln!(
                "[dsv4-ab rank=0] variant={} bailed at decode step {step}: {msg}",
                result.name
            );
        }
    }

    fn scalar_ref_text(reference: Option<&[u32]>, result: &VariantResult) -> String {
        match reference {
            Some(tokens) => match first_diff(tokens, &result.tokens) {
                None => "MATCH".to_string(),
                Some(idx) => format!("DIFF@{idx}"),
            },
            None if result.name == "scalar" => "SELF".to_string(),
            None => "NOREF".to_string(),
        }
    }

    fn set_env_var(name: &str, value: &str) {
        // SAFETY: this harness mutates env only during process startup before it
        // builds CUDA/NCCL state or spawns threads.
        unsafe { std::env::set_var(name, value) };
    }

    fn cuda_profiler_start() -> Result<()> {
        // SAFETY: CUDA profiler API has process-global side effects only.
        let code = unsafe { cudaProfilerStart() };
        anyhow::ensure!(
            code == 0,
            "cudaProfilerStart returned CUDA error code {code}"
        );
        Ok(())
    }

    fn cuda_profiler_stop() -> Result<()> {
        // SAFETY: pairs with `cuda_profiler_start`.
        let code = unsafe { cudaProfilerStop() };
        anyhow::ensure!(
            code == 0,
            "cudaProfilerStop returned CUDA error code {code}"
        );
        Ok(())
    }

    fn prefill_plan(tokens: &[u32], start_pos: usize) -> ForwardPlan {
        ForwardPlan {
            mode: ForwardMode::Prefill,
            decode_rows: Vec::new(),
            prefill_rows: vec![PrefillRow {
                slot: 0,
                tokens: tokens.to_vec(),
                start_pos,
                total_tokens: start_pos + tokens.len(),
                params: greedy(),
                penalty_history: None,
                penalty_prompt_len: 0,
            }],
        }
    }

    fn decode_plan(last_token: u32, kv_seq_len: usize) -> ForwardPlan {
        ForwardPlan {
            mode: ForwardMode::Decode,
            decode_rows: vec![infer_plan::DecodeRow {
                slot: 0,
                last_token,
                kv_seq_len,
                params: greedy(),
                penalty_history: None,
                penalty_prompt_len: 0,
            }],
            prefill_rows: Vec::new(),
        }
    }

    /// Materialize the host KV pool (set per-slot seq_len) for `plan` before
    /// submit — the seam's KvBatchDescriptor check reads `kv.seq_len(slot)`.
    /// Mirrors dsv4_parity. DSv4's real KV lives in the executor adapter; this is
    /// the seam's logical bookkeeping only.
    fn materialize_plan_kv(kv: &mut CudaKvPool, plan: &ForwardPlan) -> Result<()> {
        for row in &plan.prefill_rows {
            if row.start_pos == 0 {
                kv.free_slot(row.slot);
            }
            anyhow::ensure!(
                kv.seq_len(row.slot) == row.start_pos,
                "host KV len {} != prefill start_pos {} for slot {}",
                kv.seq_len(row.slot),
                row.start_pos,
                row.slot
            );
            kv.alloc(row.slot, row.tokens.len())?;
        }
        for row in &plan.decode_rows {
            anyhow::ensure!(
                kv.seq_len(row.slot) == row.kv_seq_len,
                "host KV len {} != decode kv_seq_len {} for slot {}",
                kv.seq_len(row.slot),
                row.kv_seq_len,
                row.slot
            );
            kv.alloc(row.slot, 1)?;
        }
        Ok(())
    }

    fn forward_once(
        exec: &mut CudaExecutor,
        kv: &mut CudaKvPool,
        plan: ForwardPlan,
    ) -> Result<u32> {
        materialize_plan_kv(kv, &plan)?;
        let inflight = exec.submit(&plan, kv as &mut dyn KvPool)?;
        match exec.poll(inflight)? {
            PollResult::Ready(out) => out
                .tokens
                .first()
                .map(|t| t.token)
                .context("DSv4 step produced no token"),
            PollResult::NotReady(_) => bail!("DSv4 executor resolves synchronously; got NotReady"),
        }
    }

    /// Like [`forward_once`] but returns ALL tokens the step emitted. MTP spec
    /// decode emits 1 (reject) or 2 (accepted base_next + bonus) and sets the
    /// executor's `pending` to the LAST emitted token, so the spec-aware decode
    /// loop must push every returned token and feed the last as the next input.
    fn forward_multi(
        exec: &mut CudaExecutor,
        kv: &mut CudaKvPool,
        plan: ForwardPlan,
    ) -> Result<Vec<u32>> {
        materialize_plan_kv(kv, &plan)?;
        let inflight = exec.submit(&plan, kv as &mut dyn KvPool)?;
        match exec.poll(inflight)? {
            PollResult::Ready(out) => {
                let toks: Vec<u32> = out.tokens.iter().map(|t| t.token).collect();
                anyhow::ensure!(!toks.is_empty(), "DSv4 step produced no token");
                Ok(toks)
            }
            PollResult::NotReady(_) => bail!("DSv4 executor resolves synchronously; got NotReady"),
        }
    }

    fn greedy() -> SamplingParams {
        SamplingParams::default()
    }

    fn parse_prompt_ids(s: &str) -> Result<Vec<u32>> {
        s.split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(|t| {
                t.parse::<u32>()
                    .with_context(|| format!("bad token id `{t}` in INFER_DSV4_PROMPT_IDS"))
            })
            .collect()
    }

    fn parse_usize_env(name: &str, default: usize) -> Result<usize> {
        match std::env::var(name) {
            Ok(raw) => raw
                .parse::<usize>()
                .with_context(|| format!("{name} must be usize, got `{raw}`")),
            Err(std::env::VarError::NotPresent) => Ok(default),
            Err(e) => bail!("{name} invalid env: {e}"),
        }
    }

    fn parse_variants(raw: &str) -> Result<Vec<Variant>> {
        let mut variants = Vec::new();
        for item in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match item {
                "baseline" | "scalar" | "bf16" | "flashmla" | "flash" => variants.push(Variant {
                    name: "baseline",
                    fused_wqkv: false,
                }),
                "fused_wqkv"
                | "fused_linear"
                | "flashmla_fused"
                | "flashmla_fused_wqkv"
                | "scalar_fused_wqkv" => variants.push(Variant {
                    name: "fused_wqkv",
                    fused_wqkv: true,
                }),
                other => bail!(
                    "unsupported INFER_DSV4_AB_VARIANTS item `{other}` \
                     (expected baseline, fused_wqkv)"
                ),
            }
        }
        anyhow::ensure!(
            !variants.is_empty(),
            "INFER_DSV4_AB_VARIANTS resolved empty"
        );
        Ok(variants)
    }

    fn compare_prefix(got: &[u32], expected: &[u32]) -> Option<usize> {
        let n = got.len().min(expected.len());
        for i in 0..n {
            if got[i] != expected[i] {
                return Some(i);
            }
        }
        if got.len() < expected.len() {
            Some(got.len())
        } else {
            None
        }
    }

    fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
        let n = a.len().min(b.len());
        for i in 0..n {
            if a[i] != b[i] {
                return Some(i);
            }
        }
        if a.len() == b.len() { None } else { Some(n) }
    }
}
