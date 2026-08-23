//! DSv4-Flash multi-GPU (TP=8/EP=8) greedy-parity harness for the R6 clean-CUDA
//! rewrite. Every rank runs the identical single-row DSv4 forward; only rank 0's
//! `clean_tokens` line is the oracle the launcher compares.
//!
//! The forward mirrors the R6 Qwen one-prefill-row harness but for DSv4. It builds
//! `CudaExecutor::from_dsv4_fp8_safetensors(path, 1)` and drives
//! `BackendExecutor::submit` with a single-row `ForwardPlan`:
//!   1. one **full-prefix prefill** at `start_pos = 0` over the whole prompt →
//!      the FIRST generated token (the prefill argmax). Every layer type
//!      (SW / CSA / HCA) recomputes its attention over `[0, len)`, so this is the
//!      runnable correctness shape on every layer today.
//!   2. then incremental decode steps (`start_pos > 0`, a single `[last_token]`
//!      row). DSv4 reallocates its SW ring caches fresh per `forward_tokens`
//!      call (see `dsv4.rs` `alloc_sw_window_caches`), so an incremental step
//!      cannot yet see prior KV. The harness attempts each step, **catches the
//!      bail**, and reports how far it got before printing `clean_tokens=[...]`
//!      (always at least the prefill argmax).
//!
//! ## NCCL bootstrap — file rendezvous (not a throwaway mint)
//!
//! `ncclGetUniqueId` embeds the bootstrap **listener socket of the calling
//! process**. A separate short-lived "mint the id and exit" helper is therefore
//! broken: its listener dies on exit and every rank fails `ncclCommInitRank`
//! with `Connection refused` / NCCL `RemoteError`. Instead, when
//! `INFER_NCCL_ID_FILE` is set, rank 0 mints the id **in this process** (the one
//! that becomes the live NCCL rank 0), writes it atomically to that file, and
//! exports `INFER_NCCL_UNIQUE_ID`; every other rank blocks on the file then
//! exports the same value. The loader's `nccl_unique_id_from_env` then decodes
//! it unchanged.
//!
//! Env (parity mode):
//!   * `INFER_DSV4_MODEL_PATH`  — DSv4 FP8 safetensors dir (required).
//!   * `INFER_DSV4_PROMPT_IDS`  — comma-separated u32 ids; default
//!     `671,6102,294,8760,344` = "The capital of France is" in the DeepSeek-V4
//!     tokenizer (matches the oracle; NOT the Qwen ids).
//!   * `INFER_DSV4_MAX_NEW`     — max new tokens to attempt (default 16).
//!   * `INFER_CUDA_DEVICES` / `INFER_TP_RANK` / `INFER_TP_SIZE` — TP world/rank
//!     (resolved inside the loader's TP runtime; the launcher binds one rank per
//!     GPU).
//!   * `INFER_NCCL_ID_FILE` — shared rendezvous path (multi-rank only). The
//!     launcher points all ranks at one file under its work dir.
//!
//! Build / run (pod, sm_90a, CUDA 12.9):
//!   See `scripts/dsv4_multigpu_parity.sh` for the full 8-rank launch + build
//!   flags (FlashMLA, `ARLE_DEEPEP_DIR`, nccl).

// This example only does anything under `cuda`; the NCCL file-rendezvous
// additionally needs `nccl`. Under a host-only build it is an explanatory no-op
// so `cargo build -p infer-cuda` (default, no-cuda) stays clean.
#![allow(clippy::print_stdout, clippy::print_stderr)]

fn main() -> anyhow::Result<()> {
    real::run()
}

#[cfg(not(feature = "cuda"))]
mod real {
    pub(super) fn run() -> anyhow::Result<()> {
        eprintln!(
            "dsv4_parity is a CUDA harness; rebuild with \
             --features cuda (single-rank) or --features cuda,nccl (multi-rank \
             NCCL file-rendezvous via INFER_NCCL_ID_FILE)."
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Context, Result};
    use infer_cuda::{
        CudaExecutor, CudaKvPool, print_dsv4_stage_profile, reset_dsv4_stage_profile,
    };
    use infer_plan::{DecodeRow, ForwardMode, ForwardPlan, PrefillRow, SamplingParams, SlotToken};
    use infer_seam::{BackendExecutor, KvAllocator, KvPool, KvQuery, PollResult};
    use std::{path::Path, time::Instant};

    /// Default prompt: "The capital of France is" in the **DeepSeek-V4** tokenizer
    /// (vocab 128000). These ids match the captured oracle, whose tokens 3-7 are
    /// exactly [671,6102,294,8760,344] = the model echoing the prompt back. The
    /// model's greedy continuation is [11111,…] = " Paris.\nThe capital of France
    /// is Paris.\n…". (The prior default 785,6722,315,9625,374 were *Qwen* ids —
    /// they decode to garbage " ar造成 thATE v" under DeepSeek, so the rewrite was
    /// being scored on a different prompt than the legacy oracle.)
    const DEFAULT_PROMPT_IDS: &str = "671,6102,294,8760,344";
    const DEFAULT_MAX_NEW: usize = 16;
    const SLOT_MAX_SEQ_HEADROOM: usize = 64;
    const DSV4_PARITY_NUM_SLOTS: usize = 1;
    const DSV4_FLASH_KV_BYTES_PER_TOKEN: usize = 584;
    const DSV4_FLASH_BASE_LAYERS: usize = 43;

    unsafe extern "C" {
        fn cudaProfilerStart() -> i32;
        fn cudaProfilerStop() -> i32;
    }

    pub(super) fn run() -> Result<()> {
        parity_forward()
    }

    /// Mint a fresh NCCL `unique_id` in THIS process and hex-encode it (128
    /// signed bytes → 256 lowercase hex chars; the loader's
    /// `nccl_unique_id_from_env` decodes exactly this layout, i8 round-tripping
    /// through u8 bit-for-bit). The caller must stay alive after minting — the id
    /// embeds this process's bootstrap listener socket.
    #[cfg(feature = "nccl")]
    fn mint_nccl_id_hex() -> Result<String> {
        use cuda_kernels::ffi::nccl;

        let mut id = nccl::ncclUniqueId {
            internal: [0i8; 128],
        };
        // SAFETY: `id` is a valid, fully-initialized 128-byte ncclUniqueId; NCCL
        // writes the rendezvous handle into it. Single-threaded, no aliasing.
        let res = unsafe { nccl::ncclGetUniqueId(&mut id) };
        nccl::check(res).context("ncclGetUniqueId failed")?;

        let mut hex = String::with_capacity(256);
        for &b in &id.internal {
            use std::fmt::Write;
            write!(hex, "{:02x}", b as u8).expect("write to String is infallible");
        }
        debug_assert_eq!(hex.len(), 256);
        Ok(hex)
    }

    /// Set `INFER_NCCL_UNIQUE_ID` for the in-process loader to pick up.
    ///
    /// SAFETY: called once at process start, single-threaded, before any NCCL
    /// init / executor build / thread spawn. Edition-2024 requires `unsafe` for
    /// env mutation; the no-other-threads precondition is what makes it sound.
    #[cfg(feature = "nccl")]
    fn export_nccl_id(hex: &str) {
        unsafe { std::env::set_var("INFER_NCCL_UNIQUE_ID", hex) };
    }

    /// File-rendezvous for the NCCL `unique_id` (multi-rank only). Rank 0 mints
    /// it in-process (staying alive as the live NCCL rank 0), writes it
    /// atomically (tmp + rename), and exports it; every other rank blocks on the
    /// file then exports the same value. Replaces the broken throwaway-process
    /// mint whose bootstrap socket died on exit (`Connection refused`).
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
            eprintln!("[dsv4-parity rank=0] minted NCCL id -> {id_file:?}");
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
                    "rank {rank} timed out (120s) waiting for NCCL id at {id_file:?}"
                );
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "nccl"))]
    fn nccl_file_rendezvous(_rank: usize, _id_file: &std::path::Path) -> Result<()> {
        anyhow::bail!(
            "INFER_NCCL_ID_FILE needs the nccl feature; rebuild with --features cuda,nccl"
        );
    }

    fn parity_forward() -> Result<()> {
        let model_path = std::env::var("INFER_DSV4_MODEL_PATH")
            .context("INFER_DSV4_MODEL_PATH must point at the DSv4 FP8 safetensors directory")?;
        let prompts = load_prompt_cases()?;
        let prompt = prompts
            .first()
            .context("DSv4 parity prompt list resolved empty")?;
        let max_new: usize = std::env::var("INFER_DSV4_MAX_NEW")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_NEW);
        anyhow::ensure!(
            prompts.iter().all(|p| !p.is_empty()),
            "DSv4 parity prompt list contains an empty prompt"
        );
        anyhow::ensure!(max_new >= 1, "INFER_DSV4_MAX_NEW must be >= 1");

        let rank: usize = std::env::var("INFER_TP_RANK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let prompt_head: Vec<u32> = prompt.iter().copied().take(16).collect();
        eprintln!(
            "[dsv4-parity rank={rank}] model={model_path} cases={} first_prompt_len={} \
             first_prompt_head={prompt_head:?} max_new={max_new}",
            prompts.len(),
            prompt.len()
        );
        let max_prompt_len = prompts.iter().map(Vec::len).max().unwrap_or(0);
        let slot_max_seq_len = configure_slot_max_seq_len(max_prompt_len, max_new, rank)?;

        // Multi-rank NCCL bootstrap: file-rendezvous BEFORE building the executor
        // (the loader's TP runtime calls ncclCommInitRank during construction).
        if let Ok(id_file) = std::env::var("INFER_NCCL_ID_FILE") {
            nccl_file_rendezvous(rank, std::path::Path::new(&id_file))?;
        }

        let batch_decode_validate = parse_batch_decode_validate()?;
        let num_slots = batch_decode_validate
            .iter()
            .copied()
            .max()
            .unwrap_or(DSV4_PARITY_NUM_SLOTS)
            .max(DSV4_PARITY_NUM_SLOTS);
        let load_t0 = Instant::now();
        let mut exec = CudaExecutor::from_dsv4_fp8_safetensors(
            &model_path,
            num_slots,
            slot_max_seq_len,
            None,
            None,
            None,
            0.5,
            0.0,
        )
        .context("from_dsv4_fp8_safetensors failed (build/config?)")?;
        let load_ms = load_t0.elapsed().as_secs_f64() * 1000.0;
        if env_flag("INFER_DSV4_MTP_STEP_A_SELFTEST") {
            exec.dsv4_verify_forward_selftest(prompt)
                .context("DSv4 MTP Step A verify-forward selftest failed")?;
        }

        if !batch_decode_validate.is_empty() {
            run_batch_decode_validate(
                &mut exec,
                prompt,
                max_new,
                load_ms,
                rank,
                &batch_decode_validate,
            )?;
            return Ok(());
        }

        for (case_idx, prompt) in prompts.iter().enumerate() {
            // Fresh pool per case = clean slot state; the device band resets on
            // the case's start_pos=0 prefill.
            let mut kv = build_host_pool(&exec, num_slots, max_prompt_len + max_new);
            run_prompt_case(
                &mut exec,
                &mut kv,
                prompt,
                max_new,
                load_ms,
                slot_max_seq_len,
                rank,
                case_idx,
                prompts.len(),
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_prompt_case(
        exec: &mut CudaExecutor,
        kv: &mut CudaKvPool,
        prompt: &[u32],
        max_new: usize,
        load_ms: f64,
        slot_max_seq_len: usize,
        rank: usize,
        case_idx: usize,
        case_count: usize,
    ) -> Result<()> {
        let prompt_head: Vec<u32> = prompt.iter().copied().take(16).collect();
        eprintln!(
            "[dsv4-parity rank={rank}] case={case_idx}/{case_count} prompt_len={} \
             prompt_head={prompt_head:?}",
            prompt.len()
        );
        // ── Step 1: full-prefix prefill at start_pos = 0 → the first token.
        let chunk1_prompt = matches!(
            std::env::var("INFER_DSV4_PROMPT_CHUNK1").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
        );
        let profile_prefill = env_flag("INFER_DSV4_PROFILE_PREFILL");
        if profile_prefill {
            cuda_profiler_start().context("cudaProfilerStart before DSv4 prefill failed")?;
        }
        reset_dsv4_stage_profile();
        let prefill_t0 = Instant::now();
        let first = if chunk1_prompt {
            let mut first = None;
            for (idx, &tok) in prompt.iter().enumerate() {
                let tokens = forward_once(exec, kv, prefill_plan(&[tok], idx))?;
                first = Some(
                    tokens
                        .first()
                        .copied()
                        .context("DSv4 chunked prefill produced no token")?,
                );
            }
            first.expect("prompt is non-empty")
        } else {
            forward_once(exec, kv, prefill_plan(prompt, 0))?
                .first()
                .copied()
                .context("DSv4 prefill produced no token")?
        };
        let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1000.0;
        print_dsv4_stage_profile("prefill", prompt.len(), prefill_ms);
        if profile_prefill {
            cuda_profiler_stop().context("cudaProfilerStop after DSv4 prefill failed")?;
        }
        let mut clean_tokens = vec![first];
        eprintln!("[dsv4-parity rank={rank}] prefill argmax (token #1) = {first}");

        // ── Step 2: incremental decode steps (start_pos > 0). The DSv4 SW ring is
        // reallocated fresh per call, so this is the gated/unproven path — attempt
        // it, but stop cleanly at the first bail and report how far we reached.
        let mut bail_at: Option<(usize, String)> = None;
        let profile_decode = env_flag("INFER_DSV4_PROFILE_DECODE");
        if profile_decode {
            cuda_profiler_start().context("cudaProfilerStart before DSv4 decode loop failed")?;
        }
        let decode_t0 = Instant::now();
        while clean_tokens.len() < max_new {
            // KV length already materialized = prompt + tokens emitted so far.
            let kv_seq_len = prompt.len() + clean_tokens.len() - 1;
            let last = *clean_tokens.last().expect("clean_tokens is non-empty");
            match forward_once(exec, kv, decode_plan(last, kv_seq_len)) {
                Ok(tokens) => {
                    for tok in tokens {
                        if clean_tokens.len() >= max_new {
                            break;
                        }
                        clean_tokens.push(tok);
                    }
                }
                Err(e) => {
                    bail_at = Some((clean_tokens.len(), format!("{e:#}")));
                    break;
                }
            }
        }
        let decode_ms = decode_t0.elapsed().as_secs_f64() * 1000.0;
        if profile_decode {
            cuda_profiler_stop().context("cudaProfilerStop after DSv4 decode loop failed")?;
        }

        if case_count == 1 {
            println!("clean_tokens={clean_tokens:?}");
        }
        println!(
            "case_result index={case_idx} prompt_len={} clean_tokens={clean_tokens:?}",
            prompt.len()
        );
        let decode_steps = clean_tokens.len().saturating_sub(1);
        if rank == 0 {
            let decode_tok_s = if decode_ms > 0.0 {
                (decode_steps as f64) / (decode_ms / 1000.0)
            } else {
                0.0
            };
            eprintln!(
                "[dsv4-parity timing rank=0] case={} prompt_tokens={} max_new={} load_ms={:.3} \
                 prefill_ms={:.3} decode_steps={} decode_ms={:.3} decode_tok_s={:.3} \
                 slot_max_seq_len={slot_max_seq_len}",
                case_idx,
                prompt.len(),
                max_new,
                load_ms,
                prefill_ms,
                decode_steps,
                decode_ms,
                decode_tok_s
            );
        }
        match bail_at {
            None => eprintln!(
                "[dsv4-parity rank={rank}] reached max_new={max_new} \
                 ({} tokens) without an incremental-decode bail",
                clean_tokens.len()
            ),
            Some((step, msg)) => eprintln!(
                "[dsv4-parity rank={rank}] incremental decode bailed at decode \
                 step {step} (token #{}): {msg}\n  → verified TODAY: prefill argmax \
                 (token #1) on every layer type (SW/CSA/HCA recompute over [0,len)).\n  \
                 → GATED: incremental start_pos>0 decode (CSA/HCA per-step KV reuse) — \
                 the full {DEFAULT_MAX_NEW}-token oracle needs the continuous-batching \
                 follow-up.",
                clean_tokens.len() + 1
            ),
        }
        Ok(())
    }

    fn parse_batch_decode_validate() -> Result<Vec<usize>> {
        let Ok(raw) = std::env::var("INFER_DSV4_BATCH_DECODE_VALIDATE") else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for part in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let value = part
                .parse::<usize>()
                .with_context(|| format!("bad INFER_DSV4_BATCH_DECODE_VALIDATE entry `{part}`"))?;
            anyhow::ensure!(value > 0, "batch size must be positive, got {value}");
            out.push(value);
        }
        Ok(out)
    }

    fn run_batch_decode_validate(
        exec: &mut CudaExecutor,
        prompt: &[u32],
        max_new: usize,
        load_ms: f64,
        rank: usize,
        batch_sizes: &[usize],
    ) -> Result<()> {
        let validate_new = std::cmp::min(max_new, 16).max(2);
        let max_batch = batch_sizes.iter().copied().max().unwrap_or(1);
        let mut kv = build_host_pool(exec, max_batch, prompt.len() + validate_new);
        // First-divergence index of `b` vs `a` (None ⇒ identical over the overlap).
        let first_div = |a: &[u32], b: &[u32]| -> Option<usize> {
            a.iter().zip(b.iter()).position(|(x, y)| x != y)
        };
        // Byte-parity is enforced only when explicitly requested. Default is
        // report-only: batched decode is CORRECT when each row is a coherent
        // continuation, not when it is bit-identical to a c=1 run (legitimate
        // batched-reduction numerics make bit-identity impossible — see the
        // ref self-parity check above).
        let strict = std::env::var_os("INFER_DSV4_BATCH_STRICT").is_some();
        let match_prefix: usize = std::env::var("INFER_DSV4_BATCH_MATCH_PREFIX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        // Clamped: a 0 would run no trial and exit 0, reporting a pass nothing checked.
        let trials: usize = std::env::var("INFER_DSV4_BATCH_VALIDATE_TRIALS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
            .max(1);
        for trial in 0..trials {
            if rank == 0 {
                println!("batch_decode_trial trial={trial}");
            }
            let ref_tokens = run_slot_sequence(exec, &mut kv, 0, prompt, validate_new)
                .context("DSv4 batch decode single-row reference failed")?;
            // NOISE FLOOR: re-run the SAME c=1 reference. If DSv4's MoE atomic-scatter /
            // all-reduce float order is non-deterministic, even c=1 self-diverges run to
            // run — which proves byte-parity-vs-reference is an invalid gate (the batched
            // per-row drift is the same phenomenon, not a logic bug).
            let ref_tokens2 = run_slot_sequence(exec, &mut kv, 0, prompt, validate_new)
                .context("DSv4 batch decode single-row reference rerun failed")?;
            let ref_self_div = first_div(&ref_tokens, &ref_tokens2);

            if rank == 0 {
                println!(
                    "batch_decode_reference batch=1 max_new={} clean_tokens={ref_tokens:?} load_ms={load_ms:.3}",
                    validate_new
                );
                println!(
                    "batch_decode_reference_rerun clean_tokens={ref_tokens2:?} ref_self_parity={} ref_self_first_div={ref_self_div:?}",
                    ref_self_div.is_none()
                );
            }

            for &batch in batch_sizes {
                let (rows, decode_ms) =
                    run_batch_sequence(exec, &mut kv, batch, prompt, validate_new).with_context(
                        || format!("DSv4 batch decode validation failed for batch={batch}"),
                    )?;
                let parity = rows.iter().all(|tokens| tokens == &ref_tokens);
                let decode_steps = validate_new.saturating_sub(1);
                let total_tokens = decode_steps * batch;
                let tok_s = if decode_ms > 0.0 {
                    (total_tokens as f64) / (decode_ms / 1000.0)
                } else {
                    0.0
                };
                let ms_per_step = if decode_steps > 0 {
                    decode_ms / decode_steps as f64
                } else {
                    0.0
                };
                if rank == 0 {
                    println!("batch_decode_validate batch={batch} byte_parity={parity}");
                    println!(
                        "  batch_decode_perf batch={batch} decode_steps={decode_steps} decode_ms={decode_ms:.3} ms_per_step={ms_per_step:.3} tok_s={tok_s:.3}"
                    );
                    for (idx, tokens) in rows.iter().enumerate() {
                        let div = first_div(&ref_tokens, tokens);
                        println!("  row{idx}_first_div_vs_ref={div:?} clean_tokens={tokens:?}");
                    }
                }
                // CORRECTNESS GATE (determined-answer / needle): with a prompt whose
                // answer is determined, every batched row must retrieve the SAME answer
                // the verified-correct c=1 reference does, over the first
                // INFER_DSV4_BATCH_MATCH_PREFIX tokens. Tail divergence past that is
                // legitimate batched-reduction numerics (see the wins derivation entry),
                // NOT a bug — so this, not byte-parity, is the real gate.
                if match_prefix > 0 {
                    for (idx, tokens) in rows.iter().enumerate() {
                        let k = match_prefix.min(ref_tokens.len()).min(tokens.len());
                        anyhow::ensure!(
                            tokens[..k] == ref_tokens[..k],
                            "DSv4 batched row {idx} (batch={batch}) diverged inside the determined answer (first {k}): {:?} vs ref {:?}",
                            &tokens[..k],
                            &ref_tokens[..k]
                        );
                    }
                }
                if strict {
                    anyhow::ensure!(
                        parity,
                        "DSv4 batch decode byte parity failed for batch={batch}"
                    );
                }
            }
        }
        Ok(())
    }

    /// Host admission pool mirroring the executor's device KV pool, exactly as
    /// the engine builds it (`infer-api` `loaded.rs`). The DSv4 executor arm
    /// builds a `KvBatchDescriptor` from this pool at every `submit`
    /// (`executor.rs`), so the pool is NOT a signature placeholder: slot count,
    /// page count, and page size must mirror the device pool (64-token FlashMLA
    /// pages, not config 16), and fixed-band models draw each slot's whole
    /// `[SW ring | compressed]` band up front via `set_fixed_pages_per_slot`
    /// so the descriptor carries the row's complete page table.
    fn build_host_pool(
        exec: &CudaExecutor,
        requested_slots: usize,
        budget_tokens: usize,
    ) -> CudaKvPool {
        let num_slots = exec.effective_num_slots().unwrap_or(requested_slots).max(1);
        let page_size = exec.effective_page_size().unwrap_or(16).max(1);
        // No device paged pool to mirror (band-arena mode) → size the pool from
        // the harness's own prompt+decode budget, one spare page per slot.
        let budget_pages = num_slots * (budget_tokens.div_ceil(page_size) + 1);
        let total_pages = exec
            .effective_total_pages()
            .filter(|&pages| pages > 0)
            .unwrap_or(budget_pages);
        let mut kv = CudaKvPool::new(num_slots, total_pages, page_size);
        if let Some(pages) = exec.effective_fixed_pages_per_slot() {
            kv.set_fixed_pages_per_slot(pages);
        }
        kv
    }

    fn run_slot_sequence(
        exec: &mut CudaExecutor,
        kv: &mut CudaKvPool,
        slot: usize,
        prompt: &[u32],
        max_new: usize,
    ) -> Result<Vec<u32>> {
        let first = forward_tokens(exec, kv, prefill_plan_for(slot, prompt, 0))?
            .into_iter()
            .find(|tok| tok.slot == slot)
            .map(|tok| tok.token)
            .context("DSv4 single-row prefill produced no token")?;
        let mut clean_tokens = vec![first];
        while clean_tokens.len() < max_new {
            let last = *clean_tokens.last().expect("clean_tokens is non-empty");
            let kv_seq_len = prompt.len() + clean_tokens.len() - 1;
            let out = forward_tokens(exec, kv, decode_plan_for(&[(slot, last, kv_seq_len)]))?;
            let token = out
                .into_iter()
                .find(|tok| tok.slot == slot)
                .map(|tok| tok.token)
                .context("DSv4 single-row decode produced no token")?;
            clean_tokens.push(token);
        }
        Ok(clean_tokens)
    }

    /// Returns (per-slot tokens, decode-loop wall ms). The decode timer excludes
    /// the per-slot prefills so the batched STEP cost is isolated — this is the
    /// throughput signal (compare BATCHED=1 grouped vs BATCHED=0 per-row executor).
    fn run_batch_sequence(
        exec: &mut CudaExecutor,
        kv: &mut CudaKvPool,
        batch: usize,
        prompt: &[u32],
        max_new: usize,
    ) -> Result<(Vec<Vec<u32>>, f64)> {
        let mut clean_tokens = vec![Vec::new(); batch];
        for (slot, slot_tokens) in clean_tokens.iter_mut().enumerate() {
            let first = forward_tokens(exec, kv, prefill_plan_for(slot, prompt, 0))?
                .into_iter()
                .find(|tok| tok.slot == slot)
                .map(|tok| tok.token)
                .with_context(|| format!("DSv4 batch prefill produced no token for slot {slot}"))?;
            slot_tokens.push(first);
        }

        let decode_t0 = Instant::now();
        while clean_tokens.iter().any(|tokens| tokens.len() < max_new) {
            let mut rows = Vec::with_capacity(batch);
            for (slot, slot_tokens) in clean_tokens.iter().enumerate() {
                if slot_tokens.len() >= max_new {
                    continue;
                }
                let last = *slot_tokens.last().expect("slot has prefill token");
                let kv_seq_len = prompt.len() + slot_tokens.len() - 1;
                rows.push((slot, last, kv_seq_len));
            }
            let out = forward_tokens(exec, kv, decode_plan_for(&rows))?;
            for tok in out {
                if tok.slot < clean_tokens.len() && clean_tokens[tok.slot].len() < max_new {
                    clean_tokens[tok.slot].push(tok.token);
                }
            }
        }
        let decode_ms = decode_t0.elapsed().as_secs_f64() * 1000.0;

        Ok((clean_tokens, decode_ms))
    }

    fn configure_slot_max_seq_len(prompt_len: usize, max_new: usize, rank: usize) -> Result<usize> {
        let slot_max_seq_len = prompt_len
            .checked_add(max_new)
            .and_then(|v| v.checked_add(SLOT_MAX_SEQ_HEADROOM))
            .context("DSv4 parity slot max_seq_len overflow")?;
        let kv_bytes = slot_max_seq_len
            .checked_mul(DSV4_FLASH_KV_BYTES_PER_TOKEN)
            .and_then(|v| v.checked_mul(DSV4_FLASH_BASE_LAYERS))
            .and_then(|v| v.checked_mul(DSV4_PARITY_NUM_SLOTS))
            .context("DSv4 parity FP8 KV arena byte estimate overflow")?;
        let kv_mib = kv_bytes as f64 / (1024.0 * 1024.0);
        eprintln!(
            "[dsv4-parity rank={rank}] slot_max_seq_len={slot_max_seq_len} \
             (= prompt_len {prompt_len} + max_new {max_new} + headroom \
             {SLOT_MAX_SEQ_HEADROOM}); fp8_kv_arena_est={kv_bytes} bytes \
             ({kv_mib:.2} MiB/rank for {DSV4_PARITY_NUM_SLOTS} slot × \
             {DSV4_FLASH_BASE_LAYERS} layers × {DSV4_FLASH_KV_BYTES_PER_TOKEN} B/token)"
        );
        // slot_max_seq_len is threaded as the from_dsv4_fp8_safetensors max_seq_len
        // PARAMETER (runtime config = explicit arg, not an env var the executor
        // reads — per the runtime-config-as-CLI-flag rule).
        Ok(slot_max_seq_len)
    }

    fn env_flag(name: &str) -> bool {
        matches!(
            std::env::var(name).as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
        )
    }

    fn cuda_profiler_start() -> Result<()> {
        // SAFETY: CUDA runtime profiler API has process-global side effects only;
        // called at the single decode-loop boundary in this short-lived harness.
        let code = unsafe { cudaProfilerStart() };
        anyhow::ensure!(
            code == 0,
            "cudaProfilerStart returned CUDA error code {code}"
        );
        Ok(())
    }

    fn cuda_profiler_stop() -> Result<()> {
        // SAFETY: pairs with `cuda_profiler_start` above.
        let code = unsafe { cudaProfilerStop() };
        anyhow::ensure!(
            code == 0,
            "cudaProfilerStop returned CUDA error code {code}"
        );
        Ok(())
    }

    /// Build a one-row prefill `ForwardPlan` over the whole prompt at `start_pos`.
    fn prefill_plan(tokens: &[u32], start_pos: usize) -> ForwardPlan {
        prefill_plan_for(0, tokens, start_pos)
    }

    fn prefill_plan_for(slot: usize, tokens: &[u32], start_pos: usize) -> ForwardPlan {
        ForwardPlan {
            mode: ForwardMode::Prefill,
            decode_rows: Vec::new(),
            prefill_rows: vec![PrefillRow {
                slot,
                tokens: tokens.to_vec(),
                start_pos,
                total_tokens: start_pos + tokens.len(),
                params: greedy(),
                penalty_history: None,
                penalty_prompt_len: 0,
            }],
        }
    }

    /// Build a one-row incremental-decode `ForwardPlan` for `last_token` at the
    /// given materialized KV length (the `start_pos > 0` path).
    fn decode_plan(last_token: u32, kv_seq_len: usize) -> ForwardPlan {
        decode_plan_for(&[(0, last_token, kv_seq_len)])
    }

    fn decode_plan_for(rows: &[(usize, u32, usize)]) -> ForwardPlan {
        ForwardPlan {
            mode: ForwardMode::Decode,
            decode_rows: rows
                .iter()
                .map(|&(slot, last_token, kv_seq_len)| DecodeRow {
                    slot,
                    last_token,
                    kv_seq_len,
                    params: greedy(),
                    penalty_history: None,
                    penalty_prompt_len: 0,
                })
                .collect(),
            prefill_rows: Vec::new(),
        }
    }

    /// Submit one plan and synchronously resolve its single sampled token.
    fn forward_once(
        exec: &mut CudaExecutor,
        kv: &mut CudaKvPool,
        plan: ForwardPlan,
    ) -> Result<Vec<u32>> {
        Ok(forward_tokens(exec, kv, plan)?
            .into_iter()
            .map(|t| t.token)
            .collect())
    }

    fn forward_tokens(
        exec: &mut CudaExecutor,
        kv: &mut CudaKvPool,
        plan: ForwardPlan,
    ) -> Result<Vec<SlotToken>> {
        materialize_plan_kv(kv, &plan)?;
        let inflight = exec.submit(&plan, kv as &mut dyn KvPool)?;
        match exec.poll(inflight)? {
            PollResult::Ready(out) => {
                let tokens = out.tokens;
                anyhow::ensure!(!tokens.is_empty(), "DSv4 step produced no token");
                Ok(tokens)
            }
            PollResult::NotReady(_) => {
                anyhow::bail!("DSv4 executor resolves synchronously; got NotReady")
            }
        }
    }

    fn materialize_plan_kv(kv: &mut CudaKvPool, plan: &ForwardPlan) -> Result<()> {
        for row in &plan.prefill_rows {
            if row.start_pos == 0 {
                kv.free_slot(row.slot);
            }
            anyhow::ensure!(
                kv.seq_len(row.slot) == row.start_pos,
                "DSv4 parity host KV len {} != prefill start_pos {} for slot {}",
                kv.seq_len(row.slot),
                row.start_pos,
                row.slot
            );
            kv.alloc(row.slot, row.tokens.len())?;
        }
        for row in &plan.decode_rows {
            anyhow::ensure!(
                kv.seq_len(row.slot) == row.kv_seq_len,
                "DSv4 parity host KV len {} != decode kv_seq_len {} for slot {}",
                kv.seq_len(row.slot),
                row.kv_seq_len,
                row.slot
            );
            kv.alloc(row.slot, 1)?;
        }
        Ok(())
    }

    fn greedy() -> SamplingParams {
        SamplingParams::default() // temperature 0.0 → greedy argmax
    }

    fn load_prompt_cases() -> Result<Vec<Vec<u32>>> {
        if let Ok(path) = std::env::var("INFER_DSV4_PROMPT_IDS_LIST_FILE") {
            let text = std::fs::read_to_string(Path::new(&path))
                .with_context(|| format!("read INFER_DSV4_PROMPT_IDS_LIST_FILE={path}"))?;
            let mut prompts = Vec::new();
            for (line_idx, raw) in text.lines().enumerate() {
                let line = raw.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                prompts.push(
                    parse_prompt_ids(line)
                        .with_context(|| format!("parse prompt ids at {path}:{}", line_idx + 1))?,
                );
            }
            anyhow::ensure!(
                !prompts.is_empty(),
                "INFER_DSV4_PROMPT_IDS_LIST_FILE={path} contained no prompts"
            );
            return Ok(prompts);
        }
        Ok(vec![parse_prompt_ids(
            &std::env::var("INFER_DSV4_PROMPT_IDS")
                .unwrap_or_else(|_| DEFAULT_PROMPT_IDS.to_string()),
        )?])
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
}
