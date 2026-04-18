//! Metal inference benchmark — prompt speed, TTFT, decode throughput, peak RSS.
//!
//! Requires the `metal` feature. Build with:
//!   cargo build --release --no-default-features --features metal,no-cuda --bin metal_bench
//!
//! # Examples
//! ```bash
//! # Basic
//! ./target/release/metal_bench --model models/Qwen3-0.6B-4bit
//!
//! # Full options
//! ./target/release/metal_bench \
//!   --model models/Qwen3-0.6B-4bit \
//!   --prompt-tokens 20 --generation-tokens 256 --warmup 3 --runs 5 --json
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{ArgAction, Parser};
use infer::backend::metal::MetalRuntimeLimits;
use serde::{Deserialize, Serialize};

/// Metal backend benchmark: prompt speed, TTFT, decode throughput, peak RSS.
#[derive(Parser)]
#[command(
    name = "metal_bench",
    about = "Metal/MLX inference benchmark (Apple Silicon)"
)]
struct Cli {
    /// Model path (local directory) or HuggingFace repo ID.
    #[arg(long, short)]
    model: String,

    /// Exact number of prompt tokens to benchmark.
    #[arg(long, default_value_t = 20)]
    prompt_tokens: usize,

    /// Exact number of generated tokens to benchmark.
    #[arg(long, visible_alias = "max-tokens", default_value_t = 256)]
    generation_tokens: usize,

    /// Drive decode via Qwen35StepDriver per-step FFI path (matches HTTP server) instead of cpp_model.generate. Qwen3.5 only.
    #[arg(long, default_value_t = false)]
    use_step_driver: bool,

    /// Number of warmup runs (excluded from statistics).
    #[arg(long, default_value_t = 3)]
    warmup: usize,

    /// Number of timed runs for statistics.
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// Print per-run timing detail to stderr.
    #[arg(long, default_value_t = false)]
    profile: bool,

    /// Emit results as JSON on stdout (suppresses human-readable table).
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Save benchmark results as a baseline JSON file.
    #[arg(long, value_name = "PATH")]
    save_baseline: Option<PathBuf>,

    /// Compare benchmark results against a previously saved baseline.
    /// Exits with code 1 if any metric regresses beyond its threshold.
    #[arg(long, value_name = "PATH")]
    compare_baseline: Option<PathBuf>,

    /// Like --save-baseline, but only overwrites if all metrics PASS
    /// (or if no baseline file exists yet).
    #[arg(long, value_name = "PATH")]
    update_baseline: Option<PathBuf>,

    /// Enable Metal DFlash with the given draft model path or HuggingFace repo.
    #[arg(long, value_name = "PATH_OR_REPO")]
    dflash_draft_model: Option<String>,

    /// Enable the experimental Metal KV pool for the Qwen3 fallback path.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "no_kv_pool")]
    kv_pool: bool,

    /// Disable the experimental Metal KV pool even if the env fallback is set.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "kv_pool")]
    no_kv_pool: bool,

    /// Override the MLX allocator memory limit in bytes before model load.
    #[arg(long, value_name = "BYTES")]
    memory_limit_bytes: Option<usize>,

    /// Override the MLX allocator cache limit in bytes before model load.
    #[arg(long, value_name = "BYTES")]
    cache_limit_bytes: Option<usize>,

    /// Override the MLX allocator wired limit in bytes before model load.
    #[arg(long, value_name = "BYTES")]
    wired_limit_bytes: Option<usize>,

    /// Override the DFlash speculative block size.
    /// Defaults to the draft config; lower values can reduce throughput.
    #[arg(long)]
    speculative_tokens: Option<usize>,

    /// Ignore EOS in --use-step-driver mode and continue until --generation-tokens.
    #[arg(long, default_value_t = false)]
    ignore_eos: bool,
}

impl Cli {
    fn kv_pool_override(&self) -> Option<bool> {
        if self.kv_pool {
            Some(true)
        } else if self.no_kv_pool {
            Some(false)
        } else {
            None
        }
    }

    fn runtime_limits(&self) -> MetalRuntimeLimits {
        MetalRuntimeLimits {
            memory_limit_bytes: self.memory_limit_bytes,
            cache_limit_bytes: self.cache_limit_bytes,
            wired_limit_bytes: self.wired_limit_bytes,
        }
    }

    fn effective_ignore_eos(&self) -> bool {
        if self.use_step_driver {
            self.ignore_eos
        } else {
            true
        }
    }
}

struct Run {
    total_time_ms: f64,
    decode_elapsed_s: f64,
    prompt_tokens: usize,
    tokens: usize,
    prompt_tps: f64,
    generation_tps: f64,
    ttft_ms: f64,
    e2e_tps: f64,
}

/// A single metric stat in a baseline file (only mean is required).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetricStat {
    mean: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    p50: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p99: Option<f64>,
}

/// Baseline JSON schema for saving / comparing benchmark results.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Baseline {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_tokens: Option<usize>,
    metrics: BTreeMap<String, MetricStat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recorded_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    notes: Option<String>,
}

/// Regression thresholds: how much worse (%) a metric can get before FAIL.
/// For throughput metrics (higher=better), a regression means the current
/// value is _lower_ than the baseline.
/// For latency metrics (lower=better), a regression means the current
/// value is _higher_ than the baseline.
fn regression_threshold(metric: &str) -> Option<(f64, bool)> {
    // Returns (threshold_pct, higher_is_better)
    match metric {
        "generation_tps" => Some((5.0, true)),
        "prompt_tps" => Some((10.0, true)),
        "ttft_ms" => Some((15.0, false)),
        _ => None,
    }
}

/// Compare current metrics against a baseline. Returns true if all checked
/// metrics pass, false if any regresses beyond its threshold.
fn compare_baseline(baseline: &Baseline, current: &BTreeMap<String, MetricStat>) -> bool {
    let mut all_pass = true;
    let mut rows: Vec<(String, f64, f64, f64, bool)> = Vec::new();

    // Only compare metrics that have regression thresholds. `total_time_ms` is
    // intentionally excluded — it's a derived quantity (prompt + decode) and the
    // individual components already have their own thresholds.
    for (metric, threshold_info) in [
        ("prompt_tps", regression_threshold("prompt_tps")),
        ("generation_tps", regression_threshold("generation_tps")),
        ("ttft_ms", regression_threshold("ttft_ms")),
    ] {
        let base_val = match baseline.metrics.get(metric) {
            Some(s) => s.mean,
            None => continue,
        };
        let cur_val = match current.get(metric) {
            Some(s) => s.mean,
            None => continue,
        };

        let (pass, delta_pct) = if let Some((thresh, higher_is_better)) = threshold_info {
            let delta = if higher_is_better {
                // throughput: regression = current < baseline
                (cur_val - base_val) / base_val * 100.0
            } else {
                // latency: regression = current > baseline (invert sign so negative = regression)
                (base_val - cur_val) / base_val * 100.0
            };
            let pass = delta >= -thresh;
            (pass, delta)
        } else {
            // No threshold — always pass, just report delta
            let delta = (cur_val - base_val) / base_val * 100.0;
            (true, delta)
        };

        if !pass {
            all_pass = false;
        }
        rows.push((metric.to_string(), base_val, cur_val, delta_pct, pass));
    }

    // Print comparison table
    eprintln!();
    eprintln!(
        "  {:<20} {:>12} {:>12} {:>10}   Status",
        "Metric", "Baseline", "Current", "Delta"
    );
    eprintln!("  {}", "-".repeat(72));
    for (metric, base_val, cur_val, delta_pct, pass) in &rows {
        let status = if *pass { "PASS" } else { "FAIL" };
        eprintln!(
            "  {:<20} {:>12.1} {:>12.1} {:>+9.1}%   {}",
            metric, base_val, cur_val, delta_pct, status
        );
    }
    eprintln!();

    all_pass
}

fn validate_baseline_compatibility(
    baseline: &Baseline,
    model: &str,
    prompt_tokens: usize,
    generation_tokens: usize,
) -> Result<()> {
    if let Some(baseline_model) = &baseline.model {
        anyhow::ensure!(
            baseline_model == model,
            "baseline model mismatch: baseline='{}', current='{}'",
            baseline_model,
            model
        );
    }
    if let Some(baseline_prompt) = baseline.prompt_tokens {
        anyhow::ensure!(
            baseline_prompt == prompt_tokens,
            "baseline prompt_tokens mismatch: baseline={}, current={}",
            baseline_prompt,
            prompt_tokens
        );
    }
    if let Some(baseline_generation) = baseline.generation_tokens {
        anyhow::ensure!(
            baseline_generation == generation_tokens,
            "baseline generation_tokens mismatch: baseline={}, current={}",
            baseline_generation,
            generation_tokens
        );
    }
    Ok(())
}

fn build_current_metrics(
    mean_prompt_tps: f64,
    mean_generation_tps: f64,
    mean_ttft_ms: f64,
    mean_total_time_ms: f64,
) -> BTreeMap<String, MetricStat> {
    let mut m = BTreeMap::new();
    m.insert(
        "prompt_tps".into(),
        MetricStat {
            mean: mean_prompt_tps,
            p50: None,
            p99: None,
        },
    );
    m.insert(
        "generation_tps".into(),
        MetricStat {
            mean: mean_generation_tps,
            p50: None,
            p99: None,
        },
    );
    m.insert(
        "ttft_ms".into(),
        MetricStat {
            mean: mean_ttft_ms,
            p50: None,
            p99: None,
        },
    );
    m.insert(
        "total_time_ms".into(),
        MetricStat {
            mean: mean_total_time_ms,
            p50: None,
            p99: None,
        },
    );
    m
}

fn main() -> Result<()> {
    #[cfg(feature = "metal")]
    return run_bench();

    #[cfg(not(feature = "metal"))]
    {
        eprintln!(
            "metal_bench requires the `metal` feature.\n\
             Rebuild: cargo build --no-default-features --features metal,no-cuda --bin metal_bench --release"
        );
        std::process::exit(1);
    }
}

#[cfg(feature = "metal")]
fn run_bench() -> Result<()> {
    use infer::backend::InferenceBackend;
    use infer::backend::metal::{
        MetalBackend, MetalBackendOptions, MetalDflashOptions, request_state::MetalRequestPhase,
        scheduler::MetalSchedulerConfig,
    };
    use infer::sampler::SamplingParams;

    let cli = Cli::parse();

    let params = SamplingParams {
        temperature: 0.0, // greedy — deterministic runs
        ignore_eos: cli.effective_ignore_eos(),
        max_new_tokens: Some(cli.generation_tokens),
        ..Default::default()
    };

    // ── Load ─────────────────────────────────────────────────────────────────
    let t_load = Instant::now();
    let mut backend = MetalBackend::with_options(MetalBackendOptions {
        dflash: cli
            .dflash_draft_model
            .as_ref()
            .map(|draft_model| MetalDflashOptions {
                draft_model: draft_model.clone(),
                speculative_tokens: cli.speculative_tokens,
            }),
        kv_pool: cli.kv_pool_override(),
        runtime_limits: cli.runtime_limits(),
    });
    backend.load(std::path::Path::new(&cli.model))?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let quant_hint = if cli.model.contains("4bit") || cli.model.contains("4-bit") {
        "4-bit"
    } else if cli.model.contains("8bit") || cli.model.contains("8-bit") {
        "8-bit"
    } else {
        "bf16"
    };

    if !cli.json {
        eprintln!("Loaded {load_ms:.0}ms  [{quant_hint}]");
    }

    let prompt_ids = backend.benchmark_prompt_ids(cli.prompt_tokens)?;

    if cli.use_step_driver {
        anyhow::ensure!(
            cli.generation_tokens > 0,
            "--use-step-driver requires --generation-tokens >= 1"
        );
        let request_state = backend.create_request_state(&prompt_ids, &params)?;
        if !request_state.is_qwen35() {
            bail!("--use-step-driver requires Qwen3.5 model");
        }
    }

    let step_driver_prefill_budget = MetalSchedulerConfig::default().prefill_chunk_size;

    // ── Warmup ───────────────────────────────────────────────────────────────
    if !cli.json {
        eprintln!("Warmup ({} runs)…", cli.warmup);
    }
    for _ in 0..cli.warmup {
        if cli.use_step_driver {
            run_step_driver_once(
                &backend,
                &prompt_ids,
                &params,
                cli.generation_tokens,
                step_driver_prefill_budget,
                MetalRequestPhase::Prefill,
            )?;
        } else {
            backend.generate_from_token_ids(&prompt_ids, &params)?;
        }
    }

    // ── Timed runs ───────────────────────────────────────────────────────────
    let mut runs: Vec<Run> = Vec::with_capacity(cli.runs);

    for i in 0..cli.runs {
        let run = if cli.use_step_driver {
            run_step_driver_once(
                &backend,
                &prompt_ids,
                &params,
                cli.generation_tokens,
                step_driver_prefill_budget,
                MetalRequestPhase::Prefill,
            )?
        } else {
            let result = backend.generate_from_token_ids(&prompt_ids, &params)?;
            if result.finish_reason != "length" || result.completion_tokens != cli.generation_tokens
            {
                anyhow::bail!(
                    "benchmark invariant failed on run {}: finish_reason={}, completion_tokens={}, expected={}",
                    i + 1,
                    result.finish_reason,
                    result.completion_tokens,
                    cli.generation_tokens,
                );
            }
            let decode_elapsed_s =
                ((result.total_time_ms - result.ttft_ms).max(0.0) / 1000.0).max(1e-9);
            let e2e_tps =
                result.completion_tokens as f64 / (result.total_time_ms / 1000.0).max(1e-9);
            Run {
                total_time_ms: result.total_time_ms,
                decode_elapsed_s,
                prompt_tokens: result.prompt_tokens,
                tokens: result.completion_tokens,
                prompt_tps: result.prompt_tps,
                generation_tps: result.generation_tps,
                ttft_ms: result.ttft_ms,
                e2e_tps,
            }
        };

        if cli.profile || !cli.json {
            eprintln!(
                "  run {:2}: prompt {:3} tok @ {:6.1} tok/s  gen {:4} tok @ {:6.1} tok/s  repo-e2e {:6.1} tok/s  ttft {:6.1}ms  total {:6.1}ms",
                i + 1,
                run.prompt_tokens,
                run.prompt_tps,
                run.tokens,
                run.generation_tps,
                run.e2e_tps,
                run.ttft_ms,
                run.total_time_ms,
            );
            if cli.use_step_driver {
                eprintln!(
                    "    [step-driver] decode tok/s = {} / {:.3}s = {:.1}",
                    run.tokens, run.decode_elapsed_s, run.generation_tps,
                );
            }
        }

        runs.push(run);
    }

    // ── Statistics ───────────────────────────────────────────────────────────
    let mut prompt_tps_sorted: Vec<f64> = runs.iter().map(|r| r.prompt_tps).collect();
    prompt_tps_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_prompt_tps = prompt_tps_sorted.iter().sum::<f64>() / prompt_tps_sorted.len() as f64;
    let p50_prompt_tps = percentile(&prompt_tps_sorted, 50.0);
    let p99_prompt_tps = percentile(&prompt_tps_sorted, 99.0);

    let mut generation_tps_sorted: Vec<f64> = runs.iter().map(|r| r.generation_tps).collect();
    generation_tps_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_generation_tps =
        generation_tps_sorted.iter().sum::<f64>() / generation_tps_sorted.len() as f64;
    let p50_generation_tps = percentile(&generation_tps_sorted, 50.0);
    let p99_generation_tps = percentile(&generation_tps_sorted, 99.0);

    let mut e2e_tps_sorted: Vec<f64> = runs.iter().map(|r| r.e2e_tps).collect();
    e2e_tps_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_e2e_tps = e2e_tps_sorted.iter().sum::<f64>() / e2e_tps_sorted.len() as f64;
    let p50_e2e_tps = percentile(&e2e_tps_sorted, 50.0);
    let p99_e2e_tps = percentile(&e2e_tps_sorted, 99.0);

    let mut ttft_sorted: Vec<f64> = runs.iter().map(|r| r.ttft_ms).collect();
    ttft_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_ttft_ms = ttft_sorted.iter().sum::<f64>() / ttft_sorted.len() as f64;
    let p50_ttft_ms = percentile(&ttft_sorted, 50.0);
    let p99_ttft_ms = percentile(&ttft_sorted, 99.0);

    let avg_tokens = runs.iter().map(|r| r.tokens).sum::<usize>() / runs.len().max(1);
    let prompt_tokens = runs.first().map_or(0, |r| r.prompt_tokens);
    let mut total_time_ms_sorted: Vec<f64> = runs.iter().map(|r| r.total_time_ms).collect();
    total_time_ms_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_total_time_ms =
        total_time_ms_sorted.iter().sum::<f64>() / total_time_ms_sorted.len() as f64;
    let p50_total_time_ms = percentile(&total_time_ms_sorted, 50.0);
    let p99_total_time_ms = percentile(&total_time_ms_sorted, 99.0);

    let peak_mb = peak_rss_kb() as f64 / 1024.0;

    // ── Output ───────────────────────────────────────────────────────────────
    if cli.json {
        let mut payload = serde_json::json!({
            "model": cli.model,
            "quantization": quant_hint,
            "prompt_tokens_requested": cli.prompt_tokens,
            "generation_tokens_requested": cli.generation_tokens,
            "warmup_runs": cli.warmup,
            "timed_runs": cli.runs,
            "load_ms": load_ms,
            "prompt_tokens": prompt_tokens,
            "avg_tokens": avg_tokens,
            "prompt_tps": { "mean": mean_prompt_tps, "p50": p50_prompt_tps, "p99": p99_prompt_tps },
            "generation_tps": { "mean": mean_generation_tps, "p50": p50_generation_tps, "p99": p99_generation_tps },
            "ttft_ms": { "mean": mean_ttft_ms, "p50": p50_ttft_ms, "p99": p99_ttft_ms },
            "total_time_ms": { "mean": mean_total_time_ms, "p50": p50_total_time_ms, "p99": p99_total_time_ms },
            "repo_e2e_tps": { "mean": mean_e2e_tps, "p50": p50_e2e_tps, "p99": p99_e2e_tps },
            "peak_rss_mb": peak_mb,
        });
        if cli.use_step_driver {
            payload["mode"] = serde_json::Value::String("step-driver".to_string());
        }
        println!("{payload}");
    } else {
        println!();
        println!("=== Metal Benchmark: {} [{}] ===", cli.model, quant_hint);
        println!("  Warmup / timed  : {} / {}", cli.warmup, cli.runs);
        println!("  Prompt tokens   : {prompt_tokens}");
        println!("  Avg tokens out  : {avg_tokens}");
        println!(
            "  Prompt speed    : {mean_prompt_tps:.1} tok/s mean  |  {p50_prompt_tps:.1} p50  |  {p99_prompt_tps:.1} p99"
        );
        if cli.use_step_driver {
            println!(
                "  [step-driver] Generation: {mean_generation_tps:.1} tok/s mean  |  {p50_generation_tps:.1} p50  |  {p99_generation_tps:.1} p99"
            );
        } else {
            println!(
                "  Generation      : {mean_generation_tps:.1} tok/s mean  |  {p50_generation_tps:.1} p50  |  {p99_generation_tps:.1} p99"
            );
        }
        println!(
            "  TTFT            : {mean_ttft_ms:.0}ms mean  |  {p50_ttft_ms:.0}ms p50  |  {p99_ttft_ms:.0}ms p99"
        );
        println!(
            "  Total wall      : {mean_total_time_ms:.0}ms mean  |  {p50_total_time_ms:.0}ms p50  |  {p99_total_time_ms:.0}ms p99"
        );
        println!(
            "  Repo E2E        : {mean_e2e_tps:.1} tok/s mean  |  {p50_e2e_tps:.1} p50  |  {p99_e2e_tps:.1} p99"
        );
        println!("  Peak RSS        : {peak_mb:.0}MB");
        println!(
            "==={}===",
            "=".repeat(cli.model.len() + quant_hint.len() + 4)
        );
    }

    // ── Baseline operations ─────────────────────────────────────────────
    let current_metrics = build_current_metrics(
        mean_prompt_tps,
        mean_generation_tps,
        mean_ttft_ms,
        mean_total_time_ms,
    );

    let make_baseline = |notes: Option<String>| -> Baseline {
        Baseline {
            model: Some(cli.model.clone()),
            prompt_tokens: Some(cli.prompt_tokens),
            generation_tokens: Some(cli.generation_tokens),
            metrics: current_metrics.clone(),
            recorded_at: None,
            notes,
        }
    };

    // --save-baseline: unconditionally write current results
    if let Some(ref path) = cli.save_baseline {
        let bl = make_baseline(None);
        let json = serde_json::to_string_pretty(&bl)?;
        std::fs::write(path, &json)?;
        eprintln!("Baseline saved to {}", path.display());
    }

    // --compare-baseline: load baseline and compare
    let comparison_passed = if let Some(ref path) = cli.compare_baseline {
        let data = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read baseline {}: {}", path.display(), e))?;
        let baseline: Baseline = serde_json::from_str(&data)
            .map_err(|e| anyhow::anyhow!("failed to parse baseline {}: {}", path.display(), e))?;
        eprintln!("Comparing against baseline: {}", path.display());
        validate_baseline_compatibility(
            &baseline,
            &cli.model,
            cli.prompt_tokens,
            cli.generation_tokens,
        )?;
        let passed = compare_baseline(&baseline, &current_metrics);
        if passed {
            eprintln!("All metrics within threshold.");
        } else {
            eprintln!("REGRESSION DETECTED — one or more metrics exceeded threshold.");
        }
        Some(passed)
    } else {
        None
    };

    // --update-baseline: only overwrite if comparison passes (or no prior baseline)
    if let Some(ref path) = cli.update_baseline {
        let should_write = if path.exists() {
            // Load existing baseline and compare
            let data = std::fs::read_to_string(path).map_err(|e| {
                anyhow::anyhow!("failed to read baseline {}: {}", path.display(), e)
            })?;
            let baseline: Baseline = serde_json::from_str(&data).map_err(|e| {
                anyhow::anyhow!("failed to parse baseline {}: {}", path.display(), e)
            })?;
            eprintln!("Checking update eligibility against: {}", path.display());
            validate_baseline_compatibility(
                &baseline,
                &cli.model,
                cli.prompt_tokens,
                cli.generation_tokens,
            )?;
            let passed = compare_baseline(&baseline, &current_metrics);
            if passed {
                eprintln!("All metrics PASS — updating baseline.");
            } else {
                eprintln!("Regression detected — baseline NOT updated.");
            }
            passed
        } else {
            eprintln!("No existing baseline at {} — creating new.", path.display());
            true
        };

        if should_write {
            let bl = make_baseline(None);
            let json = serde_json::to_string_pretty(&bl)?;
            std::fs::write(path, &json)?;
            eprintln!("Baseline written to {}", path.display());
        }
    }

    // Exit with code 1 if comparison failed
    if comparison_passed == Some(false) {
        anyhow::bail!("benchmark regression detected");
    }

    Ok(())
}

#[cfg(feature = "metal")]
fn run_step_driver_once(
    backend: &infer::backend::metal::MetalBackend,
    prompt_ids: &[u32],
    params: &infer::sampler::SamplingParams,
    expected_generation_tokens: usize,
    prefill_budget: usize,
    expected_prefill_phase: infer::backend::metal::request_state::MetalRequestPhase,
) -> Result<Run> {
    use infer::backend::metal::request_state::MetalRequestPhase;

    let mut request_state = backend.create_request_state(prompt_ids, params)?;
    if !request_state.is_qwen35() {
        bail!("--use-step-driver requires Qwen3.5 model");
    }
    anyhow::ensure!(
        request_state.phase() == expected_prefill_phase,
        "step-driver benchmark expected {:?} phase, got {:?}",
        expected_prefill_phase,
        request_state.phase()
    );

    let started_at = Instant::now();
    while request_state.phase() == MetalRequestPhase::Prefill {
        let result = request_state.prefill_chunk(prefill_budget)?;
        anyhow::ensure!(
            result.processed_tokens > 0,
            "step-driver prefill made no forward progress"
        );
    }

    let ttft_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    let decode_started = request_state.phase() == MetalRequestPhase::Decode;
    let decode_t0 = Instant::now();
    while request_state.phase() == MetalRequestPhase::Decode {
        request_state
            .decode_step()?
            .context("step-driver decode_step did not emit a token")?;
    }
    let total_time_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    let decode_elapsed_s = if decode_started {
        decode_t0.elapsed().as_secs_f64()
    } else {
        0.0
    };

    let finish_reason = request_state.finish_reason().unwrap_or("unknown");
    if params.ignore_eos {
        anyhow::ensure!(
            finish_reason == "length"
                && request_state.generated_tokens() == expected_generation_tokens,
            "step-driver invariant failed: finish_reason={}, completion_tokens={}, expected={}",
            finish_reason,
            request_state.generated_tokens(),
            expected_generation_tokens,
        );
    } else {
        anyhow::ensure!(
            matches!(finish_reason, "length" | "stop"),
            "step-driver invariant failed: unexpected finish_reason={finish_reason}",
        );
        anyhow::ensure!(
            request_state.generated_tokens() <= expected_generation_tokens,
            "step-driver invariant failed: completion_tokens={} exceeded expected={}",
            request_state.generated_tokens(),
            expected_generation_tokens,
        );
    }

    let prompt_tps = if ttft_ms > 0.0 && !prompt_ids.is_empty() {
        prompt_ids.len() as f64 / (ttft_ms / 1000.0).max(1e-9)
    } else {
        0.0
    };
    let generation_tps = if request_state.generated_tokens() > 0 && decode_elapsed_s > 0.0 {
        request_state.generated_tokens() as f64 / decode_elapsed_s
    } else {
        0.0
    };
    let e2e_tps = if request_state.generated_tokens() > 0 {
        request_state.generated_tokens() as f64 / (total_time_ms / 1000.0).max(1e-9)
    } else {
        0.0
    };

    Ok(Run {
        total_time_ms,
        decode_elapsed_s,
        prompt_tokens: prompt_ids.len(),
        tokens: request_state.generated_tokens(),
        prompt_tps,
        generation_tps,
        ttft_ms,
        e2e_tps,
    })
}

/// Nearest-rank percentile over a *sorted* slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((p.clamp(0.0, 100.0) / 100.0) * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1);
    sorted[idx.min(sorted.len() - 1)]
}

/// Peak resident-set size in KB (macOS: `ru_maxrss` is in bytes).
#[allow(unused)]
fn peak_rss_kb() -> u64 {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut ru) };
        ru.ru_maxrss as u64 / 1024
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Baseline, MetricStat, compare_baseline, percentile, validate_baseline_compatibility,
    };
    use std::collections::BTreeMap;

    fn metric(mean: f64) -> MetricStat {
        MetricStat {
            mean,
            p50: None,
            p99: None,
        }
    }

    #[test]
    fn baseline_compatibility_rejects_model_mismatch() {
        let baseline = Baseline {
            model: Some("mlx-community/Qwen3-0.6B-4bit".into()),
            prompt_tokens: Some(20),
            generation_tokens: Some(64),
            metrics: BTreeMap::new(),
            recorded_at: None,
            notes: None,
        };
        let err =
            validate_baseline_compatibility(&baseline, "mlx-community/Qwen3.5-4B-MLX-4bit", 20, 64)
                .unwrap_err()
                .to_string();
        assert!(err.contains("baseline model mismatch"), "err={err}");
    }

    #[test]
    fn baseline_compatibility_rejects_token_mismatch() {
        let baseline = Baseline {
            model: None,
            prompt_tokens: Some(20),
            generation_tokens: Some(64),
            metrics: BTreeMap::new(),
            recorded_at: None,
            notes: None,
        };
        let err = validate_baseline_compatibility(&baseline, "model", 32, 64)
            .unwrap_err()
            .to_string();
        assert!(err.contains("baseline prompt_tokens mismatch"), "err={err}");
    }

    #[test]
    fn compare_baseline_fails_regression_beyond_threshold() {
        let mut baseline_metrics = BTreeMap::new();
        baseline_metrics.insert("prompt_tps".into(), metric(100.0));
        baseline_metrics.insert("generation_tps".into(), metric(50.0));
        baseline_metrics.insert("ttft_ms".into(), metric(10.0));

        let baseline = Baseline {
            model: None,
            prompt_tokens: None,
            generation_tokens: None,
            metrics: baseline_metrics,
            recorded_at: None,
            notes: None,
        };

        let mut current = BTreeMap::new();
        current.insert("prompt_tps".into(), metric(95.0));
        current.insert("generation_tps".into(), metric(40.0));
        current.insert("ttft_ms".into(), metric(10.5));

        assert!(!compare_baseline(&baseline, &current));
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let sorted = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&sorted, 50.0), 2.0);
        assert_eq!(percentile(&sorted, 99.0), 4.0);
    }
}
