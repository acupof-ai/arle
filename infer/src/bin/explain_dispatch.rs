//! `explain-dispatch` — operator-facing introspection for the resolved
//! [`DispatchPolicy`] **and** the oplib dispatch plan a hypothetical shape would
//! select.
//!
//! Governance background:
//! [`docs/reviews/2026-05-29-gpu-dispatch-governance-analysis.md`] and
//! [`docs/plans/gpu-dispatch-governance.md`]. `dispatch_policy.rs` is the
//! **Declare** gate: every dispatch-affecting env knob is parsed exactly once
//! into one inspectable struct. This binary is the operator-facing answer to
//! "which dispatch knobs are active right now" — it resolves
//! [`DispatchPolicy::from_env`] in the current environment and prints each
//! field with its env var name and active/default state, one readable line per
//! knob, e.g.:
//!
//! ```text
//! INFER_MARLIN_W4_FP8_PREFILL          marlin_w4_fp8_prefill = false (default)
//! ```
//!
//! With **no query flags** this prints exactly that policy dump and nothing
//! else.
//!
//! When a query is supplied (`--weight-format …`, `--qo-heads …`, etc.) the
//! binary additionally prints a *resolved dispatch plan* — the concrete answer
//! to the governance analysis's "每个 GPU 走的链路不清晰" (per-SKU path
//! unclear). For the queried `weight_format × batch × alignment-bools`, it calls
//! the backend-neutral [`oplib::linear::plan`] for **both** the decode and
//! prefill phases and prints the selected [`LinearKernel`] label; for the
//! queried `(qo,kv,head-dim)` it calls [`oplib::attention::head_config`] /
//! [`head_config_hd128`](oplib::attention::head_config_hd128) and prints the
//! resolved head specialization, or the canonical "no precompiled kernel for
//! this config" hard-fail string — demonstrating the runtime hard-fail is now
//! answerable on CPU.
//!
//! No feature gate: `infer::dispatch_policy` and `infer::oplib::{linear,
//! attention}` are backend-independent (their structs and `plan`/`head_config`
//! functions are pure), so this resolves, prints, and *answers the query* under
//! every feature set, including the host-only `no-cuda` / `cpu` builds an
//! operator (or a Mac with no nvcc/GPU) would use to inspect a deployment.

use infer::dispatch_policy::DispatchPolicy;
use infer::oplib::attention::{head_config, head_config_hd128};
use infer::oplib::linear::{LinearPhase, LinearPlanInputs, WeightFormat, plan};

/// Render a boolean knob: env var name, field name, value, and whether the
/// value is the compiled-in default (`false`) or an active opt-in (`true`).
fn line_bool(env_var: &str, field: &str, value: bool) -> String {
    let state = if value { "active" } else { "default" };
    format!("{env_var:<36} {field} = {value} ({state})")
}

/// Render the numeric DSv4 threshold knob. The default is `4`; any other
/// resolved value reflects an active override (legacy `< 1` / unparseable
/// inputs already fold back to `4` inside `DispatchPolicy::from_env`).
fn line_usize(env_var: &str, field: &str, value: usize, default: usize) -> String {
    let state = if value == default {
        "default"
    } else {
        "active"
    };
    format!("{env_var:<36} {field} = {value} ({state})")
}

/// Print the resolved [`DispatchPolicy`] — the Declare gate dump. This is the
/// original (and, with no query flags, the *only*) output of this binary.
fn print_policy(policy: &DispatchPolicy) {
    println!("resolved DispatchPolicy (Declare gate):");
    println!(
        "{}",
        line_bool(
            "INFER_MARLIN_W4_FP8_PREFILL",
            "marlin_w4_fp8_prefill",
            policy.marlin_w4_fp8_prefill,
        )
    );
    println!(
        "{}",
        line_bool(
            "INFER_HYBRID_W4A8_PREFILL",
            "hybrid_w4a8_prefill",
            policy.hybrid_w4a8_prefill,
        )
    );
    println!(
        "{}",
        line_bool(
            "INFER_MARLIN_W4A8_AUTOCONFIG",
            "marlin_w4a8_autoconfig",
            policy.marlin_w4a8_autoconfig,
        )
    );
    println!(
        "{}",
        line_bool(
            "INFER_R4_W4A16_GEMV_OVERRIDE",
            "r4_w4a16_gemv_override",
            policy.r4_w4a16_gemv_override,
        )
    );
    println!(
        "{}",
        line_bool(
            "INFER_DETERMINISTIC",
            "deterministic_gemm",
            policy.deterministic_gemm,
        )
    );
    println!(
        "{}",
        line_bool(
            "INFER_TILELANG_BF16_SPLIT_KV",
            "tilelang_bf16_split_kv",
            policy.tilelang_bf16_split_kv,
        )
    );
    println!(
        "{}",
        line_bool("INFER_PREFILL_GRAPH", "prefill_graph", policy.prefill_graph)
    );
    println!(
        "{}",
        line_usize(
            "ARLE_DSV4_GROUPED_GEMM_M_THRESHOLD",
            "dsv4_grouped_gemm_m_threshold",
            policy.dsv4_grouped_gemm_m_threshold,
            4,
        )
    );
}

/// Parsed `--weight-format <name>` value, mapped onto the backend-neutral
/// [`WeightFormat`] the oplib selector reads. Accepts the enum variant names
/// verbatim (case-insensitive), so the flag mirrors `WeightFormat`'s spelling.
fn parse_weight_format(name: &str) -> Result<WeightFormat, String> {
    let wf = match name.to_ascii_lowercase().as_str() {
        "densebf16" | "bf16" => WeightFormat::DenseBf16,
        "w8a16" => WeightFormat::W8A16,
        "w4a16" => WeightFormat::W4A16,
        "w2a16" => WeightFormat::W2A16,
        "marlinw4a8" => WeightFormat::MarlinW4A8,
        "ggufq3k" | "q3k" => WeightFormat::GgufQ3K,
        "ggufq4k" | "q4k" => WeightFormat::GgufQ4K,
        "ggufq5k" | "q5k" => WeightFormat::GgufQ5K,
        "ggufq6k" | "q6k" => WeightFormat::GgufQ6K,
        "turboquant" => WeightFormat::TurboQuant,
        "dsv4fp8blockscaled" | "dsv4fp8" => WeightFormat::Dsv4Fp8BlockScaled,
        "dsv4fp4blockscaled" | "dsv4fp4" => WeightFormat::Dsv4Fp4BlockScaled,
        other => {
            return Err(format!(
                "unknown --weight-format '{other}'; valid: DenseBf16, W8A16, W4A16, \
                 W2A16, MarlinW4A8, GgufQ3K, GgufQ4K, GgufQ5K, GgufQ6K, TurboQuant, \
                 Dsv4Fp8BlockScaled, Dsv4Fp4BlockScaled"
            ));
        }
    };
    Ok(wf)
}

/// Parse a flag's `--name value` argument as the requested type, surfacing a
/// clear error if the value is missing or unparseable.
fn parse_arg<T: std::str::FromStr>(flag: &str, value: Option<&str>) -> Result<T, String> {
    let raw = value.ok_or_else(|| format!("{flag} requires a value"))?;
    raw.parse::<T>()
        .map_err(|_| format!("{flag}: could not parse '{raw}'"))
}

/// Parse a `--flag true|false|1|0` boolean. The alignment flags default to the
/// common (aligned) case, so a bare run that only sets `--weight-format`
/// reflects the typical CUDA-side alignment outcome.
fn parse_bool(flag: &str, value: Option<&str>) -> Result<bool, String> {
    match value.map(str::to_ascii_lowercase) {
        Some(v) if v == "true" || v == "1" || v == "yes" => Ok(true),
        Some(v) if v == "false" || v == "0" || v == "no" => Ok(false),
        Some(other) => Err(format!("{flag}: expected true|false|1|0, got '{other}'")),
        None => Err(format!("{flag} requires a value")),
    }
}

/// The collected, parsed dispatch-query inputs. `None` for a sub-query means it
/// was not requested (no flags for that operator family were given).
struct Query {
    /// Linear inputs, present when `--weight-format` was supplied. `batch` and
    /// the alignment bools default per the documented common case.
    linear: Option<LinearQuery>,
    /// Attention inputs, present when `--qo-heads` and `--kv-heads` were
    /// supplied.
    attention: Option<AttentionQuery>,
}

/// Linear query fields (mirrors [`LinearPlanInputs`] minus the per-phase
/// `phase`, which the binary sweeps over Decode + Prefill itself).
struct LinearQuery {
    weight_format: WeightFormat,
    batch: usize,
    is_hybrid_w4_marlin: bool,
    has_marlin: bool,
    marlin_prefill_aligned: bool,
    hybrid_w4a8_aligned: bool,
    marlin_w4a8_aligned: bool,
    hybrid_w4_fp8_aligned: bool,
}

/// Attention query fields.
struct AttentionQuery {
    qo_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

const USAGE: &str = "\
explain-dispatch — print the resolved DispatchPolicy (Declare gate), and,
when a query is supplied, the oplib dispatch plan for a hypothetical
shape/quant/heads. Backend-neutral; runs on CPU with no GPU/nvcc.

USAGE:
  explain-dispatch                       # policy dump only (no query)
  explain-dispatch [LINEAR-QUERY] [ATTENTION-QUERY]

LINEAR QUERY (triggered by --weight-format):
  --weight-format <FMT>   DenseBf16 | W8A16 | W4A16 | W2A16 | MarlinW4A8 |
                          GgufQ3K | GgufQ4K | GgufQ5K | GgufQ6K | TurboQuant |
                          Dsv4Fp8BlockScaled | Dsv4Fp4BlockScaled
  --batch <N>             rows/tokens in the batch (default 1)
  --is-hybrid-w4-marlin <b>     weight.is_hybrid_w4_marlin()        (default false)
  --has-marlin <b>              weight.has_marlin()                 (default false)
  --marlin-prefill-aligned <b>  marlin_prefill_aligned(w).is_ok()   (default true)
  --hybrid-w4a8-aligned <b>     hybrid_w4a8_aligned(w).is_ok()      (default true)
  --marlin-w4a8-aligned <b>     marlin_w4a8_aligned(w).is_ok()      (default true)
  --hybrid-w4-fp8-aligned <b>   hybrid_w4_fp8_aligned(w).is_ok()    (default true)
    (the *-aligned / *-marlin bools mirror the CUDA-side checks normally
     computed off a &DeviceMatrix; exposed here so the CPU query is complete.)

ATTENTION QUERY (triggered by --qo-heads + --kv-heads):
  --qo-heads <N>          number of query/output heads
  --kv-heads <N>          number of key/value heads
  --head-dim <128|256>    selects head_config_hd128() vs head_config() (default 256)

  -h, --help              print this usage and exit

The query section is purely additive; with no query flags the output is the
policy dump alone.";

/// The outcome of CLI parsing. `Help` short-circuits to the usage banner;
/// `Resolved(None)` is the no-arg, policy-dump-only run (identical to the
/// original behavior); `Resolved(Some(_))` carries a dispatch query.
enum ParseOutcome {
    Help,
    Resolved(Option<Query>),
}

/// Parse the CLI into a [`ParseOutcome`]. Returns `Resolved(None)` for a no-arg
/// run (policy-dump-only — identical to the original behavior) and `Help` when
/// `-h`/`--help` is requested; the caller (`main`) owns the process exit.
fn parse_query(args: &[String]) -> Result<ParseOutcome, String> {
    if args.is_empty() {
        return Ok(ParseOutcome::Resolved(None));
    }

    // Linear inputs (with documented defaults; alignment defaults to the common
    // aligned case so a bare `--weight-format X` reflects the typical SKU).
    let mut weight_format: Option<WeightFormat> = None;
    let mut batch: usize = 1;
    let mut is_hybrid_w4_marlin = false;
    let mut has_marlin = false;
    let mut marlin_prefill_aligned = true;
    let mut hybrid_w4a8_aligned = true;
    let mut marlin_w4a8_aligned = true;
    let mut hybrid_w4_fp8_aligned = true;

    // Attention inputs.
    let mut qo_heads: Option<usize> = None;
    let mut kv_heads: Option<usize> = None;
    let mut head_dim: usize = 256;

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        // `next` is the value token; `None` past the end (parse helpers error).
        let next = args.get(i + 1).map(String::as_str);
        match flag {
            "-h" | "--help" => return Ok(ParseOutcome::Help),
            "--weight-format" => {
                weight_format =
                    Some(parse_weight_format(next.ok_or_else(|| {
                        "--weight-format requires a value".to_string()
                    })?)?);
                i += 2;
            }
            "--batch" => {
                batch = parse_arg("--batch", next)?;
                i += 2;
            }
            "--is-hybrid-w4-marlin" => {
                is_hybrid_w4_marlin = parse_bool(flag, next)?;
                i += 2;
            }
            "--has-marlin" => {
                has_marlin = parse_bool(flag, next)?;
                i += 2;
            }
            "--marlin-prefill-aligned" => {
                marlin_prefill_aligned = parse_bool(flag, next)?;
                i += 2;
            }
            "--hybrid-w4a8-aligned" => {
                hybrid_w4a8_aligned = parse_bool(flag, next)?;
                i += 2;
            }
            "--marlin-w4a8-aligned" => {
                marlin_w4a8_aligned = parse_bool(flag, next)?;
                i += 2;
            }
            "--hybrid-w4-fp8-aligned" => {
                hybrid_w4_fp8_aligned = parse_bool(flag, next)?;
                i += 2;
            }
            "--qo-heads" => {
                qo_heads = Some(parse_arg("--qo-heads", next)?);
                i += 2;
            }
            "--kv-heads" => {
                kv_heads = Some(parse_arg("--kv-heads", next)?);
                i += 2;
            }
            "--head-dim" => {
                head_dim = parse_arg("--head-dim", next)?;
                i += 2;
            }
            other => {
                return Err(format!(
                    "unknown flag '{other}' (run with --help for usage)"
                ));
            }
        }
    }

    let linear = weight_format.map(|weight_format| LinearQuery {
        weight_format,
        batch,
        is_hybrid_w4_marlin,
        has_marlin,
        marlin_prefill_aligned,
        hybrid_w4a8_aligned,
        marlin_w4a8_aligned,
        hybrid_w4_fp8_aligned,
    });

    let attention = match (qo_heads, kv_heads) {
        (Some(qo_heads), Some(kv_heads)) => Some(AttentionQuery {
            qo_heads,
            kv_heads,
            head_dim,
        }),
        (None, None) => None,
        _ => return Err("attention query needs BOTH --qo-heads and --kv-heads".to_string()),
    };

    if linear.is_none() && attention.is_none() {
        return Err(
            "a query needs --weight-format (linear) and/or --qo-heads + --kv-heads \
             (attention); run with --help for usage"
                .to_string(),
        );
    }

    Ok(ParseOutcome::Resolved(Some(Query { linear, attention })))
}

/// Build [`LinearPlanInputs`] for a given phase from the parsed linear query.
fn linear_inputs(q: &LinearQuery, phase: LinearPhase) -> LinearPlanInputs {
    LinearPlanInputs {
        weight_format: q.weight_format,
        batch: q.batch,
        phase,
        is_hybrid_w4_marlin: q.is_hybrid_w4_marlin,
        has_marlin: q.has_marlin,
        marlin_prefill_aligned: q.marlin_prefill_aligned,
        hybrid_w4a8_aligned: q.hybrid_w4a8_aligned,
        marlin_w4a8_aligned: q.marlin_w4a8_aligned,
        hybrid_w4_fp8_aligned: q.hybrid_w4_fp8_aligned,
    }
}

/// Print the resolved oplib dispatch plan for the query, under the resolved
/// `policy`. Linear: both phases via `oplib::linear::plan`. Attention: the
/// selected head specialization or the canonical hard-fail string.
fn print_plan(query: &Query, policy: &DispatchPolicy) {
    println!();
    println!("resolved dispatch plan (oplib selection — CPU, backend-neutral):");

    if let Some(q) = &query.linear {
        let decode = plan(&linear_inputs(q, LinearPhase::Decode), policy);
        let prefill = plan(&linear_inputs(q, LinearPhase::Prefill), policy);
        println!(
            "  linear decode(batch={})   -> {}",
            q.batch,
            decode.kernel_label()
        );
        println!(
            "  linear prefill(batch={})  -> {}",
            q.batch,
            prefill.kernel_label()
        );
    }

    if let Some(q) = &query.attention {
        match q.head_dim {
            256 => match head_config(q.qo_heads, q.kv_heads) {
                Ok(cfg) => println!(
                    "  attention hd256 (qo={}, kv={}) -> {:?}",
                    q.qo_heads, q.kv_heads, cfg
                ),
                Err(msg) => println!(
                    "  attention hd256 (qo={}, kv={}) -> ERR: {}",
                    q.qo_heads, q.kv_heads, msg
                ),
            },
            128 => match head_config_hd128(q.qo_heads, q.kv_heads) {
                Ok(cfg) => println!(
                    "  attention hd128 (qo={}, kv={}) -> {:?}",
                    q.qo_heads, q.kv_heads, cfg
                ),
                Err(msg) => println!(
                    "  attention hd128 (qo={}, kv={}) -> ERR: {}",
                    q.qo_heads, q.kv_heads, msg
                ),
            },
            other => println!(
                "  attention (qo={}, kv={}) -> ERR: unsupported --head-dim {} \
                 (precompiled head specializations exist only for 128 and 256)",
                q.qo_heads, q.kv_heads, other
            ),
        }
    }
}

fn main() -> std::process::ExitCode {
    // Skip argv[0] (the binary path).
    let args: Vec<String> = std::env::args().skip(1).collect();
    let outcome = match parse_query(&args) {
        Ok(outcome) => outcome,
        Err(msg) => {
            eprintln!("error: {msg}");
            return std::process::ExitCode::from(2);
        }
    };

    // `--help` short-circuits to the usage banner — no policy dump, clean exit.
    let query = match outcome {
        ParseOutcome::Help => {
            println!("{USAGE}");
            return std::process::ExitCode::SUCCESS;
        }
        ParseOutcome::Resolved(query) => query,
    };

    // Resolve directly (not via the process-wide `dispatch_policy()` cache) so
    // this is a clean, side-effect-free read of the current environment.
    let policy = DispatchPolicy::from_env();

    // The policy dump is always printed first — unchanged from the original
    // behavior; with no query this is the entire output.
    print_policy(&policy);

    // The plan section is purely additive — emitted only when a query is given.
    if let Some(query) = query {
        print_plan(&query, &policy);
    }

    std::process::ExitCode::SUCCESS
}
