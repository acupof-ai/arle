use std::path::PathBuf;

use clap::{ArgGroup, Args as ClapArgs, Parser, Subcommand, ValueEnum};

pub(crate) const DEFAULT_OPD_LOGITS_WINDOW_SIZE: usize = 32;

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("expected a positive integer, got '{value}'"))?;
    if parsed == 0 {
        return Err("value must be at least 1".to_string());
    }
    Ok(parsed)
}

/// Like `parse_positive_usize` but also accepts `0` as an "auto" sentinel —
/// used by `--max-tokens` so users can ask the CLI to read the model's
/// `max_position_embeddings` (or `context_length`) at startup instead of
/// pinning a fixed cap. Negative or non-integer input is still rejected.
fn parse_max_tokens_or_auto(value: &str) -> Result<usize, String> {
    let trimmed = value.trim();
    if trimmed == "auto" || trimmed == "0" {
        return Ok(0);
    }
    parse_positive_usize(trimmed)
}

/// `--kv-disk` spill root. Clap's stock `PathBuf` parser rejects empty input,
/// which would reject the bare-flag `default_missing_value = ""` sentinel the
/// serve lowering resolves to the default root — so accept any string here.
fn parse_kv_disk(value: &str) -> Result<PathBuf, String> {
    Ok(PathBuf::from(value))
}

/// KV tier budget (`--kv-dram` / `--kv-disk-limit`): "off"/"0" disables,
/// "N%" or a bare float in (0,1) is a fraction of the measured resource, and
/// anything else is bytes with an optional binary suffix (K/KB/KiB … T/TB/TiB,
/// all treated as binary multiples).
fn parse_kv_budget(s: &str) -> Result<infer_api::KvTierBudget, String> {
    use infer_api::KvTierBudget;
    let s = s.trim();
    if s.eq_ignore_ascii_case("off") || s == "0" {
        return Ok(KvTierBudget::Off);
    }
    if let Some(pct) = s.strip_suffix('%') {
        let pct: f64 = pct
            .trim()
            .parse()
            .map_err(|_| format!("expected a percentage like \"50%\", got '{s}'"))?;
        if !(pct > 0.0 && pct <= 100.0) {
            return Err(format!("percentage must be in (0, 100], got '{s}'"));
        }
        return Ok(KvTierBudget::Fraction(pct / 100.0));
    }
    if let Ok(frac) = s.parse::<f64>()
        && frac > 0.0
        && frac < 1.0
    {
        return Ok(KvTierBudget::Fraction(frac));
    }
    const SUFFIXES: [(&str, usize); 12] = [
        ("kib", 1 << 10),
        ("kb", 1 << 10),
        ("k", 1 << 10),
        ("mib", 1 << 20),
        ("mb", 1 << 20),
        ("m", 1 << 20),
        ("gib", 1 << 30),
        ("gb", 1 << 30),
        ("g", 1 << 30),
        ("tib", 1 << 40),
        ("tb", 1 << 40),
        ("t", 1 << 40),
    ];
    let lower = s.to_ascii_lowercase();
    let (digits, mult) = SUFFIXES
        .iter()
        .find_map(|&(suffix, mult)| lower.strip_suffix(suffix).map(|d| (d, mult)))
        .unwrap_or((lower.as_str(), 1));
    let value: usize = digits.trim().parse().map_err(|_| {
        format!(
            "expected bytes (\"16GiB\"), a percentage (\"50%\"), a fraction in (0,1), \
             or 0/off; got '{s}'"
        )
    })?;
    let bytes = value
        .checked_mul(mult)
        .ok_or_else(|| format!("byte budget overflows: '{s}'"))?;
    if bytes == 0 {
        return Err("byte budget must be positive (use 0/off to disable)".to_string());
    }
    Ok(KvTierBudget::Bytes(bytes))
}

/// `--speculative-tokens` draft depth. The Metal DFlash runtime clamps the
/// effective block size into `[2, draft_head_block_size]` and rejects a depth
/// below 2, so reject `0`/`1` at the CLI boundary with a clear message.
fn parse_speculative_tokens(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("expected a positive integer, got '{value}'"))?;
    if parsed < 2 {
        return Err("speculative draft depth must be at least 2".to_string());
    }
    Ok(parsed)
}

fn parse_temperature(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("expected a finite number, got '{value}'"))?;
    if !parsed.is_finite() {
        return Err("temperature must be finite".to_string());
    }
    if parsed < 0.0 {
        return Err("temperature must be >= 0.0".to_string());
    }
    Ok(parsed)
}

fn parse_positive_temperature(value: &str) -> Result<f32, String> {
    let parsed = parse_temperature(value)?;
    if parsed <= 0.0 {
        return Err("temperature must be > 0.0".to_string());
    }
    Ok(parsed)
}

fn parse_unit_interval(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("expected a finite number, got '{value}'"))?;
    if parsed.is_finite() && (0.0..=1.0).contains(&parsed) {
        return Ok(parsed);
    }
    Err("value must be finite and in [0.0, 1.0]".to_string())
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum TracePromptsMode {
    On,
    Off,
}

// `keep_prompts` is only consumed by the trajectory writer, which is currently
// compiled for the interactive CUDA/Metal/CPU front door. Mirror that gate here
// so `cargo clippy -p cli -- -D warnings` on serve-only builds stays clean.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl TracePromptsMode {
    pub(crate) fn keep_prompts(self) -> bool {
        matches!(self, Self::On)
    }
}

fn parse_trace_path(value: &str) -> Result<PathBuf, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("trace path must not be empty".to_string());
    }
    Ok(PathBuf::from(trimmed))
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum ServeBackendArg {
    #[value(alias = "arle", alias = "native")]
    Auto,
    Cpu,
    Metal,
    Cuda,
    Hip,
    Vulkan,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum ModelFamilyArg {
    Auto,
    Qwen35,
}

impl ModelFamilyArg {
    pub(crate) fn as_train_family(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Qwen35 => "qwen35",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum SaveDtypeArg {
    F32,
    Bf16,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum TapeDtypeArg {
    F32,
    Bf16,
}

impl TapeDtypeArg {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn as_tape_dtype(self) -> autograd::TapeDtype {
        match self {
            Self::F32 => autograd::TapeDtype::F32,
            Self::Bf16 => autograd::TapeDtype::Bf16,
        }
    }
}

/// Teacher source for `agent-opd --gkd` (per-token teacher-KL) replay.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum GkdTeacherArg {
    /// A slow EMA of the student (a stabilizing self-teacher, zero extra model).
    Ema,
    /// The student's frozen initial adapter+base snapshot (never EMA-updated).
    #[value(name = "self")]
    SelfFrozen,
}

/// `agent-opd --sync`: when the trained LoRA re-merges into the serve engine.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum SyncArg {
    /// After every task group (strict on-policy; default).
    EveryGroup,
    /// Once at round end (fewer merges; intra-round rollouts go stale).
    EveryRound,
}

/// Policy-update preset for `agent-opd` (`--update-strategy`). Each value maps
/// to a `train::update_strategy::UpdatePreset` constructor — the algorithm is
/// data, not a code path; see `TrainAgentOpdArgs::update_preset`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum UpdateStrategyArg {
    /// Reject failures, masked CE on full passes (default, unchanged).
    RejectionCe,
    /// SAO Phase 1: hard-gated per-token PG with a batch-centered advantage.
    SaoDis,
    /// SAO Phase 2: per-token Skip-Obs GAE from a learned value critic.
    SaoValue,
    /// GRPO: group-normalized advantage, clamped per-token ratio.
    Grpo,
    /// DAPO: clip-higher + dynamic sampling + token-level (batch-mean) loss.
    Dapo,
    /// Dr.GRPO: GRPO minus its length/std biases (fixed-constant normalizer).
    DrGrpo,
    /// GSPO: sequence-level clipped importance ratio.
    Gspo,
    /// CISPO: detached one-sided clamped-IS weight, batch-mean loss.
    Cispo,
}

impl SaveDtypeArg {
    pub(crate) fn as_train_dtype(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum PretrainPresetArg {
    #[value(name = "tiny-3m")]
    Tiny3m,
    #[value(name = "small-25m")]
    Small25m,
    #[value(name = "small-30m")]
    Small30m,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum KlDirectionArg {
    Forward,
    Reverse,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum OpdKlMaskArg {
    /// Mask prompt-prefix rows and train only on sampled completion rows.
    #[value(name = "completion", alias = "completion-only")]
    Completion,
    /// Include prompt-prefix and completion rows in the KL objective.
    Full,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum OpdTeacherRuntimeArg {
    /// Load the teacher through the train-side Qwen3.5 autograd loader.
    #[value(name = "in-process", alias = "in_process")]
    InProcess,
    /// Load the teacher through infer-api and score raw logits through the runtime.
    Infer,
    /// Score raw logits over HTTP against a separately-served teacher (--teacher-url).
    /// Keeps the teacher off the training GPU — the only way the full FP8 teacher
    /// fits alongside an FP8 student on one card.
    Api,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum OpdSftAnchorArg {
    /// Use the sampled student rollout as the optional GKD hard-token CE anchor.
    #[value(name = "student-rollout", alias = "student_rollout")]
    StudentRollout,
    /// Use completion/target tokens from --prompts-file as the hard-token CE anchor.
    #[value(name = "corpus-truth", alias = "corpus_truth")]
    CorpusTruth,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum LrScheduleArg {
    /// Keep AdamW at the fixed --lr value for every step.
    Fixed,
    /// Linear warmup, then cosine decay to 10% of --lr.
    Cosine,
}

#[derive(Debug, Clone, ClapArgs)]
pub(crate) struct RenderArgs {
    /// Print the fully resolved execution plan without running the job.
    #[arg(long, default_value_t = false)]
    pub(crate) dry_run: bool,

    /// Render `--dry-run` output as JSON for scripts and CI.
    #[arg(long, default_value_t = false, requires = "dry_run")]
    pub(crate) json: bool,
}

#[derive(Parser)]
#[command(
    name = "arle",
    about = "ARLE local agent, training, and dataset CLI",
    after_help = "Common flows:\n  arle                                       Start the interactive agent REPL.\n  arle run                                   Explicit alias for the interactive agent REPL.\n  arle run --prompt \"Summarize this repo\"    Run one prompt and exit.\n  arle run --stdin --json < prompt.txt       Read one prompt from stdin and emit JSON.\n  arle serve --model-path /path/to/model      Start the OpenAI-compatible server.\n  arle --doctor                              Inspect the local environment and model resolution.\n  arle train env                             Print train-time environment diagnostics.\n  arle train opd --smoke --steps 5            Run the OPD smoke path.",
    group(ArgGroup::new("inspection_mode").args(["doctor", "list_models"]))
)]
pub(crate) struct Args {
    #[arg(long, hide = true, default_value_t = false)]
    pub(crate) kernel_build_id: bool,

    /// Path to model directory or HuggingFace model ID.
    /// If omitted, the CLI auto-detects a local model from common directories and HF cache.
    #[arg(long)]
    pub(crate) model_path: Option<String>,

    /// Print a local environment/model-resolution diagnostic report and exit.
    #[arg(long, default_value_t = false)]
    pub(crate) doctor: bool,

    /// Print discovered and recommended models, then exit.
    #[arg(long, default_value_t = false)]
    pub(crate) list_models: bool,

    /// Render `--doctor` / `--list-models` output as JSON for scripts and CI.
    #[arg(long, default_value_t = false, requires = "inspection_mode")]
    pub(crate) json: bool,

    /// Fail with a non-zero exit code when `--doctor` reports warnings.
    #[arg(
        long,
        default_value_t = false,
        requires = "doctor",
        conflicts_with = "list_models"
    )]
    pub(crate) strict: bool,

    #[command(subcommand)]
    pub(crate) command: Option<CliCommand>,

    /// Maximum agent turns (generate-execute cycles) per query.
    /// 250 lets multi-step tool plans run to completion on long tasks
    /// (project surveys, refactors, audits). The agent still stops as
    /// soon as it produces a final answer, so a high cap costs nothing
    /// on short turns. Override with `--max-turns N`.
    #[arg(long, default_value_t = 250, value_parser = parse_positive_usize)]
    pub(crate) max_turns: usize,

    /// Maximum tokens to generate per turn. Default `0` means "auto" —
    /// the CLI reads `max_position_embeddings` (or `context_length` for
    /// GGUF) from the model's config at startup and uses that as the
    /// per-turn cap. Pass `--max-tokens N` to pin an explicit value.
    /// Pass `--max-tokens auto` (or `0`) to make the auto-resolution
    /// explicit. If config can't be read, falls back to 262144 (256K).
    #[arg(long, default_value_t = 0, value_parser = parse_max_tokens_or_auto)]
    pub(crate) max_tokens: usize,

    /// Sampling temperature (0.0 = greedy)
    #[arg(
        long,
        default_value_t = 0.0,
        value_parser = parse_temperature,
        allow_hyphen_values = true
    )]
    pub(crate) temperature: f32,

    /// Disable CUDA graph capture (CUDA only; no-op on other backends).
    #[arg(long, default_value_t = false)]
    pub(crate) no_cuda_graph: bool,

    /// Disable built-in shell/python tools for the local agent runtime.
    /// Also honored per-run via `arle run --no-tools`.
    #[arg(long, default_value_t = false)]
    pub(crate) no_tools: bool,

    /// Skip interactive model selection (use auto-discovery)
    #[arg(long, default_value_t = false)]
    pub(crate) non_interactive: bool,

    /// Path to a JSONL file that will receive one trajectory record per
    /// agent turn (Phase 1 / v1 schema). When unset, no trajectory is
    /// written. See `docs/projects/agent-trajectory-export.md` for the
    /// canonical schema.
    #[arg(long, value_parser = parse_trace_path)]
    pub(crate) trace: Option<PathBuf>,

    /// Whether to record the full ChatML prompt in each trajectory's
    /// `sub_turns[].prompt_text`. `off` writes JSON `null` for that
    /// field — useful when the prompt would dominate trace size or
    /// leak operator data.
    #[arg(long, value_enum, default_value_t = TracePromptsMode::On)]
    pub(crate) trace_prompts: TracePromptsMode,

    /// Front-end agent to launch on a no-args interactive start. `eli` serves
    /// the picked model locally and hands the session to the sibling Eli agent
    /// framework; `arle` uses the built-in REPL. When omitted, the choice is
    /// read from `~/.config/arle/agent.toml` (set on the first Eli launch so
    /// subsequent runs default to Eli).
    #[arg(long, value_enum)]
    pub(crate) agent: Option<AgentFrontendArg>,

    /// Launch the Eli front-end in gateway (serve) mode instead of the
    /// interactive chat REPL. Only meaningful with `--agent eli` (or a saved
    /// Eli default). Persisted as the launch mode for subsequent runs.
    #[arg(long, default_value_t = false)]
    pub(crate) gateway: bool,
}

/// Which front-end agent the no-args `arle` start launches.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum AgentFrontendArg {
    /// Built-in ARLE REPL (in-process model load).
    Arle,
    /// Sibling Eli agent framework against a local `arle serve`.
    Eli,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum CliCommand {
    /// Agent REPL and one-shot prompt execution.
    Run(Box<RunArgs>),
    /// One-shot image OCR with DeepSeek-OCR (auto-downloads the model).
    Ocr(Box<OcrArgs>),
    /// OpenAI-compatible serving through the matching backend binary.
    Serve(Box<ServeArgs>),
    /// Training jobs.
    Train(Box<TrainArgs>),
    /// Model utilities (download from Hugging Face).
    Model(Box<ModelArgs>),
}

/// OCR mode → prompt preset for DeepSeek-OCR.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum OcrMode {
    /// Plain text extraction (`<|grounding|>Free OCR.`).
    Free,
    /// Text + bounding-box grounding (`<|grounding|>OCR this image.`).
    Grounding,
    /// Convert the document to Markdown (`<|grounding|>Convert the document to markdown.`).
    Markdown,
}

#[derive(Debug, Clone, PartialEq, Eq, ClapArgs)]
#[command(
    after_help = "Examples:\n  arle ocr page.png\n  arle ocr --mode markdown scan.jpg\n  arle ocr --mode free https://example.com/receipt.png\n  arle ocr --prompt \"<|grounding|>Extract the table.\" table.png\n  arle ocr --pages 1,3,5-8 report.pdf\n  arle ocr --output out.md report.pdf\n\nThe DeepSeek-OCR model auto-downloads on first use (Metal/Apple Silicon only). PDFs are rendered page-by-page via pdftoppm."
)]
pub(crate) struct OcrArgs {
    /// Image to read — a local path or http(s) URL.
    pub(crate) image: String,

    /// OCR preset. Ignored when `--prompt` is given.
    #[arg(long, value_enum, default_value_t = OcrMode::Free)]
    pub(crate) mode: OcrMode,

    /// Override the OCR instruction prompt verbatim (include any `<|grounding|>`
    /// marker you want; do NOT add a literal `<image>` marker — the image is
    /// spliced in automatically).
    #[arg(long)]
    pub(crate) prompt: Option<String>,

    /// Model directory or HuggingFace id. Defaults to the bundled DeepSeek-OCR
    /// model and downloads it if missing.
    #[arg(long)]
    pub(crate) model_path: Option<String>,

    /// Max tokens to generate per page. `0`/`auto` uses the model context limit.
    #[arg(long, default_value_t = 0, value_parser = parse_max_tokens_or_auto)]
    pub(crate) max_tokens: usize,

    /// PDF pages to OCR: `1`, `1,3,5-8`. Omit to process every page.
    #[arg(long)]
    pub(crate) pages: Option<String>,

    /// Write the final OCR text to a file instead of stdout.
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,

    /// Emit a JSON document ({ text, model, usage }) for scripts.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

// Used only by the backend-gated `ocr` module — gate to match (the `OcrMode`
// enum + `OcrArgs` stay ungated for CLI parsing under any feature set).
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl OcrMode {
    /// The DeepSeek-OCR instruction prompt for this mode (no `<image>` marker —
    /// the engine splices the image automatically).
    pub(crate) fn prompt(self) -> &'static str {
        match self {
            Self::Free => "<|grounding|>Free OCR.",
            Self::Grounding => "<|grounding|>OCR this image.",
            Self::Markdown => "<|grounding|>Convert the document to markdown.",
        }
    }
}

#[derive(Debug, Clone, clap::Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  arle model download Qwen/Qwen3-0.6B\n  arle model download mlx-community/Qwen3.6-35B-A3B-4bit"
)]
pub(crate) struct ModelArgs {
    #[command(subcommand)]
    pub(crate) command: ModelCommand,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum ModelCommand {
    /// Download a model from Hugging Face Hub or ModelScope (config + tokenizer + sharded weights).
    Download(ModelDownloadArgs),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum ModelSourceArg {
    /// Hugging Face Hub (default, global; uses `hf-hub` + `~/.cache/huggingface/`).
    Hf,
    /// ModelScope (魔搭) — PRC-friendly mirror used by the OPD substrate.
    /// Cache lives at `~/.cache/modelscope/hub/models/{org}/{name}/`.
    Modelscope,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Examples:\n  arle model download Qwen/Qwen3-0.6B\n  arle model download --source modelscope Qwen/Qwen3-0.6B"
)]
pub(crate) struct ModelDownloadArgs {
    /// Model ID (e.g. "Qwen/Qwen3-0.6B" or "mlx-community/Qwen3.6-35B-A3B-4bit").
    /// Both HF and ModelScope accept the same `org/name` shape for Qwen-family
    /// and other dual-published repos.
    pub(crate) model_id: String,

    /// Where to download from. Defaults to `hf`; use `modelscope` from PRC
    /// networks or to feed the OPD substrate without HF reach.
    #[arg(long, value_enum, default_value_t = ModelSourceArg::Hf)]
    pub(crate) source: ModelSourceArg,

    #[command(flatten)]
    pub(crate) render: RenderArgs,
}

#[derive(Debug, Clone, PartialEq, Eq, ClapArgs)]
#[command(
    group(ArgGroup::new("run_input").args(["prompt", "stdin"])),
    after_help = "Output:\n  Plain text is written to stdout by default.\n  `--json` emits one machine-readable document with model, backend, usage, and tool-call stats.\n\nExamples:\n  arle --model-path /path/to/model run\n  arle --model-path /path/to/model run --prompt \"Summarize this repo\"\n  arle --model-path /path/to/gemma4 run --prompt \"What is in this image?\" --image https://example.com/cat.jpg\n  arle --model-path /path/to/model run --stdin --json < prompt.txt\n  arle --model-path /path/to/model run --no-tools --prompt \"No tool execution\""
)]
pub(crate) struct RunArgs {
    /// Run a single prompt and exit.
    #[arg(long)]
    pub(crate) prompt: Option<String>,

    /// Read one prompt from stdin, run it, and exit.
    #[arg(long, default_value_t = false)]
    pub(crate) stdin: bool,

    /// Attach an image to the one-shot prompt. Accepts a local path or http(s) URL.
    /// Repeat to attach multiple images. Interactive REPL uses `/image <path-or-url>`.
    #[arg(long = "image", value_name = "PATH_OR_URL")]
    pub(crate) image: Vec<String>,

    /// Render one-shot output as JSON for scripts and CI.
    #[arg(long, default_value_t = false, requires = "run_input")]
    pub(crate) json: bool,

    /// Disable built-in shell/python tools for this run.
    #[arg(long, default_value_t = false)]
    pub(crate) no_tools: bool,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "This is a thin front door over the in-process ARLE-native backend compiled into this binary.\n\nExamples:\n  arle serve --model-path /path/to/Qwen3-4B\n  arle serve --backend arle --model-path /models/Qwen3-4B --port 8000\n  arle serve --backend metal --model-path mlx-community/Qwen3-0.6B-4bit --port 8010\n  arle serve --backend vulkan --model-path /models/qwen3.gguf --port 8020\n  arle serve --backend cuda --model-path /models/Qwen3-4B"
)]
pub(crate) struct ServeArgs {
    /// Model directory or HuggingFace model ID. Defaults to the top-level --model-path.
    #[arg(long)]
    pub(crate) model_path: Option<String>,

    /// Serving backend to launch; `auto` selects the compiled backend.
    #[arg(long, value_enum, default_value_t = ServeBackendArg::Auto)]
    pub(crate) backend: ServeBackendArg,

    /// Port to listen on.
    #[arg(long, default_value_t = 8000)]
    pub(crate) port: u16,

    /// Host or IP address to bind to when the backend binary supports it.
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) bind: String,

    /// Use conservative local-machine engine budgets for smoother foreground UX.
    #[arg(long, default_value_t = false)]
    pub(crate) low_impact: bool,

    /// L2 (host DRAM) KV tier budget for the WHOLE deployment, split across
    /// TP ranks: bytes ("64GiB"), a fraction of available DRAM ("50%"), or
    /// 0/off to disable. A cap, not a reservation. DSv4 further splits each
    /// rank's share between the whole-slot park store and the prefix-state
    /// pool (50/50; park floored at one 16 MiB chunk so a small budget never
    /// silently disables it).
    #[arg(long, value_parser = parse_kv_budget, default_value = "50%")]
    pub(crate) kv_dram: infer_api::KvTierBudget,

    /// L3 (SSD/NVMe) KV spill root. A bare --kv-disk uses ARLE_KV_SSD_PATH or
    /// the platform cache dir. Every engine — including each multiproc TP
    /// worker — attaches at build; backends without a tier store fail closed.
    #[arg(long, value_name = "DIR", num_args = 0..=1, default_missing_value = "", value_parser = parse_kv_disk)]
    pub(crate) kv_disk: Option<PathBuf>,

    /// L3 cap for the WHOLE deployment (bytes or a fraction of free disk),
    /// split across TP ranks. Requires --kv-disk; default 50% of free disk.
    #[arg(long, value_parser = parse_kv_budget)]
    pub(crate) kv_disk_limit: Option<infer_api::KvTierBudget>,

    /// KV cache storage dtype. `auto` lets the backend choose its default.
    #[arg(long, value_enum, default_value_t = ServeKvCacheDtypeArg::Auto)]
    pub(crate) kv_cache_dtype: ServeKvCacheDtypeArg,

    /// Opt into write-through tiered KV memory ("infinite memory"): HBM is a
    /// bounded write-through cache, the host tier (DRAM→NVMe) is the source of
    /// truth. When a session exceeds the GPU working set, decode attends a
    /// budget-bounded resident subset (sink + top-k mean-key + local) instead of
    /// the full cache. Metal, and CUDA dense-Qwen3 (eager decode), with
    /// `--kv-cache-dtype bf16` only; default off → baseline byte-identical.
    /// See `docs/plans/2026-06-23-writethrough-tiered-kv-memory.md`.
    #[arg(long, default_value_t = false)]
    pub(crate) kv_recall: bool,

    /// Whole-process memory budget in bytes for unified-memory backends.
    /// Metal applies this before loading weights and clamps KV capacity to fit.
    #[arg(long, value_parser = parse_positive_usize, value_name = "BYTES")]
    pub(crate) memory_budget_bytes: Option<usize>,

    /// Physical memory bytes to reserve for macOS and foreground apps.
    #[arg(long, value_parser = parse_positive_usize, value_name = "BYTES")]
    pub(crate) system_reserve_bytes: Option<usize>,

    /// Allow Metal startup even when macOS swap is already materially active.
    #[arg(long, default_value_t = false)]
    pub(crate) allow_swap: bool,

    /// Soft cap on concurrently-running requests (SGLang `max_running_requests`).
    /// The executor derives hot-workspace capacity from this plus model/VRAM budget.
    #[arg(long, value_parser = parse_positive_usize)]
    pub(crate) max_running_requests: Option<usize>,

    /// Fraction of total VRAM for the static KV pool, profiled from MEASURED
    /// free VRAM after weights load (SGLang's `mem_fraction_static`). The KV
    /// token pool gets `free − total × (1 − frac)`; the rest is headroom for
    /// activations/scratch. Clamped to `[0.05, 0.97]`. Wired for CUDA dense Qwen3
    /// + Qwen3.6 + Metal (one shared pool); DSv4 keeps per-slot sizing for now.
    ///   Default 0.9.
    #[arg(long, default_value_t = 0.9)]
    pub(crate) mem_fraction_static: f64,

    /// Max prompt tokens accepted at ingress.
    #[arg(long, value_parser = parse_positive_usize)]
    pub(crate) max_prompt_tokens: Option<usize>,

    /// Max prompt plus generated tokens for one request.
    #[arg(long, value_parser = parse_positive_usize)]
    pub(crate) max_total_tokens: Option<usize>,

    /// Bound on generated tokens for chat requests that enable thinking
    /// (`chat_template_kwargs.enable_thinking=true`). `0` (the default) leaves
    /// thinking unbounded — byte-identical to before this flag. A positive
    /// value caps such requests so a slow think can't run away / time out.
    #[arg(long, default_value_t = 0)]
    pub(crate) max_thinking_tokens: usize,

    /// Per-request prefill chunk size.
    #[arg(long, value_parser = parse_positive_usize)]
    pub(crate) chunked_prefill_size: Option<usize>,

    /// Token budget for one scheduler tick (all prefill chunks in that tick
    /// combined). Sets the M dimension of the step's GEMMs, so it trades step
    /// latency against GEMM efficiency; a decode row sharing the tick waits a
    /// whole step. Unset = the shipped constant.
    #[arg(long, value_parser = parse_positive_usize)]
    pub(crate) max_num_batched_tokens: Option<usize>,

    /// Rotate waiters past the running cap by parking whole decode slots into
    /// the KV tier (whole-slot-tier models only; others fail closed at start).
    #[arg(long, default_value_t = false)]
    pub(crate) kv_oversubscription: bool,

    /// Dump every raw `/v1/messages` request body to
    /// `<dir>/<epoch_ms>_<seq>.json` (CC-trajectory capture for
    /// `arle train cc-convert`). Fire-and-forget; unset = zero cost.
    #[arg(long, value_name = "DIR")]
    pub(crate) dump_messages_dir: Option<PathBuf>,

    /// LoRA adapter safetensors (train `--save-lora-adapters` output) re-merged
    /// into the resident student weights once at startup. CUDA Qwen3.5/3.6 only.
    #[arg(long, value_name = "FILE")]
    pub(crate) lora_adapters: Option<PathBuf>,

    /// LoRA alpha for --lora-adapters (`scale = alpha / rank`; rank is read
    /// from the adapter tensor shapes).
    #[arg(long, default_value_t = 32.0)]
    pub(crate) lora_alpha: f32,

    /// Speculative decode route (CUDA): checkpoint-native `mtp` (DSv4) or the
    /// external `dspark` block drafter (Qwen3.6, dir via `--mtp-draft-model`).
    #[arg(long, value_enum, default_value_t = ServeSpecTypeArg::None)]
    pub(crate) spec_type: ServeSpecTypeArg,

    /// Multi-GPU collective backend for small decode-path messages (CUDA
    /// multi-rank only). `nccl` (default): plain NCCL — the 2026-06-10 matched
    /// A/B measured the one-shot path wall-neutral on single-node H20 (the
    /// decode wall is rank-skew-bound, not protocol-bound). `auto` boots the
    /// one-shot custom AR/AG with a self-test and loud degrade — kept for
    /// multi-node and the fused-AR+norm follow-up.
    #[arg(long, value_enum, default_value_t = ServeCommBackendArg::Nccl)]
    pub(crate) comm_backend: ServeCommBackendArg,

    /// External drafter checkpoint dir. Consumed by `--spec-type dspark`
    /// (DSpark/DFlash block drafter, CUDA Qwen3.6); rejected otherwise.
    #[arg(long, value_name = "PATH_OR_REPO")]
    pub(crate) mtp_draft_model: Option<String>,

    /// Verify-step cost model `step_ms = bias + row · verify_rows` driving the
    /// DSpark goodput budget (needs a confidence head in the drafter). Default
    /// is the H20 ThinkingCap-27B c=16 measurement.
    #[arg(long, default_value_t = 211.0, value_name = "MS")]
    pub(crate) dspark_sps_bias_ms: f32,

    /// Marginal verify-row cost for the DSpark goodput budget.
    #[arg(long, default_value_t = 0.53, value_name = "MS")]
    pub(crate) dspark_sps_row_ms: f32,

    /// Cap the DSpark draft block length (checkpoint value if unset). The chain
    /// stops at the first rejection, so every position past the accepted prefix
    /// costs a draft forward and a verify row that can never commit: TC-27B +
    /// DFlash at block 16 keeps 3.28 tokens per chain and discards 79.5% of the
    /// drafted work. This is the fixed-length stand-in for the confidence head's
    /// adaptive truncation.
    #[arg(long, value_name = "N")]
    pub(crate) dspark_block_size: Option<usize>,

    /// Confidence-head early stop: a DSpark block is cut at the first position
    /// whose cumulative survival falls under this. 0 turns the gate off and
    /// drafts the whole block, which is how the reference evaluates (DeepSpec
    /// `eval.py --confidence-threshold`, default 0.0) — it measures a drafter
    /// apart from its head, and a saturated-low head truncates every block to
    /// nothing. Unset keeps the goodput budget, which is what ships today.
    /// A positive value cuts earlier than DeepSpec's, which thresholds each
    /// position's own sigmoid rather than the running product.
    #[arg(long, value_parser = parse_unit_interval, value_name = "P")]
    pub(crate) dspark_confidence_threshold: Option<f32>,

    /// Install a Markov head from a safetensors file over the draft
    /// checkpoint's. This is the only way to put a trained head into a serve —
    /// copying the file into the draft dir does nothing, since the loader reads
    /// only the shards `model.safetensors.index.json` lists.
    #[arg(long, value_name = "FILE")]
    pub(crate) dspark_markov_init: Option<PathBuf>,

    /// Number of MTP draft tokens to propose per verify block on CUDA.
    #[arg(long, value_name = "N")]
    pub(crate) mtp_draft_tokens: Option<usize>,

    /// Metal speculative-decode draft head (HF id or local dir). When set, the
    /// Metal backend drafts with this NextN/MTP head and verifies on the base
    /// (bit-identical to greedy). Auto-resolved for Qwen3.6-27B unless
    /// `--no-speculative`.
    #[arg(long, value_name = "HF_ID_OR_PATH")]
    pub(crate) draft_model: Option<String>,

    /// Metal speculative draft depth (NextN/MTP tokens proposed per verify
    /// block). Default 2 is the measured optimum on Qwen3.6-27B (~18.1 tok/s).
    #[arg(long, default_value_t = 2, value_parser = parse_speculative_tokens, value_name = "N")]
    pub(crate) speculative_tokens: usize,

    /// Disable Metal speculative decode, including the Qwen3.6-27B default-on
    /// auto-enable. Forces plain single-token decode.
    #[arg(long, default_value_t = false)]
    pub(crate) no_speculative: bool,

    /// Metal speculative-decode acceptance width. A drafted token is accepted when
    /// it lies in the target model's top-K for that position. `1` (default) = exact
    /// greedy verify, bit-identical to no-speculation output. `K>1` raises the
    /// acceptance rate (longer accepted prefixes → faster) but is LOSSY: the
    /// committed token may be the target's 2nd/3rd choice, so the output deviates
    /// from exact greedy — validate with the correct-inference (needle) gate, not
    /// byte-identity vs baseline. Verify cost is unchanged (the target logits are
    /// already computed; top-K is a free membership test).
    #[arg(long, default_value_t = 1, value_name = "K")]
    pub(crate) spec_accept_topk: usize,

    /// MTP root-branch top-k width on CUDA D2; D2/T2 verifies root + candidates.
    #[arg(long, value_name = "K")]
    pub(crate) mtp_draft_topk: Option<usize>,

    /// Inference-analysis probe: write per-token entropy/NLL + decode logit-lens
    /// JSONL to this path (TP rank 0 only; truncated on open). Activates the
    /// probe; unset = probe fully off, zero hot-path cost. CUDA backend
    /// (DSv4/GLM lens lane). Env transport: `ARLE_PROBE_*` (docs/environment.md).
    #[arg(long, value_name = "PATH")]
    pub(crate) probe_out: Option<PathBuf>,

    /// Logit-lens depth: after each of the last N layers of a decode step,
    /// project the residual stream through the head and record entropy/NLL/
    /// top-1 agreement. `0` disables the lens (per-token entropy stays on).
    #[arg(long, default_value_t = 10, value_name = "N")]
    pub(crate) probe_lens_layers: usize,

    /// Per-token entropy/NLL records (all prefill positions + every decode
    /// token, raw T=1 logits). Requires --probe-out.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) probe_token_entropy: bool,

    // ── CUDA runtime toggles (EngineLoadConfig.cuda; defaults = shipped behavior) ──
    /// Whole-step Qwen3.5/3.6 decode graph (paged lane licensed 2026-08-03:
    /// −7.9% ITL, byte-identical greedy, MMLU 84/100 vs 80-81 baseline).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) qwen35_decode_graph: bool,

    /// Batched rows>1 Qwen3.5/3.6 decode; false = sequential per-row A/B arm.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) qwen35_batched_decode: bool,

    /// DeepGEMM grouped expert GEMMs (read at load: builds grouped weight caches).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) qwen35_deepgemm: bool,

    /// Decode-band batch grouped MoE kernels; false = hand batch kernels A/B arm.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) qwen35_moe_decode_kernel: bool,

    /// On-device MoE router; false = host `infer_moe::route` reference.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) qwen35_gpu_router: bool,

    /// FA3 full-attention prefill (silently in-tree on stub builds).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) qwen35_fa3: bool,

    /// FA3 decode split count; 0 derives it from the device SM count so the
    /// kv-head × split tiles fill the machine. Explicit values clamp to [2, 256].
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub(crate) qwen35_fa3_decode_splits: usize,
    /// Routed-row floor for the DeepGEMM grouped expert path (default 1024).
    /// Batched decode is `R = top_k * B`, so 1024 keeps decode off DeepGEMM;
    /// lower it to reach the uncharacterized mid-band.
    #[arg(long, default_value_t = 1024)]
    pub(crate) qwen35_deepgemm_min_routes: usize,

    /// FlashQLA chunked GDN prefill (sm_90, per-geometry AOT instantiations;
    /// runtime-probes kernel availability and falls back to the recurrent
    /// scan). Known trade: raw-completion few-shot can flip knife-edge
    /// boundary tokens (chat-format quality is parity — see the 2026-08-02
    /// verdict entry).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) qwen35_gdr_chunked: bool,

    /// Fast page-16 decode-metadata kernel path.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) decode_metadata_fast_page16: bool,

    /// Marlin W4 FP8 prefill weights at load.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) marlin_w4_fp8_prefill: bool,

    /// Retain the cuMemAllocAsync pool across syncs (caching allocator).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) cuda_mempool_retain: bool,

    /// Safetensor shard read-ahead cache budget in bytes (unset = built-in default).
    #[arg(long, value_name = "BYTES")]
    pub(crate) shard_cache_bytes: Option<usize>,

    /// Pin multi-rank CUDA workers to their GPU's NUMA node.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) numa_pin: bool,

    /// DSv4 FlashMLA sparse decode; unset = default (on when compiled in),
    /// false = eager scalar fallback (A/B lever).
    #[arg(long, value_name = "BOOL")]
    pub(crate) dsv4_flashmla_decode: Option<bool>,

    /// DSv4 DSA indexer SM budget.
    #[arg(long, default_value_t = 78, value_name = "N")]
    pub(crate) dsv4_dsa_indexer_sms: usize,

    /// DSv4 contiguous-decode MoE path (KILLED as default; A/B lever).
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) dsv4_moe_contig_decode: bool,

    /// DSv4 decode-region prefix reuse: restore a later turn to the exact prior
    /// finish position. Default ON (2026-07-11 pod license: multi-turn
    /// concurrent +25% throughput at c=16, no single-shot regression); pass
    /// `false` to restore the pre-reuse path.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) dsv4_decode_reuse: bool,

    /// Adaptive MTP gate at B=1: skip speculation below the accept break-even.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) mtp_adaptive: bool,

    /// Minimum accept-rate EMA to keep speculating under --mtp-adaptive.
    #[arg(long, default_value_t = 0.55, value_name = "F")]
    pub(crate) mtp_min_accept: f32,

    /// Speculate (MTP/DSpark) only when the decode batch is ≤ this; above it
    /// decode routes to the plain batched path. Default 16 = the measured
    /// DSpark envelope (+77% at c=2, +8.5% at c=16); MTP, which does not batch
    /// its draft, still wants 1.
    #[arg(long, default_value_t = 16, value_name = "N")]
    pub(crate) spec_max_batch: usize,

    /// DeepEP intranode SM budget (positive, even).
    #[arg(long, default_value_t = 20, value_name = "N")]
    pub(crate) deepep_num_sms: u32,

    /// DeepEP LL per-rank dispatch-token cap (unset = SGLANG env or 256).
    #[arg(long, value_name = "N")]
    pub(crate) deepep_max_dispatch_tokens_per_rank: Option<u32>,

    // ── Metal runtime toggles (EngineLoadConfig.metal) ──
    /// Overlapped c=1 greedy decode pipeline.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) metal_pipeline: bool,

    /// Load-time JIT warmup forward.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) metal_warmup: bool,

    /// Paged-prefix SDPA read path for single-token decode.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) metal_paged_kv_read: bool,

    /// Host (blocking D2H) non-greedy sampler; false = device greedy argmax.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) metal_host_sampling: bool,

    /// Cap block-diffusion denoising steps per row (unset = checkpoint default).
    #[arg(long, value_name = "N")]
    pub(crate) diffusion_max_denoising_steps: Option<usize>,

    /// Max compute dispatches per Vulkan command buffer (TDR/latency safety
    /// valve; unset = whole token in one submit).
    #[arg(long, value_name = "N")]
    pub(crate) vulkan_submit_cap: Option<usize>,

    /// Additional engine-pool model metadata to expose from the serving control plane.
    ///
    /// Format: `id=path[,type=text-generation|embedding|reranker][,aliases=a|b][,pinned=true][,memory_bytes=N][,ttl_secs=N]`.
    /// The first implementation is metadata/control-plane only; non-primary
    /// embedding and reranker entries are explicit stubs, not generation routes.
    #[arg(long = "pool-model", value_name = "SPEC")]
    pub(crate) pool_models: Vec<String>,

    /// Forward additional backend-specific flags after `--`.
    ///
    /// Always rejected — the in-process serve stack does not forward to a
    /// standalone binary. Kept as a parse target so users migrating from
    /// vLLM/SGLang get a clear error instead of a generic clap "unexpected
    /// argument".
    #[arg(last = true, allow_hyphen_values = true)]
    pub(crate) extra_args: Vec<String>,
}

impl ServeArgs {
    /// CUDA runtime toggles for `EngineLoadConfig.cuda`.
    pub(crate) fn cuda_runtime_flags(&self) -> infer_api::CudaRuntimeFlags {
        infer_api::CudaRuntimeFlags {
            qwen35_decode_graph: self.qwen35_decode_graph,
            qwen35_batched_decode: self.qwen35_batched_decode,
            qwen35_deepgemm: self.qwen35_deepgemm,
            qwen35_moe_decode_kernel: self.qwen35_moe_decode_kernel,
            qwen35_gpu_router: self.qwen35_gpu_router,
            qwen35_fa3: self.qwen35_fa3,
            qwen35_fa3_decode_splits: self.qwen35_fa3_decode_splits,
            qwen35_deepgemm_min_routes: self.qwen35_deepgemm_min_routes,
            qwen35_gdr_chunked: self.qwen35_gdr_chunked,
            decode_metadata_fast_page16: self.decode_metadata_fast_page16,
            marlin_w4_fp8_prefill: self.marlin_w4_fp8_prefill,
            mempool_retain: self.cuda_mempool_retain,
            shard_cache_bytes: self.shard_cache_bytes,
            numa_pin: self.numa_pin,
            comm_backend: match self.comm_backend {
                ServeCommBackendArg::Auto => infer_api::CommBackend::Auto,
                ServeCommBackendArg::Nccl => infer_api::CommBackend::Nccl,
            },
            dsv4_flashmla_decode: self.dsv4_flashmla_decode,
            dsv4_dsa_indexer_sms: self.dsv4_dsa_indexer_sms,
            dsv4_moe_contig_decode: self.dsv4_moe_contig_decode,
            dsv4_decode_reuse: self.dsv4_decode_reuse,
            mtp_adaptive: self.mtp_adaptive,
            mtp_min_accept: self.mtp_min_accept,
            spec_max_batch: self.spec_max_batch,
            dspark_confidence_threshold: self.dspark_confidence_threshold,
            deepep_num_sms: self.deepep_num_sms,
            deepep_max_dispatch_tokens_per_rank: self.deepep_max_dispatch_tokens_per_rank,
        }
    }

    /// Metal runtime toggles for `EngineLoadConfig.metal`. The speculative
    /// fields ride here (not env) so the executor's resolver sees the flags.
    pub(crate) fn metal_runtime_flags(&self) -> infer_api::MetalRuntimeFlags {
        infer_api::MetalRuntimeFlags {
            pipeline: self.metal_pipeline,
            warmup: self.metal_warmup,
            paged_kv_read: self.metal_paged_kv_read,
            host_sampling: self.metal_host_sampling,
            speculative: !self.no_speculative,
            draft_model: self.draft_model.clone(),
            speculative_tokens: self
                .draft_model
                .is_some()
                .then_some(self.speculative_tokens),
            spec_accept_topk: i32::try_from(self.spec_accept_topk).unwrap_or(i32::MAX),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ServeCommBackendArg {
    /// One-shot custom collectives with automatic loud degrade to NCCL.
    Auto,
    /// Plain NCCL everywhere.
    Nccl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ServeSpecTypeArg {
    None,
    Auto,
    Mtp,
    /// DSpark/DFlash block drafter (CUDA Qwen3.6; draft checkpoint dir via
    /// `--mtp-draft-model`).
    Dspark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ServeKvCacheDtypeArg {
    Auto,
    Bf16,
    Int8,
    /// FP8 (E4M3) paged KV — CUDA only, wired one mode at a time under #68 T3
    /// (fails loud at engine construction until its path lands).
    Fp8,
    /// Trellis 4-bit paged KV — CUDA only, same #68 T3 staging as `Fp8`.
    Tq4,
}

/// OPD runtime toggles shared by every OPD subcommand
/// (`train::apply_runtime_flags`; defaults = shipped behavior).
#[derive(Debug, Clone, ClapArgs)]
pub(crate) struct OpdRuntimeArgs {
    /// Offload per-layer grad-checkpoints to host RAM during the writeback
    /// forward; false keeps them on-device (short sequences, GPU-bound).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) writeback_offload: bool,

    /// Idle-engine offload time-share between scoring phases.
    #[arg(long, value_enum, default_value_t = OpdEngineOffloadArg::Off)]
    pub(crate) engine_offload: OpdEngineOffloadArg,

    /// Enable per-layer gradient checkpointing when needed.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) gradient_checkpointing: bool,

    /// Min tensor bytes for checkpoint host offload.
    #[arg(long, default_value_t = 2 << 20, value_name = "BYTES")]
    pub(crate) checkpoint_offload_min_bytes: usize,

    /// Reload a host-offloaded checkpoint to device before its backward replay,
    /// so the recompute forward takes its device path instead of repacking and
    /// re-uploading host buffers.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) checkpoint_reload_device: bool,

    /// Pinned-host budget for parked checkpoint activations (0 = off, pageable
    /// path). Pinned buffers make both copy legs async and non-blocking; the
    /// pages are unswappable, so this is a hard ceiling. An activation it pins is
    /// always reloaded to device, whatever `--checkpoint-reload-device` says.
    #[arg(long, default_value_t = 0, value_name = "BYTES")]
    pub(crate) checkpoint_pinned_offload_bytes: usize,

    /// Skip longer update records (0 = unlimited).
    #[arg(long, default_value_t = 23_000, value_name = "TOKENS")]
    pub(crate) max_update_seq: usize,

    /// Rollout tensor retain interval (steps).
    #[arg(long, default_value_t = 2, value_name = "N")]
    pub(crate) rollout_retain_interval: usize,

    /// Rollout progress log interval (steps; the log line itself is gated by
    /// the ARLE_OPD_ROLLOUT_PROGRESS instrumentation env).
    #[arg(long, default_value_t = 16, value_name = "N")]
    pub(crate) rollout_progress_interval: usize,

    /// Expert tile for the MoE LoRA backward.
    #[arg(long, default_value_t = 16, value_name = "N")]
    pub(crate) moe_lora_bwd_expert_tile: usize,

    /// Row tile for the LoRA linear backward.
    #[arg(long, default_value_t = 1024, value_name = "N")]
    pub(crate) lora_linear_bwd_tile_rows: usize,

    /// Frozen-prompt-KV writeback path.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) writeback_frozen_prompt_kv: bool,

    /// FlashQLA chunkwise GDN prefill in the autograd CUDA backend.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) gdr_chunkwise_prefill: bool,

    /// Native FP8 DeepGEMM for frozen-weight forward projections (opt-in; gated on FD parity).
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) fp8_native_gemm: bool,

    /// Force the monolithic chunked-scan linear-attention backward (A/B arm).
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) la_backward_mono: bool,

    /// Force the legacy two-pass decode attention kernel (A/B arm).
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) autograd_decode_attn_legacy: bool,

    /// Retain the cuMemAllocAsync pool across syncs (caching allocator).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) cuda_mempool_retain: bool,

    /// Autograd tape storage precision. bf16 halves retained-activation + emitted-grad
    /// VRAM (compute + accumulators + fp32 islands unchanged). CUDA-only; no-op on Metal.
    #[arg(long, value_enum, default_value_t = TapePrecisionArg::Fp32)]
    pub(crate) tape_precision: TapePrecisionArg,

    /// DSpark/DFlash block-drafter checkpoint dir for the in-process rollout
    /// engine (spec decode; CUDA Qwen3.5/3.6, single-GPU only — mirrors serve's
    /// `--spec-type dspark --mtp-draft-model`).
    #[arg(long, value_name = "DIR")]
    pub(crate) dspark_draft_model: Option<PathBuf>,

    /// DSpark goodput-budget cost model for --dspark-draft-model (mirrors
    /// serve's `--dspark-sps-bias-ms` / `--dspark-sps-row-ms`).
    #[arg(long, default_value_t = 211.0, value_name = "MS")]
    pub(crate) dspark_sps_bias_ms: f32,

    /// Marginal verify-row cost (mirrors serve's `--dspark-sps-row-ms`).
    #[arg(long, default_value_t = 0.53, value_name = "MS")]
    pub(crate) dspark_sps_row_ms: f32,

    /// Whole-step Qwen3.5/3.6 decode graph for the in-process rollout engine
    /// (mirrors serve's flag). Default off = unchanged behavior; flipping it
    /// on waits for the co-residency license.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) qwen35_decode_graph: bool,

    /// FlashQLA chunked GDN prefill for the in-process rollout engine (mirrors
    /// serve's flag, same shipped default).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) qwen35_gdr_chunked: bool,

    /// MTP spec-decode draft depth for the in-process rollout engine (mirrors
    /// serve's `--spec-type mtp --mtp-draft-tokens`). None (default) = off.
    #[arg(long, value_name = "N")]
    pub(crate) mtp_draft_tokens: Option<usize>,

    /// Share of TOTAL VRAM for the rollout engine (see
    /// `infer_seam::profile_kv_pool_tokens`). Below the weights' own share of
    /// total the pool collapses to the token floor; above the co-resident
    /// student's headroom the writeback OOMs.
    #[arg(long, default_value_t = 0.5, value_name = "F")]
    pub(crate) rollout_mem_fraction: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum OpdEngineOffloadArg {
    /// Resident weights (default).
    Off,
    /// Offload both idle engines.
    All,
    /// Offload only the rollout student.
    Student,
    /// Offload only the scoring teacher.
    Teacher,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum TapePrecisionArg {
    /// fp32 storage (default; byte-identical to shipped behavior).
    Fp32,
    /// bf16 storage for retained activations + emitted grads; compute stays fp32. CUDA-only.
    Bf16,
}

impl OpdRuntimeArgs {
    pub(crate) fn to_flags(&self) -> train::TrainRuntimeFlags {
        train::TrainRuntimeFlags {
            writeback_offload: self.writeback_offload,
            engine_offload: match self.engine_offload {
                OpdEngineOffloadArg::Off => train::opd::EngineOffloadMode::Off,
                OpdEngineOffloadArg::All => train::opd::EngineOffloadMode::All,
                OpdEngineOffloadArg::Student => train::opd::EngineOffloadMode::Student,
                OpdEngineOffloadArg::Teacher => train::opd::EngineOffloadMode::Teacher,
            },
            gradient_checkpointing: self.gradient_checkpointing,
            writeback_frozen_prompt_kv: self.writeback_frozen_prompt_kv,
            rollout_retain_interval: self.rollout_retain_interval,
            rollout_progress_interval: self.rollout_progress_interval,
            max_update_seq: self.max_update_seq,
            autograd: autograd::AutogradRuntimeFlags {
                checkpoint_offload_min_bytes: self.checkpoint_offload_min_bytes,
                checkpoint_reload_device: self.checkpoint_reload_device,
                checkpoint_pinned_offload_bytes: self.checkpoint_pinned_offload_bytes,
                lora_linear_bwd_tile_rows: self.lora_linear_bwd_tile_rows,
                moe_lora_bwd_expert_tile: self.moe_lora_bwd_expert_tile,
                gdr_chunkwise_prefill: self.gdr_chunkwise_prefill,
                fp8_native_gemm: self.fp8_native_gemm,
                la_backward_mono: self.la_backward_mono,
                decode_attn_legacy: self.autograd_decode_attn_legacy,
                cuda_mempool_retain: self.cuda_mempool_retain,
                tape_precision: match self.tape_precision {
                    TapePrecisionArg::Fp32 => autograd::TapePrecision::Fp32,
                    TapePrecisionArg::Bf16 => autograd::TapePrecision::Bf16,
                },
            },
        }
    }

    /// CUDA toggles for the rollout engine. Both OPD engines build them here.
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_runtime_flags(&self) -> infer_api::CudaRuntimeFlags {
        infer_api::CudaRuntimeFlags {
            qwen35_decode_graph: self.qwen35_decode_graph,
            qwen35_gdr_chunked: self.qwen35_gdr_chunked,
            mempool_retain: self.cuda_mempool_retain,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, clap::Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  arle train env\n  arle train opd --smoke --steps 5\n  arle train estimate-memory --tokenizer tokenizer.json --preset small-25m"
)]
pub(crate) struct TrainArgs {
    #[command(subcommand)]
    pub(crate) command: TrainCommand,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum TrainCommand {
    /// Print train-time environment diagnostics.
    Env(TrainEnvArgs),
    /// Estimate parameter count and rough memory.
    EstimateMemory(TrainEstimateMemoryArgs),
    /// On-policy distillation.
    Opd(TrainOpdArgs),
    /// Self-training LoRA OPD (SOPD): the model self-updates its own LoRA
    /// adapter against an EMA self-teacher, with snapshot/revert on regress.
    SelfOpd(TrainSelfOpdArgs),
    /// Rubric-OPD: the student samples N rollouts per prompt, a strong teacher
    /// (DeepSeek-V4-Flash) judges each against a text-level rubric, and the
    /// accepted rollouts are written back as CE targets (RFT). Cross-vocab-free.
    RubricOpd(TrainRubricOpdArgs),
    /// Agent-OPD: the student drives the read/write/replace/bash tool loop
    /// against a per-task repo sandbox (SWE-bench-Pro). The reward is execution
    /// (run the hidden tests), no text judge — passing trajectories are written
    /// back as CE targets.
    // Boxed: the largest command variant — keeps `TrainCommand` compact
    // (clippy::large_enum_variant).
    AgentOpd(Box<TrainAgentOpdArgs>),
    /// Convert captured `/v1/messages` dumps (`arle serve --dump-messages-dir`)
    /// into verl-style token records for the agent-OPD masked-CE replay
    /// (`agent-opd --replay-records`).
    CcConvert(TrainCcConvertArgs),
    /// Perplexity (PPL) over a text corpus via teacher-forced logits — a quality
    /// calibration for FP8 / quant checkpoints. CUDA-only (forward_token_logits).
    Ppl(TrainPplArgs),
    /// Offline DSpark speculative-decoding draft training: anchored blocks
    /// against the trunk's own taps and logits. CUDA-only (the trunk forward is
    /// `LoadedInferenceEngine::forward_training_taps`).
    SpecDraft(TrainSpecDraftArgs),
    /// Weak-to-strong (w2s) online distillation: distill the policy shift
    /// (post-RL − pre-RL log-prob difference) from two weak auxiliary models
    /// into a strong student via a proxy teacher and reverse KL.
    W2s(TrainW2sArgs),
}

/// Every objective/optimizer default is the trainer's, so `--help` and the
/// recipe cannot drift apart.
fn spec_draft_cfg() -> spec_train::trainer::Config {
    spec_train::trainer::Config::default()
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Examples:\n  arle train spec-draft --model-path checkpoints/base --draft drafts/dspark --data conversations.jsonl --out drafts/dspark-trained\n  arle train spec-draft --model-path checkpoints/base --draft drafts/dspark --data conversations.jsonl --out /tmp/ab-tv0 --loss-tv 0.0 --seed 42 --log-every 1"
)]
pub(crate) struct TrainSpecDraftArgs {
    #[command(flatten)]
    pub(crate) runtime: OpdRuntimeArgs,

    /// Target model directory. Supplies the trunk forward, the tokenizer, and
    /// the embeddings + lm_head the draft shares with it.
    #[arg(long, value_name = "PATH")]
    pub(crate) model_path: PathBuf,

    /// Draft directory: `config.json` always, weights only to warm-start.
    /// Config alone trains from scratch, which is what the reference does.
    #[arg(long, value_name = "PATH")]
    pub(crate) draft: PathBuf,

    /// Conversations JSONL from `scripts/spec_train_data.py regen`.
    #[arg(long, value_name = "PATH")]
    pub(crate) data: PathBuf,

    /// Directory to write the trained draft into: `config.json` +
    /// `model.safetensors`, the layout `--draft` and the serve both read.
    #[arg(long, value_name = "DIR")]
    pub(crate) out: PathBuf,

    #[arg(long, default_value_t = 1000, value_parser = parse_positive_usize)]
    pub(crate) steps: usize,

    /// Samples per optimizer step.
    #[arg(long, default_value_t = 8, value_parser = parse_positive_usize)]
    pub(crate) batch: usize,

    /// Tokens kept per conversation. 4096 is the reference's `data.max_length`
    /// (DeepSpec `config/dspark/dspark_qwen3_14b.py`). Truncation drops the
    /// tail, so a conversation over the cap loses the end of the supervised
    /// turn; raising this also lengthens every trunk forward.
    #[arg(long, default_value_t = 4096, value_parser = parse_positive_usize)]
    pub(crate) max_len: usize,

    #[arg(long, default_value_t = spec_draft_cfg().num_anchors, value_parser = parse_positive_usize)]
    pub(crate) num_anchors: usize,

    /// Blocks per backward — the VRAM knob. Attention scores and logits both
    /// scale with it; all 512 anchors at once is ~82 GiB of activation.
    #[arg(long, default_value_t = spec_draft_cfg().blocks_per_backward, value_parser = parse_positive_usize)]
    pub(crate) blocks_per_backward: usize,

    /// Fraction of VRAM the trunk's KV pool may claim. The serving default
    /// (0.9) leaves nothing for the draft: a training job runs one short
    /// sequence, so the pool is pure waste here.
    #[arg(long, default_value_t = 0.45)]
    pub(crate) trunk_mem_fraction: f64,

    #[arg(long, default_value_t = spec_draft_cfg().lr)]
    pub(crate) lr: f32,

    #[arg(long, default_value_t = spec_draft_cfg().warmup_ratio, value_parser = parse_unit_interval)]
    pub(crate) warmup_ratio: f32,

    /// Gradient-norm clip. The step log prints the pre-clip norm, so a run that
    /// clips every step — a silent lr cut — is visible as one.
    #[arg(long, default_value_t = spec_draft_cfg().max_grad_norm)]
    pub(crate) max_grad_norm: f32,

    #[arg(long, default_value_t = spec_draft_cfg().weight_decay)]
    pub(crate) weight_decay: f32,

    /// Row weight `exp(-t / gamma)` across a block; later rows count less.
    #[arg(long, default_value_t = spec_draft_cfg().loss_decay_gamma)]
    pub(crate) loss_decay_gamma: f32,

    /// Objective mix `α_ce·ℒ_ce + α_tv·ℒ_tv + α_conf·ℒ_conf`.
    #[arg(long, default_value_t = spec_draft_cfg().weights.ce)]
    pub(crate) loss_ce: f32,

    #[arg(long, default_value_t = spec_draft_cfg().weights.tv)]
    pub(crate) loss_tv: f32,

    #[arg(long, default_value_t = spec_draft_cfg().weights.confidence)]
    pub(crate) loss_confidence: f32,

    /// Fixes the anchor sample and the epoch shuffle, so two runs are comparable.
    #[arg(long, default_value_t = spec_draft_cfg().seed)]
    pub(crate) seed: u64,

    /// Markov head rank; 0 trains a draft without one.
    #[arg(long, default_value_t = 256)]
    pub(crate) markov_rank: usize,

    /// Steps between log lines.
    #[arg(long, default_value_t = 1, value_parser = parse_positive_usize)]
    pub(crate) log_every: usize,

    #[arg(long, default_value_t = 100, value_parser = parse_positive_usize)]
    pub(crate) save_every: usize,

    /// Continue a run from a `--save-every` checkpoint directory: draft weights,
    /// AdamW moments and the step counter, so the LR schedule, the anchor draw
    /// and the sample order pick up where they stopped. Takes precedence over
    /// the `--draft` warm start.
    #[arg(long, value_name = "DIR")]
    pub(crate) resume: Option<PathBuf>,

    /// Check the trunk supervision signal on the first sample and exit without
    /// training. One `forward_training_taps` — the trainer's own call — then
    /// the `last_hidden @ lm_head` reconstruction against the engine's own
    /// logits at 16 positions spread over the sequence, gated on argmax
    /// agreement 0.75, top-64 overlap 0.50, and mean |Δ log p(next token)|
    /// 1.0 nat; plus every tap non-degenerate at pairwise cosine under 0.99.
    /// The bars are loose because the engine's head is FP8/Marlin against an
    /// f32 recompute, and still strict because a wrong tap layer or a
    /// pre/post-norm slip puts all three near zero while the loss curve keeps
    /// falling. Non-zero exit on FAIL. Materializes `[seq, vocab]` logits, so
    /// `--max-len` bounds the cost.
    #[arg(long, default_value_t = false)]
    pub(crate) probe: bool,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Weak-to-strong (w2s) online distillation.\n\nDistill the policy shift ΔT = log π_post-RL − log π_pre-RL from two weak auxiliary models into a strong student. The proxy teacher is z_student.detach() + α·T·(ΔT₁+ΔT₂)/2, trained with reverse KL.\n\nExamples:\n  arle train w2s --student-model <dir> --aux1-pre <dir> --aux1-post <dir> --aux2-pre <dir> --aux2-post <dir> --steps 100"
)]
pub(crate) struct TrainW2sArgs {
    #[command(flatten)]
    pub(crate) runtime: OpdRuntimeArgs,

    /// Student (strong) model directory. Loaded with LoRA adapters.
    #[arg(long, value_name = "PATH")]
    pub(crate) student_model: PathBuf,

    /// Auxiliary model 1 pre-RL (base) checkpoint.
    #[arg(long, value_name = "PATH")]
    pub(crate) aux1_pre: PathBuf,

    /// Auxiliary model 1 post-RL (instruct) checkpoint.
    #[arg(long, value_name = "PATH")]
    pub(crate) aux1_post: PathBuf,

    /// Auxiliary model 2 pre-RL (base) checkpoint.
    #[arg(long, value_name = "PATH")]
    pub(crate) aux2_pre: PathBuf,

    /// Auxiliary model 2 post-RL (instruct) checkpoint.
    #[arg(long, value_name = "PATH")]
    pub(crate) aux2_post: PathBuf,

    /// Prompt token ids (comma-separated). Defaults to a short BOS sequence.
    #[arg(long, value_name = "IDS")]
    pub(crate) prompt_ids: Option<String>,

    /// Distillation strength α. α·T·‖ΔT‖ should be comparable to ‖z_student‖.
    #[arg(long, default_value_t = 0.5)]
    pub(crate) alpha: f32,

    /// Softmax temperature T (2–4).
    #[arg(long, default_value_t = 2.0, value_parser = parse_positive_temperature)]
    pub(crate) temperature: f32,

    /// Confidence filter: skip samples where student max prob exceeds this.
    #[arg(long, default_value_t = 0.9, value_parser = parse_unit_interval)]
    pub(crate) confidence_threshold: f32,

    /// Consistency gate: skip samples where cos(ΔT₁, ΔT₂) is below this.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) consistency_threshold: f32,

    /// Local KL regularizer weight β₁ (vs previous adapter).
    #[arg(long, default_value_t = 0.01)]
    pub(crate) beta_local: f32,

    /// Global KL regularizer weight β₂ (vs base model).
    #[arg(long, default_value_t = 0.001)]
    pub(crate) beta_global: f32,

    /// Total training steps.
    #[arg(long, default_value_t = 10)]
    pub(crate) steps: usize,

    /// AdamW learning rate.
    #[arg(long, default_value_t = 1.0e-5)]
    pub(crate) lr: f32,

    /// Gradient L2 norm clip threshold.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) grad_clip: f32,

    /// LoRA rank.
    #[arg(long, default_value_t = 16)]
    pub(crate) lora_rank: usize,

    /// LoRA alpha.
    #[arg(long, default_value_t = 32.0)]
    pub(crate) lora_alpha: f32,

    /// LoRA target set.
    #[arg(long, default_value = "attention-qv")]
    pub(crate) lora_target_set: String,

    /// Compute backend.
    #[arg(long, value_enum, default_value_t = OpdBackendArg::Auto)]
    pub(crate) backend: OpdBackendArg,

    /// Auxiliary model backend: `in-process` loads the aux models in the train
    /// process (small models only); `infer` runs them via LoadedInferenceEngine
    /// (supports TP / large models that would OOM the train process).
    #[arg(long, value_enum, default_value_t = W2sAuxBackendArg::InProcess)]
    pub(crate) aux_backend: W2sAuxBackendArg,

    /// Render output as JSON.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, ClapArgs)]
pub(crate) struct TrainPplArgs {
    /// Model directory (local path or HF id) to score.
    #[arg(long, value_name = "PATH")]
    pub(crate) model_path: PathBuf,

    /// UTF-8 text corpus to compute perplexity over.
    #[arg(long, value_name = "PATH")]
    pub(crate) corpus: PathBuf,

    /// Context window (non-overlapping) in tokens.
    #[arg(long, default_value_t = 4096, value_parser = parse_positive_usize)]
    pub(crate) ctx: usize,

    /// Cap the number of windows scored (quick run); unset scores the whole corpus.
    #[arg(long, value_parser = parse_positive_usize)]
    pub(crate) max_windows: Option<usize>,

    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Groups the dumps into attempts via --window/--windows (whole dir when omitted),\npicks each window's session-final request (largest `messages` array), maps it\nthrough the serve's own /v1/messages adapter, renders ChatML with supervised\nspans, and emits one JSONL token record per attempt:\n  {\"label\":…, \"prompt_ids\":[…], \"response_ids\":[…], \"response_mask\":[…], …}"
)]
pub(crate) struct TrainCcConvertArgs {
    /// Directory of raw /v1/messages dumps (`<epoch_ms>_<seq>.json`).
    #[arg(long, value_name = "DIR")]
    pub(crate) dump_dir: PathBuf,

    /// tokenizer.json used to tokenize the rendered conversation.
    #[arg(long, value_name = "FILE")]
    pub(crate) tokenizer: PathBuf,

    /// Output JSONL (one record per attempt window).
    #[arg(long, value_name = "FILE")]
    pub(crate) out: PathBuf,

    /// Attempt window `<t_start_ms>:<t_end_ms>[:<label>]` (repeatable).
    #[arg(long = "window", value_name = "START:END[:LABEL]")]
    pub(crate) window: Vec<String>,

    /// JSONL of windows: rows `{"label":…, "t_start_ms":…, "t_end_ms":…}`.
    #[arg(long, value_name = "FILE")]
    pub(crate) windows: Option<PathBuf>,
}

#[derive(Debug, Clone, ClapArgs)]
pub(crate) struct TrainEnvArgs {
    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Examples:\n  arle train estimate-memory --tokenizer tokenizer.json --preset small-25m\n  arle train estimate-memory --model checkpoints/base --lora-rank 32 --json"
)]
pub(crate) struct TrainEstimateMemoryArgs {
    /// Existing model directory to inspect for LoRA adaptation / eval-style runs.
    #[arg(long, alias = "model-path")]
    pub(crate) model: Option<PathBuf>,

    /// Scratch tokenizer source (`tokenizer.json` or a local model dir containing it).
    #[arg(long)]
    pub(crate) tokenizer: Option<PathBuf>,

    /// Optional scratch preset for OPD-side scratch estimates.
    #[arg(long, value_enum)]
    pub(crate) preset: Option<PretrainPresetArg>,

    /// Override the scratch model family.
    #[arg(long, value_enum)]
    pub(crate) model_family: Option<ModelFamilyArg>,

    /// Token batch width used for the rough activation estimate.
    #[arg(long, default_value_t = 1, value_parser = parse_positive_usize)]
    pub(crate) batch: usize,

    /// Sequence length used for the rough activation estimate.
    #[arg(long, default_value_t = 512, value_parser = parse_positive_usize)]
    pub(crate) seq: usize,

    /// LoRA rank used for model-dir estimates.
    #[arg(long, default_value_t = 16, value_parser = parse_positive_usize)]
    pub(crate) lora_rank: usize,

    /// Save dtype used for checkpoint-size estimates.
    #[arg(long, value_enum, default_value_t = SaveDtypeArg::Bf16)]
    pub(crate) save_dtype: SaveDtypeArg,

    /// Scratch vocab size override.
    #[arg(long)]
    pub(crate) vocab_size: Option<usize>,

    /// Scratch hidden width override.
    #[arg(long)]
    pub(crate) hidden: Option<usize>,

    /// Scratch transformer layer count override.
    #[arg(long)]
    pub(crate) layers: Option<usize>,

    /// Scratch attention head count override.
    #[arg(long)]
    pub(crate) heads: Option<usize>,

    /// Scratch KV head count override.
    #[arg(long)]
    pub(crate) kv_heads: Option<usize>,

    /// Scratch per-head dimension override.
    #[arg(long)]
    pub(crate) head_dim: Option<usize>,

    /// Scratch MLP intermediate width override.
    #[arg(long)]
    pub(crate) intermediate: Option<usize>,

    /// Scratch maximum position embeddings override.
    #[arg(long)]
    pub(crate) max_pos: Option<usize>,

    /// For Qwen3.5 scratch estimates, insert one linear-attention layer every N layers.
    #[arg(long)]
    pub(crate) linear_attn_every: Option<usize>,

    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "On-Policy Distillation. The student samples a rollout greedily, the\nteacher re-scores it, and forward KL drives backward through the student.\n\nSmoke (no model load, tiny embedded Qwen3.5 config):\n  arle train opd --smoke --steps 5\n\nReal (HF / ModelScope-cached model directory, pending loader):\n  arle train opd --student-model ~/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \\\n                 --teacher-model ~/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \\\n                 --steps 20 --rollout-len 16"
)]
pub(crate) struct TrainOpdArgs {
    #[command(flatten)]
    pub(crate) runtime: OpdRuntimeArgs,

    /// Student model directory in HF safetensors layout. If unset and
    /// `--smoke` is given, uses the embedded tiny Qwen3.5 config.
    #[arg(long, alias = "student")]
    pub(crate) student_model: Option<PathBuf>,

    /// Teacher model directory. If unset, clones the student (self-distill).
    #[arg(long, alias = "teacher")]
    pub(crate) teacher_model: Option<PathBuf>,

    /// Teacher scoring runtime. `infer` supports runtime-only teacher checkpoints.
    #[arg(long, value_enum, default_value_t = OpdTeacherRuntimeArg::InProcess)]
    pub(crate) teacher_runtime: OpdTeacherRuntimeArg,

    /// Raw-logits endpoint for `--teacher-runtime api` (e.g.
    /// `http://127.0.0.1:8137/v1/raw_logits`). Required with `api`, ignored otherwise.
    #[arg(long, value_name = "URL")]
    pub(crate) teacher_url: Option<String>,

    /// Initial prompt token ids, comma-separated. Defaults to `1,3,8`.
    #[arg(long, value_name = "IDS")]
    pub(crate) prompt_ids: Option<String>,

    /// Prompt corpus JSONL. Each row may contain `text` or tokenized `prompt_ids`.
    #[arg(long, value_name = "FILE")]
    pub(crate) prompts_file: Option<PathBuf>,

    /// Max tokens per prompt row when loading --prompts-file.
    #[arg(long, default_value_t = 512)]
    pub(crate) prompt_max_tokens: usize,

    /// Deterministic prompt-corpus sampling seed for --prompts-file.
    #[arg(long, default_value_t = 0)]
    pub(crate) prompt_seed: u64,

    /// Fixed held-out token-id sequence (comma-separated) for periodic NLL
    /// observation. Defaults to the prompt ids when unset.
    #[arg(long, value_name = "IDS")]
    pub(crate) eval_ids: Option<String>,

    /// Tokens to roll out greedily from the student per step.
    #[arg(long, default_value_t = 8)]
    pub(crate) rollout_len: usize,

    /// Rollout sampling temperature (0.0 = greedy argmax).
    #[arg(
        long,
        default_value_t = 0.0,
        value_parser = parse_temperature,
        allow_hyphen_values = true
    )]
    pub(crate) rollout_temperature: f32,

    /// Rollout nucleus sampling threshold (1.0 disables top-p filtering).
    #[arg(long, default_value_t = 1.0)]
    pub(crate) rollout_top_p: f32,

    /// Rollout top-k filter (0 disables top-k filtering).
    #[arg(long, default_value_t = 0)]
    pub(crate) rollout_top_k: i32,

    /// Optional deterministic rollout sampling seed.
    #[arg(long)]
    pub(crate) rollout_seed: Option<u64>,

    /// KL objective direction: forward = KL(teacher||student), reverse = KL(student||teacher).
    #[arg(long, value_enum, default_value_t = KlDirectionArg::Forward)]
    pub(crate) kl_direction: KlDirectionArg,

    /// KL softening temperature (>0). Values other than 1.0 are pure-OPD only.
    #[arg(long, default_value_t = 1.0, value_parser = parse_positive_temperature)]
    pub(crate) kl_temperature: f32,

    /// Generalized JSD beta in [0,1]. Unset keeps --kl-direction.
    #[arg(long, value_parser = parse_unit_interval)]
    pub(crate) kl_beta: Option<f32>,

    /// Sparse teacher target top-k. Requires H20 Piece A engine-side top-k.
    #[arg(long, value_parser = parse_positive_usize)]
    pub(crate) teacher_topk: Option<usize>,

    /// Opt in to the fused lm_head+loss path (computes full-vocab logits + KD loss
    /// in one op to save the [window, vocab] tensor). DEFAULT IS DENSE: on H20 the
    /// fused path ran the lm_head on the HOST (~205 s/step for the 27B, GPU idle)
    /// vs ~3.9 s GPU-bound for dense — only enable for windows too large to
    /// materialize [window, vocab]. See errors/2026-06-23-opd-fused-distill-default-host-bound.
    #[arg(long, default_value_t = false)]
    pub(crate) fused_distill: bool,

    /// Deprecated no-op: dense logits+KL is now the default. Kept so existing
    /// `--no-fused-distill` invocations still run (it forces the now-default dense path).
    #[arg(long, default_value_t = false)]
    pub(crate) no_fused_distill: bool,

    /// KL token mask. Completion-only matches the validated GKD/OPD recipe.
    #[arg(long, value_enum, default_value_t = OpdKlMaskArg::Completion)]
    pub(crate) kl_mask: OpdKlMaskArg,

    /// Logits/backward window size for OPD scoring (0 disables windowing).
    #[arg(long, default_value_t = DEFAULT_OPD_LOGITS_WINDOW_SIZE)]
    pub(crate) logits_window_size: usize,

    /// Total OPD training steps.
    #[arg(long, default_value_t = 5)]
    pub(crate) steps: usize,

    /// AdamW learning rate.
    #[arg(long, default_value_t = 1.0e-4)]
    pub(crate) lr: f32,

    /// Learning-rate schedule for OPD optimizer steps.
    #[arg(long, value_enum, default_value_t = LrScheduleArg::Fixed)]
    pub(crate) lr_schedule: LrScheduleArg,

    /// Warmup steps for --lr-schedule cosine. Defaults to ceil(3% of --steps).
    #[arg(long)]
    pub(crate) lr_warmup_steps: Option<usize>,

    /// GKD blend lambda. Default 0.0 is pure KL OPD; lambda=1 with
    /// --sft-anchor corpus-truth runs off-policy SFT on prompt completions.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) gkd_lambda: f32,

    /// Hard-token SFT anchor used when --gkd-lambda > 0.
    #[arg(long, value_enum, default_value_t = OpdSftAnchorArg::StudentRollout)]
    pub(crate) sft_anchor: OpdSftAnchorArg,

    /// Observe held-out NLL every N steps (0 disables it).
    #[arg(long, default_value_t = 0)]
    pub(crate) gate_every_n: usize,

    /// Gradient L2 norm clip threshold.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) grad_clip: f32,

    /// LoRA rank for the real-checkpoint student adapter.
    #[arg(long, default_value_t = 16)]
    pub(crate) lora_rank: usize,

    /// LoRA alpha for the real-checkpoint student adapter.
    #[arg(long, default_value_t = 32.0)]
    pub(crate) lora_alpha: f32,

    /// LoRA target set: `attention-qv`, `attention-full` (q/k/v/o + linear-attn), or `all-linear`.
    #[arg(long, default_value = "attention-qv")]
    pub(crate) lora_target_set: String,

    /// First transformer layer index that receives LoRA adapters.
    #[arg(long)]
    pub(crate) lora_layer_start: Option<usize>,

    /// Directory where servable full-materialized OPD checkpoints are written.
    #[arg(long, value_name = "DIR")]
    pub(crate) save_checkpoint: Option<PathBuf>,

    /// Save a servable full checkpoint every N steps (0 = final checkpoint only).
    #[arg(long, default_value_t = 0)]
    pub(crate) save_every: usize,

    /// Smoke mode — use the embedded tiny Qwen3.5 config instead of
    /// loading weights from disk. Useful for hardware smoke + CI gating.
    #[arg(long, default_value_t = false)]
    pub(crate) smoke: bool,

    /// Compute backend for autograd. `auto` picks CUDA when the binary
    /// is built with the cuda feature, else CPU. `cpu` forces the CPU
    /// reference path even on a CUDA-built binary.
    #[arg(long, value_enum, default_value_t = OpdBackendArg::Auto)]
    pub(crate) backend: OpdBackendArg,

    /// Storage dtype for tape-saved activations (f32 default is byte-identical).
    #[arg(long, value_enum, default_value_t = TapeDtypeArg::F32)]
    pub(crate) tape_dtype: TapeDtypeArg,

    /// Rollout engine (CUDA only). `infer` = in-process infer engine (~5×
    /// faster; default); `train` = train-crate O(n²) A/B arm.
    #[arg(long, value_enum)]
    pub(crate) rollout_engine: Option<OpdRolloutEngineArg>,

    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

/// Rollout engine selector for `arle train opd --rollout-engine`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum OpdRolloutEngineArg {
    /// In-process infer engine (CUDA graph + paged KV). Default when unset.
    Infer,
    /// Train-crate hand-written decode (A/B baseline arm).
    Train,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum OpdBackendArg {
    /// Pick CUDA when available, else CPU.
    Auto,
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum W2sAuxBackendArg {
    /// Load aux models in the train process (small models only).
    InProcess,
    /// Run aux models via LoadedInferenceEngine (supports TP / large models).
    Infer,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Self-OPD (SOPD): inline self-training. The student samples a rollout,\nan EMA self-teacher (base + lagged EMA adapter) re-scores it, GKD blends a\nλ·CE self-anchor with (1−λ)·KL(student‖EMA), and the EMA adapter trails the\nstudent θ_ema ← α·θ_ema + (1−α)·θ_student. Every N steps a held-out NLL gate\nreverts {student, EMA, AdamW} together on regress.\n\nSmoke (no model load, embedded tiny Qwen3.5 config):\n  arle train self-opd --smoke --steps 5\n\nReal:\n  arle train self-opd --student-model <dir> --steps 20 --rollout-len 16 --gate-every-n 10"
)]
pub(crate) struct TrainSelfOpdArgs {
    #[command(flatten)]
    pub(crate) runtime: OpdRuntimeArgs,

    /// Student model directory in HF safetensors layout. Required unless --smoke.
    #[arg(long, alias = "student")]
    pub(crate) student_model: Option<PathBuf>,

    /// Initial prompt token ids, comma-separated. Defaults to `1,3,8`.
    #[arg(long, value_name = "IDS")]
    pub(crate) prompt_ids: Option<String>,

    /// Fixed held-out token-id sequence (comma-separated) for the no-regression
    /// NLL gate. Defaults to the prompt ids when unset.
    #[arg(long, value_name = "IDS")]
    pub(crate) eval_ids: Option<String>,

    /// Tokens to roll out greedily from the student per step.
    #[arg(long, default_value_t = 8)]
    pub(crate) rollout_len: usize,

    /// Rollout sampling temperature (0.0 = greedy argmax).
    #[arg(
        long,
        default_value_t = 0.0,
        value_parser = parse_temperature,
        allow_hyphen_values = true
    )]
    pub(crate) rollout_temperature: f32,

    /// Rollout nucleus sampling threshold (1.0 disables top-p filtering).
    #[arg(long, default_value_t = 1.0)]
    pub(crate) rollout_top_p: f32,

    /// Rollout top-k filter (0 disables top-k filtering).
    #[arg(long, default_value_t = 0)]
    pub(crate) rollout_top_k: i32,

    /// Optional deterministic rollout sampling seed.
    #[arg(long)]
    pub(crate) rollout_seed: Option<u64>,

    /// KL objective direction: forward = KL(teacher||student), reverse = KL(student||teacher).
    #[arg(long, value_enum, default_value_t = KlDirectionArg::Forward)]
    pub(crate) kl_direction: KlDirectionArg,

    /// KL softening temperature (>0). Values other than 1.0 are pure-OPD only.
    #[arg(long, default_value_t = 1.0, value_parser = parse_positive_temperature)]
    pub(crate) kl_temperature: f32,

    /// Generalized JSD beta in [0,1]. Unset keeps --kl-direction.
    #[arg(long, value_parser = parse_unit_interval)]
    pub(crate) kl_beta: Option<f32>,

    /// Sparse teacher target top-k. Requires H20 Piece A engine-side top-k.
    #[arg(long, value_parser = parse_positive_usize)]
    pub(crate) teacher_topk: Option<usize>,

    /// Opt in to the fused lm_head+loss path (computes full-vocab logits + KD loss
    /// in one op to save the [window, vocab] tensor). DEFAULT IS DENSE: on H20 the
    /// fused path ran the lm_head on the HOST (~205 s/step for the 27B, GPU idle)
    /// vs ~3.9 s GPU-bound for dense — only enable for windows too large to
    /// materialize [window, vocab]. See errors/2026-06-23-opd-fused-distill-default-host-bound.
    #[arg(long, default_value_t = false)]
    pub(crate) fused_distill: bool,

    /// Total SOPD training steps.
    #[arg(long, default_value_t = 5)]
    pub(crate) steps: usize,

    /// AdamW learning rate.
    #[arg(long, default_value_t = 1.0e-4)]
    pub(crate) lr: f32,

    /// Learning-rate schedule for SOPD optimizer steps.
    #[arg(long, value_enum, default_value_t = LrScheduleArg::Fixed)]
    pub(crate) lr_schedule: LrScheduleArg,

    /// Warmup steps for --lr-schedule cosine. Defaults to ceil(3% of --steps).
    #[arg(long)]
    pub(crate) lr_warmup_steps: Option<usize>,

    /// Gradient L2 norm clip threshold.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) grad_clip: f32,

    /// EMA decay α for the self-teacher: θ_ema ← α·θ_ema + (1−α)·θ_student.
    #[arg(long, default_value_t = 0.999)]
    pub(crate) ema_alpha: f32,

    /// GKD blend λ: loss = λ·CE(student‖its own rollout) + (1−λ)·KL(student‖EMA).
    /// MUST be > 0 for cold start — at λ=0 with a fresh adapter the student, EMA,
    /// and base coincide so the KL gradient is identically zero (dead on arrival).
    #[arg(long, default_value_t = 0.5)]
    pub(crate) gkd_lambda: f32,

    /// Run the held-out NLL no-regression gate every N steps (0 disables it).
    #[arg(long, default_value_t = 0)]
    pub(crate) gate_every_n: usize,

    /// Relative held-out-NLL tolerance: revert the window if NLL rises above
    /// baseline·(1 + tol).
    #[arg(long, default_value_t = 0.02)]
    pub(crate) gate_regress_tol: f32,

    /// LoRA rank for the student/EMA adapters.
    #[arg(long, default_value_t = 16)]
    pub(crate) lora_rank: usize,

    /// LoRA alpha for the student/EMA adapters.
    #[arg(long, default_value_t = 32.0)]
    pub(crate) lora_alpha: f32,

    /// LoRA target set: `attention-qv`, `attention-full` (q/k/v/o + linear-attn), or `all-linear`.
    #[arg(long, default_value = "attention-qv")]
    pub(crate) lora_target_set: String,

    /// Directory where servable full-materialized SOPD checkpoints are written.
    #[arg(long, value_name = "DIR")]
    pub(crate) save_checkpoint: Option<PathBuf>,

    /// Save a servable full checkpoint every N steps (0 = final checkpoint only).
    #[arg(long, default_value_t = 0)]
    pub(crate) save_every: usize,

    /// Smoke mode — embedded tiny Qwen3.5 config, no disk weights. The gate is
    /// skipped (random tiny weights make held-out NLL meaningless).
    #[arg(long, default_value_t = false)]
    pub(crate) smoke: bool,

    /// Compute backend for autograd (auto picks CUDA when built with cuda).
    #[arg(long, value_enum, default_value_t = OpdBackendArg::Auto)]
    pub(crate) backend: OpdBackendArg,

    /// Storage dtype for tape-saved activations (f32 default is byte-identical).
    #[arg(long, value_enum, default_value_t = TapeDtypeArg::F32)]
    pub(crate) tape_dtype: TapeDtypeArg,

    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

/// Which built-in rubric the Flash judge grades rollouts against.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum RubricTaskArg {
    /// Math reasoning: factual answer-correctness gates acceptance.
    Math,
    /// Agentic tool-use (BFCL): correct call/abstain gates acceptance.
    Agentic,
}

/// cc-harness reward transform (`--reward-shape`). Default `Dense` reproduces
/// the historical reward exactly.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub(crate) enum RewardShapeArg {
    /// Pytest pass-fraction as-is (unchanged default).
    #[default]
    Dense,
    /// 1.0 only when every test passes, else 0.0.
    Binary,
    /// Pass/fail anchor with a small tie-break fraction for fails; errors → 0.
    Anchored,
}

impl From<RewardShapeArg> for train::cc_harness::RewardShape {
    fn from(arg: RewardShapeArg) -> Self {
        match arg {
            RewardShapeArg::Dense => Self::Dense,
            RewardShapeArg::Binary => Self::Binary,
            RewardShapeArg::Anchored => Self::Anchored,
        }
    }
}

/// What the accepted set writes back as CE targets.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum RubricWritebackArg {
    /// Mode A — train on the student's own accepted rollouts (RFT; caps at best-of-N).
    Accepted,
    /// Mode B — train on the teacher's corrected solution for rejected rollouts
    /// (breaks the best-of-N ceiling). Requires the judge `correct()` path.
    Correction,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Rubric-OPD: the student samples N rollouts/prompt; DeepSeek-V4-Flash judges\neach against a text-level rubric (vocab-agnostic, so a DeepSeek teacher can grade a\nQwen student); accepted rollouts are written back as CE targets (RFT). Selection-only\n(--writeback accepted) caps at the student's best-of-N; --writeback correction lets the\nteacher rewrite rejected rollouts to break that ceiling.\n\n  arle train rubric-opd --student-model <qwen3.6-27b-dir> --teacher-model <dsv4-flash-dir> \\\n    --prompts-file <jsonl> --rubric-task math --rounds 3 --samples-per-prompt 4"
)]
pub(crate) struct TrainRubricOpdArgs {
    #[command(flatten)]
    pub(crate) runtime: OpdRuntimeArgs,

    /// Student model directory (HF safetensors layout). The trained LoRA student.
    #[arg(long, alias = "student")]
    pub(crate) student_model: PathBuf,

    /// Judge/teacher model directory (DeepSeek-V4-Flash). Used as a text judge only.
    #[arg(long, alias = "teacher")]
    pub(crate) teacher_model: PathBuf,

    /// JSONL prompts file; each line `{"text": "<problem>"}`.
    #[arg(long, value_name = "FILE")]
    pub(crate) prompts_file: PathBuf,

    /// Which built-in rubric the judge grades against.
    #[arg(long, value_enum, default_value_t = RubricTaskArg::Math)]
    pub(crate) rubric_task: RubricTaskArg,

    /// What the accepted set writes back (Mode A = accepted, Mode B = correction).
    #[arg(long, value_enum, default_value_t = RubricWritebackArg::Accepted)]
    pub(crate) writeback: RubricWritebackArg,

    /// RFT rounds (generate → judge → select → writeback-CE).
    #[arg(long, default_value_t = 3)]
    pub(crate) rounds: usize,

    /// On-policy samples generated per prompt each round.
    #[arg(long, default_value_t = 4)]
    pub(crate) samples_per_prompt: usize,

    /// Cap on CE writeback steps per round (unset = all accepted). The 27B-dense
    /// autograd CE is ~minutes/step, so bounding the accepted set keeps a round
    /// tractable (first N accepted).
    #[arg(long, value_name = "N")]
    pub(crate) writeback_cap: Option<usize>,

    /// CE writeback micro-batch size. The CE is overhead-bound (GPU ~0% util), so
    /// batching B accepted pairs into one forward+backward amortizes the host
    /// op-dispatch (~B× faster); B bounds the [B, seq, vocab] logit VRAM.
    #[arg(long, default_value_t = 4)]
    pub(crate) writeback_batch: usize,

    /// Row chunk for the fused indexed-CE lm_head projection in the batched
    /// writeback; bounds the transient `[window, vocab]` logits tile.
    #[arg(long, default_value_t = 2048)]
    pub(crate) writeback_window: usize,

    /// Mode B: max Flash corrections written back per round (0 = select-only Mode A).
    #[arg(long, default_value_t = 0)]
    pub(crate) correction_cap: usize,

    /// Max new tokens for a Mode B correction solution.
    #[arg(long, default_value_t = 1024)]
    pub(crate) correction_max_tokens: usize,

    /// Max new tokens per sampled rollout.
    #[arg(long, default_value_t = 1024)]
    pub(crate) max_new_tokens: usize,

    /// Max tokens the judge may emit per verdict.
    #[arg(long, default_value_t = 512)]
    pub(crate) max_verdict_tokens: usize,

    /// KV slots for the judge engine (>1 batches judging; VRAM-gated at 3-model residency).
    #[arg(long, default_value_t = 1)]
    pub(crate) judge_num_slots: usize,

    /// Self-consistency mode: one model judges itself via majority-vote on \boxed
    /// (no 35B judge loaded; frees ~35GB). teacher-model is ignored.
    #[arg(long, default_value_t = false)]
    pub(crate) self_consistency: bool,

    /// KV slots for the rollout engine (self-consistency frees VRAM -> set higher,
    /// e.g. 16, for batched sampling/eval).
    #[arg(long, default_value_t = 2)]
    pub(crate) rollout_num_slots: usize,

    /// Disable train-infer FP8 weight sharing (训推一体). Sharing is ON by default:
    /// the autograd LoRA student's FROZEN FP8 base projections point (zero-copy) at
    /// the resident FP8 base of the co-resident rollout engine (same GPU/primary
    /// context), removing one ~27 GB copy (verified 61→44 GB, loss-identical, on
    /// Qwen3.6-27B-FP8). It is a graceful no-op for a non-FP8 single-GPU student
    /// (empty pointer table → normal load). Pass --no-share-frozen-base to force the
    /// byte-identical two-copy load (e.g. multi-GPU/TP, which sharing rejects).
    #[arg(long, default_value_t = false)]
    pub(crate) no_share_frozen_base: bool,

    /// SOPD 对且短: among the accepted (self-consistency-correct) rollouts for a
    /// prompt, distill ONLY the shortest (fewest tokens) instead of all of them —
    /// teaches concise correct reasoning (thinking efficiency). Off by default =
    /// write back every accepted rollout (correctness only).
    #[arg(long, default_value_t = false)]
    pub(crate) distill_shortest: bool,

    /// Rollout sampling temperature. 0.3 (not 1.0): hd256/FP8 sampling corrupts at
    /// temp>0 above ~0.3 (#48; b4b293f0c fixed greedy only). 0.3 stays coherent and
    /// >0. Restore 1.0 once the hd256/FP8 temp>0 fix lands.
    #[arg(long, default_value_t = 0.3, value_parser = parse_temperature, allow_hyphen_values = true)]
    pub(crate) rollout_temperature: f32,

    /// Rollout nucleus sampling threshold (1.0 disables top-p).
    #[arg(long, default_value_t = 0.95)]
    pub(crate) rollout_top_p: f32,

    /// Rollout top-k filter (0 disables top-k).
    #[arg(long, default_value_t = 20)]
    pub(crate) rollout_top_k: i32,

    /// Optional deterministic base rollout seed (per-sample = base + i).
    #[arg(long)]
    pub(crate) rollout_seed: Option<u64>,

    /// AdamW learning rate.
    #[arg(long, default_value_t = 1.0e-5)]
    pub(crate) lr: f32,

    /// LoRA rank for the student adapter.
    #[arg(long, default_value_t = 32)]
    pub(crate) lora_rank: usize,

    /// LoRA alpha for the student adapter.
    #[arg(long, default_value_t = 64.0)]
    pub(crate) lora_alpha: f32,

    /// LoRA target set: `attention-qv`, `attention-full` (q/k/v/o + linear-attn), or `all-linear`.
    #[arg(long, default_value = "all-linear")]
    pub(crate) lora_target_set: String,

    /// Suffix-detach: train LoRA only on layers >= N and detach the autograd
    /// backward before layer N (cuts the backward to the top suffix). Needed to fit
    /// a 27B dense student CE backward; unset = all layers (OOMs at 27B/seq~1k).
    #[arg(long, value_name = "N")]
    pub(crate) lora_layer_start: Option<usize>,

    /// Directory where servable full-materialized checkpoints are written.
    #[arg(long, value_name = "DIR")]
    pub(crate) save_checkpoint: Option<PathBuf>,

    /// Save a checkpoint every N rounds (0 = final round only).
    #[arg(long, default_value_t = 1)]
    pub(crate) save_every: usize,

    /// Directory for fast adapter-only (LoRA) safetensors saves
    /// (adapters_round{N}.safetensors). The full-materialize save host-loops the
    /// merged 27B weights and hangs; this writes only the tiny adapter matrices.
    #[arg(long, value_name = "DIR")]
    pub(crate) save_lora_adapters: Option<PathBuf>,

    /// Optional eval set (jsonl with `problem` + `answer` gold). Scored in-process
    /// via the rollout engine BEFORE training (base) and after each round — no
    /// checkpoint save (full-materialize is host-loop pathological at 27B).
    #[arg(long, value_name = "FILE")]
    pub(crate) eval_prompts_file: Option<PathBuf>,

    /// Directory for per-round eval answer dumps (eval_round_base.jsonl, eval_round{N}.jsonl).
    #[arg(long, value_name = "DIR")]
    pub(crate) eval_out_dir: Option<PathBuf>,

    /// Number of eval problems (from the head of --eval-prompts-file).
    #[arg(long, default_value_t = 50)]
    pub(crate) eval_n: usize,

    /// Max new tokens for EVAL generation (separate from rollout --max-new-tokens).
    /// Must be large: the think-on student needs room to reach the final \boxed{}
    /// answer; too small truncates mid-reasoning and scores a fake 0.
    #[arg(long, default_value_t = 4096)]
    pub(crate) eval_max_new_tokens: usize,

    /// Compute backend for autograd (auto picks CUDA when built with cuda).
    #[arg(long, value_enum, default_value_t = OpdBackendArg::Auto)]
    pub(crate) backend: OpdBackendArg,

    /// Storage dtype for tape-saved activations (f32 default is byte-identical).
    #[arg(long, value_enum, default_value_t = TapeDtypeArg::F32)]
    pub(crate) tape_dtype: TapeDtypeArg,

    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Agent-OPD: the student drives the read/write/replace/bash tool loop against a\nper-task repo sandbox (SWE-bench-Pro). The reward is EXECUTION — `git diff` is the\ncandidate patch, the hidden test_patch + fail_to_pass tests are run, exit-0 = pass.\nNo text judge/teacher is loaded; passing trajectories are written back as CE targets\n(verl-style response_mask keeps tool/environment tokens out of the loss).\n\n  arle train agent-opd --student-model <qwen3.6-27b-dir> \\\n    --dataset <swe-bench-pro.jsonl> --staged-root <repos-checked-out-at-base_commit> \\\n    --rounds 3 --samples-per-prompt 4"
)]
pub(crate) struct TrainAgentOpdArgs {
    #[command(flatten)]
    pub(crate) runtime: OpdRuntimeArgs,

    /// Student model directory (HF safetensors layout). The trained LoRA student.
    #[arg(long, alias = "student")]
    pub(crate) student_model: PathBuf,

    /// SWE-bench-Pro dataset JSONL (one task object per line: instance_id,
    /// problem_statement, repo, base_commit, test_patch, fail_to_pass, ...).
    /// Not used (and not required) with --replay-records.
    #[arg(long, value_name = "FILE", required_unless_present = "replay_records")]
    pub(crate) dataset: Option<PathBuf>,

    /// Directory holding each task's repo already checked out at its
    /// `base_commit`, addressed as `<staged-root>/<instance_id>/`.
    /// Not used (and not required) with --replay-records.
    #[arg(long, value_name = "DIR", required_unless_present = "replay_records")]
    pub(crate) staged_root: Option<PathBuf>,

    /// Replay pre-converted token records (`arle train cc-convert` output JSONL)
    /// through masked writeback instead of rolling out. Ratio-weighted presets
    /// require a finite generation-time `gen_logprobs` sidecar per masked token;
    /// CE/GKD permits records without it. Validated before model/store creation.
    #[arg(long, value_name = "FILE")]
    pub(crate) replay_records: Option<PathBuf>,

    /// Epochs over --replay-records.
    #[arg(long, default_value_t = 1)]
    pub(crate) replay_epochs: usize,

    /// Experience replay: after each fresh group's update, additionally train
    /// on up to N groups drawn from an age-bounded (≤10 rounds),
    /// |A|-prioritized buffer. IS-corrected against each trajectory's stored
    /// generation-time behavior sidecar; requires a ratio-weighted strategy.
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub(crate) replay_reuse: usize,

    /// Root dir under which each task's per-rollout sandbox is built (copied from
    /// the staged tree, reset between samples).
    #[arg(long, value_name = "DIR", default_value = "/tmp/agent-opd")]
    pub(crate) work_root: PathBuf,

    /// Optional cap on the number of tasks (from the head of --dataset) for a
    /// smoke run.
    #[arg(long, value_name = "N")]
    pub(crate) task_limit: Option<usize>,

    /// Pass-rate task selection: zero-variance skip (GRESO-style, P(explore) >=
    /// 0.1, fixed-seed RNG) + retirement at EMA pass >= 0.9 for 3 consecutive
    /// rounds. Round 0 always runs all tasks. `false` = every task every round.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) task_selection: bool,

    /// DAPO dynamic sampling: boot replacement groups (from the round's
    /// unscheduled tasks) for dead ones — zero-variance or all-truncated — until
    /// the round trains as many effective groups as it scheduled tasks. Capped by
    /// `--dynamic-sampling-max-factor`× launches and, if set, cumulative rollout
    /// tokens; an all-dead corpus terminates at the cap without an optimizer step.
    #[arg(long, default_value_t = false)]
    pub(crate) dynamic_sampling: bool,

    /// Launch cap as a multiple of the scheduled task count (attempt budget).
    #[arg(long, default_value_t = 3.0, value_name = "X")]
    pub(crate) dynamic_sampling_max_factor: f32,

    /// Cumulative rollout-completion-token budget for refills (0 = unbounded).
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub(crate) dynamic_sampling_token_budget: u64,

    /// HELD-OUT eval task set (SWE-bench-Pro JSONL, SAME schema as --dataset but
    /// SEPARATE tasks — no overlap, so the pass-rate measures generalization).
    /// When set, an eval-only rollout+score pass runs at round boundaries (a
    /// round-0 baseline BEFORE any training, then every --eval-every rounds) and
    /// the held-out PASS-RATE is logged next to mean_train_loss. Eval never
    /// trains (no writeback, no optimizer step, greedy sampling).
    #[arg(long, value_name = "FILE")]
    pub(crate) eval_dataset: Option<PathBuf>,

    /// Staged repos for the held-out --eval-dataset tasks
    /// (`<eval-staged-root>/<instance_id>/`). Defaults to --staged-root when the
    /// eval tasks live under the same staged tree.
    #[arg(long, value_name = "DIR")]
    pub(crate) eval_staged_root: Option<PathBuf>,

    /// Cap on the number of held-out eval tasks (from the head of
    /// --eval-dataset).
    #[arg(long, value_name = "N")]
    pub(crate) eval_n: Option<usize>,

    /// Run the held-out eval every N rounds (plus the round-0 baseline). 0 = off
    /// (only the baseline runs, if --eval-dataset is set). Default 1 (every round).
    #[arg(long, default_value_t = 1)]
    pub(crate) eval_every: usize,

    /// Held-out eval task-level concurrency. Eval rollouts share the
    /// continuous-batching serve engine but each blocks on host tool-exec, so a
    /// serial eval leaves the GPU idle; N concurrent tasks keep it fed (~N×
    /// faster). No policy update during eval → semantically identical to serial.
    /// Keep ≥2: at 1 there is no sandbox boot-ahead, so the GPU idles during each
    /// task's boot (the concurrent path relies on other workers to hide it).
    #[arg(long, default_value_t = 8)]
    pub(crate) eval_concurrency: usize,

    /// Directory for the per-round eval dumps (eval_round{N}.jsonl with per-task
    /// pass/fail + the aggregate pass-rate). Defaults to --save-lora-adapters,
    /// then --save-checkpoint, then the current dir.
    #[arg(long, value_name = "DIR")]
    pub(crate) eval_out_dir: Option<PathBuf>,

    /// Structured per-round metrics sink (one JSON line per round: the full
    /// round report + phase timers + held-out pass-rate). Defaults to
    /// <eval-out-dir>/metrics.jsonl.
    #[arg(long, value_name = "PATH")]
    pub(crate) metrics_out: Option<PathBuf>,

    /// Agent rollouts generated per task each round (best-of-N).
    #[arg(long, default_value_t = 4)]
    pub(crate) samples_per_prompt: usize,

    /// Prompts (task groups) rolled per policy update. All G roll concurrently
    /// at ONE policy version, then a single update trains their merged batch —
    /// G × samples_per_prompt sessions in flight instead of one group's worth,
    /// with the straggler tail amortized across the window. Staleness stays 0
    /// at any G. Slots/KV are sized from G × samples_per_prompt.
    #[arg(long, default_value_t = 1)]
    pub(crate) prompts_per_update: usize,

    /// Port the in-process serve binds on 127.0.0.1 for the cc rollout harness.
    #[arg(long, default_value_t = 8000)]
    pub(crate) serve_port: u16,

    /// Serve-side temperature for requests that omit it (CC sends none). 0.3, not
    /// 1.0: hd256/FP8 (Qwen3.6-27B) sampling corrupts at temp>0 above ~0.3 (#48;
    /// b4b293f0c fixed greedy/argmax only, the temp>0 distribution still degrades)
    /// — 0.3 stays coherent AND keeps behavior logprobs non-empty. Ratio-weighted
    /// strategies require T > 0; rejection-ce permits greedy T <= 0. Restore
    /// 1.0 once the hd256/FP8 temp>0 fix lands.
    #[arg(long, default_value_t = 0.3, value_name = "T")]
    pub(crate) rollout_temperature: f32,

    /// Per-sample `claude -p` wall-clock cap (seconds); the child process
    /// group is killed on expiry.
    #[arg(long, default_value_t = 600)]
    pub(crate) cc_timeout: u64,

    /// Weight-sync cadence into the rollout serve.
    #[arg(long, value_enum, default_value_t = SyncArg::EveryGroup)]
    pub(crate) sync: SyncArg,

    /// Rollout/train overlap: 0 = task groups strictly sequential (on-policy);
    /// 1 = admit the next group's rollouts before this group's train+merge.
    /// Ratio-weighted strategies use the generation-time behavior sidecar for
    /// both fresh and stale groups, so overlap does not change the denominator.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u8).range(0..=1))]
    pub(crate) staleness: u8,

    /// `PYTHONPATH` (sandbox-relative) for the bash tool + test runs, e.g.
    /// `lib:test`. Optional.
    #[arg(long, value_name = "PATH")]
    pub(crate) pythonpath: Option<String>,

    /// Test-run (scoring) timeout (seconds).
    #[arg(long, default_value_t = 300)]
    pub(crate) test_timeout_secs: u64,

    /// How the pytest pass-fraction becomes the reward. `dense` (default) is the
    /// fraction as-is, byte-identical to prior runs; `binary` = 1 only on a full
    /// pass; `anchored` = binary pass with a small tie-break fraction for fails.
    #[arg(long, value_enum, default_value_t = RewardShapeArg::Dense)]
    pub(crate) reward_shape: RewardShapeArg,

    /// RFT rounds (rollout → execute → score → writeback-CE).
    #[arg(long, default_value_t = 3)]
    pub(crate) rounds: usize,

    /// Cap on trained trajectories per round/epoch (unset = all accepted). Replay
    /// PG keeps task groups atomic: if the next full trainable group exceeds the
    /// remaining cap, replay stops instead of truncating that group.
    #[arg(long, value_name = "N")]
    pub(crate) writeback_cap: Option<usize>,

    /// Sequence-window size for the masked single-trajectory CE writeback. Each
    /// window re-forwards the prefix up to its end and materializes only a
    /// `[window, vocab]` logits tile, bounding VRAM on long agentic trajectories.
    #[arg(long, default_value_t = 2048)]
    pub(crate) writeback_window: usize,

    /// Policy-update preset. `rejection-ce` (default) = reject failures +
    /// masked CE. Ratio-weighted presets require finite generation-time
    /// behavior logprobs for every masked token and fail closed otherwise.
    #[arg(long, value_enum, default_value_t = UpdateStrategyArg::RejectionCe)]
    pub(crate) update_strategy: UpdateStrategyArg,

    /// SAO DIS lower clip on the importance ratio `exp(logπθ − logπ_rollout)`
    /// (gate opens above `1 − eps_low`). Only used by `--update-strategy sao-dis`.
    #[arg(long, default_value_t = 0.8)]
    pub(crate) sao_eps_low: f32,

    /// SAO DIS upper clip on the importance ratio (gate closes above
    /// `1 + eps_high`). Only used by `--update-strategy sao-dis`.
    #[arg(long, default_value_t = 3.0)]
    pub(crate) sao_eps_high: f32,

    /// SAO Phase 2 GAE discount γ. Only used by `--update-strategy sao-value`.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) sao_gamma: f32,

    /// SAO Phase 2 GAE trace λ. Only used by `--update-strategy sao-value`.
    #[arg(long, default_value_t = 0.95)]
    pub(crate) sao_lambda: f32,

    /// SAO Phase 2 value-critic learning rate. Only used by `sao-value`.
    #[arg(long, default_value_t = 1.0e-3)]
    pub(crate) value_lr: f32,

    /// Disable train-infer FP8 weight sharing (训推一体). Sharing is ON by default
    /// (the autograd student's frozen FP8 base points zero-copy at the rollout
    /// engine's resident base; graceful no-op for non-FP8 single-GPU). Pass
    /// --no-share-frozen-base to force the byte-identical two-copy load.
    #[arg(long, default_value_t = false)]
    pub(crate) no_share_frozen_base: bool,

    /// AdamW learning rate.
    #[arg(long, default_value_t = 1.0e-5)]
    pub(crate) lr: f32,

    /// LoRA rank for the student adapter.
    #[arg(long, default_value_t = 32)]
    pub(crate) lora_rank: usize,

    /// LoRA alpha for the student adapter.
    #[arg(long, default_value_t = 64.0)]
    pub(crate) lora_alpha: f32,

    /// LoRA target set: `attention-qv`, `attention-full` (q/k/v/o + linear-attn), or `all-linear`.
    #[arg(long, default_value = "all-linear")]
    pub(crate) lora_target_set: String,

    /// Suffix-detach: train LoRA only on layers >= N and detach the autograd
    /// backward before layer N (cuts the backward to the top suffix). Needed to
    /// fit a 27B dense student CE backward; unset = all layers.
    #[arg(long, value_name = "N")]
    pub(crate) lora_layer_start: Option<usize>,

    /// Directory where servable full-materialized checkpoints are written.
    #[arg(long, value_name = "DIR")]
    pub(crate) save_checkpoint: Option<PathBuf>,

    /// Save a checkpoint every N rounds (0 = final round only).
    #[arg(long, default_value_t = 1)]
    pub(crate) save_every: usize,

    /// Directory for fast adapter-only (LoRA) safetensors saves
    /// (adapters_round{N}.safetensors).
    #[arg(long, value_name = "DIR")]
    pub(crate) save_lora_adapters: Option<PathBuf>,

    /// Crash-resume from a previously-saved PEFT LoRA adapter dir (a
    /// `--save-lora-adapters` round output, containing `adapter_model.safetensors`):
    /// overlays its A/B weights onto the fresh student before training/eval.
    /// Fresh zero-B LoRA if unset. Round-to-round chaining is in-process (the
    /// adapter stays in the TensorStore), not a use of this flag.
    #[arg(long, value_name = "DIR")]
    pub(crate) lora_adapters: Option<PathBuf>,

    /// Compute backend for autograd (auto picks CUDA when built with cuda).
    #[arg(long, value_enum, default_value_t = OpdBackendArg::Auto)]
    pub(crate) backend: OpdBackendArg,

    /// Storage dtype for tape-saved activations (f32 default is byte-identical).
    #[arg(long, value_enum, default_value_t = TapeDtypeArg::F32)]
    pub(crate) tape_dtype: TapeDtypeArg,

    /// Diagnostic: skip the rollout and run ONE masked-CE writeback on a synthetic trajectory of this token length (find writeback OOM fast). 0 = off.
    #[arg(long, default_value_t = 0)]
    pub(crate) synthetic_writeback_seq: usize,

    /// Context-parallel group size (sequence sharded across N GPUs, one process
    /// per rank). 1 = single-card (default, byte-identical). >1 spawns N ranks and
    /// requires the nccl feature. The 256K writeback needs this — single-card peaks
    /// past one card at ~seq 49152.
    #[arg(long, default_value_t = 1)]
    pub(crate) cp_size: usize,

    /// Comma-separated CUDA device ordinals, one per world rank of the CP×DP
    /// mesh (world = cp_size*dp_size, CP inner; e.g. "0,1,2,3"). Defaults to
    /// 0..world. Synonym of --dp-devices; --cp-devices wins when both are set.
    #[arg(long)]
    pub(crate) cp_devices: Option<String>,

    /// Data-parallel group size (batch sharded across N GPUs, one process per
    /// rank; weights replicated, grads all-reduced). 1 = single-card (default,
    /// byte-identical). >1 spawns N ranks and requires the nccl feature.
    /// Composes with cp_size>1 (world = cp*dp; seq collectives run on a
    /// ncclCommSplit CP subgroup, grad/count all-reduces on the world comm).
    #[arg(long, default_value_t = 1)]
    pub(crate) dp_size: usize,

    /// Comma-separated CUDA device ordinals, one per world rank of the CP×DP
    /// mesh. Defaults to 0..world. Synonym of --cp-devices.
    #[arg(long)]
    pub(crate) dp_devices: Option<String>,

    /// Skip LoRA on all routed MoE expert projections (gate/up/down per expert).
    /// Keeps LoRA only on attention projections (q/k/v/o) and shared experts.
    /// Drops LoRA params from ~6.1B to ~59M and Adam state from ~49 GB to ~0.5 GB,
    /// enabling multiple writebacks per round without --writeback-cap.
    #[arg(long, default_value_t = false)]
    pub(crate) lora_skip_experts: bool,

    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    /// GKD mode for --replay-records: distil a TEACHER's per-position
    /// distribution on the trajectory tokens (forward-KL) instead of masked
    /// next-token CE. Denser signal than reproduction-only RFT. CUDA only.
    #[arg(long, default_value_t = false)]
    pub(crate) gkd: bool,

    /// GKD teacher source (with --gkd): `ema` (a slow EMA of the student) or
    /// `self` (the student's frozen initial adapter+base snapshot).
    #[arg(long, value_enum, default_value_t = GkdTeacherArg::Ema)]
    pub(crate) gkd_teacher: GkdTeacherArg,

    /// GKD softmax temperature for the teacher-KL distill loss.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) gkd_temperature: f32,

    /// AEPO entropy weighting for --gkd: scale each position's KL by
    /// `(1 + w * normalized_student_entropy)`. 0 = off. NOTE: not yet wired —
    /// a positive value logs a TODO and runs unweighted KL.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) gkd_entropy_weight: f32,

    /// EMA decay for `--gkd-teacher ema` (θ_ema ← α·θ_ema + (1−α)·θ_student).
    #[arg(long, default_value_t = 0.999)]
    pub(crate) gkd_ema_alpha: f32,
}

impl TrainAgentOpdArgs {
    /// Per-rank CUDA device ordinals for the CP×DP mesh (world = cp*dp, CP
    /// inner). `--cp-devices` (or `--dp-devices`) if set — one entry per world
    /// rank — else `0..world`.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn mesh_devices(&self) -> Vec<usize> {
        let world = self.cp_size.max(1) * self.dp_size.max(1);
        match self.cp_devices.as_ref().or(self.dp_devices.as_ref()) {
            Some(spec) => spec
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect(),
            None => (0..world).collect(),
        }
    }

    /// The `--update-strategy` preset; the sao-* flags override the SAO presets'
    /// clip/GAE fields (their clap defaults ARE the preset defaults).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn update_preset(&self) -> train::update_strategy::UpdatePreset {
        use train::update_strategy::UpdatePreset;
        match self.update_strategy {
            UpdateStrategyArg::RejectionCe => UpdatePreset::rejection_ce(),
            UpdateStrategyArg::SaoDis => UpdatePreset::sao_dis(self.sao_eps_low, self.sao_eps_high),
            UpdateStrategyArg::SaoValue => UpdatePreset::sao_value(
                self.sao_eps_low,
                self.sao_eps_high,
                self.sao_gamma,
                self.sao_lambda,
            ),
            UpdateStrategyArg::Grpo => UpdatePreset::grpo(),
            UpdateStrategyArg::Dapo => UpdatePreset::dapo(),
            // Dr.GRPO's fixed normalizer = the rollout generation budget
            // (cc owns sampling/turns; the session token ceiling is the budget).
            UpdateStrategyArg::DrGrpo => {
                UpdatePreset::dr_grpo(train::cc_harness::CC_SESSION_TOKENS)
            }
            UpdateStrategyArg::Gspo => UpdatePreset::gspo(),
            UpdateStrategyArg::Cispo => UpdatePreset::cispo(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Args, CliCommand, LrScheduleArg, OpdKlMaskArg, OpdSftAnchorArg, OpdTeacherRuntimeArg,
        RunArgs, TrainCommand,
    };
    use clap::{CommandFactory, Parser};

    /// The two divergences are deliberate; a third fails this test.
    #[cfg(feature = "cuda")]
    #[test]
    fn serve_defaults_match_the_seam_defaults() {
        // `ServeArgs` is a `clap::Args` group, not a `Parser` — build a bare
        // command around it to get the defaults clap would apply.
        let cmd = <super::ServeArgs as clap::Args>::augment_args(clap::Command::new("serve"));
        let serve = <super::ServeArgs as clap::FromArgMatches>::from_arg_matches(
            &cmd.get_matches_from(["serve"]),
        )
        .expect("ServeArgs defaults parse")
        .cuda_runtime_flags();
        assert!(serve.dsv4_decode_reuse, "licensed on serve only");
        assert_eq!(serve.comm_backend, infer_api::CommBackend::Nccl);
        let rest = infer_api::CudaRuntimeFlags {
            dsv4_decode_reuse: false,
            comm_backend: infer_api::CommBackend::Auto,
            ..serve
        };
        assert_eq!(rest, infer_api::CudaRuntimeFlags::default());
    }

    #[test]
    fn parse_kv_budget_accepts_percent_fraction_bytes_and_off() {
        use infer_api::KvTierBudget;

        use super::parse_kv_budget;
        assert_eq!(parse_kv_budget("50%"), Ok(KvTierBudget::Fraction(0.5)));
        assert_eq!(parse_kv_budget("0.5"), Ok(KvTierBudget::Fraction(0.5)));
        assert_eq!(parse_kv_budget("16GiB"), Ok(KvTierBudget::Bytes(16 << 30)));
        assert_eq!(
            parse_kv_budget("1073741824"),
            Ok(KvTierBudget::Bytes(1 << 30))
        );
        assert_eq!(parse_kv_budget("0"), Ok(KvTierBudget::Off));
        assert_eq!(parse_kv_budget("off"), Ok(KvTierBudget::Off));
        assert!(parse_kv_budget("junk").is_err());
    }

    #[test]
    fn rejects_removed_max_gpu_kv_flag() {
        let err = Args::try_parse_from(["arle", "--max-gpu-kv", "256"])
            .err()
            .expect("removed flag should be rejected");
        let rendered = err.to_string();
        assert!(rendered.contains("--max-gpu-kv"));
    }

    #[test]
    fn rejects_removed_tools_flag() {
        let err = Args::try_parse_from(["arle", "--tools"])
            .err()
            .expect("removed flag should be rejected");
        assert!(err.to_string().contains("--tools"));
    }

    #[test]
    fn accepts_no_tools_flag() {
        let args = Args::try_parse_from(["arle", "--no-tools"])
            .expect("global no-tools flag should parse");
        assert!(args.no_tools);
    }

    #[test]
    fn accepts_run_no_tools_flag() {
        let args = Args::try_parse_from(["arle", "run", "--no-tools"])
            .expect("run no-tools flag should parse");
        match args.command.expect("run command") {
            CliCommand::Run(run) => assert!(run.no_tools),
            other => panic!("expected run command, got {other:?}"),
        }
    }

    #[test]
    fn accepts_serve_command() {
        let args = Args::try_parse_from([
            "arle",
            "serve",
            "--backend",
            "cpu",
            "--model-path",
            "models/tiny",
            "--port",
            "8010",
            "--",
            "--max-waiting",
            "8",
        ])
        .expect("serve command should parse");
        match args.command.expect("serve command") {
            CliCommand::Serve(serve) => {
                assert_eq!(serve.backend, super::ServeBackendArg::Cpu);
                assert_eq!(serve.model_path.as_deref(), Some("models/tiny"));
                assert_eq!(serve.port, 8010);
                assert_eq!(serve.extra_args, ["--max-waiting", "8"]);
            }
            other => panic!("expected serve command, got {other:?}"),
        }
    }

    #[test]
    fn accepts_serve_speculative_flags() {
        let args = Args::try_parse_from([
            "arle",
            "serve",
            "--backend",
            "metal",
            "--model-path",
            "models/qwen36-27b",
            "--draft-model",
            "mlx-community/Qwen3.6-27B-MTP-4bit",
            "--speculative-tokens",
            "3",
        ])
        .expect("serve speculative flags should parse");
        match args.command.expect("serve command") {
            CliCommand::Serve(serve) => {
                assert_eq!(
                    serve.draft_model.as_deref(),
                    Some("mlx-community/Qwen3.6-27B-MTP-4bit")
                );
                assert_eq!(serve.speculative_tokens, 3);
                assert!(!serve.no_speculative);
            }
            other => panic!("expected serve command, got {other:?}"),
        }
    }

    #[test]
    fn serve_speculative_tokens_defaults_to_two() {
        let args = Args::try_parse_from(["arle", "serve", "--model-path", "models/qwen36-27b"])
            .expect("serve should parse");
        match args.command.expect("serve command") {
            CliCommand::Serve(serve) => {
                assert_eq!(serve.speculative_tokens, 2);
                assert!(serve.draft_model.is_none());
                assert!(!serve.no_speculative);
            }
            other => panic!("expected serve command, got {other:?}"),
        }
    }

    #[test]
    fn accepts_serve_no_speculative_flag() {
        let args = Args::try_parse_from([
            "arle",
            "serve",
            "--model-path",
            "models/qwen36-27b",
            "--no-speculative",
        ])
        .expect("serve --no-speculative should parse");
        match args.command.expect("serve command") {
            CliCommand::Serve(serve) => assert!(serve.no_speculative),
            other => panic!("expected serve command, got {other:?}"),
        }
    }

    #[test]
    fn rejects_serve_speculative_tokens_below_two() {
        for bad in ["0", "1"] {
            let err = Args::try_parse_from([
                "arle",
                "serve",
                "--model-path",
                "models/qwen36-27b",
                "--speculative-tokens",
                bad,
            ])
            .err()
            .unwrap_or_else(|| panic!("speculative depth `{bad}` should be rejected"));
            assert!(
                err.to_string().contains("at least 2"),
                "unexpected error for `{bad}`: {err}"
            );
        }
    }

    #[test]
    fn rejects_zero_max_turns() {
        let err = Args::try_parse_from(["arle", "--max-turns", "0"])
            .err()
            .expect("zero max-turns should be rejected");
        assert!(err.to_string().contains("at least 1"));
    }

    #[test]
    fn accepts_zero_max_tokens_as_auto_sentinel() {
        // 0 is reserved as the "auto-resolve from model config" sentinel —
        // the CLI substitutes `max_position_embeddings` at startup before
        // any inference call, so the engine never sees a literal 0.
        let args = Args::try_parse_from(["arle", "--max-tokens", "0"])
            .expect("0 max-tokens should parse as auto");
        assert_eq!(args.max_tokens, 0);
    }

    #[test]
    fn accepts_auto_keyword_for_max_tokens() {
        let args = Args::try_parse_from(["arle", "--max-tokens", "auto"])
            .expect("'auto' should parse as the same sentinel");
        assert_eq!(args.max_tokens, 0);
    }

    #[test]
    fn rejects_negative_or_garbage_max_tokens() {
        // Anything other than 0/auto/positive-integer must still error out.
        for bad in ["-1", "1.5", "abc", " "] {
            let err = Args::try_parse_from(["arle", "--max-tokens", bad])
                .err()
                .unwrap_or_else(|| panic!("garbage value `{bad}` should be rejected"));
            let msg = err.to_string();
            assert!(
                msg.contains("expected") || msg.contains("at least 1"),
                "unexpected error for `{bad}`: {msg}"
            );
        }
    }

    #[test]
    fn rejects_negative_temperature() {
        let err = Args::try_parse_from(["arle", "--temperature", "-0.1"])
            .err()
            .expect("negative temperature should be rejected");
        assert!(err.to_string().contains("temperature must be >= 0.0"));
    }

    #[test]
    fn rejects_non_finite_temperature() {
        let err = Args::try_parse_from(["arle", "--temperature", "NaN"])
            .err()
            .expect("NaN temperature should be rejected");
        assert!(err.to_string().contains("temperature must be finite"));
    }

    #[test]
    fn accepts_doctor_flag() {
        let args = Args::try_parse_from(["arle", "--doctor"]).expect("doctor flag should parse");
        assert!(args.doctor);
    }

    #[test]
    fn accepts_list_models_flag() {
        let args =
            Args::try_parse_from(["arle", "--list-models"]).expect("list-models flag should parse");
        assert!(args.list_models);
    }

    #[test]
    fn rejects_doctor_and_list_models_together() {
        let err = Args::try_parse_from(["arle", "--doctor", "--list-models"])
            .err()
            .expect("doctor and list-models should conflict");
        assert!(err.to_string().contains("--list-models"));
    }

    #[test]
    fn accepts_doctor_json_flag() {
        let args = Args::try_parse_from(["arle", "--doctor", "--json"])
            .expect("doctor json flag should parse");
        assert!(args.doctor);
        assert!(args.json);
    }

    #[test]
    fn accepts_doctor_strict_flag() {
        let args = Args::try_parse_from(["arle", "--doctor", "--strict"])
            .expect("doctor strict flag should parse");
        assert!(args.doctor);
        assert!(args.strict);
    }

    #[test]
    fn accepts_list_models_json_flag() {
        let args = Args::try_parse_from(["arle", "--list-models", "--json"])
            .expect("list-models json flag should parse");
        assert!(args.list_models);
        assert!(args.json);
    }

    #[test]
    fn rejects_json_without_inspection_mode() {
        let err = Args::try_parse_from(["arle", "--json"])
            .err()
            .expect("--json without inspection mode should fail");
        assert!(err.to_string().contains("--doctor"));
    }

    #[test]
    fn rejects_strict_without_doctor() {
        let err = Args::try_parse_from(["arle", "--strict"])
            .err()
            .expect("--strict without doctor should fail");
        assert!(err.to_string().contains("--doctor"));
    }

    #[test]
    fn rejects_strict_with_list_models() {
        let err = Args::try_parse_from(["arle", "--list-models", "--strict"])
            .err()
            .expect("--strict with list-models should fail");
        let rendered = err.to_string();
        assert!(rendered.contains("--list-models"));
        assert!(rendered.contains("--strict"));
    }

    #[test]
    fn command_tree_is_valid() {
        Args::command().debug_assert();
    }

    #[test]
    fn accepts_run_prompt() {
        let args = Args::try_parse_from(["arle", "run", "--prompt", "hello"])
            .expect("run prompt should parse");
        let Some(CliCommand::Run(run)) = args.command else {
            panic!("expected run command");
        };
        assert_eq!(
            *run,
            RunArgs {
                prompt: Some("hello".to_string()),
                stdin: false,
                image: Vec::new(),
                json: false,
                no_tools: false,
            }
        );
    }

    #[test]
    fn accepts_run_stdin_json() {
        let args = Args::try_parse_from(["arle", "run", "--stdin", "--json"])
            .expect("run stdin json should parse");
        let Some(CliCommand::Run(run)) = args.command else {
            panic!("expected run command");
        };
        assert!(run.stdin);
        assert!(run.json);
        assert!(run.prompt.is_none());
    }

    #[test]
    fn accepts_run_image() {
        let args = Args::try_parse_from([
            "arle",
            "run",
            "--prompt",
            "describe",
            "--image",
            "https://example.com/cat.jpg",
            "--image",
            "/tmp/local.png",
        ])
        .expect("run image should parse");
        let Some(CliCommand::Run(run)) = args.command else {
            panic!("expected run command");
        };
        assert_eq!(
            run.image,
            vec![
                "https://example.com/cat.jpg".to_string(),
                "/tmp/local.png".to_string()
            ]
        );
    }

    #[test]
    fn rejects_run_json_without_input() {
        let err = Args::try_parse_from(["arle", "run", "--json"])
            .err()
            .expect("run --json without input should fail");
        assert!(err.to_string().contains("--prompt"));
    }

    #[test]
    fn rejects_retired_train_test_stub() {
        assert!(Args::try_parse_from(["arle", "train", "test"]).is_err());
    }

    #[test]
    fn accepts_train_opd_eval_gate_and_explicit_zero_gkd_lambda() {
        let args = Args::try_parse_from([
            "arle",
            "train",
            "opd",
            "--student-model",
            "models/qwen",
            "--teacher-model",
            "models/qwen-teacher",
            "--teacher-runtime",
            "infer",
            "--eval-ids",
            "1,2,3",
            "--gate-every-n",
            "5",
            "--gkd-lambda",
            "0.0",
            "--sft-anchor",
            "student-rollout",
            "--save-checkpoint",
            "/tmp/opd-save",
            "--save-every",
            "5",
            "--kl-temperature",
            "2.0",
            "--kl-beta",
            "0.5",
            "--kl-mask",
            "full",
            "--prompts-file",
            "examples/opd/prompts.jsonl",
            "--prompt-max-tokens",
            "256",
            "--prompt-seed",
            "42",
            "--lr-schedule",
            "cosine",
            "--lr-warmup-steps",
            "300",
        ])
        .expect("pure OPD eval flags should parse");
        let Some(CliCommand::Train(train)) = args.command else {
            panic!("expected train command");
        };
        let TrainCommand::Opd(opd) = train.command else {
            panic!("expected train opd command");
        };
        assert_eq!(
            opd.teacher_model.as_deref(),
            Some(std::path::Path::new("models/qwen-teacher"))
        );
        assert_eq!(opd.teacher_runtime, OpdTeacherRuntimeArg::Infer);
        assert_eq!(opd.eval_ids.as_deref(), Some("1,2,3"));
        assert_eq!(opd.gate_every_n, 5);
        assert_eq!(opd.gkd_lambda, 0.0);
        assert_eq!(opd.sft_anchor, OpdSftAnchorArg::StudentRollout);
        assert_eq!(opd.kl_temperature, 2.0);
        assert_eq!(opd.kl_beta, Some(0.5));
        assert_eq!(opd.teacher_topk, None);
        assert!(!opd.no_fused_distill);
        assert_eq!(opd.kl_mask, OpdKlMaskArg::Full);
        assert_eq!(
            opd.prompts_file.as_deref(),
            Some(std::path::Path::new("examples/opd/prompts.jsonl"))
        );
        assert_eq!(opd.prompt_max_tokens, 256);
        assert_eq!(opd.prompt_seed, 42);
        assert_eq!(opd.lr_schedule, LrScheduleArg::Cosine);
        assert_eq!(opd.lr_warmup_steps, Some(300));
        assert_eq!(opd.lora_rank, 16);
        assert_eq!(opd.lora_alpha, 32.0);
        assert_eq!(opd.lora_target_set, "attention-qv");
        assert_eq!(opd.lora_layer_start, None);
        assert_eq!(
            opd.save_checkpoint.as_deref(),
            Some(std::path::Path::new("/tmp/opd-save"))
        );
        assert_eq!(opd.save_every, 5);
    }

    #[test]
    fn train_opd_defaults_to_completion_kl_mask() {
        let args = Args::try_parse_from(["arle", "train", "opd", "--student-model", "models/qwen"])
            .expect("train opd should parse");
        let Some(CliCommand::Train(train)) = args.command else {
            panic!("expected train command");
        };
        let TrainCommand::Opd(opd) = train.command else {
            panic!("expected train opd command");
        };
        assert_eq!(opd.kl_mask, OpdKlMaskArg::Completion);
        assert_eq!(opd.teacher_runtime, OpdTeacherRuntimeArg::InProcess);
        assert!(opd.prompts_file.is_none());
        assert_eq!(opd.prompt_max_tokens, 512);
        assert_eq!(opd.prompt_seed, 0);
        assert_eq!(opd.lr_schedule, LrScheduleArg::Fixed);
        assert_eq!(opd.lr_warmup_steps, None);
        assert_eq!(opd.sft_anchor, OpdSftAnchorArg::StudentRollout);
        assert_eq!(opd.kl_beta, None);
        assert_eq!(opd.teacher_topk, None);
        assert!(!opd.no_fused_distill);
    }

    #[test]
    fn accepts_train_opd_teacher_topk_flag() {
        let args = Args::try_parse_from([
            "arle",
            "train",
            "opd",
            "--student-model",
            "models/qwen",
            "--teacher-topk",
            "64",
        ])
        .expect("train opd teacher top-k flag should parse");
        let Some(CliCommand::Train(train)) = args.command else {
            panic!("expected train command");
        };
        let TrainCommand::Opd(opd) = train.command else {
            panic!("expected train opd command");
        };
        assert_eq!(opd.teacher_topk, Some(64));
    }

    #[test]
    fn accepts_train_opd_no_fused_distill_flag() {
        let args = Args::try_parse_from([
            "arle",
            "train",
            "opd",
            "--student-model",
            "models/qwen",
            "--no-fused-distill",
        ])
        .expect("train opd no-fused-distill flag should parse");
        let Some(CliCommand::Train(train)) = args.command else {
            panic!("expected train command");
        };
        let TrainCommand::Opd(opd) = train.command else {
            panic!("expected train opd command");
        };
        assert!(opd.no_fused_distill);
    }

    #[test]
    fn train_opd_defaults_to_dense_fused_distill_opt_in() {
        // Default (no flag) must be dense: --fused-distill is opt-in, off by default.
        let parse = |extra: &[&str]| {
            let mut argv = vec!["arle", "train", "opd", "--student-model", "models/qwen"];
            argv.extend_from_slice(extra);
            let args = Args::try_parse_from(argv).expect("train opd should parse");
            let Some(CliCommand::Train(train)) = args.command else {
                panic!("expected train command");
            };
            let TrainCommand::Opd(opd) = train.command else {
                panic!("expected train opd command");
            };
            opd
        };
        // Absent => dense (fused_distill stays false).
        assert!(!parse(&[]).fused_distill);
        // Opt-in flag flips it on.
        assert!(parse(&["--fused-distill"]).fused_distill);
    }

    #[test]
    fn accepts_train_opd_corpus_truth_sft_anchor() {
        let args = Args::try_parse_from([
            "arle",
            "train",
            "opd",
            "--student-model",
            "models/qwen",
            "--prompts-file",
            "examples/opd/teacher-rollouts.jsonl",
            "--gkd-lambda",
            "1.0",
            "--sft-anchor",
            "corpus-truth",
        ])
        .expect("off-policy corpus-truth SFT flags should parse");
        let Some(CliCommand::Train(train)) = args.command else {
            panic!("expected train command");
        };
        let TrainCommand::Opd(opd) = train.command else {
            panic!("expected train opd command");
        };
        assert_eq!(opd.gkd_lambda, 1.0);
        assert_eq!(opd.sft_anchor, OpdSftAnchorArg::CorpusTruth);
    }

    #[test]
    fn accepts_train_self_opd_save_checkpoint_flags() {
        let args = Args::try_parse_from([
            "arle",
            "train",
            "self-opd",
            "--student-model",
            "models/qwen",
            "--save-checkpoint",
            "/tmp/self-opd-save",
            "--save-every",
            "3",
            "--lr-schedule",
            "cosine",
            "--lr-warmup-steps",
            "9",
        ])
        .expect("self-opd save flags should parse");
        let Some(CliCommand::Train(train)) = args.command else {
            panic!("expected train command");
        };
        let TrainCommand::SelfOpd(opd) = train.command else {
            panic!("expected train self-opd command");
        };
        assert_eq!(
            opd.save_checkpoint.as_deref(),
            Some(std::path::Path::new("/tmp/self-opd-save"))
        );
        assert_eq!(opd.save_every, 3);
        assert_eq!(opd.kl_temperature, 1.0);
        assert_eq!(opd.kl_beta, None);
        assert_eq!(opd.teacher_topk, None);
        assert_eq!(opd.lr_schedule, LrScheduleArg::Cosine);
        assert_eq!(opd.lr_warmup_steps, Some(9));
    }

    #[test]
    fn rejects_train_opd_non_positive_kl_temperature() {
        let err = Args::try_parse_from([
            "arle",
            "train",
            "opd",
            "--student-model",
            "models/qwen",
            "--kl-temperature",
            "0",
        ])
        .err()
        .expect("zero KL temperature should be rejected");
        assert!(err.to_string().contains("temperature must be > 0.0"));
    }

    #[test]
    fn rejects_train_opd_kl_beta_outside_unit_interval() {
        let err = Args::try_parse_from([
            "arle",
            "train",
            "opd",
            "--student-model",
            "models/qwen",
            "--kl-beta",
            "1.1",
        ])
        .err()
        .expect("out-of-range KL beta should be rejected");
        assert!(err.to_string().contains("[0.0, 1.0]"));
    }

    #[test]
    fn rejects_train_opd_zero_teacher_topk() {
        let err = Args::try_parse_from([
            "arle",
            "train",
            "opd",
            "--student-model",
            "models/qwen",
            "--teacher-topk",
            "0",
        ])
        .err()
        .expect("zero teacher top-k should be rejected");
        assert!(err.to_string().contains("at least 1"));
    }

    #[test]
    fn accepts_trace_with_prompts_off() {
        let args = Args::try_parse_from([
            "arle",
            "--trace",
            "/tmp/trace.jsonl",
            "--trace-prompts",
            "off",
        ])
        .expect("trace + trace-prompts should parse");
        assert_eq!(
            args.trace.as_deref(),
            Some(std::path::Path::new("/tmp/trace.jsonl"))
        );
        assert_eq!(args.trace_prompts, super::TracePromptsMode::Off);
    }

    #[test]
    fn trace_prompts_defaults_to_on() {
        let args = Args::try_parse_from(["arle"]).expect("default args");
        assert_eq!(args.trace_prompts, super::TracePromptsMode::On);
        assert!(args.trace.is_none());
    }

    /// Unset and `0` must stay distinguishable all the way into
    /// `CudaRuntimeFlags`: unset keeps the goodput budget, `0` proposes the
    /// whole block. Collapsing them flips every serve's default silently.
    #[test]
    fn dspark_confidence_threshold_separates_unset_from_zero() {
        let flags = |extra: &[&str]| {
            let argv = [&["arle", "serve", "--backend", "cuda"][..], extra].concat();
            match Args::try_parse_from(argv)
                .expect("serve should parse")
                .command
                .expect("serve command")
            {
                CliCommand::Serve(serve) => serve.cuda_runtime_flags().dspark_confidence_threshold,
                other => panic!("expected serve command, got {other:?}"),
            }
        };
        assert_eq!(flags(&[]), None);
        assert_eq!(flags(&["--dspark-confidence-threshold", "0"]), Some(0.0));
        assert_eq!(
            flags(&["--dspark-confidence-threshold", "0.75"]),
            Some(0.75)
        );
        assert!(
            Args::try_parse_from([
                "arle",
                "serve",
                "--backend",
                "cuda",
                "--dspark-confidence-threshold",
                "1.5",
            ])
            .is_err()
        );
    }

    #[test]
    fn rejects_empty_trace_path() {
        let err = Args::try_parse_from(["arle", "--trace", ""])
            .err()
            .expect("empty trace path should be rejected");
        assert!(err.to_string().contains("trace path must not be empty"));
    }
}
