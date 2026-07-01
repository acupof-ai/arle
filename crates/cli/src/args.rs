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
pub(crate) enum BackendArg {
    Auto,
    Cpu,
    Metal,
    Cuda,
    Vulkan,
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

    /// Disable CUDA graph (useful for debugging)
    #[arg(long, default_value_t = false)]
    pub(crate) no_cuda_graph: bool,

    /// Maximum CUDA decode batch size to pre-capture for graph replay.
    #[arg(long)]
    pub(crate) cuda_graph_max_bs: Option<usize>,

    /// Disable built-in shell/python tools for the local agent runtime.
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

    /// Opt into the SSD/NVMe KV tier using the default local cache root.
    #[arg(long, default_value_t = false)]
    pub(crate) kv_ssd: bool,

    /// Opt into the SSD/NVMe KV tier root. The rewrite stack validates this
    /// request and fails closed until a backend exposes real SSD recall.
    #[arg(long, value_name = "DIR")]
    pub(crate) kv_ssd_path: Option<PathBuf>,

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

    /// Maximum bytes this serve process may use under the SSD KV tier root.
    #[arg(long, value_parser = parse_positive_usize)]
    pub(crate) kv_ssd_max_bytes: Option<usize>,

    /// Host-RAM budget in bytes for the T1 prefix-KV tier (default-on:
    /// CUDA dense keeps 4 GiB when unset). `0` disables the tier.
    #[arg(long, value_name = "BYTES")]
    pub(crate) kv_t1_budget_bytes: Option<usize>,

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
    /// activations/scratch. Clamped to `[0.5, 0.97]`. Wired for CUDA dense Qwen3
    /// + Qwen3.6 + Metal (one shared pool); DSv4 keeps per-slot sizing for now.
    ///   Default 0.9.
    #[arg(long, default_value_t = 0.9)]
    pub(crate) mem_fraction_static: f64,

    /// Fraction of *available* host DRAM the L2 (host-DRAM) KV tier may claim.
    /// The store is pageable host memory on a (shared) box, so the default 0.5
    /// is deliberately conservative — the budget is
    /// `clamp(frac × MemAvailable, [4 GiB, MemAvailable − max(64 GiB, 0.15 ×
    /// MemTotal)])`, leaving a reserve so the OS never swaps/OOM-kills. CUDA
    /// only; `--kv-t1-budget-bytes` is an explicit override that wins. Default 0.5.
    #[arg(long, default_value_t = 0.5)]
    pub(crate) dram_fraction: f64,

    /// Fraction of *free* disk the L3 (NVMe) KV tier may claim when `--kv-ssd-path`
    /// is set without `--kv-ssd-max-bytes`. Budget is `clamp(frac × free_disk,
    /// [8 GiB, free_disk − max(50 GiB, 0.1 × total_disk)])`, reserving room for
    /// the FS + co-tenants. Default 0.5.
    #[arg(long, default_value_t = 0.5)]
    pub(crate) ssd_fraction: f64,

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

    /// Opt into running-cap oversubscription: rotate waiters in by parking the
    /// longest-running decode's whole-slot KV image to the tier store. Requires
    /// a whole-slot tier backend; default off (byte-identical).
    #[arg(long, default_value_t = false)]
    pub(crate) slot_oversubscription: bool,

    /// Optional upstream train control-plane URL to expose under `/v1/train/*`.
    #[arg(long)]
    pub(crate) train_control_url: Option<String>,

    /// Speculative decode route. Currently only CUDA accepts checkpoint-native
    /// `mtp`; external draft-model routes are not re-ported in the rewrite.
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

    /// External split MTP drafter model path or HuggingFace repo.
    /// Parsed for compatibility, but rejected by the rewrite serve path until
    /// external draft-model speculative decode is re-ported.
    #[arg(long, value_name = "PATH_OR_REPO")]
    pub(crate) mtp_draft_model: Option<String>,

    /// Number of MTP draft tokens to propose per verify block on CUDA.
    #[arg(long, value_name = "N")]
    pub(crate) mtp_draft_tokens: Option<usize>,

    /// Metal speculative-decode draft head (HF id or local dir). When set, the
    /// Metal backend drafts with this NextN/MTP head and verifies on the base
    /// (bit-identical to greedy). Overrides the `INFER_METAL_DFLASH_DRAFT_MODEL`
    /// env fallback. Auto-resolved for Qwen3.6-27B unless `--no-speculative`.
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

    /// Additional engine-pool model metadata to expose from the serving control plane.
    ///
    /// Format: `id=path[,type=text-generation|embedding|reranker][,aliases=a|b][,pinned=true][,memory_bytes=N][,ttl_secs=N]`.
    /// The first implementation is metadata/control-plane only; non-primary
    /// embedding and reranker entries are explicit stubs, not generation routes.
    #[arg(long = "pool-model", value_name = "SPEC")]
    pub(crate) pool_models: Vec<String>,

    /// Forward additional backend-specific flags after `--`.
    #[arg(last = true, allow_hyphen_values = true)]
    pub(crate) extra_args: Vec<String>,
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
    AgentOpd(TrainAgentOpdArgs),
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

    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum OpdBackendArg {
    /// Pick CUDA when available, else CPU.
    Auto,
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Self-OPD (SOPD): inline self-training. The student samples a rollout,\nan EMA self-teacher (base + lagged EMA adapter) re-scores it, GKD blends a\nλ·CE self-anchor with (1−λ)·KL(student‖EMA), and the EMA adapter trails the\nstudent θ_ema ← α·θ_ema + (1−α)·θ_student. Every N steps a held-out NLL gate\nreverts {student, EMA, AdamW} together on regress.\n\nSmoke (no model load, embedded tiny Qwen3.5 config):\n  arle train self-opd --smoke --steps 5\n\nReal:\n  arle train self-opd --student-model <dir> --steps 20 --rollout-len 16 --gate-every-n 10"
)]
pub(crate) struct TrainSelfOpdArgs {
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

    /// Rollout sampling temperature (>0 for rejection-sampling diversity).
    #[arg(long, default_value_t = 1.0, value_parser = parse_temperature, allow_hyphen_values = true)]
    pub(crate) rollout_temperature: f32,

    /// Rollout nucleus sampling threshold (1.0 disables top-p).
    #[arg(long, default_value_t = 1.0)]
    pub(crate) rollout_top_p: f32,

    /// Rollout top-k filter (0 disables top-k).
    #[arg(long, default_value_t = 0)]
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

    /// Gradient checkpointing for the CE backward (recompute activations). On by
    /// default; `--grad-checkpointing false` to disable.
    #[arg(long, default_value_t = true)]
    pub(crate) grad_checkpointing: bool,

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

    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, ClapArgs)]
#[command(
    after_help = "Agent-OPD: the student drives the read/write/replace/bash tool loop against a\nper-task repo sandbox (SWE-bench-Pro). The reward is EXECUTION — `git diff` is the\ncandidate patch, the hidden test_patch + fail_to_pass tests are run, exit-0 = pass.\nNo text judge/teacher is loaded; passing trajectories are written back as CE targets\n(verl-style response_mask keeps tool/environment tokens out of the loss).\n\n  arle train agent-opd --student-model <qwen3.6-27b-dir> \\\n    --dataset <swe-bench-pro.jsonl> --staged-root <repos-checked-out-at-base_commit> \\\n    --rounds 3 --samples-per-prompt 4"
)]
pub(crate) struct TrainAgentOpdArgs {
    /// Student model directory (HF safetensors layout). The trained LoRA student.
    #[arg(long, alias = "student")]
    pub(crate) student_model: PathBuf,

    /// SWE-bench-Pro dataset JSONL (one task object per line: instance_id,
    /// problem_statement, repo, base_commit, test_patch, fail_to_pass, ...).
    #[arg(long, value_name = "FILE")]
    pub(crate) dataset: PathBuf,

    /// Directory holding each task's repo already checked out at its
    /// `base_commit`, addressed as `<staged-root>/<instance_id>/`.
    #[arg(long, value_name = "DIR")]
    pub(crate) staged_root: PathBuf,

    /// Root dir under which each task's per-rollout sandbox is built (copied from
    /// the staged tree, reset between samples).
    #[arg(long, value_name = "DIR", default_value = "/tmp/agent-opd")]
    pub(crate) work_root: PathBuf,

    /// Optional cap on the number of tasks (from the head of --dataset) for a
    /// smoke run.
    #[arg(long, value_name = "N")]
    pub(crate) task_limit: Option<usize>,

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

    /// Directory for the per-round eval dumps (eval_round{N}.jsonl with per-task
    /// pass/fail + the aggregate pass-rate). Defaults to --save-lora-adapters,
    /// then --save-checkpoint, then the current dir.
    #[arg(long, value_name = "DIR")]
    pub(crate) eval_out_dir: Option<PathBuf>,

    /// Sampling temperature for the held-out eval rollout (0.0 = greedy). Kept
    /// separate from --rollout-temperature so eval is deterministic-ish while
    /// training samples diversely.
    #[arg(long, default_value_t = 0.0, value_parser = parse_temperature, allow_hyphen_values = true)]
    pub(crate) eval_temperature: f32,

    /// Agent rollouts generated per task each round (best-of-N).
    #[arg(long, default_value_t = 4)]
    pub(crate) samples_per_prompt: usize,

    /// Max agent turns (tool sub-turns) per rollout.
    #[arg(long, default_value_t = 30)]
    pub(crate) max_turns: usize,

    /// Max tokens the engine generates per agent sub-turn.
    #[arg(long, default_value_t = 2048)]
    pub(crate) max_tokens: usize,

    /// `PYTHONPATH` (sandbox-relative) for the bash tool + test runs, e.g.
    /// `lib:test`. Optional.
    #[arg(long, value_name = "PATH")]
    pub(crate) pythonpath: Option<String>,

    /// Bash tool timeout (seconds).
    #[arg(long, default_value_t = 150)]
    pub(crate) bash_timeout_secs: u64,

    /// Test-run (scoring) timeout (seconds).
    #[arg(long, default_value_t = 300)]
    pub(crate) test_timeout_secs: u64,

    /// RFT rounds (rollout → execute → score → writeback-CE).
    #[arg(long, default_value_t = 3)]
    pub(crate) rounds: usize,

    /// Cap on CE writeback pairs trained per round (unset = all accepted).
    #[arg(long, value_name = "N")]
    pub(crate) writeback_cap: Option<usize>,

    /// CE writeback micro-batch size (amortizes host op-dispatch; bounds logit VRAM).
    #[arg(long, default_value_t = 4)]
    pub(crate) writeback_batch: usize,

    /// Sequence-window size for the masked single-trajectory CE writeback. Each
    /// window re-forwards the prefix up to its end and materializes only a
    /// `[window, vocab]` logits tile, bounding VRAM on long agentic trajectories.
    #[arg(long, default_value_t = 2048)]
    pub(crate) writeback_window: usize,

    /// KV slots for the rollout engine (>1 batches sampling/eval; VRAM-gated).
    #[arg(long, default_value_t = 2)]
    pub(crate) rollout_num_slots: usize,

    /// Disable train-infer FP8 weight sharing (训推一体). Sharing is ON by default
    /// (the autograd student's frozen FP8 base points zero-copy at the rollout
    /// engine's resident base; graceful no-op for non-FP8 single-GPU). Pass
    /// --no-share-frozen-base to force the byte-identical two-copy load.
    #[arg(long, default_value_t = false)]
    pub(crate) no_share_frozen_base: bool,

    /// Rollout sampling temperature (>0 for rejection-sampling diversity).
    #[arg(long, default_value_t = 1.0, value_parser = parse_temperature, allow_hyphen_values = true)]
    pub(crate) rollout_temperature: f32,

    /// Rollout nucleus sampling threshold (1.0 disables top-p).
    #[arg(long, default_value_t = 1.0)]
    pub(crate) rollout_top_p: f32,

    /// Rollout top-k filter (0 disables top-k).
    #[arg(long, default_value_t = 0)]
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
    /// backward before layer N (cuts the backward to the top suffix). Needed to
    /// fit a 27B dense student CE backward; unset = all layers.
    #[arg(long, value_name = "N")]
    pub(crate) lora_layer_start: Option<usize>,

    /// Gradient checkpointing for the CE backward (recompute activations). On by
    /// default; `--grad-checkpointing false` to disable.
    #[arg(long, default_value_t = true)]
    pub(crate) grad_checkpointing: bool,

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

    /// Compute backend for autograd (auto picks CUDA when built with cuda).
    #[arg(long, value_enum, default_value_t = OpdBackendArg::Auto)]
    pub(crate) backend: OpdBackendArg,

    /// Diagnostic: skip the rollout and run ONE masked-CE writeback on a synthetic trajectory of this token length (find writeback OOM fast). 0 = off.
    #[arg(long, default_value_t = 0)]
    pub(crate) synthetic_writeback_seq: usize,

    /// Skip LoRA on all routed MoE expert projections (gate/up/down per expert).
    /// Keeps LoRA only on attention projections (q/k/v/o) and shared experts.
    /// Drops LoRA params from ~6.1B to ~59M and Adam state from ~49 GB to ~0.5 GB,
    /// enabling multiple writebacks per round without --writeback-cap.
    #[arg(long, default_value_t = false)]
    pub(crate) lora_skip_experts: bool,

    /// Render output as JSON for scripts and CI.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[cfg(test)]
mod tests {
    use super::{
        Args, CliCommand, LrScheduleArg, OpdKlMaskArg, OpdSftAnchorArg, OpdTeacherRuntimeArg,
        RunArgs, TrainCommand,
    };
    use clap::{CommandFactory, Parser};

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

    #[test]
    fn rejects_empty_trace_path() {
        let err = Args::try_parse_from(["arle", "--trace", ""])
            .err()
            .expect("empty trace path should be rejected");
        assert!(err.to_string().contains("trace path must not be empty"));
    }
}
