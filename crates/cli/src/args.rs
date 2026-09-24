use std::path::PathBuf;

use clap::{ArgGroup, Args as ClapArgs, Parser, Subcommand, ValueEnum};

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

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum TracePromptsMode {
    On,
    Off,
}

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
    about = "ARLE: local inference server for coding agents (Anthropic + OpenAI APIs) and agent REPL",
    after_help = "Common flows:\n  arle                                       Start the interactive agent REPL.\n  arle run                                   Explicit alias for the interactive agent REPL.\n  arle run --prompt \"Summarize this repo\"    Run one prompt and exit.\n  arle run --stdin --json < prompt.txt       Read one prompt from stdin and emit JSON.\n  arle serve --model-path /path/to/model      Serve over the Anthropic and OpenAI APIs.\n  arle --doctor                              Inspect the local environment and model resolution.",
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

    /// Disable built-in shell/python tools for the local agent runtime.
    /// Also honored per-run via `arle run --no-tools`.
    #[arg(long, default_value_t = false)]
    pub(crate) no_tools: bool,

    /// Skip interactive model selection (use auto-discovery)
    #[arg(long, default_value_t = false)]
    pub(crate) non_interactive: bool,

    /// Path to a JSONL file that will receive one trajectory record per
    /// agent turn (v1 schema). When unset, no trajectory is
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
    /// Serve a model over the Anthropic Messages and OpenAI APIs, in-process on the chosen backend.
    Serve(Box<ServeArgs>),
    /// Model utilities (download from Hugging Face).
    Model(Box<ModelArgs>),
    /// Run one registered CUDA kernel against a CPU f32 reference.
    Kernel(Box<KernelArgs>),
}

/// `arle kernel <name> --shape M,N,K`: one kernel run alone, printing its
/// device time and max relative error against a CPU f32 reference.
#[derive(Debug, Clone, ClapArgs)]
pub(crate) struct KernelArgs {
    /// Registered kernel name (an unknown name lists the registry).
    pub(crate) name: String,

    /// Problem dims as M,N,K (e.g. 1,34816,5120).
    #[arg(long)]
    pub(crate) shape: String,

    /// Reference implementation. Only `cpu` exists today.
    #[arg(long = "ref", default_value = "cpu")]
    pub(crate) reference: String,

    /// Timed launches; the mean device time is reported.
    #[arg(long, default_value_t = 20)]
    pub(crate) iters: usize,
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
    after_help = "Output:\n  Plain text is written to stdout by default.\n  `--json` emits one machine-readable document with model, backend, usage, and tool-call stats.\n\nExamples:\n  arle --model-path /path/to/model run\n  arle --model-path /path/to/model run --prompt \"Summarize this repo\"\n  arle --model-path /path/to/model run --stdin --json < prompt.txt\n  arle --model-path /path/to/model run --no-tools --prompt \"No tool execution\""
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
    after_help = "This is a thin front door over the in-process ARLE-native backend compiled into this binary.\n\nExamples:\n  arle serve --model-path /path/to/Qwen3-4B\n  arle serve --backend arle --model-path /models/Qwen3-4B --port 8000\n  arle serve --backend metal --model-path mlx-community/Qwen3-0.6B-4bit --port 8010\n  arle serve --backend cuda --model-path /models/Qwen3-4B"
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

    /// Serve-wide sampling defaults for fields the request omits; both override
    /// the checkpoint's `generation_config.json`. A checkpoint that ships no
    /// sampling config serves greedy, which loops on long generations:
    /// LFM2.5-8B-A1B repeated a 10-gram 10x over 2500 tokens and never finished
    /// a count-to-200, and `--repetition-penalty 1.05` alone finished it in 807.
    #[arg(long, value_name = "F")]
    pub(crate) temperature: Option<f32>,

    /// See `--temperature`. `1.0` disables; 1.05-1.1 is the useful band.
    #[arg(long, value_name = "F")]
    pub(crate) repetition_penalty: Option<f32>,

    /// KV cache storage dtype. `auto` lets the backend choose its default.
    #[arg(long, value_enum, default_value_t = ServeKvCacheDtypeArg::Auto)]
    pub(crate) kv_cache_dtype: ServeKvCacheDtypeArg,

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

    /// Exit when this process disappears. A supervising app that is killed
    /// (SIGKILL, crash) never runs its own cleanup, so the engine — holding
    /// tens of GiB of weights — has to notice on its own.
    #[arg(long, value_name = "PID")]
    pub(crate) parent_pid: Option<i32>,

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

    /// Reasoning token budget for thinking requests. `0` (the default) uses
    /// the model default (32768 for DeepSeek-V4, unlimited otherwise). A
    /// positive value forces `</think>` after this many reasoning tokens,
    /// then the model continues with content.
    #[arg(long, default_value_t = 0)]
    pub(crate) max_thinking_tokens: usize,

    /// Per-request prefill chunk size.
    #[arg(long, value_parser = parse_positive_usize)]
    pub(crate) chunked_prefill_size: Option<usize>,

    /// Rotate waiters past the running cap by parking whole decode slots into
    /// the KV tier (whole-slot-tier models only; others fail closed at start).
    #[arg(long, default_value_t = false)]
    pub(crate) kv_oversubscription: bool,

    /// Minimum decode tokens before an oversubscribed request may be parked again.
    #[arg(long, value_parser = parse_positive_usize)]
    pub(crate) kv_oversubscription_min_slice: Option<usize>,

    /// Dump every raw `/v1/messages` request body to
    /// `<dir>/<epoch_ms>_<seq>.json` (CC-trajectory capture).
    /// Fire-and-forget; unset = zero cost.
    #[arg(long, value_name = "DIR")]
    pub(crate) dump_messages_dir: Option<PathBuf>,

    /// LoRA adapter safetensors (OPD `--save-lora-adapters` output) re-merged
    /// into the resident student weights once at startup. CUDA Qwen3.5/3.6 only.
    #[arg(long, value_name = "FILE")]
    pub(crate) lora_adapters: Option<PathBuf>,

    /// LoRA alpha for --lora-adapters (`scale = alpha / rank`; rank is read
    /// from the adapter tensor shapes).
    #[arg(long, default_value_t = 32.0)]
    pub(crate) lora_alpha: f32,

    /// Speculative decode route (CUDA): `auto` (default) speculates whenever the
    /// checkpoint declares an MTP head; `mtp` forces the checkpoint-native head;
    /// `dspark` is the external block drafter (dir via `--mtp-draft-model`).
    #[arg(long, value_enum, default_value_t = ServeSpecTypeArg::Auto)]
    pub(crate) spec_type: ServeSpecTypeArg,

    /// Multi-GPU collective backend for small decode-path messages (CUDA
    /// multi-rank only). `nccl` (default): plain NCCL — the 2026-06-10 matched
    /// A/B measured the one-shot path wall-neutral on single-node H20 (the
    /// decode wall is rank-skew-bound, not protocol-bound), and the 2026-08-17
    /// probe measured one-shot 51-53 tok/s vs NCCL 70-80 on Qwen3.6-27B. `auto`
    /// boots the one-shot custom AR/AG with a self-test and loud degrade.
    #[arg(long, value_enum, default_value_t = ServeCommBackendArg::Nccl)]
    pub(crate) comm_backend: ServeCommBackendArg,

    /// Tensor-parallel degree: attention heads sharded across N GPUs. World
    /// size = `tensor_parallel_size × context_parallel_size`. Default 1.
    #[arg(long, default_value_t = 1, value_parser = parse_positive_usize, value_name = "N")]
    pub(crate) tensor_parallel_size: usize,

    /// Context-parallel degree: the sequence dimension sharded across N GPUs
    /// (ring prefill + flash-decoding decode). World size =
    /// `tensor_parallel_size × context_parallel_size`. Default 1.
    #[arg(long, default_value_t = 1, value_parser = parse_positive_usize, value_name = "N")]
    pub(crate) context_parallel_size: usize,

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

    /// Metal speculative draft depth (NextN/MTP/DSpark tokens proposed per
    /// verify block). Unset = model-specific default (2 for Qwen3.6, 4 for
    /// DSpark on Metal).
    #[arg(long, value_parser = parse_speculative_tokens, value_name = "N")]
    pub(crate) speculative_tokens: Option<usize>,

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

    /// Routed-row floor for the DeepGEMM grouped expert path (default 1024).
    /// Batched decode is `R = top_k * B`; the FP8 mid-band (R=129..1023) was
    /// measured a wash on H20 (2026-08-23), so 1024 keeps decode on the hand
    /// kernels. Lower it only to re-probe on new hardware/kernel versions.
    #[arg(long, default_value_t = 1024)]
    pub(crate) qwen35_deepgemm_min_routes: usize,

    /// FlashQLA chunked GDN prefill (sm_90, per-geometry AOT instantiations;
    /// runtime-probes kernel availability and falls back to the recurrent
    /// scan). Known trade: raw-completion few-shot can flip knife-edge
    /// boundary tokens (chat-format quality is parity — see the 2026-08-02
    /// verdict entry).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) qwen35_gdr_chunked: bool,

    /// Retain the cuMemAllocAsync pool across syncs (caching allocator).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) cuda_mempool_retain: bool,

    /// Safetensor shard read-ahead cache budget in bytes (unset = built-in default).
    #[arg(long, value_name = "BYTES")]
    pub(crate) shard_cache_bytes: Option<usize>,

    /// Pin multi-rank CUDA workers to their GPU's NUMA node.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) numa_pin: bool,

    /// DSv4 DSA indexer SM budget.
    #[arg(long, default_value_t = 78, value_name = "N")]
    pub(crate) dsv4_dsa_indexer_sms: usize,

    /// Speculate (MTP/DSpark) only when the decode batch is ≤ this; above it
    /// decode routes to the plain batched path. Default 16 = the measured
    /// DSpark envelope (+77% at c=2, +8.5% at c=16). On DSv4 the gate is
    /// pinned to 1 (the draft runs per slot and speculation loses above c=1);
    /// on qwen35 it governs the batched greedy DSpark (BF16 paged KV) and MTP
    /// paths.
    #[arg(long, default_value_t = 16, value_name = "N")]
    pub(crate) spec_max_batch: usize,

    /// DeepEP intranode SM budget (positive, even).
    #[arg(long, default_value_t = 20, value_name = "N")]
    pub(crate) deepep_num_sms: u32,

    /// DeepEP LL per-rank dispatch-token cap (unset = SGLANG env or 256).
    #[arg(long, value_name = "N")]
    pub(crate) deepep_max_dispatch_tokens_per_rank: Option<u32>,

    /// DSv4 MoE transport: `allreduce` (default), `deepep`, `deepep_ll`,
    /// `mega_moe`. Unset = `ARLE_DSV4_MOE_TRANSPORT` env or allreduce. The
    /// `--deepep-*` flags take effect only under a deepep transport.
    #[arg(long, value_name = "NAME")]
    pub(crate) dsv4_moe_transport: Option<String>,

    /// Load-time JIT warmup forward. Off by default — the first request pays
    /// the JIT + embed dequant cost, which is faster overall for cold-start
    /// scenarios. Serving deployments should opt in.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub(crate) metal_warmup: bool,

    /// Cap block-diffusion denoising steps per row (unset = checkpoint default).
    #[arg(long, value_name = "N")]
    pub(crate) diffusion_max_denoising_steps: Option<usize>,
}

impl ServeArgs {
    /// CUDA runtime toggles for `EngineLoadConfig.cuda`.
    pub(crate) fn cuda_runtime_flags(&self) -> infer_api::CudaRuntimeFlags {
        infer_api::CudaRuntimeFlags {
            qwen35_decode_graph: true,
            qwen35_deepgemm_min_routes: self.qwen35_deepgemm_min_routes,
            qwen35_gdr_chunked: self.qwen35_gdr_chunked,
            mempool_retain: self.cuda_mempool_retain,
            shard_cache_bytes: self.shard_cache_bytes,
            numa_pin: self.numa_pin,
            comm_backend: match self.comm_backend {
                ServeCommBackendArg::Auto => infer_api::CommBackend::Auto,
                ServeCommBackendArg::Nccl => infer_api::CommBackend::Nccl,
            },
            dsv4_dsa_indexer_sms: self.dsv4_dsa_indexer_sms,
            spec_max_batch: self.spec_max_batch,
            deepep_num_sms: self.deepep_num_sms,
            deepep_max_dispatch_tokens_per_rank: self.deepep_max_dispatch_tokens_per_rank,
            dsv4_moe_transport: self.dsv4_moe_transport.clone(),
        }
    }

    /// Metal runtime toggles for `EngineLoadConfig.metal`. The speculative
    /// fields ride here (not env) so the executor's resolver sees the flags.
    pub(crate) fn metal_runtime_flags(&self) -> infer_api::MetalRuntimeFlags {
        infer_api::MetalRuntimeFlags {
            warmup: self.metal_warmup,
            speculative: !self.no_speculative,
            draft_model: self.draft_model.clone(),
            speculative_tokens: self.speculative_tokens,
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
