---
title: Task #34 rescope — HF Hub model download substrate FULLY EXISTS, only CLI subcommand surface is missing (~30-50 LOC)
date: 2026-05-10
type: research
status: rescope-pending-codex-pickup-post-phase1
---

# Task #34 rescope — HF Hub model download substrate FULLY EXISTS, only CLI subcommand surface is missing (~30-50 LOC)

> Per `61c9666` revised priority post Phase 0 P0.A KILL: P0 next-axis
> = #28 spec decoding (-50%+ ITL via amortized weight read), blocked
> on #34 "arle data download HF Hub library blocker". P0 survey
> this tick reveals #34 is **structurally complete** for substrate;
> only the `arle model download` CLI subcommand is missing. ~30-50
> LOC unblocks the entire P0 ITL win path.

## §0 Direct evidence (raw `grep` + `wc -l` this tick, NOT memory recall per skill v1.10.0 #28)

### HF Hub dependency present

```bash
$ grep -nE "hf-hub|huggingface|hf_hub" Cargo.toml crates/*/Cargo.toml
crates/train/Cargo.toml:25:hf-hub = { version = "0.5", features = ["tokio"] }
crates/cli/Cargo.toml:26:hf-hub = { version = "0.5", features = ["tokio"] }
```

Two crates have hf-hub 0.5 wired. NOT a dep blocker.

### Model download function ALREADY EXISTS

```bash
$ wc -l crates/cli/src/download.rs
160 crates/cli/src/download.rs

$ grep -nE "^pub(\(crate\))? fn|model" crates/cli/src/download.rs
14: /// Download a model from HuggingFace Hub with per-file progress bars.
17: pub(crate) fn download_model_with_progress(model_id: &str) -> Result<PathBuf> {
```

**`download_model_with_progress(model_id: &str) -> Result<PathBuf>`**
exists at `crates/cli/src/download.rs:17` with full features:
- Sharded weights handling (model.safetensors.index.json detection)
- Mandatory file list (config.json, tokenizer.json, tokenizer_config.json)
- Progress bars via indicatif
- hf-hub sync API integration
- Error reporting with model-id context

This is production-quality code, NOT a stub.

### But ONLY callable via interactive startup wizard

```bash
$ grep -rn "download_model_with_progress" crates/ src/
crates/cli/src/startup.rs:68: download::download_model_with_progress(&hf_id)?;
crates/cli/src/download.rs:17: pub(crate) fn download_model_with_progress(model_id: &str) -> Result<PathBuf> {
```

Only one call site: `crates/cli/src/startup.rs:68` (interactive
hardware detection + model picker wizard, runs when `arle` launched
without subcommand per lib.rs:95 comment).

### `arle data download` is dataset-only

```bash
$ grep -nE "DataDownload|run_data_download" crates/cli/src/{args,train_cli}.rs
crates/cli/src/args.rs:389: Download(DataDownloadArgs),
crates/cli/src/args.rs:1145: pub(crate) struct DataDownloadArgs {
crates/cli/src/train_cli.rs:55: pub(crate) fn run_data(data: DataArgs) -> ExitCode {
crates/cli/src/train_cli.rs:57: DataCommand::Download(args) => run_data_download(args),
crates/cli/src/train_cli.rs:348: fn run_data_download(args: DataDownloadArgs) -> ExitCode {
```

`DataDownloadArgs` (args.rs:1145) takes `--repo` + `--file` for **dataset
file download**. Dispatches to `train::commands::download_dataset` (NOT
the model download path).

### So what's missing?

A `ModelDownload` subcommand that wraps `download_model_with_progress`,
mirroring the `DataDownload` pattern. Estimated ~30-50 LOC:

- `args.rs`: new `ModelCommand::Download(ModelDownloadArgs)` enum variant
- `args.rs`: new `ModelDownloadArgs { model_id: String }` struct
- `train_cli.rs` (or new `model_cli.rs`): `run_model_download()` dispatch
- `lib.rs`: route `CliCommand::Model(...)` to dispatch

Wall time: ~1h codex.

## §1 Why this matters — unblocks P0 ITL win path

Per `61c9666` revised priority post Phase 0 P0.A KILL:

| Priority | Path | LOC | Wall | Predicted gain |
|----------|------|-----|------|----------------|
| **P0** | Spec decoding (#28, blocked on #34) | 500 | 1 wk + training | **-50%+ ITL** via amortized weight read |
| **P1** | W3/W2 quantization research | TBD | 1 wk research | -25-50% ITL ceiling |
| **P2** | Phase 1 dequant.h port (in flight #42) | 687 | 1.5-2 days | -3-8% ITL |
| **P3** | Prefill-only FP8 (new) | 700 | 2 days | -8-16% TTFT |

P0 is the only path with `-50%+` predicted gain on sm_89 W4 decode
(per `61c9666` architectural analysis: amortizes the HBM-bound weight
read across speculative tokens, so binding constraint shifts from
weight bandwidth to throughput).

#28 brief description (from task list): "M_medusa scaffold (~500 LOC
+ 1 week training, blocked on #34)". The "blocked on #34" was the
download infra — but #34 substrate EXISTS, so the actual blocker is
just the CLI subcommand.

## §2 Pickup recipe (for codex, post Phase 1)

```rust
// crates/cli/src/args.rs

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum ModelCommand {
    /// Download a model from Hugging Face Hub.
    Download(ModelDownloadArgs),
}

#[derive(Debug, Clone, ClapArgs)]
#[command(after_help = "Example:\n  arle model download Qwen/Qwen3-0.6B")]
pub(crate) struct ModelDownloadArgs {
    /// HuggingFace model ID (e.g. "Qwen/Qwen3-0.6B")
    pub(crate) model_id: String,

    #[command(flatten)]
    pub(crate) render: RenderArgs,
}

// Wire ModelCommand into CliCommand enum

// crates/cli/src/train_cli.rs (or new crates/cli/src/model_cli.rs)

pub(crate) fn run_model(model: ModelArgs) -> ExitCode {
    match model.command {
        ModelCommand::Download(args) => run_model_download(args),
    }
}

fn run_model_download(args: ModelDownloadArgs) -> ExitCode {
    // Reuse existing download::download_model_with_progress
    match crate::download::download_model_with_progress(&args.model_id) {
        Ok(path) => {
            eprintln!("Downloaded to: {}", path.display());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("Download failed: {err:#}");
            ExitCode::FAILURE
        }
    }
}

// crates/cli/src/lib.rs

Some(CliCommand::Model(args)) => return train_cli::run_model(*args),
```

LOC delta: ~30-50. Wall time: ~1h codex (likely faster).

## §3 Decision

If codex's Phase 1 dequant.h port lands cleanly (greedy_consistency
PASS + bench LICENSE), this #34 unblock is the natural next pickup —
small scope, high leverage, opens P0 #28 spec decoding work.

If Phase 1 KILLs: this #34 work stands as P3' fallback (codex
keeps momentum on a small-scope ship while Claude figures out the
ITL pivot).

## §4 Cross-references

- Phase 0 KILL architectural synthesis: `docs/research/2026-05-10-phase0a-decode-kill-architectural-implication.md` (61c9666)
- Phase 1 in flight: `docs/research/2026-05-10-phase1-substep1.1-codex-impl-audit-clean.md` (70b4d7b)
- Phase 1 wins skeleton: `docs/experience/wins/SKELETON-2026-05-10-path-b-phase1-substep1.1-dequant-port.md` (48c6e49)
- Existing model download substrate: `crates/cli/src/download.rs:17` (160 LOC)
- Existing call site: `crates/cli/src/startup.rs:68`
- Existing `arle data download` (dataset, not model): `crates/cli/src/args.rs:1145`, `train_cli.rs:348`
- hf-hub deps: `crates/cli/Cargo.toml:26`, `crates/train/Cargo.toml:25`
- Skill v1.10.0 anti-pattern #28 (verify raw output not memory recall): `.claude/skills/kernel-optimization/SKILL.md`

## §5 Status

#34 rescope COMPLETE: substrate fully exists, only CLI subcommand
surface is missing (~30-50 LOC, ~1h codex). Pickup deferred to
post-Phase 1 — keeps codex on the active Phase 1 tranche without
context-switching mid-bench. Once landed, unblocks #28 spec decoding
(P0 -50%+ ITL win path). Per skill v1.10.0 #28: all claims this
tick verified by raw `grep` / `wc -l` output, NOT memory recall.
