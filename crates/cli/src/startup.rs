//! Interactive startup flow orchestrator.
//!
//! Ties together hardware detection, model catalog, banner, picker, and
//! download into a single cohesive startup experience.

use std::io::IsTerminal;

use anyhow::Result;
use console::style;

use crate::args::Args;
use crate::banner;
use crate::download;
use crate::hardware;
use crate::hub_discovery;
use crate::model_catalog;
use crate::model_picker::{self, PickerResult};

/// Run the interactive startup flow and return the resolved model source.
///
/// Falls back to `infer::hf_hub::resolve_model_source` when:
/// - `--model-path` is provided
/// - `--non-interactive` flag is set
/// - stdin/stdout is not a TTY (piped)
pub(crate) fn resolve_model_interactive(args: &Args) -> Result<String> {
    // Fast path: explicit model path bypasses everything.
    if let Some(ref model_path) = args.model_path {
        if !model_path.trim().is_empty() {
            return Ok(model_path.clone());
        }
    }

    // Non-interactive: fall back to existing auto-discovery.
    if args.non_interactive || !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        return infer::hf_hub::resolve_model_source(args.model_path.as_deref());
    }

    // ── Interactive startup ──────────────────────────────────────────────
    let info = hardware::detect_system();
    banner::print_startup_banner(&info);

    // Discover locally available models.
    let local_models = discover_all_local_models();

    // Get catalog recommendations.
    let recommended = model_catalog::recommend_models(&info);
    let all_backend = model_catalog::all_for_backend(&info);

    // If we found exactly one local model and nothing else makes sense,
    // just confirm and go.
    if local_models.len() == 1 && recommended.is_empty() {
        let (name, _path) = &local_models[0];
        eprintln!(
            "  {} {}",
            style("auto-selected").green(),
            style(name).bold()
        );
        eprintln!();
        return Ok(name.clone());
    }

    // Show the interactive picker.
    match model_picker::pick_model(&local_models, &recommended, &all_backend)? {
        PickerResult::LocalModel(name) => Ok(name),
        PickerResult::RemoteModel(hf_id) => {
            download::download_model_with_progress(&hf_id)?;
            Ok(hf_id)
        }
    }
}

/// Fallback wizard: scan the HF hub cache for supported-family snapshots and
/// show a `dialoguer::Select`. Called from `lib::run` when the main resolve
/// path returned nothing (no curated candidate matched).
///
/// Returns `Ok(Some(path))` on a user selection, `Ok(None)` on Esc / empty
/// cache, and propagates IO errors only from the picker interaction.
pub(crate) fn run_hub_wizard() -> Result<Option<String>> {
    use dialoguer::Select;

    let snapshots = hub_discovery::discover_hub_snapshots();
    if snapshots.is_empty() {
        return Ok(None);
    }

    let items: Vec<String> = snapshots
        .iter()
        .map(|s| {
            format!(
                "{}  {}",
                style(&s.model_id).bold(),
                style(s.path.display().to_string()).dim()
            )
        })
        .collect();

    let selection = Select::new()
        .with_prompt("Select a model (or press Esc to cancel):")
        .items(&items)
        .default(0)
        .interact_opt()?;

    match selection {
        Some(idx) => Ok(Some(snapshots[idx].path.display().to_string())),
        None => Ok(None),
    }
}

/// Discover all locally available models from the default candidate list.
fn discover_all_local_models() -> Vec<(String, std::path::PathBuf)> {
    let candidates = [
        "mlx-community/Qwen3-0.6B-4bit",
        "mlx-community/Qwen3-0.6B-bf16",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3.5-4B",
        "mlx-community/Qwen3-4B-4bit",
        "mlx-community/Qwen3-8B-4bit",
        "THUDM/glm-4-9b-chat",
    ];

    candidates
        .iter()
        .filter_map(|candidate| {
            infer::hf_hub::resolve_local_model_path(candidate)
                .map(|path| ((*candidate).to_string(), path))
        })
        .collect()
}
