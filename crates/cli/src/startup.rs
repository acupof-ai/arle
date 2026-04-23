//! Interactive startup flow orchestrator.
//!
//! Ties together hardware detection, model catalog, banner, picker, and
//! download into a single cohesive startup experience.

use std::collections::HashSet;
use std::io::IsTerminal;
use std::path::PathBuf;

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
    let local_snapshots = discover_local_snapshots();
    let local_models = local_models_from_snapshots(&local_snapshots);

    // Get catalog recommendations.
    let recommended = model_catalog::recommend_models(&info);

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
    match model_picker::pick_model(&local_models, &recommended)? {
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

    let snapshots = discover_local_snapshots();
    if snapshots.is_empty() {
        return Ok(None);
    }

    let items: Vec<String> = snapshots
        .iter()
        .map(|s| model_picker::name_path_item(&s.model_id, &s.path))
        .collect();

    let selection = Select::new()
        .with_prompt("Select a model (or press Esc to cancel):")
        .items(&items)
        .max_length(model_picker::picker_page_len(items.len()))
        .default(0)
        .interact_opt()?;

    match selection {
        Some(idx) => Ok(Some(snapshots[idx].path.display().to_string())),
        None => Ok(None),
    }
}

fn discover_local_snapshots() -> Vec<hub_discovery::HubSnapshot> {
    dedupe_snapshots_by_model_id(hub_discovery::discover_hub_snapshots())
}

fn local_models_from_snapshots(snapshots: &[hub_discovery::HubSnapshot]) -> Vec<(String, PathBuf)> {
    snapshots
        .iter()
        .map(|snapshot| (snapshot.model_id.clone(), snapshot.path.clone()))
        .collect()
}

fn dedupe_snapshots_by_model_id(
    snapshots: Vec<hub_discovery::HubSnapshot>,
) -> Vec<hub_discovery::HubSnapshot> {
    let mut seen = HashSet::new();
    snapshots
        .into_iter()
        .filter(|snapshot| seen.insert(snapshot.model_id.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{dedupe_snapshots_by_model_id, local_models_from_snapshots};
    use crate::hub_discovery::HubSnapshot;
    use std::path::PathBuf;

    #[test]
    fn local_models_preserve_snapshot_identity() {
        let snapshots = vec![
            HubSnapshot {
                model_id: "mlx-community/Qwen3.6-35B-A3B-4bit".to_string(),
                path: PathBuf::from("/tmp/qwen35b"),
            },
            HubSnapshot {
                model_id: "mlx-community/Qwen3.5-4B-MLX-4bit".to_string(),
                path: PathBuf::from("/tmp/qwen35-4b"),
            },
        ];

        let models = local_models_from_snapshots(&snapshots);
        assert_eq!(
            models,
            vec![
                (
                    "mlx-community/Qwen3.6-35B-A3B-4bit".to_string(),
                    PathBuf::from("/tmp/qwen35b"),
                ),
                (
                    "mlx-community/Qwen3.5-4B-MLX-4bit".to_string(),
                    PathBuf::from("/tmp/qwen35-4b"),
                ),
            ]
        );
    }

    #[test]
    fn dedupe_snapshots_by_model_id_keeps_newest_snapshot() {
        let snapshots = vec![
            HubSnapshot {
                model_id: "Qwen/Qwen3-4B".to_string(),
                path: PathBuf::from("/tmp/newer"),
            },
            HubSnapshot {
                model_id: "mlx-community/Qwen3.6-35B-A3B-4bit".to_string(),
                path: PathBuf::from("/tmp/moe"),
            },
            HubSnapshot {
                model_id: "Qwen/Qwen3-4B".to_string(),
                path: PathBuf::from("/tmp/older"),
            },
        ];

        assert_eq!(
            dedupe_snapshots_by_model_id(snapshots),
            vec![
                HubSnapshot {
                    model_id: "Qwen/Qwen3-4B".to_string(),
                    path: PathBuf::from("/tmp/newer"),
                },
                HubSnapshot {
                    model_id: "mlx-community/Qwen3.6-35B-A3B-4bit".to_string(),
                    path: PathBuf::from("/tmp/moe"),
                },
            ]
        );
    }
}
