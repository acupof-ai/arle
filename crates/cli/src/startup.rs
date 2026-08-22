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

pub(crate) fn resolve_model_interactive(args: &Args) -> Result<String> {
    if let Some(ref model_path) = args.model_path
        && !model_path.trim().is_empty()
    {
        return Ok(model_path.clone());
    }

    if args.non_interactive || !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        return infer_util::hf_hub::resolve_model_source(args.model_path.as_deref());
    }

    let info = hardware::detect_system();
    banner::print_startup_banner(&info);

    let local_snapshots = discover_local_snapshots();
    let local_models = local_models_from_snapshots(&local_snapshots);

    let recommended = model_catalog::recommend_models(&info);

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

    let default_index = model_picker::default_picker_index(&local_models, &recommended);
    match model_picker::pick_model(&local_models, &recommended, default_index)? {
        PickerResult::LocalModel(name) => Ok(name),
        PickerResult::RemoteModel(hf_id) => {
            download::download_model_with_progress(&hf_id)?;
            Ok(hf_id)
        }
    }
}

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
