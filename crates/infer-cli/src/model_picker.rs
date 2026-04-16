//! Interactive model selection UI.
//!
//! Uses `dialoguer::Select` for the main picker and `dialoguer::Input` +
//! nucleo fuzzy filtering for HuggingFace search.

use std::path::PathBuf;

use anyhow::Result;
use console::style;
use dialoguer::{Confirm, Input, Select};

use crate::hf_search;
use crate::model_catalog::CatalogEntry;

/// Result of the model picker interaction.
pub(crate) enum PickerResult {
    /// User selected a locally available model.
    LocalModel(String),
    /// User selected a remote model to download.
    RemoteModel(String),
}

/// Display the interactive model picker.
///
/// Shows local models first, then recommended downloads, then a search option.
pub(crate) fn pick_model(
    local_models: &[(String, PathBuf)],
    recommended: &[&CatalogEntry],
    _all_for_backend: &[&CatalogEntry],
) -> Result<PickerResult> {
    let mut items: Vec<String> = Vec::new();
    let mut actions: Vec<PickerAction> = Vec::new();

    // ── Local models ─────────────────────────────────────────────────────
    if !local_models.is_empty() {
        for (name, path) in local_models {
            let display_path = abbreviate_path(path);
            items.push(format!(
                "{}  {}  {}",
                style("local").green(),
                style(name).bold(),
                style(display_path).dim()
            ));
            actions.push(PickerAction::Local(name.clone()));
        }
    }

    // ── Recommended downloads ────────────────────────────────────────────
    if !recommended.is_empty() {
        // Separator
        if !local_models.is_empty() {
            items.push(format!("{}", style("── download ──").dim()));
            actions.push(PickerAction::Separator);
        }

        for entry in recommended {
            let quant = entry
                .quantization
                .map(|q| format!(" {}", style(q).yellow()))
                .unwrap_or_default();
            items.push(format!(
                "{}  {}{:<4}  {}",
                style("fetch").cyan(),
                style(entry.display_name).bold(),
                quant,
                style(format!("{:.1} GB", entry.size_gb)).dim()
            ));
            actions.push(PickerAction::Remote(entry.hf_id.to_string()));
        }
    }

    // ── Search option ────────────────────────────────────────────────────
    items.push(format!(
        "{}  {}",
        style("  >> ").dim(),
        style("Search HuggingFace...").italic()
    ));
    actions.push(PickerAction::Search);

    // ── Show picker ──────────────────────────────────────────────────────
    loop {
        let selection = Select::new()
            .with_prompt(format!("{}", style("select model").bold()))
            .items(&items)
            .default(0)
            .interact()?;

        match &actions[selection] {
            PickerAction::Local(name) => return Ok(PickerResult::LocalModel(name.clone())),
            PickerAction::Remote(hf_id) => {
                let entry = recommended
                    .iter()
                    .find(|e| e.hf_id == hf_id)
                    .expect("action points to catalog entry");
                if confirm_download(entry)? {
                    return Ok(PickerResult::RemoteModel(hf_id.clone()));
                }
                // User declined, show picker again
                continue;
            }
            PickerAction::Search => {
                if let Some(model_id) = run_search_flow()? {
                    return Ok(PickerResult::RemoteModel(model_id));
                }
                // User cancelled search, show picker again
                continue;
            }
            PickerAction::Separator => {
                // User selected the separator line, ignore
                continue;
            }
        }
    }
}

/// Interactive HuggingFace search flow.
fn run_search_flow() -> Result<Option<String>> {
    let query: String = Input::new()
        .with_prompt(format!("{}", style("search query").bold()))
        .interact_text()?;

    if query.trim().is_empty() {
        return Ok(None);
    }

    eprintln!("  {} ...", style("searching").dim());

    match hf_search::search_hf_models(query.trim()) {
        Ok(results) if results.is_empty() => {
            eprintln!("  {}", style("no results found").yellow());
            Ok(None)
        }
        Ok(results) => {
            let display_items: Vec<String> = results.iter().map(|r| r.display_line()).collect();

            let selection = Select::new()
                .with_prompt(format!("{}", style("pick model").bold()))
                .items(&display_items)
                .default(0)
                .interact_opt()?;

            match selection {
                Some(idx) => {
                    let model_id = &results[idx].model_id;
                    let confirmed = Confirm::new()
                        .with_prompt(format!("Download {}?", style(model_id).bold()))
                        .default(true)
                        .interact()?;

                    if confirmed {
                        Ok(Some(model_id.clone()))
                    } else {
                        Ok(None)
                    }
                }
                None => Ok(None),
            }
        }
        Err(e) => {
            eprintln!(
                "  {} {}",
                style("search failed:").yellow(),
                style(format!("{e:#}")).dim()
            );
            Ok(None)
        }
    }
}

fn confirm_download(entry: &CatalogEntry) -> Result<bool> {
    let confirmed = Confirm::new()
        .with_prompt(format!(
            "Download {}? ({:.1} GB)",
            style(entry.hf_id).bold(),
            entry.size_gb
        ))
        .default(true)
        .interact()?;
    Ok(confirmed)
}

fn abbreviate_path(path: &PathBuf) -> String {
    let s = path.display().to_string();
    if let Some(home) = std::env::var_os("HOME") {
        let home_str = home.to_string_lossy();
        if let Some(rest) = s.strip_prefix(home_str.as_ref()) {
            return format!("~{rest}");
        }
    }
    s
}

#[derive(Clone)]
enum PickerAction {
    Local(String),
    Remote(String),
    Search,
    Separator,
}
