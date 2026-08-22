//! `dialoguer::Select` assumes one terminal row per item; long model paths
//! wrap and desync cursor math, so every item is forced onto one truncated
//! line and the picker paginates to fit the terminal.

use std::path::{Path, PathBuf};

use anyhow::Result;
use console::{Term, style, truncate_str};
use dialoguer::{Confirm, Input, Select};

use crate::hf_search;
use crate::model_catalog::CatalogEntry;

pub(crate) enum PickerResult {
    LocalModel(String),
    RemoteModel(String),
}

const PICKER_HEIGHT_MARGIN: usize = 6;
const PICKER_MAX_VISIBLE_ROWS: usize = 12;
const PICKER_ITEM_MARGIN: usize = 6;
const PICKER_MIN_WIDTH: usize = 24;

/// Item layout: `[local…, "── download ──"?, recommended…, search]`;
/// `default_index` pre-selects the best option so Enter lands on a great model.
pub(crate) fn pick_model(
    local_models: &[(String, PathBuf)],
    recommended: &[&CatalogEntry],
    default_index: usize,
) -> Result<PickerResult> {
    let mut items: Vec<String> = Vec::new();
    let mut actions: Vec<PickerAction> = Vec::new();

    if !local_models.is_empty() {
        for (name, path) in local_models {
            items.push(local_model_item(name, path));
            actions.push(PickerAction::Local(name.clone()));
        }
    }

    if !recommended.is_empty() {
        if !local_models.is_empty() {
            items.push(separator_item("── download ──"));
            actions.push(PickerAction::Separator);
        }

        for entry in recommended {
            items.push(remote_download_item(entry));
            actions.push(PickerAction::Remote(entry.hf_id.to_string()));
        }
    }

    items.push(search_item());
    actions.push(PickerAction::Search);

    // Clamp the default into range and never land it on the separator (which
    // would loop).
    let default_index = sanitize_default_index(default_index, &actions);
    loop {
        let selection = Select::new()
            .with_prompt(format!("{}", style("select model").bold()))
            .items(&items)
            .max_length(picker_page_len(items.len()))
            .default(default_index)
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
                continue;
            }
            PickerAction::Search => {
                if let Some(model_id) = run_search_flow()? {
                    return Ok(PickerResult::RemoteModel(model_id));
                }
                continue;
            }
            PickerAction::Separator => {
                continue;
            }
        }
    }
}

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
            let display_items: Vec<String> = results
                .iter()
                .map(|r| fit_picker_item(r.display_line()))
                .collect();

            let selection = Select::new()
                .with_prompt(format!("{}", style("pick model").bold()))
                .items(&display_items)
                .max_length(picker_page_len(display_items.len()))
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

/// Computed against `pick_model`'s item layout:
/// `[local…, "── download ──"?, recommended…, search]`.
pub(crate) fn default_picker_index(
    local_models: &[(String, PathBuf)],
    recommended: &[&CatalogEntry],
) -> usize {
    if let Some(idx) = local_models
        .iter()
        .position(|(name, _)| crate::model_catalog::find_by_hf_id(name).is_some_and(is_flagship))
    {
        return idx;
    }

    if !recommended.is_empty() {
        let separator = usize::from(!local_models.is_empty());
        return local_models.len() + separator;
    }

    0
}

fn is_flagship(entry: &CatalogEntry) -> bool {
    entry.recommended.is_some()
}

fn sanitize_default_index(index: usize, actions: &[PickerAction]) -> usize {
    if actions.is_empty() {
        return 0;
    }
    let index = index.min(actions.len() - 1);
    if matches!(actions[index], PickerAction::Separator) {
        return (index + 1).min(actions.len() - 1);
    }
    index
}

pub(crate) fn picker_page_len(item_count: usize) -> usize {
    picker_page_len_for_rows(Term::stderr().size().0 as usize, item_count)
}

pub(crate) fn fit_picker_item(text: impl Into<String>) -> String {
    fit_picker_item_for_width(text.into(), Term::stderr().size().1 as usize)
}

pub(crate) fn name_path_item(name: &str, path: &Path) -> String {
    format_name_path_item(None, name, path)
}

fn picker_page_len_for_rows(rows: usize, item_count: usize) -> usize {
    rows.saturating_sub(PICKER_HEIGHT_MARGIN)
        .clamp(1, PICKER_MAX_VISIBLE_ROWS)
        .min(item_count.max(1))
}

fn fit_picker_item_for_width(text: String, columns: usize) -> String {
    let max_width = columns
        .saturating_sub(PICKER_ITEM_MARGIN)
        .max(PICKER_MIN_WIDTH);
    truncate_str(&text, max_width, "...").into_owned()
}

fn local_model_item(name: &str, path: &Path) -> String {
    let label = if crate::model_catalog::find_by_hf_id(name).is_some_and(is_flagship) {
        // A flagship the user already has — same green "local" tag, plus a
        // star so the best pick reads at a glance.
        format!("{} {}", style("local").green(), style("★").yellow())
    } else {
        format!("{}", style("local").green())
    };
    format_name_path_item(Some(label), name, path)
}

fn format_name_path_item(label: Option<String>, name: &str, path: &Path) -> String {
    let prefix = label.map(|label| format!("{label}  ")).unwrap_or_default();
    fit_picker_item(format!(
        "{prefix}{}  {}",
        style(name).bold(),
        style(abbreviate_path(path)).dim()
    ))
}

fn remote_download_item(entry: &CatalogEntry) -> String {
    let quant = entry
        .quantization
        .map(|q| format!(" {}", style(q).yellow()))
        .unwrap_or_default();

    // Size is always shown (load-bearing before a download). Flagship picks
    // append a ★ and a one-line "why" — truncation discipline keeps it to a
    // single line on narrow terminals.
    let note = match entry.recommended {
        Some(reason) => format!(
            "{}  {} {}",
            style(format!("{:.1} GB", entry.size_gb)).dim(),
            style("★").yellow(),
            style(reason).cyan(),
        ),
        None => style(format!("{:.1} GB", entry.size_gb)).dim().to_string(),
    };

    fit_picker_item(format!(
        "{}  {}{:<4}  {}",
        style("fetch").cyan(),
        style(entry.display_name).bold(),
        quant,
        note,
    ))
}

fn separator_item(label: &str) -> String {
    fit_picker_item(format!("{}", style(label).dim()))
}

fn search_item() -> String {
    fit_picker_item(format!(
        "{}  {}",
        style("  >> ").dim(),
        style("Search HuggingFace...").italic()
    ))
}

fn abbreviate_path(path: &Path) -> String {
    let display = replace_home_prefix(path.display().to_string());
    let separator = std::path::MAIN_SEPARATOR;
    let root = if display.starts_with(separator) {
        separator.to_string()
    } else {
        String::new()
    };
    let components: Vec<&str> = display
        .split(separator)
        .filter(|part| !part.is_empty())
        .collect();

    if components.len() <= 5 {
        return display;
    }

    let head = components[..2].join(&separator.to_string());
    let tail = components[components.len() - 3..].join(&separator.to_string());
    format!("{root}{head}{separator}...{separator}{tail}")
}

fn replace_home_prefix(path: String) -> String {
    if let Some(home) = std::env::var_os("HOME") {
        let home_str = home.to_string_lossy();
        if let Some(rest) = path.strip_prefix(home_str.as_ref()) {
            return format!("~{rest}");
        }
    }
    path
}

#[derive(Clone)]
enum PickerAction {
    Local(String),
    Remote(String),
    Search,
    Separator,
}
