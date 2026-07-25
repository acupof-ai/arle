//! Interactive model selection UI.
//!
//! Uses `dialoguer::Select` for the main picker and `dialoguer::Input` +
//! nucleo fuzzy filtering for HuggingFace search.
//!
//! `dialoguer::Select` assumes each item occupies one terminal row. Long model
//! paths can wrap and desync cursor math, so every displayed item is forced
//! onto a single truncated line and the picker always paginates to fit the
//! current terminal.

use std::path::{Path, PathBuf};

use anyhow::Result;
use console::{Term, style, truncate_str};
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

const PICKER_HEIGHT_MARGIN: usize = 6;
const PICKER_MAX_VISIBLE_ROWS: usize = 12;
const PICKER_ITEM_MARGIN: usize = 6;
const PICKER_MIN_WIDTH: usize = 24;

/// Display the interactive model picker.
///
/// Shows local models first, then recommended downloads, then a search option.
/// `default_index` pre-selects the best available option so that pressing
/// Enter lands on a great model (see [`default_picker_index`]).
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
    // would loop). Callers pass the best available option; this is the seat
    // belt for an out-of-range or unlucky index.
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

/// Index the picker should pre-select so that pressing Enter lands on a great
/// model. Prefers a locally-available flagship (no download), else the first
/// recommended download. The index is computed against the same item layout
/// `pick_model` builds: `[local…, "── download ──"?, recommended…, search]`.
pub(crate) fn default_picker_index(
    local_models: &[(String, PathBuf)],
    recommended: &[&CatalogEntry],
) -> usize {
    // 1. A flagship the user already has on disk — zero-wait, best pick.
    if let Some(idx) = local_models
        .iter()
        .position(|(name, _)| crate::model_catalog::find_by_hf_id(name).is_some_and(is_flagship))
    {
        return idx;
    }

    // 2. Otherwise the first recommended download (the catalog already orders
    //    flagships first). It sits past the local block and the separator.
    if !recommended.is_empty() {
        let separator = usize::from(!local_models.is_empty());
        return local_models.len() + separator;
    }

    // 3. No recommendations: fall back to the first item (first local model).
    0
}

fn is_flagship(entry: &CatalogEntry) -> bool {
    entry.recommended.is_some()
}

/// Keep a default index in range and off the separator row (selecting it would
/// just loop the picker). On a separator hit, nudge to the next selectable row.
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

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{
        PickerAction, abbreviate_path, default_picker_index, fit_picker_item_for_width,
        picker_page_len_for_rows, sanitize_default_index,
    };
    use crate::model_catalog::{self, CatalogEntry};
    use console::measure_text_width;

    fn recs(ids: &[&str]) -> Vec<&'static CatalogEntry> {
        ids.iter()
            .map(|id| model_catalog::find_by_hf_id(id).expect("catalog id"))
            .collect()
    }

    fn locals(ids: &[&str]) -> Vec<(String, PathBuf)> {
        ids.iter()
            .map(|id| (id.to_string(), PathBuf::from("/tmp").join(id)))
            .collect()
    }

    #[test]
    fn default_index_prefers_local_flagship() {
        // A flagship already on disk wins outright — no download needed.
        let local = locals(&["some/random-model", "mlx-community/Qwen3.6-27B-OptiQ-4bit"]);
        let recommended = recs(&["mlx-community/Qwen3.6-35B-A3B-4bit"]);
        assert_eq!(default_picker_index(&local, &recommended), 1);
    }

    #[test]
    fn default_index_points_at_first_download_past_separator() {
        // No local flagship → first recommended download. Layout is
        // [local(1), separator(1), recommended…], so index 2.
        let local = locals(&["some/random-model"]);
        let recommended = recs(&[
            "mlx-community/Qwen3.6-27B-OptiQ-4bit",
            "mlx-community/Qwen3.6-35B-A3B-4bit",
        ]);
        assert_eq!(default_picker_index(&local, &recommended), 2);
    }

    #[test]
    fn default_index_no_separator_when_no_local_models() {
        let recommended = recs(&["mlx-community/Qwen3.6-27B-OptiQ-4bit"]);
        assert_eq!(default_picker_index(&[], &recommended), 0);
    }

    #[test]
    fn default_index_falls_back_to_zero_without_recommendations() {
        let local = locals(&["some/random-model", "another/model"]);
        assert_eq!(default_picker_index(&local, &[]), 0);
    }

    #[test]
    fn sanitize_default_index_skips_separator_and_clamps() {
        let actions = vec![
            PickerAction::Local("a".into()),
            PickerAction::Separator,
            PickerAction::Remote("b".into()),
            PickerAction::Search,
        ];
        // Out of range → clamp to last.
        assert_eq!(sanitize_default_index(99, &actions), 3);
        // Separator → next selectable.
        assert_eq!(sanitize_default_index(1, &actions), 2);
        // Already valid → unchanged.
        assert_eq!(sanitize_default_index(0, &actions), 0);
        // Empty → 0.
        assert_eq!(sanitize_default_index(3, &[]), 0);
    }

    #[test]
    fn picker_page_len_uses_terminal_budget() {
        assert_eq!(picker_page_len_for_rows(24, 30), 12);
        assert_eq!(picker_page_len_for_rows(10, 30), 4);
        assert_eq!(picker_page_len_for_rows(4, 30), 1);
        assert_eq!(picker_page_len_for_rows(24, 3), 3);
    }

    #[test]
    fn fit_picker_item_for_width_truncates_to_single_line() {
        let item = fit_picker_item_for_width(
            "local  mlx-community/Qwen3-0.6B-4bit  ~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8".to_string(),
            48,
        );

        assert!(!item.contains('\n'));
        assert!(measure_text_width(&item) <= 42);
        assert!(item.ends_with("..."));
    }

    #[test]
    fn abbreviate_path_preserves_head_and_tail_components() {
        let abbreviated = abbreviate_path(Path::new(
            "/opt/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8",
        ));

        assert!(abbreviated.starts_with("/opt/huggingface/"));
        assert!(abbreviated.contains("/.../"));
        assert!(abbreviated.ends_with(
            "models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8"
        ));
    }
}

#[derive(Clone)]
enum PickerAction {
    Local(String),
    Remote(String),
    Search,
    Separator,
}
