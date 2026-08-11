//! First-run welcome banner + per-user marker file.
//!
//! - First launch: prints a 3-line banner and writes
//!   `${XDG_CONFIG_HOME:-$HOME/.config}/arle/seen` with a timestamp.
//! - Subsequent launches: prints a single info line so the model stays visible.
//! - Non-writable config dir → silently fall back to the short one-liner.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use console::Style;

fn config_home() -> Option<PathBuf> {
    if let Some(x) = std::env::var_os("XDG_CONFIG_HOME")
        && !x.is_empty()
    {
        Some(PathBuf::from(x))
    } else {
        Some(PathBuf::from(std::env::var_os("HOME")?).join(".config"))
    }
}

/// Compute the `seen` marker file path honouring `$XDG_CONFIG_HOME`.
///
/// Returns `None` only when `$HOME` is unset AND `$XDG_CONFIG_HOME` is
/// unset — on any sane dev environment this is always `Some`.
pub(crate) fn banner_marker_path() -> Option<PathBuf> {
    Some(config_home()?.join("arle").join("seen"))
}

fn legacy_banner_marker_path() -> Option<PathBuf> {
    Some(config_home()?.join("arle").join("seen"))
}

fn marker_exists(path: &Path) -> bool {
    path.exists()
}

fn write_marker(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    fs::write(path, format!("{now}\n"))
}

/// Print the welcome banner. First run: 3-line banner + marker write.
/// Subsequent runs: 1-line model reminder. Non-writable config dir falls
/// back to the 1-liner.
pub(crate) fn print_welcome_banner(model_id: &str) {
    let dim = Style::new().dim();
    let marker = banner_marker_path();
    let legacy_marker = legacy_banner_marker_path();
    let marker_seen = marker.as_ref().is_some_and(|path| marker_exists(path));
    let legacy_seen = legacy_marker
        .as_ref()
        .is_some_and(|path| marker_exists(path));
    let first_run = !marker_seen && !legacy_seen;

    if first_run {
        eprintln!(
            "{}",
            dim.apply_to("▎ Welcome to ARLE — type a message to begin, or /help for commands.")
        );
        eprintln!("{}", dim.apply_to(format!("▎ model: {model_id}")));
        eprintln!(
            "{}",
            dim.apply_to("▎ /help  /reset  /tools  /quit · end a line with \\ for multi-line")
        );
        eprintln!(
            "{}",
            dim.apply_to("▎ Ctrl-C cancels a reply · Ctrl-D exits")
        );
    } else {
        eprintln!("{}", dim.apply_to(format!("▎ ARLE · model: {model_id}")));
    }

    // Attempt the marker write. A failure here (read-only $HOME, etc.)
    // is swallowed — next launch will just show the banner again, which
    // is strictly better than erroring out.
    if !marker_seen
        && let Some(path) = marker
        && write_marker(&path).is_err()
    {
        log::debug!("could not write welcome marker");
    }
}
