//! First-run welcome banner + per-user marker file.
//!
//! - First launch: prints a 3-line banner and writes
//!   `${XDG_CONFIG_HOME:-$HOME/.config}/agent-infer/seen` with a timestamp.
//! - Subsequent launches: prints a single info line so the model stays visible.
//! - Non-writable config dir → silently fall back to the short one-liner.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use console::Style;

/// Compute the `seen` marker file path honouring `$XDG_CONFIG_HOME`.
///
/// Returns `None` only when `$HOME` is unset AND `$XDG_CONFIG_HOME` is
/// unset — on any sane dev environment this is always `Some`.
pub(crate) fn banner_marker_path() -> Option<PathBuf> {
    let base = if let Some(x) = std::env::var_os("XDG_CONFIG_HOME")
        && !x.is_empty()
    {
        PathBuf::from(x)
    } else {
        PathBuf::from(std::env::var_os("HOME")?).join(".config")
    };
    Some(base.join("agent-infer").join("seen"))
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
    let first_run = match marker.as_ref() {
        Some(p) => !marker_exists(p),
        None => false,
    };

    if first_run {
        eprintln!(
            "{}",
            dim.apply_to(format!("▎ agent-infer · model: {model_id}"))
        );
        eprintln!(
            "{}",
            dim.apply_to("▎ commands: /help  /reset  /tools  /quit      \\ at EOL = multi-line")
        );
        eprintln!(
            "{}",
            dim.apply_to("▎ Ctrl-C to cancel generation · Ctrl-D to exit")
        );

        // Attempt the marker write. A failure here (read-only $HOME, etc.)
        // is swallowed — next launch will just show the banner again, which
        // is strictly better than erroring out.
        if let Some(p) = marker
            && write_marker(&p).is_err()
        {
            log::debug!("could not write welcome marker");
        }
    } else {
        eprintln!(
            "{}",
            dim.apply_to(format!("▎ agent-infer · model: {model_id}"))
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EnvGuard {
        key: &'static str,
        prior: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let prior = std::env::var_os(key);
            match value {
                Some(v) => unsafe { std::env::set_var(key, v) },
                None => unsafe { std::env::remove_var(key) },
            }
            Self { key, prior }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prior {
                Some(v) => unsafe { std::env::set_var(self.key, v) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    #[test]
    fn banner_marker_path_respects_xdg() {
        let _xdg = EnvGuard::set("XDG_CONFIG_HOME", Some("/tmp/xdgtest"));
        let _home = EnvGuard::set("HOME", Some("/home/ignored"));
        let p = banner_marker_path().expect("xdg set => Some");
        assert_eq!(p, PathBuf::from("/tmp/xdgtest/agent-infer/seen"));
    }

    #[test]
    fn banner_marker_path_falls_back_to_home() {
        let _xdg = EnvGuard::set("XDG_CONFIG_HOME", None);
        let _home = EnvGuard::set("HOME", Some("/home/u"));
        let p = banner_marker_path().expect("home set => Some");
        assert_eq!(p, PathBuf::from("/home/u/.config/agent-infer/seen"));
    }

    #[test]
    fn banner_marker_path_empty_xdg_treated_as_unset() {
        let _xdg = EnvGuard::set("XDG_CONFIG_HOME", Some(""));
        let _home = EnvGuard::set("HOME", Some("/home/u"));
        let p = banner_marker_path().expect("home set => Some");
        assert_eq!(p, PathBuf::from("/home/u/.config/agent-infer/seen"));
    }
}
