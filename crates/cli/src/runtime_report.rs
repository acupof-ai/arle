#[cfg(unix)]
use std::mem::MaybeUninit;

use sysinfo::{ProcessesToUpdate, System, get_current_pid};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MemorySnapshot {
    pub(crate) current_rss_bytes: Option<u64>,
    pub(crate) peak_rss_bytes: Option<u64>,
    pub(crate) system_available_bytes: Option<u64>,
}

pub(crate) struct ExitResourceReport {
    enabled: bool,
}

impl ExitResourceReport {
    #[must_use]
    pub(crate) fn enabled() -> Self {
        Self { enabled: true }
    }
}

impl Drop for ExitResourceReport {
    fn drop(&mut self) {
        if self.enabled {
            print_exit_memory_report();
        }
    }
}

fn print_exit_memory_report() {
    let snapshot = capture_memory_snapshot();
    eprintln!("{}", format_exit_memory_report(snapshot));
}

fn capture_memory_snapshot() -> MemorySnapshot {
    let mut system = System::new();
    system.refresh_memory();
    let system_available_bytes = Some(system.available_memory());

    let current_rss_bytes = get_current_pid().ok().and_then(|pid| {
        system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        system.process(pid).map(|process| process.memory())
    });

    MemorySnapshot {
        current_rss_bytes,
        peak_rss_bytes: peak_rss_bytes(),
        system_available_bytes,
    }
}

fn format_exit_memory_report(snapshot: MemorySnapshot) -> String {
    let peak = format_bytes_or_unknown(snapshot.peak_rss_bytes);
    let current = format_bytes_or_unknown(snapshot.current_rss_bytes);
    let available = format_bytes_or_unknown(snapshot.system_available_bytes);
    format!(
        "[ARLE] exit memory: peak_rss={peak} current_rss={current} system_available={available}"
    )
}

fn format_bytes_or_unknown(bytes: Option<u64>) -> String {
    bytes
        .map(format_gib)
        .unwrap_or_else(|| "unknown".to_string())
}

fn format_gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / 1024.0 / 1024.0 / 1024.0)
}

#[cfg(unix)]
fn peak_rss_bytes() -> Option<u64> {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` points to valid writable storage for `getrusage`.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: `getrusage` returned success, so it initialized `usage`.
    let raw = unsafe { usage.assume_init() }.ru_maxrss;
    if raw <= 0 {
        return None;
    }
    Some(ru_maxrss_to_bytes(raw as u64))
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> Option<u64> {
    None
}

#[cfg(target_os = "macos")]
fn ru_maxrss_to_bytes(raw: u64) -> u64 {
    raw
}

#[cfg(all(unix, not(target_os = "macos")))]
fn ru_maxrss_to_bytes(raw: u64) -> u64 {
    raw.saturating_mul(1024)
}
