//! Auto-budgeting for Metal in-memory prefix KV snapshots.
//!
//! This budget is intentionally separate from `set_wired_limit`: wired memory
//! controls how much MLX may keep resident; this controls how many completed
//! request KV snapshots ARLE keeps alive for same-server prefix replay.

#[cfg(target_os = "macos")]
use std::process::Command;

use crate::{model_arch::ModelArchInfo, model_source::ResolvedModelSource};

use super::{
    config::{apply_gguf_metadata_overrides, load_metal_config, load_metal_config_from_gguf},
    wired_limit::{model_weight_size, model_weight_size_for_path},
};

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
const AUTO_KV_MEMORY_MAX_BYTES: u64 = 8 * GIB;
const AUTO_KV_MEMORY_MIN_BYTES: u64 = 256 * MIB;
const AUTO_KV_MEMORY_MIN_SYSTEM_HEADROOM_BYTES: u64 = 4 * GIB;
const AUTO_KV_MEMORY_LIVE_KV_RESERVE_MULTIPLIER: u64 = 2;
const AUTO_KV_MEMORY_SNAPSHOT_SPARE_DIVISOR: u64 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AutoKvMemoryBudget {
    budget_bytes: u64,
    available_bytes: u64,
    total_bytes: u64,
    model_weight_bytes: u64,
    weight_reserve_bytes: u64,
    kv_bytes_per_token: u64,
    live_kv_estimate_bytes: u64,
    live_kv_reserve_bytes: u64,
    system_headroom_bytes: u64,
    spare_after_reserve_bytes: u64,
}

/// Compute the default in-memory prefix snapshot budget. Returns `None` only
/// when the local model or system memory cannot be sized; a successful estimate
/// may return `0`, which deliberately disables memory snapshots under pressure.
pub fn auto_kv_memory_max_bytes(
    model_path: &str,
    max_running_requests: usize,
    max_batch_tokens: usize,
    weight_reserve_limit_bytes: Option<u64>,
) -> Option<u64> {
    let memory = system_memory_snapshot_bytes()?;
    let source = ResolvedModelSource::resolve(model_path).ok()?;
    let weights = model_weight_size_for_path(source.resolved_path())
        .or_else(|| model_weight_size(model_path))?;
    let kv_bytes_per_token = estimate_kv_bytes_per_token(&source)?;
    let weight_reserve_bytes = weight_reserve_limit_bytes
        .map(|limit| weights.bytes.min(limit))
        .unwrap_or(0);
    let budget = compute_auto_kv_memory_budget(
        memory.available_bytes,
        memory.total_bytes,
        weights.bytes,
        weight_reserve_bytes,
        kv_bytes_per_token,
        max_running_requests,
        max_batch_tokens,
    );
    log::info!(
        "Metal in-memory KV snapshot auto-budget: budget={} bytes ({:.2} GiB), available={} bytes ({:.2} GiB), total={} bytes ({:.2} GiB), model_weights={} bytes ({:.2} GiB), weight_reserve={} bytes ({:.2} GiB), live_kv_estimate={} bytes ({:.2} GiB), live_kv_reserve={} bytes ({:.2} GiB), system_headroom={} bytes ({:.2} GiB), spare_after_reserve={} bytes ({:.2} GiB), kv_bytes_per_token={}, model_source={}",
        budget.budget_bytes,
        gib(budget.budget_bytes),
        budget.available_bytes,
        gib(budget.available_bytes),
        budget.total_bytes,
        gib(budget.total_bytes),
        budget.model_weight_bytes,
        gib(budget.model_weight_bytes),
        budget.weight_reserve_bytes,
        gib(budget.weight_reserve_bytes),
        budget.live_kv_estimate_bytes,
        gib(budget.live_kv_estimate_bytes),
        budget.live_kv_reserve_bytes,
        gib(budget.live_kv_reserve_bytes),
        budget.system_headroom_bytes,
        gib(budget.system_headroom_bytes),
        budget.spare_after_reserve_bytes,
        gib(budget.spare_after_reserve_bytes),
        budget.kv_bytes_per_token,
        weights.source.display(),
    );
    Some(budget.budget_bytes)
}

fn compute_auto_kv_memory_budget(
    available_bytes: u64,
    total_bytes: u64,
    model_weight_bytes: u64,
    weight_reserve_bytes: u64,
    kv_bytes_per_token: u64,
    max_running_requests: usize,
    max_batch_tokens: usize,
) -> AutoKvMemoryBudget {
    let live_tokens =
        (max_running_requests.max(1) as u64).saturating_mul(max_batch_tokens.max(1) as u64);
    let live_kv_estimate_bytes = kv_bytes_per_token.saturating_mul(live_tokens);
    let live_kv_reserve_bytes =
        live_kv_estimate_bytes.saturating_mul(AUTO_KV_MEMORY_LIVE_KV_RESERVE_MULTIPLIER);
    let system_headroom_bytes = AUTO_KV_MEMORY_MIN_SYSTEM_HEADROOM_BYTES.max(total_bytes / 10);
    let reserved_bytes = weight_reserve_bytes
        .saturating_add(live_kv_reserve_bytes)
        .saturating_add(system_headroom_bytes);
    let spare_after_reserve_bytes = available_bytes.saturating_sub(reserved_bytes);
    let mut budget_bytes = (spare_after_reserve_bytes / AUTO_KV_MEMORY_SNAPSHOT_SPARE_DIVISOR)
        .min(AUTO_KV_MEMORY_MAX_BYTES);
    if budget_bytes < AUTO_KV_MEMORY_MIN_BYTES {
        budget_bytes = 0;
    }

    AutoKvMemoryBudget {
        budget_bytes,
        available_bytes,
        total_bytes,
        model_weight_bytes,
        weight_reserve_bytes,
        kv_bytes_per_token,
        live_kv_estimate_bytes,
        live_kv_reserve_bytes,
        system_headroom_bytes,
        spare_after_reserve_bytes,
    }
}

fn estimate_kv_bytes_per_token(source: &ResolvedModelSource) -> Option<u64> {
    let mut config = if let Some(dir) = source.config_dir() {
        load_metal_config(dir).ok()?
    } else {
        let gguf = source.gguf()?;
        load_metal_config_from_gguf(gguf).ok()?
    };
    if let Some(gguf) = source.gguf() {
        apply_gguf_metadata_overrides(&mut config, gguf);
    }
    Some(config.kv_cache_bytes_per_token() as u64)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SystemMemorySnapshot {
    available_bytes: u64,
    total_bytes: u64,
}

fn system_memory_snapshot_bytes() -> Option<SystemMemorySnapshot> {
    #[cfg(target_os = "macos")]
    {
        let total_bytes = sysctl_u64("hw.memsize")?;
        let available_bytes = macos_available_memory_bytes()?;
        Some(SystemMemorySnapshot {
            available_bytes,
            total_bytes,
        })
    }

    #[cfg(target_os = "linux")]
    {
        return linux_memory_snapshot_bytes();
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
fn sysctl_u64(name: &str) -> Option<u64> {
    let raw = command_output_trimmed("sysctl", &["-n", name])?;
    raw.parse::<u64>().ok()
}

#[cfg(target_os = "macos")]
fn macos_available_memory_bytes() -> Option<u64> {
    let raw = command_output_trimmed("vm_stat", &[])?;
    parse_macos_vm_stat_available_bytes(&raw)
}

#[cfg(target_os = "macos")]
fn parse_macos_vm_stat_available_bytes(raw: &str) -> Option<u64> {
    let mut page_size = None;
    let mut pages_free = 0u64;
    let mut pages_inactive = 0u64;
    let mut pages_speculative = 0u64;

    for line in raw.lines() {
        if line.contains("page size of") {
            page_size = line
                .split_whitespace()
                .find_map(|word| word.parse::<u64>().ok());
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let pages = parse_vm_stat_pages(value);
        match key.trim() {
            "Pages free" => pages_free = pages,
            "Pages inactive" => pages_inactive = pages,
            "Pages speculative" => pages_speculative = pages,
            _ => {}
        }
    }

    let page_size = page_size?;
    Some(
        pages_free
            .saturating_add(pages_inactive)
            .saturating_add(pages_speculative)
            .saturating_mul(page_size),
    )
}

#[cfg(target_os = "macos")]
fn parse_vm_stat_pages(value: &str) -> u64 {
    let digits = value
        .chars()
        .filter(char::is_ascii_digit)
        .collect::<String>();
    digits.parse::<u64>().unwrap_or(0)
}

#[cfg(target_os = "linux")]
fn linux_memory_snapshot_bytes() -> Option<SystemMemorySnapshot> {
    let raw = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut available_kib = None;
    let mut total_kib = None;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available_kib = parse_meminfo_kib(rest);
        } else if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kib = parse_meminfo_kib(rest);
        }
    }
    Some(SystemMemorySnapshot {
        available_bytes: available_kib?.saturating_mul(1024),
        total_bytes: total_kib?.saturating_mul(1024),
    })
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kib(value: &str) -> Option<u64> {
    value
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<u64>().ok())
}

#[cfg(target_os = "macos")]
fn command_output_trimmed(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_budget_reserves_weights_live_kv_and_headroom() {
        let budget = compute_auto_kv_memory_budget(
            48 * GIB,
            64 * GIB,
            20 * GIB,
            20 * GIB,
            64 * 1024,
            2,
            4096,
        );
        assert_eq!(budget.model_weight_bytes, 20 * GIB);
        assert_eq!(budget.weight_reserve_bytes, 20 * GIB);
        assert_eq!(budget.live_kv_estimate_bytes, 512 * MIB);
        assert_eq!(budget.live_kv_reserve_bytes, GIB);
        assert_eq!(budget.system_headroom_bytes, 64 * GIB / 10);
        assert_eq!(budget.budget_bytes, 8 * GIB);
    }

    #[test]
    fn auto_budget_disables_memory_snapshots_when_spare_is_tiny() {
        let budget = compute_auto_kv_memory_budget(
            24 * GIB,
            32 * GIB,
            20 * GIB,
            20 * GIB,
            64 * 1024,
            4,
            4096,
        );
        assert_eq!(budget.budget_bytes, 0);
    }

    #[test]
    fn auto_budget_uses_half_spare_below_cap() {
        let budget = compute_auto_kv_memory_budget(
            30 * GIB,
            48 * GIB,
            20 * GIB,
            20 * GIB,
            64 * 1024,
            1,
            1024,
        );
        assert_eq!(budget.live_kv_estimate_bytes, 64 * MIB);
        assert_eq!(
            budget.budget_bytes,
            (30 * GIB - 20 * GIB - 128 * MIB - 48 * GIB / 10) / 2
        );
    }

    #[test]
    fn auto_budget_does_not_hard_reserve_unwired_weights() {
        let budget =
            compute_auto_kv_memory_budget(20 * GIB, 48 * GIB, 19 * GIB, 0, 20 * 1024, 1, 4096);
        assert_eq!(budget.model_weight_bytes, 19 * GIB);
        assert_eq!(budget.weight_reserve_bytes, 0);
        assert!(budget.budget_bytes > 7 * GIB);
        assert!(budget.budget_bytes < 8 * GIB);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parses_vm_stat_available_memory() {
        let raw = "\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               10.
Pages active:                             99.
Pages inactive:                           20.
Pages speculative:                        30.
";
        assert_eq!(parse_macos_vm_stat_available_bytes(raw), Some(60 * 16384));
    }
}
