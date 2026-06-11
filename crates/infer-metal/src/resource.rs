//! Startup resource budgeting for the Metal executor.
//!
//! Apple Silicon uses unified memory: over-admitting the model process can stall
//! the whole desktop, not just fail a device allocation. This module keeps the
//! guard below the backend seam: infer-api passes neutral budget knobs, Metal
//! turns them into MLX allocator limits and a KV-token capacity.

use std::path::Path;
use std::process::Command;

use crate::{config, wired_limit};

use crate::executor::MetalKvCacheDtype;

const GIB: usize = 1 << 30;
const MIB: usize = 1 << 20;
const WIRED_HEADROOM_BYTES: usize = GIB;
const DEFAULT_RUNTIME_HEADROOM_BYTES: usize = 6 * GIB;
const LOW_IMPACT_RUNTIME_HEADROOM_BYTES: usize = 8 * GIB;
const DEFAULT_AVAILABLE_RESERVE_BYTES: usize = 6 * GIB;
const LOW_IMPACT_AVAILABLE_RESERVE_BYTES: usize = 8 * GIB;
const DEFAULT_CACHE_LIMIT_BYTES: usize = GIB;
const LOW_IMPACT_CACHE_LIMIT_BYTES: usize = 512 * MIB;
const SWAP_USED_GUARD_BYTES: usize = 512 * MIB;

#[derive(Debug, Clone, Copy)]
pub struct MetalResourceRequest {
    pub kv_cache_dtype: MetalKvCacheDtype,
    pub num_slots: usize,
    pub total_pages: usize,
    pub page_size: usize,
    pub low_impact: bool,
    pub memory_budget_bytes: Option<usize>,
    pub system_reserve_bytes: Option<usize>,
    pub allow_swap: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct MetalResourcePlan {
    pub total_memory_bytes: Option<usize>,
    pub available_memory_bytes: Option<usize>,
    pub swap_used_bytes: Option<usize>,
    pub memory_limit_bytes: usize,
    pub cache_limit_bytes: usize,
    pub wired_limit_bytes: usize,
    pub weight_bytes: usize,
    pub runtime_headroom_bytes: usize,
    pub static_state_bytes: usize,
    pub kv_budget_bytes: usize,
    pub kv_bytes_per_token_all_slots: usize,
    pub requested_total_pages: usize,
    pub planned_total_pages: usize,
    pub capacity_tokens: usize,
    pub clamped: bool,
}

impl MetalResourcePlan {
    pub fn describe(&self) -> String {
        format!(
            "memory_limit={}GiB wired={}GiB cache={}MiB weights={}GiB runtime_headroom={}GiB static_state={}MiB kv_budget={}GiB kv_capacity_tokens={} pages={}{}",
            self.memory_limit_bytes / GIB,
            self.wired_limit_bytes / GIB,
            self.cache_limit_bytes / MIB,
            self.weight_bytes / GIB,
            self.runtime_headroom_bytes / GIB,
            self.static_state_bytes / MIB,
            self.kv_budget_bytes / GIB,
            self.capacity_tokens,
            self.planned_total_pages,
            if self.clamped { " clamped" } else { "" },
        )
    }
}

pub fn plan_resource_budget(
    model_dir: &Path,
    request: MetalResourceRequest,
) -> anyhow::Result<MetalResourcePlan> {
    let model_config = config::load_metal_config(model_dir)?;
    let weight_bytes = wired_limit::model_weight_bytes(model_dir).ok_or_else(|| {
        anyhow::anyhow!(
            "Metal resource guard could not estimate model weight bytes for {}; \
             pass a local safetensors/bin/gguf/npz model directory or set an explicit budget after verifying memory headroom",
            model_dir.display()
        )
    })?;

    let total_memory_bytes = physical_memory_bytes();
    let available_memory_bytes = available_memory_bytes();
    let swap_used_bytes = swap_used_bytes();
    if !request.allow_swap {
        if let Some(used) = swap_used_bytes {
            anyhow::ensure!(
                used <= SWAP_USED_GUARD_BYTES,
                "Metal resource guard rejected startup because macOS swap is already active: used={} MiB. \
                 This path can spill unified memory to SSD and stall the system. Close memory-heavy apps, reboot to clear swap if needed, or pass --allow-swap after accepting the risk.",
                used / MIB,
            );
        }
    }
    let runtime_headroom_bytes = if request.low_impact {
        LOW_IMPACT_RUNTIME_HEADROOM_BYTES
    } else {
        DEFAULT_RUNTIME_HEADROOM_BYTES
    };
    let cache_limit_bytes = if request.low_impact {
        LOW_IMPACT_CACHE_LIMIT_BYTES
    } else {
        DEFAULT_CACHE_LIMIT_BYTES
    };
    let memory_limit_bytes = resolve_memory_limit(
        request.memory_budget_bytes,
        request.system_reserve_bytes,
        request.low_impact,
        total_memory_bytes,
        available_memory_bytes,
    )?;

    let wired_limit_bytes = weight_bytes
        .checked_add(WIRED_HEADROOM_BYTES)
        .ok_or_else(|| anyhow::anyhow!("Metal wired-limit estimate overflowed"))?;
    let static_state_bytes = gdr_state_bytes_per_slot(&model_config)
        .checked_mul(request.num_slots.max(1))
        .ok_or_else(|| anyhow::anyhow!("Metal GDR state estimate overflowed"))?;
    let fixed_bytes = weight_bytes
        .checked_add(runtime_headroom_bytes)
        .and_then(|v| v.checked_add(static_state_bytes))
        .ok_or_else(|| anyhow::anyhow!("Metal fixed memory estimate overflowed"))?;

    anyhow::ensure!(
        memory_limit_bytes > fixed_bytes,
        "Metal resource guard rejected startup: memory budget {} GiB is below fixed requirement {} GiB \
         (weights {} GiB + runtime headroom {} GiB + static state {} MiB). \
         Close other memory-heavy apps, use a smaller model, or pass --memory-budget-bytes after verifying headroom.",
        memory_limit_bytes / GIB,
        fixed_bytes / GIB,
        weight_bytes / GIB,
        runtime_headroom_bytes / GIB,
        static_state_bytes / MIB,
    );
    anyhow::ensure!(
        memory_limit_bytes > wired_limit_bytes,
        "Metal resource guard rejected startup: memory budget {} GiB is below wired requirement {} GiB \
         (weights + {} GiB).",
        memory_limit_bytes / GIB,
        wired_limit_bytes / GIB,
        WIRED_HEADROOM_BYTES / GIB,
    );

    let kv_budget_bytes = memory_limit_bytes - fixed_bytes;
    let kv_bytes_per_token_all_slots =
        kv_bytes_per_token_per_slot(&model_config, request.kv_cache_dtype)?
            .checked_mul(request.num_slots.max(1))
            .ok_or_else(|| anyhow::anyhow!("Metal KV byte estimate overflowed"))?;
    anyhow::ensure!(
        kv_bytes_per_token_all_slots > 0,
        "Metal KV byte estimate was zero"
    );
    let requested_total_pages = request.total_pages.max(1);
    let requested_capacity_tokens = requested_total_pages
        .checked_mul(request.page_size.max(1))
        .ok_or_else(|| anyhow::anyhow!("Metal requested KV capacity overflowed"))?;
    let max_capacity_tokens = kv_budget_bytes / kv_bytes_per_token_all_slots;
    let max_total_pages = max_capacity_tokens / request.page_size.max(1);
    anyhow::ensure!(
        max_total_pages > 0,
        "Metal resource guard rejected startup: remaining KV budget {} MiB cannot hold one {}-token page \
         ({} bytes/token across {} slot(s)).",
        kv_budget_bytes / MIB,
        request.page_size.max(1),
        kv_bytes_per_token_all_slots,
        request.num_slots.max(1),
    );

    let planned_total_pages = requested_total_pages.min(max_total_pages).max(1);
    let capacity_tokens = planned_total_pages
        .checked_mul(request.page_size.max(1))
        .ok_or_else(|| anyhow::anyhow!("Metal planned KV capacity overflowed"))?;

    Ok(MetalResourcePlan {
        total_memory_bytes,
        available_memory_bytes,
        swap_used_bytes,
        memory_limit_bytes,
        cache_limit_bytes,
        wired_limit_bytes,
        weight_bytes,
        runtime_headroom_bytes,
        static_state_bytes,
        kv_budget_bytes,
        kv_bytes_per_token_all_slots,
        requested_total_pages,
        planned_total_pages,
        capacity_tokens: capacity_tokens.min(requested_capacity_tokens),
        clamped: planned_total_pages < requested_total_pages,
    })
}

fn resolve_memory_limit(
    explicit_budget: Option<usize>,
    explicit_reserve: Option<usize>,
    low_impact: bool,
    total_memory_bytes: Option<usize>,
    available_memory_bytes: Option<usize>,
) -> anyhow::Result<usize> {
    if let Some(budget) = explicit_budget {
        anyhow::ensure!(
            budget >= GIB,
            "--memory-budget-bytes must be at least 1 GiB"
        );
        if let Some(total) = total_memory_bytes {
            let reserve =
                explicit_reserve.unwrap_or_else(|| default_system_reserve_bytes(total, low_impact));
            anyhow::ensure!(
                reserve < total,
                "Metal system reserve {} GiB is >= total memory {} GiB",
                reserve / GIB,
                total / GIB,
            );
            anyhow::ensure!(
                budget <= total - reserve,
                "--memory-budget-bytes {} GiB exceeds physical budget after system reserve {} GiB",
                budget / GIB,
                (total - reserve) / GIB,
            );
        }
        if let Some(available) = available_memory_bytes {
            let reserve = if low_impact {
                LOW_IMPACT_AVAILABLE_RESERVE_BYTES
            } else {
                DEFAULT_AVAILABLE_RESERVE_BYTES
            };
            anyhow::ensure!(
                available > reserve,
                "Metal resource guard rejected startup: available memory {} GiB is <= anti-swap reserve {} GiB. \
                 macOS would likely compress/swap to SSD under model load; close memory-heavy apps or use a smaller model.",
                available / GIB,
                reserve / GIB,
            );
            anyhow::ensure!(
                budget <= available - reserve,
                "--memory-budget-bytes {} GiB exceeds current anti-swap budget {} GiB \
                 (available memory minus reserve).",
                budget / GIB,
                (available - reserve) / GIB,
            );
        }
        return Ok(budget);
    }

    let mut candidates = Vec::new();
    if let Some(total) = total_memory_bytes {
        let reserve =
            explicit_reserve.unwrap_or_else(|| default_system_reserve_bytes(total, low_impact));
        anyhow::ensure!(
            reserve < total,
            "Metal system reserve {} GiB is >= total memory {} GiB",
            reserve / GIB,
            total / GIB,
        );
        candidates.push(total - reserve);
    }
    if let Some(available) = available_memory_bytes {
        let reserve = if low_impact {
            LOW_IMPACT_AVAILABLE_RESERVE_BYTES
        } else {
            DEFAULT_AVAILABLE_RESERVE_BYTES
        };
        anyhow::ensure!(
            available > reserve,
            "Metal resource guard rejected startup: available memory {} GiB is <= anti-swap reserve {} GiB. \
             macOS would likely compress/swap to SSD under model load; close memory-heavy apps or use a smaller model.",
            available / GIB,
            reserve / GIB,
        );
        candidates.push(available - reserve);
    }
    candidates
        .into_iter()
        .min()
        .ok_or_else(|| anyhow::anyhow!("Metal resource guard could not determine a memory budget"))
}

fn default_system_reserve_bytes(total_memory_bytes: usize, low_impact: bool) -> usize {
    let fraction = if low_impact {
        total_memory_bytes / 3
    } else {
        total_memory_bytes / 4
    };
    let floor = if low_impact { 18 * GIB } else { 14 * GIB };
    fraction
        .max(floor)
        .min(total_memory_bytes.saturating_sub(GIB))
}

fn kv_bytes_per_token_per_slot(
    config: &config::MetalModelConfig,
    kv_cache_dtype: MetalKvCacheDtype,
) -> anyhow::Result<usize> {
    let full_layers = config.arch.num_full_attention_layers();
    let nkv = config.num_key_value_heads;
    let hd = config.head_dim;
    let bytes_per_layer = match kv_cache_dtype {
        MetalKvCacheDtype::Bf16 => {
            // K + V, both BF16.
            2usize
                .checked_mul(nkv)
                .and_then(|v| v.checked_mul(hd))
                .and_then(|v| v.checked_mul(2))
        }
        MetalKvCacheDtype::Int8 => {
            let group = crate::executor::int8_kv_group_size(config.head_dim)?;
            // Per K/V: packed uint32 data uses head_dim bytes; scale+bias use
            // two BF16 vectors, i.e. 4 * head_dim/group bytes.
            let per_k_or_v = hd
                .checked_add(4usize.saturating_mul(hd / group))
                .and_then(|v| v.checked_mul(nkv));
            per_k_or_v.and_then(|v| v.checked_mul(2))
        }
    }
    .ok_or_else(|| anyhow::anyhow!("Metal KV byte estimate overflowed"))?;
    bytes_per_layer
        .checked_mul(full_layers)
        .ok_or_else(|| anyhow::anyhow!("Metal KV byte estimate overflowed"))
}

fn gdr_state_bytes_per_slot(config: &config::MetalModelConfig) -> usize {
    let layers = config.arch.num_linear_attention_layers();
    let linear = &config.arch.linear;
    let recurrent = linear
        .num_value_heads
        .saturating_mul(linear.value_dim)
        .saturating_mul(linear.key_dim)
        .saturating_mul(4);
    let conv = linear
        .conv_kernel
        .saturating_sub(1)
        .saturating_mul(linear.qkv_dim())
        .saturating_mul(2);
    layers.saturating_mul(recurrent.saturating_add(conv))
}

fn physical_memory_bytes() -> Option<usize> {
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse::<usize>()
        .ok()
}

fn page_size_bytes() -> Option<usize> {
    let output = Command::new("sysctl")
        .args(["-n", "hw.pagesize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse::<usize>()
        .ok()
}

fn available_memory_bytes() -> Option<usize> {
    let page_size = page_size_bytes().unwrap_or(4096);
    let output = Command::new("vm_stat").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let pages = ["Pages free", "Pages inactive", "Pages speculative"]
        .into_iter()
        .filter_map(|key| parse_vm_stat_pages(&text, key))
        .sum::<usize>();
    (pages > 0).then_some(pages.saturating_mul(page_size))
}

fn swap_used_bytes() -> Option<usize> {
    let output = Command::new("sysctl")
        .args(["-n", "vm.swapusage"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    parse_swap_used_bytes(&text)
}

fn parse_vm_stat_pages(text: &str, key: &str) -> Option<usize> {
    let line = text.lines().find(|line| line.starts_with(key))?;
    let value = line
        .split(':')
        .nth(1)?
        .trim()
        .trim_end_matches('.')
        .replace([',', '.'], "");
    value.parse::<usize>().ok()
}

fn parse_swap_used_bytes(text: &str) -> Option<usize> {
    let used = text.split("used =").nth(1)?.trim();
    let raw = used.split_whitespace().next()?;
    parse_memory_quantity(raw)
}

fn parse_memory_quantity(raw: &str) -> Option<usize> {
    let raw = raw.trim();
    let unit = raw.chars().last()?;
    let (number, multiplier) = match unit {
        'K' | 'k' => (&raw[..raw.len() - 1], 1024f64),
        'M' | 'm' => (&raw[..raw.len() - 1], MIB as f64),
        'G' | 'g' => (&raw[..raw.len() - 1], GIB as f64),
        _ if unit.is_ascii_digit() => (raw, 1f64),
        _ => return None,
    };
    let value = number.parse::<f64>().ok()?;
    Some((value * multiplier) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        MetalGdrConfig, MetalModelConfig, MetalQwen35ArchConfig, MetalQwen35LayerType,
    };

    fn tiny_config() -> MetalModelConfig {
        MetalModelConfig {
            hidden_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 32,
            stop_token_ids: vec![],
            quantization: None,
            arch: MetalQwen35ArchConfig {
                layer_types: vec![
                    MetalQwen35LayerType::FullAttention,
                    MetalQwen35LayerType::LinearAttention,
                    MetalQwen35LayerType::FullAttention,
                    MetalQwen35LayerType::LinearAttention,
                ],
                rotary_dim: 32,
                linear: MetalGdrConfig {
                    num_key_heads: 4,
                    key_dim: 32,
                    num_value_heads: 2,
                    value_dim: 32,
                    conv_kernel: 4,
                    rms_norm_eps: 1e-6,
                },
                moe: None,
            },
        }
    }

    #[test]
    fn int8_kv_estimate_is_smaller_than_bf16() {
        let cfg = tiny_config();
        let bf16 = kv_bytes_per_token_per_slot(&cfg, MetalKvCacheDtype::Bf16).unwrap();
        let int8 = kv_bytes_per_token_per_slot(&cfg, MetalKvCacheDtype::Int8).unwrap();
        assert!(int8 < bf16, "int8={int8}, bf16={bf16}");
    }

    #[test]
    fn default_reserve_leaves_headroom() {
        let total = 48 * GIB;
        assert_eq!(default_system_reserve_bytes(total, false), 14 * GIB);
        assert_eq!(default_system_reserve_bytes(total, true), 18 * GIB);
    }

    #[test]
    fn explicit_memory_budget_wins() {
        let budget =
            resolve_memory_limit(Some(22 * GIB), None, false, Some(48 * GIB), Some(40 * GIB))
                .unwrap();
        assert_eq!(budget, 22 * GIB);
    }

    #[test]
    fn available_memory_below_reserve_is_rejected() {
        let err = resolve_memory_limit(None, None, false, Some(48 * GIB), Some(4 * GIB))
            .expect_err("low available memory must fail closed");
        assert!(err.to_string().contains("anti-swap reserve"));
    }

    #[test]
    fn vm_stat_parser_accepts_macos_format() {
        let text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:                               12,345.\nPages inactive:                           20.\nPages speculative:                        7.\n";
        assert_eq!(parse_vm_stat_pages(text, "Pages free"), Some(12345));
        assert_eq!(parse_vm_stat_pages(text, "Pages inactive"), Some(20));
        assert_eq!(parse_vm_stat_pages(text, "Pages speculative"), Some(7));
    }

    #[test]
    fn swap_usage_parser_accepts_macos_format() {
        let text = "total = 2048.00M  used = 768.25M  free = 1279.75M  (encrypted)";
        assert_eq!(parse_swap_used_bytes(text), Some(768 * MIB + 256 * 1024));
    }
}
