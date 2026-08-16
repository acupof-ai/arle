//! Startup resource budgeting for the Metal executor.
//!
//! Apple Silicon uses unified memory: over-admitting the model process can stall
//! the whole desktop, not just fail a device allocation. This module keeps the
//! guard below the backend seam: infer-api passes neutral budget knobs, Metal
//! turns them into MLX allocator limits and a KV-token capacity.

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use crate::{config, mlx, wired_limit};

use crate::executor::MetalKvCacheDtype;

const GIB: usize = 1 << 30;
const MIB: usize = 1 << 20;
const WIRED_HEADROOM_BYTES: usize = GIB;
// Non-KV runtime scratch reserved ABOVE weights (activations, MLX command buffers).
// KV is budgeted separately (`kv_budget = memory_limit − fixed`) and clamped to fit,
// and `*_AVAILABLE_RESERVE_BYTES` below is a *second*, independent anti-swap buffer
// kept free for the OS. So this term only needs to cover non-KV transients. A c=1 27B
// load measured ~3–5 GiB of runtime+KV TOTAL above the 18 GiB weights (non-KV < 3),
// yet 6 GiB here stacked on the 6 GiB anti-swap reserve required available ≈ weights
// + 12 GiB — over-reserving ~2× and falsely rejecting loads that fit (the 27B was
// rejected by a 0.6 GiB margin at 29.4 GiB available). 4 GiB covers the scratch with
// margin; the anti-swap reserve remains the swap backstop. (TODO: scale by
// num_slots/context instead of a flat constant.)
const DEFAULT_RUNTIME_HEADROOM_BYTES: usize = 4 * GIB;
const LOW_IMPACT_RUNTIME_HEADROOM_BYTES: usize = 6 * GIB;
const DEFAULT_AVAILABLE_RESERVE_BYTES: usize = 6 * GIB;
const LOW_IMPACT_AVAILABLE_RESERVE_BYTES: usize = 8 * GIB;
const DEFAULT_CACHE_LIMIT_BYTES: usize = GIB;
const LOW_IMPACT_CACHE_LIMIT_BYTES: usize = 512 * MIB;
const SWAP_USED_WARN_BYTES: usize = 512 * MIB;
const PAGING_SAMPLE_MILLIS: u64 = 1_000;
const ACTIVE_PAGEOUT_GUARD_BYTES: usize = 64 * MIB;
const ACTIVE_SWAPOUT_GUARD_BYTES: usize = 16 * MIB;

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
    /// Fraction of the anti-swap-clamped `memory_limit` the KV pool may claim
    /// (SGLang `mem_fraction_static`); the rest is non-KV headroom.
    pub mem_fraction_static: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct MetalWeightOnlyResourceRequest {
    pub low_impact: bool,
    pub memory_budget_bytes: Option<usize>,
    pub system_reserve_bytes: Option<usize>,
    pub allow_swap: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct MetalSystemStatus {
    pub total_memory_bytes: Option<usize>,
    pub available_memory_bytes: Option<usize>,
    pub recommended_max_working_set_bytes: Option<usize>,
    pub swap_used_bytes: Option<usize>,
    pub pageouts_delta_bytes: Option<usize>,
    pub swapouts_delta_bytes: Option<usize>,
    pub paging_sample_millis: u64,
}

impl MetalSystemStatus {
    pub fn current() -> Self {
        Self {
            total_memory_bytes: physical_memory_bytes(),
            available_memory_bytes: available_memory_bytes(),
            recommended_max_working_set_bytes: mlx::recommended_max_working_set_size_bytes(),
            swap_used_bytes: swap_used_bytes(),
            pageouts_delta_bytes: None,
            swapouts_delta_bytes: None,
            paging_sample_millis: 0,
        }
    }

    pub fn describe(&self) -> String {
        let paging = if self.paging_sample_millis == 0 {
            "paging_delta=not_sampled".to_string()
        } else {
            format!(
                "pageouts_delta={} swapouts_delta={} sample={}ms",
                format_mib(self.pageouts_delta_bytes),
                format_mib(self.swapouts_delta_bytes),
                self.paging_sample_millis,
            )
        };
        format!(
            "system total={} available={} gpu_working_set={} swap_used={} {paging}",
            format_gib(self.total_memory_bytes),
            format_gib(self.available_memory_bytes),
            format_gib(self.recommended_max_working_set_bytes),
            format_mib(self.swap_used_bytes),
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct PagingActivity {
    pageouts_delta_bytes: Option<usize>,
    swapouts_delta_bytes: Option<usize>,
    sample_millis: u64,
}

impl PagingActivity {
    fn is_active_pressure(self) -> bool {
        self.pageouts_delta_bytes
            .is_some_and(|bytes| bytes >= ACTIVE_PAGEOUT_GUARD_BYTES)
            || self
                .swapouts_delta_bytes
                .is_some_and(|bytes| bytes >= ACTIVE_SWAPOUT_GUARD_BYTES)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MetalResourcePlan {
    pub total_memory_bytes: Option<usize>,
    pub available_memory_bytes: Option<usize>,
    pub recommended_max_working_set_bytes: Option<usize>,
    pub swap_used_bytes: Option<usize>,
    pub pageouts_delta_bytes: Option<usize>,
    pub swapouts_delta_bytes: Option<usize>,
    pub paging_sample_millis: u64,
    pub residual_swap_warning: bool,
    pub memory_limit_bytes: usize,
    pub cache_limit_bytes: usize,
    pub wired_limit_bytes: usize,
    pub weight_bytes: usize,
    pub runtime_headroom_bytes: usize,
    pub static_state_bytes: usize,
    pub kv_budget_bytes: usize,
    pub kv_bytes_per_token: usize,
    pub requested_total_pages: usize,
    pub planned_total_pages: usize,
    pub capacity_tokens: usize,
    pub clamped: bool,
}

impl MetalResourcePlan {
    pub fn describe(&self) -> String {
        format!(
            "{} memory_limit={}GiB wired={}GiB cache={}MiB weights={}GiB runtime_headroom={}GiB static_state={}MiB kv_budget={}GiB kv_capacity_tokens={} pages={}{}",
            MetalSystemStatus {
                total_memory_bytes: self.total_memory_bytes,
                available_memory_bytes: self.available_memory_bytes,
                recommended_max_working_set_bytes: self.recommended_max_working_set_bytes,
                swap_used_bytes: self.swap_used_bytes,
                pageouts_delta_bytes: self.pageouts_delta_bytes,
                swapouts_delta_bytes: self.swapouts_delta_bytes,
                paging_sample_millis: self.paging_sample_millis,
            }
            .describe(),
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

pub(crate) fn apply_startup_mlx_limits(
    model_dir: &Path,
    resource_plan: Option<&MetalResourcePlan>,
    model_label: Option<&str>,
    print_plan: bool,
) {
    if let Some(plan) = resource_plan {
        let previous_memory = mlx::set_memory_limit_bytes(plan.memory_limit_bytes as u64);
        let previous_cache = mlx::set_cache_limit_bytes(plan.cache_limit_bytes as u64);
        let previous_wired = mlx::set_wired_limit_bytes(plan.wired_limit_bytes as u64);
        if print_plan {
            let label = model_label.map_or(String::new(), |label| format!("{label} "));
            eprintln!("[infer-metal] {label}resource guard: {}", plan.describe());
        }
        let label = model_label.map_or_else(
            || "Metal resource guard".to_string(),
            |label| format!("{label} Metal resource guard"),
        );
        log::info!(
            "{label} set MLX limits: memory={} (previous {}), cache={} (previous {}), wired={} (previous {})",
            plan.memory_limit_bytes,
            previous_memory,
            plan.cache_limit_bytes,
            previous_cache,
            plan.wired_limit_bytes,
            previous_wired
        );
    } else if let Some(limit) = wired_limit::auto_wired_limit_bytes(model_dir) {
        let previous = mlx::set_wired_limit_bytes(limit as u64);
        let label = model_label.map_or("Metal executor", |label| label);
        log::info!(
            "{label} wired limit set to {} bytes (previous {})",
            limit,
            previous
        );
    }
}

pub fn plan_resource_budget(
    model_dir: &Path,
    request: MetalResourceRequest,
) -> anyhow::Result<MetalResourcePlan> {
    let model_config = config::load_metal_config(model_dir)?;
    let weight_bytes = wired_limit::model_weight_bytes(model_dir).ok_or_else(|| {
        anyhow::anyhow!(
            "Metal resource guard could not estimate model weight bytes for {}; system={}. \
             pass a local safetensors/bin/gguf/npz model directory or set an explicit budget after verifying memory headroom",
            model_dir.display(),
            MetalSystemStatus::current().describe(),
        )
    })?;

    let system_status = MetalSystemStatus::current();
    let total_memory_bytes = system_status.total_memory_bytes;
    let available_memory_bytes = system_status.available_memory_bytes;
    let recommended_max_working_set_bytes = system_status.recommended_max_working_set_bytes;
    let swap_used_bytes = system_status.swap_used_bytes;
    let paging = sample_paging_activity();
    let system_status = MetalSystemStatus {
        pageouts_delta_bytes: paging.and_then(|p| p.pageouts_delta_bytes),
        swapouts_delta_bytes: paging.and_then(|p| p.swapouts_delta_bytes),
        paging_sample_millis: paging.map_or(0, |p| p.sample_millis),
        ..system_status
    };
    let system_status_line = system_status.describe();
    let residual_swap_warning = swap_used_bytes.is_some_and(|used| used >= SWAP_USED_WARN_BYTES);
    if residual_swap_warning {
        log::warn!(
            "Metal resource guard: residual macOS swap is present but not a hard failure: {}",
            system_status_line
        );
    }
    if !request.allow_swap && paging.is_some_and(PagingActivity::is_active_pressure) {
        anyhow::bail!(
            "Metal resource guard rejected startup: {system_status_line}; active pageout/swapout activity is above the guardrail \
             (pageout >= {} MiB/s or swapout >= {} MiB/s). This path can spill unified memory to SSD and stall the system. \
             Close memory-heavy apps and retry, or pass --allow-swap after accepting the risk.",
            ACTIVE_PAGEOUT_GUARD_BYTES / MIB,
            ACTIVE_SWAPOUT_GUARD_BYTES / MIB,
        );
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
        recommended_max_working_set_bytes,
        &system_status_line,
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

    // Shared budget kernel (infer-seam), same policy as the CUDA executors:
    // the reject-below-fixed guard is `memory_limit > fixed`.
    anyhow::ensure!(
        infer_seam::SlotBudget::fits_fixed(memory_limit_bytes, fixed_bytes),
        "Metal resource guard rejected startup: {system_status_line}; memory budget {} GiB is below fixed requirement {} GiB \
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
        "Metal resource guard rejected startup: {system_status_line}; memory budget {} GiB is below wired requirement {} GiB \
         (weights + {} GiB).",
        memory_limit_bytes / GIB,
        wired_limit_bytes / GIB,
        WIRED_HEADROOM_BYTES / GIB,
    );

    // ONE shared num_slots-independent token pool (mirrors CUDA): per-token cost
    // is NOT multiplied by num_slots — num_slots is a zero-HBM soft cap.
    let kv_bytes_per_token = kv_bytes_per_token_per_slot(&model_config, request.kv_cache_dtype)?;
    anyhow::ensure!(kv_bytes_per_token > 0, "Metal KV byte estimate was zero");
    // Shared budget kernel with `mem_fraction_static`: kv_budget = floor(limit ×
    // frac) − fixed, the anti-swap clamp already baked into `memory_limit`.
    let mem_fraction = infer_seam::clamp_mem_fraction_static(request.mem_fraction_static);
    let token_budget = infer_seam::SlotBudget::from_free(
        memory_limit_bytes,
        mem_fraction,
        fixed_bytes,
        kv_bytes_per_token,
    );
    let kv_budget_bytes = token_budget.budget_bytes;
    let max_capacity_tokens = token_budget.affordable().unwrap_or(0);
    let requested_total_pages = request.total_pages.max(1);
    let requested_capacity_tokens = requested_total_pages
        .checked_mul(request.page_size.max(1))
        .ok_or_else(|| anyhow::anyhow!("Metal requested KV capacity overflowed"))?;
    let max_total_pages = max_capacity_tokens / request.page_size.max(1);
    anyhow::ensure!(
        max_total_pages > 0,
        "Metal resource guard rejected startup: {system_status_line}; remaining KV budget {} MiB cannot hold one {}-token page \
         ({} bytes/token, shared pool).",
        kv_budget_bytes / MIB,
        request.page_size.max(1),
        kv_bytes_per_token,
    );

    // Shared clamp: planned = min(requested, max_total_pages); both ≥1 here, so
    // the legacy `.max(1)` was a no-op. `clamped` flags a reduced request.
    let (planned_total_pages, clamped) =
        infer_seam::clamp_to_affordable(requested_total_pages, max_total_pages);
    let capacity_tokens = planned_total_pages
        .checked_mul(request.page_size.max(1))
        .ok_or_else(|| anyhow::anyhow!("Metal planned KV capacity overflowed"))?;

    Ok(MetalResourcePlan {
        total_memory_bytes,
        available_memory_bytes,
        recommended_max_working_set_bytes,
        swap_used_bytes,
        pageouts_delta_bytes: system_status.pageouts_delta_bytes,
        swapouts_delta_bytes: system_status.swapouts_delta_bytes,
        paging_sample_millis: system_status.paging_sample_millis,
        residual_swap_warning,
        memory_limit_bytes,
        cache_limit_bytes,
        wired_limit_bytes,
        weight_bytes,
        runtime_headroom_bytes,
        static_state_bytes,
        kv_budget_bytes,
        kv_bytes_per_token,
        requested_total_pages,
        planned_total_pages,
        capacity_tokens: capacity_tokens.min(requested_capacity_tokens),
        clamped,
    })
}

pub fn plan_weight_only_resource_budget(
    model_dir: &Path,
    request: MetalWeightOnlyResourceRequest,
) -> anyhow::Result<MetalResourcePlan> {
    let weight_bytes = wired_limit::model_weight_bytes(model_dir).ok_or_else(|| {
        anyhow::anyhow!(
            "Metal resource guard could not estimate model weight bytes for {}; system={}. \
             pass a local safetensors/bin/gguf/npz model directory or set an explicit budget after verifying memory headroom",
            model_dir.display(),
            MetalSystemStatus::current().describe(),
        )
    })?;

    let system_status = MetalSystemStatus::current();
    let total_memory_bytes = system_status.total_memory_bytes;
    let available_memory_bytes = system_status.available_memory_bytes;
    let recommended_max_working_set_bytes = system_status.recommended_max_working_set_bytes;
    let swap_used_bytes = system_status.swap_used_bytes;
    let paging = sample_paging_activity();
    let system_status = MetalSystemStatus {
        pageouts_delta_bytes: paging.and_then(|p| p.pageouts_delta_bytes),
        swapouts_delta_bytes: paging.and_then(|p| p.swapouts_delta_bytes),
        paging_sample_millis: paging.map_or(0, |p| p.sample_millis),
        ..system_status
    };
    let system_status_line = system_status.describe();
    let residual_swap_warning = swap_used_bytes.is_some_and(|used| used >= SWAP_USED_WARN_BYTES);
    if residual_swap_warning {
        log::warn!(
            "Metal resource guard: residual macOS swap is present but not a hard failure: {}",
            system_status_line
        );
    }
    if !request.allow_swap && paging.is_some_and(PagingActivity::is_active_pressure) {
        anyhow::bail!(
            "Metal resource guard rejected startup: {system_status_line}; active pageout/swapout activity is above the guardrail \
             (pageout >= {} MiB/s or swapout >= {} MiB/s). This path can spill unified memory to SSD and stall the system. \
             Close memory-heavy apps and retry, or pass --allow-swap after accepting the risk.",
            ACTIVE_PAGEOUT_GUARD_BYTES / MIB,
            ACTIVE_SWAPOUT_GUARD_BYTES / MIB,
        );
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
        recommended_max_working_set_bytes,
        &system_status_line,
    )?;

    let wired_limit_bytes = weight_bytes
        .checked_add(WIRED_HEADROOM_BYTES)
        .ok_or_else(|| anyhow::anyhow!("Metal wired-limit estimate overflowed"))?;
    let fixed_bytes = weight_bytes
        .checked_add(runtime_headroom_bytes)
        .ok_or_else(|| anyhow::anyhow!("Metal fixed memory estimate overflowed"))?;
    anyhow::ensure!(
        infer_seam::SlotBudget::fits_fixed(memory_limit_bytes, fixed_bytes),
        "Metal resource guard rejected startup: {system_status_line}; memory budget {} GiB is below fixed requirement {} GiB \
         (weights {} GiB + runtime headroom {} GiB). \
         Close other memory-heavy apps, use a smaller model, or pass --memory-budget-bytes after verifying headroom.",
        memory_limit_bytes / GIB,
        fixed_bytes / GIB,
        weight_bytes / GIB,
        runtime_headroom_bytes / GIB,
    );
    anyhow::ensure!(
        memory_limit_bytes > wired_limit_bytes,
        "Metal resource guard rejected startup: {system_status_line}; memory budget {} GiB is below wired requirement {} GiB \
         (weights + {} GiB).",
        memory_limit_bytes / GIB,
        wired_limit_bytes / GIB,
        WIRED_HEADROOM_BYTES / GIB,
    );

    Ok(MetalResourcePlan {
        total_memory_bytes,
        available_memory_bytes,
        recommended_max_working_set_bytes,
        swap_used_bytes,
        pageouts_delta_bytes: system_status.pageouts_delta_bytes,
        swapouts_delta_bytes: system_status.swapouts_delta_bytes,
        paging_sample_millis: system_status.paging_sample_millis,
        residual_swap_warning,
        memory_limit_bytes,
        cache_limit_bytes,
        wired_limit_bytes,
        weight_bytes,
        runtime_headroom_bytes,
        static_state_bytes: 0,
        kv_budget_bytes: memory_limit_bytes.saturating_sub(fixed_bytes),
        kv_bytes_per_token: 0,
        requested_total_pages: 0,
        planned_total_pages: 0,
        capacity_tokens: usize::MAX,
        clamped: false,
    })
}

fn resolve_memory_limit(
    explicit_budget: Option<usize>,
    explicit_reserve: Option<usize>,
    low_impact: bool,
    total_memory_bytes: Option<usize>,
    available_memory_bytes: Option<usize>,
    recommended_max_working_set_bytes: Option<usize>,
    system_status_line: &str,
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
                "Metal system reserve {} GiB is >= total memory {} GiB; {system_status_line}",
                reserve / GIB,
                total / GIB,
            );
            anyhow::ensure!(
                budget <= total - reserve,
                "--memory-budget-bytes {} GiB exceeds physical budget after system reserve {} GiB; {system_status_line}",
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
                "Metal resource guard rejected startup: {system_status_line}; available memory {} GiB is <= anti-swap reserve {} GiB. \
                 macOS would likely compress/swap to SSD under model load; close memory-heavy apps or use a smaller model.",
                available / GIB,
                reserve / GIB,
            );
            // The rejection message tells the operator to pass this flag "after
            // verifying headroom", so an explicit budget overrides the
            // available-memory heuristic; the physical bound above still holds.
            if budget > available - reserve {
                log::warn!(
                    "Metal resource guard: --memory-budget-bytes {} GiB is above the anti-swap \
                     budget {} GiB (available memory minus reserve). Honouring the explicit \
                     budget; macOS may compress or swap under load. {system_status_line}",
                    budget / GIB,
                    (available - reserve) / GIB,
                );
            }
        }
        return Ok(budget);
    }

    let mut candidates = Vec::new();
    if let Some(working_set) = recommended_max_working_set_bytes {
        candidates.push(working_set);
    }
    if let Some(total) = total_memory_bytes {
        let reserve =
            explicit_reserve.unwrap_or_else(|| default_system_reserve_bytes(total, low_impact));
        anyhow::ensure!(
            reserve < total,
            "Metal system reserve {} GiB is >= total memory {} GiB; {system_status_line}",
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
            "Metal resource guard rejected startup: {system_status_line}; available memory {} GiB is <= anti-swap reserve {} GiB. \
             macOS would likely compress/swap to SSD under model load; close memory-heavy apps or use a smaller model.",
            available / GIB,
            reserve / GIB,
        );
        candidates.push(available - reserve);
    }
    candidates.into_iter().min().ok_or_else(|| {
        anyhow::anyhow!(
            "Metal resource guard could not determine a memory budget; {system_status_line}"
        )
    })
}

fn format_gib(bytes: Option<usize>) -> String {
    bytes.map_or_else(
        || "unknown".to_string(),
        |bytes| format!("{:.1}GiB", bytes as f64 / GIB as f64),
    )
}

fn format_mib(bytes: Option<usize>) -> String {
    bytes.map_or_else(
        || "unknown".to_string(),
        |bytes| format!("{}MiB", bytes / MIB),
    )
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
        MetalKvCacheDtype::Bf16 => 2usize
            .checked_mul(nkv)
            .and_then(|v| v.checked_mul(hd))
            .and_then(|v| v.checked_mul(2)),
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
    let text = vm_stat_text()?;
    let pages = ["Pages free", "Pages inactive", "Pages speculative"]
        .into_iter()
        .filter_map(|key| parse_vm_stat_pages(&text, key))
        .sum::<usize>();
    (pages > 0).then_some(pages.saturating_mul(page_size))
}

fn sample_paging_activity() -> Option<PagingActivity> {
    let page_size = page_size_bytes().unwrap_or(4096);
    let before = VmStatCounters::current()?;
    std::thread::sleep(Duration::from_millis(PAGING_SAMPLE_MILLIS));
    let after = VmStatCounters::current()?;
    Some(PagingActivity {
        pageouts_delta_bytes: after
            .pageouts
            .checked_sub(before.pageouts)
            .map(|pages| pages.saturating_mul(page_size)),
        swapouts_delta_bytes: after
            .swapouts
            .checked_sub(before.swapouts)
            .map(|pages| pages.saturating_mul(page_size)),
        sample_millis: PAGING_SAMPLE_MILLIS,
    })
}

#[derive(Debug, Clone, Copy)]
struct VmStatCounters {
    pageouts: usize,
    swapouts: usize,
}

impl VmStatCounters {
    fn current() -> Option<Self> {
        let text = vm_stat_text()?;
        Some(Self {
            pageouts: parse_vm_stat_pages(&text, "Pageouts")?,
            swapouts: parse_vm_stat_pages(&text, "Swapouts")?,
        })
    }
}

fn vm_stat_text() -> Option<String> {
    let output = Command::new("vm_stat").output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
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
