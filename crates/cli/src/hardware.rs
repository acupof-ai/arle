//! GPU detection uses subprocess calls / sysctl, not the CUDA/Metal runtimes,
//! to keep startup light.

use std::process::Command;

use sysinfo::System;

#[derive(Debug, Clone)]
pub(crate) enum GpuInfo {
    Cuda {
        name: String,
        vram_gb: f64,
    },
    Metal {
        chip: String,
        unified_memory_gb: f64,
        recommended_working_set_gb: Option<f64>,
    },
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompiledBackend {
    Cuda,
    Metal,
    #[cfg(feature = "hip")]
    Hip,
    #[cfg(feature = "vulkan")]
    Vulkan,
    Cpu,
    #[cfg(not(any(
        feature = "cuda",
        feature = "metal",
        feature = "hip",
        feature = "vulkan",
        feature = "cpu"
    )))]
    None,
}

impl CompiledBackend {
    #[allow(clippy::needless_return)] // cfg arms are additive: cuda+metal both active needs explicit returns.
    pub(crate) fn detect() -> Self {
        #[cfg(feature = "cuda")]
        {
            return Self::Cuda;
        }
        #[cfg(all(not(feature = "cuda"), feature = "metal"))]
        {
            return Self::Metal;
        }
        #[cfg(all(not(feature = "cuda"), not(feature = "metal"), feature = "hip"))]
        {
            return Self::Hip;
        }
        #[cfg(all(
            not(feature = "cuda"),
            not(feature = "metal"),
            not(feature = "hip"),
            feature = "vulkan"
        ))]
        {
            return Self::Vulkan;
        }
        #[cfg(all(
            not(feature = "cuda"),
            not(feature = "metal"),
            not(feature = "hip"),
            not(feature = "vulkan"),
            feature = "cpu"
        ))]
        {
            return Self::Cpu;
        }
        #[cfg(not(any(
            feature = "cuda",
            feature = "metal",
            feature = "hip",
            feature = "vulkan",
            feature = "cpu"
        )))]
        {
            Self::None
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            #[cfg(feature = "hip")]
            Self::Hip => "hip",
            #[cfg(feature = "vulkan")]
            Self::Vulkan => "vulkan",
            Self::Cpu => "cpu",
            #[cfg(not(any(
                feature = "cuda",
                feature = "metal",
                feature = "hip",
                feature = "vulkan",
                feature = "cpu"
            )))]
            Self::None => "none",
        }
    }

    pub(crate) fn supports_inference(self) -> bool {
        let _ = self;
        #[cfg(any(
            feature = "cuda",
            feature = "metal",
            feature = "hip",
            feature = "vulkan",
            feature = "cpu"
        ))]
        {
            true
        }
        #[cfg(not(any(
            feature = "cuda",
            feature = "metal",
            feature = "hip",
            feature = "vulkan",
            feature = "cpu"
        )))]
        {
            false
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SystemInfo {
    pub(crate) cpu_name: String,
    pub(crate) cpu_cores: usize,
    pub(crate) total_ram_gb: f64,
    pub(crate) available_ram_gb: f64,
    pub(crate) gpu: GpuInfo,
    pub(crate) compiled_backend: CompiledBackend,
}

impl SystemInfo {
    /// Keyed to the backend compiled into this binary, not just the host
    /// accelerator.
    pub(crate) fn effective_memory_gb(&self) -> f64 {
        match self.compiled_backend {
            CompiledBackend::Cuda => match &self.gpu {
                GpuInfo::Cuda { vram_gb, .. } => *vram_gb,
                _ => 0.0,
            },
            CompiledBackend::Metal => match &self.gpu {
                GpuInfo::Metal {
                    unified_memory_gb,
                    recommended_working_set_gb,
                    ..
                } => {
                    let working_set =
                        recommended_working_set_gb.unwrap_or(*unified_memory_gb * 0.75);
                    let physical_budget = metal_physical_budget_gb(*unified_memory_gb);
                    let available_budget =
                        metal_available_budget_gb(*unified_memory_gb, self.available_ram_gb);
                    working_set.min(physical_budget).min(available_budget)
                }
                _ => 0.0,
            },
            // No host AMD-GPU probe yet (the catalog has no HIP entries either);
            // report 0 rather than a fictional VRAM figure.
            #[cfg(feature = "hip")]
            CompiledBackend::Hip => 0.0,
            // No generic Vulkan VRAM probe yet; report 0 rather than a
            // fictional figure.
            #[cfg(feature = "vulkan")]
            CompiledBackend::Vulkan => 0.0,
            CompiledBackend::Cpu => {
                if self.available_ram_gb > 0.0 {
                    self.available_ram_gb
                } else {
                    self.total_ram_gb * 0.75
                }
            }
            #[cfg(not(any(
                feature = "cuda",
                feature = "metal",
                feature = "hip",
                feature = "vulkan",
                feature = "cpu"
            )))]
            CompiledBackend::None => 0.0,
        }
    }
}

/// Best-effort; never panics.
pub(crate) fn detect_system() -> SystemInfo {
    let mut sys = System::new();
    sys.refresh_cpu_all();
    sys.refresh_memory();

    let cpu_name = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let cpu_cores = sys.cpus().len();
    let total_ram_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    let available_ram_gb = sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

    let compiled_backend = CompiledBackend::detect();
    let gpu = detect_gpu(total_ram_gb);

    SystemInfo {
        cpu_name,
        cpu_cores,
        total_ram_gb,
        available_ram_gb,
        gpu,
        compiled_backend,
    }
}

fn detect_gpu(total_ram_gb: f64) -> GpuInfo {
    let nvidia = detect_nvidia_gpu();
    if !matches!(nvidia, GpuInfo::None) {
        return nvidia;
    }

    detect_apple_gpu(total_ram_gb)
}

fn detect_nvidia_gpu() -> GpuInfo {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let line = stdout.trim();
            if let Some((name, vram_str)) = line.split_once(',') {
                let vram_mb: f64 = vram_str.trim().parse().unwrap_or(0.0);
                return GpuInfo::Cuda {
                    name: name.trim().to_string(),
                    vram_gb: vram_mb / 1024.0,
                };
            }
            GpuInfo::None
        }
        _ => GpuInfo::None,
    }
}

fn detect_apple_gpu(total_ram_gb: f64) -> GpuInfo {
    if !cfg!(target_os = "macos") {
        return GpuInfo::None;
    }

    let output = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output();

    let chip = match output {
        Ok(out) if out.status.success() => {
            let raw = String::from_utf8_lossy(&out.stdout);
            raw.trim().to_string()
        }
        _ => "Apple Silicon".to_string(),
    };

    if chip.contains("Apple") {
        GpuInfo::Metal {
            chip,
            unified_memory_gb: total_ram_gb,
            recommended_working_set_gb: recommended_metal_working_set_gb(),
        }
    } else {
        GpuInfo::None
    }
}

#[cfg(feature = "metal")]
fn recommended_metal_working_set_gb() -> Option<f64> {
    infer_api::metal_recommended_max_working_set_size_bytes()
        .map(|bytes| bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

#[cfg(not(feature = "metal"))]
fn recommended_metal_working_set_gb() -> Option<f64> {
    None
}

fn metal_physical_budget_gb(total_memory_gb: f64) -> f64 {
    let reserve = (total_memory_gb / 4.0).max(14.0).min(total_memory_gb - 1.0);
    (total_memory_gb - reserve).max(0.0)
}

fn metal_available_budget_gb(total_memory_gb: f64, available_memory_gb: f64) -> f64 {
    let reserve = if total_memory_gb >= 32.0 { 8.0 } else { 6.0 };
    (available_memory_gb - reserve).max(0.0)
}
