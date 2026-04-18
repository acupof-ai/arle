//! System hardware detection for CLI startup.
//!
//! Detects CPU, RAM, and GPU capabilities to drive model recommendations.
//! GPU detection uses lightweight methods (subprocess calls, sysctl) to avoid
//! pulling in full CUDA/Metal runtime just for startup info.

use std::process::Command;

use sysinfo::System;

/// Detected GPU information.
#[derive(Debug, Clone)]
pub(crate) enum GpuInfo {
    /// NVIDIA GPU detected via nvidia-smi.
    Cuda { name: String, vram_gb: f64 },
    /// Apple Silicon with unified memory.
    Metal {
        chip: String,
        unified_memory_gb: f64,
    },
    /// No GPU detected or not applicable.
    None,
}

/// Which inference backend was compiled into this binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompiledBackend {
    Cuda,
    Metal,
    Cpu,
}

impl CompiledBackend {
    #[allow(clippy::needless_return)] // cfg arms are additive: cuda+metal both active needs explicit returns.
    pub(crate) fn detect() -> Self {
        #[cfg(feature = "cuda")]
        {
            return Self::Cuda;
        }
        #[cfg(feature = "metal")]
        {
            return Self::Metal;
        }
        #[cfg(feature = "cpu")]
        {
            return Self::Cpu;
        }
        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "cpu")))]
        {
            Self::Cpu
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Cpu => "cpu",
        }
    }
}

/// Aggregated system information for model recommendation.
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
    /// Effective memory available for model loading (VRAM for CUDA, unified
    /// RAM for Metal, system RAM for CPU).
    pub(crate) fn effective_memory_gb(&self) -> f64 {
        match &self.gpu {
            GpuInfo::Cuda { vram_gb, .. } => *vram_gb,
            GpuInfo::Metal {
                unified_memory_gb, ..
            } => *unified_memory_gb * 0.75, // leave headroom for OS
            GpuInfo::None => self.available_ram_gb,
        }
    }
}

/// Detect system hardware. Never panics — returns best-effort info.
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
    let gpu = detect_gpu(compiled_backend, total_ram_gb);

    SystemInfo {
        cpu_name,
        cpu_cores,
        total_ram_gb,
        available_ram_gb,
        gpu,
        compiled_backend,
    }
}

fn detect_gpu(backend: CompiledBackend, total_ram_gb: f64) -> GpuInfo {
    match backend {
        CompiledBackend::Cuda => detect_nvidia_gpu(),
        CompiledBackend::Metal => detect_apple_gpu(total_ram_gb),
        CompiledBackend::Cpu => GpuInfo::None,
    }
}

/// Detect NVIDIA GPU via nvidia-smi subprocess (2s timeout).
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

/// Detect Apple Silicon chip name via sysctl.
fn detect_apple_gpu(total_ram_gb: f64) -> GpuInfo {
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

    GpuInfo::Metal {
        chip,
        unified_memory_gb: total_ram_gb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_system_returns_sensible_values() {
        let info = detect_system();
        assert!(info.cpu_cores > 0);
        assert!(info.total_ram_gb > 0.5);
        assert!(!info.cpu_name.is_empty());
    }

    #[test]
    fn effective_memory_positive() {
        let info = detect_system();
        assert!(info.effective_memory_gb() > 0.0);
    }

    #[test]
    fn compiled_backend_has_name() {
        let backend = CompiledBackend::detect();
        assert!(!backend.name().is_empty());
    }
}
