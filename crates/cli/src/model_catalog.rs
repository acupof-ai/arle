//! Curated model catalog with hardware requirements and metadata.
//!
//! Provides instant (no-network) model recommendations based on detected
//! hardware. Extended by live HuggingFace search for discovery beyond the
//! curated set.

use crate::hardware::{CompiledBackend, SystemInfo};

/// A curated model entry with download size and hardware requirements.
#[derive(Debug, Clone)]
pub(crate) struct CatalogEntry {
    pub(crate) hf_id: &'static str,
    pub(crate) display_name: &'static str,
    #[allow(dead_code)]
    pub(crate) family: &'static str,
    #[allow(dead_code)]
    pub(crate) param_count: &'static str,
    pub(crate) quantization: Option<&'static str>,
    /// Approximate download size in GB.
    pub(crate) size_gb: f64,
    /// Minimum memory (RAM/VRAM) needed to load the model.
    pub(crate) min_memory_gb: f64,
    /// Which backends can run this model.
    pub(crate) backends: &'static [CompiledBackend],
    /// Whether ARLE has a working implementation for this arch.
    pub(crate) implemented: bool,
}

impl CatalogEntry {
    /// Whether this entry is runnable on the given system.
    pub(crate) fn fits(&self, info: &SystemInfo) -> bool {
        self.implemented
            && self.backends.contains(&info.compiled_backend)
            && self.min_memory_gb <= info.effective_memory_gb()
    }
}

use CompiledBackend::{Cpu, Cuda, Metal};

/// Curated catalog of recommended models — ordered by preference (smallest
/// viable first for each family).
pub(crate) const CATALOG: &[CatalogEntry] = &[
    // ── MLX quantized (Metal-optimized) ──────────────────────────────────
    CatalogEntry {
        hf_id: "mlx-community/Qwen3-0.6B-4bit",
        display_name: "Qwen3 0.6B",
        family: "Qwen3",
        param_count: "0.6B",
        quantization: Some("4-bit"),
        size_gb: 0.5,
        min_memory_gb: 1.0,
        backends: &[Metal],
        implemented: true,
    },
    CatalogEntry {
        hf_id: "mlx-community/Qwen3-0.6B-bf16",
        display_name: "Qwen3 0.6B",
        family: "Qwen3",
        param_count: "0.6B",
        quantization: Some("bf16"),
        size_gb: 1.2,
        min_memory_gb: 2.0,
        backends: &[Metal],
        implemented: true,
    },
    // ── Full-precision (CUDA + CPU) ──────────────────────────────────────
    CatalogEntry {
        hf_id: "Qwen/Qwen3-0.6B",
        display_name: "Qwen3 0.6B",
        family: "Qwen3",
        param_count: "0.6B",
        quantization: None,
        size_gb: 1.6,
        min_memory_gb: 2.5,
        backends: &[Cuda, Metal, Cpu],
        implemented: true,
    },
    CatalogEntry {
        hf_id: "Qwen/Qwen3-4B",
        display_name: "Qwen3 4B",
        family: "Qwen3",
        param_count: "4B",
        quantization: None,
        size_gb: 9.4,
        min_memory_gb: 10.0,
        backends: &[Cuda, Metal, Cpu],
        implemented: true,
    },
    CatalogEntry {
        hf_id: "Qwen/Qwen3-8B",
        display_name: "Qwen3 8B",
        family: "Qwen3",
        param_count: "8B",
        quantization: None,
        size_gb: 17.0,
        min_memory_gb: 18.0,
        backends: &[Cuda, Metal],
        implemented: true,
    },
    CatalogEntry {
        hf_id: "Qwen/Qwen3.5-4B",
        display_name: "Qwen3.5 4B",
        family: "Qwen3.5",
        param_count: "4B",
        quantization: None,
        size_gb: 9.8,
        min_memory_gb: 10.5,
        backends: &[Cuda, Metal],
        implemented: true,
    },
    CatalogEntry {
        hf_id: "THUDM/glm-4-9b-chat",
        display_name: "GLM-4 9B",
        family: "GLM4",
        param_count: "9B",
        quantization: None,
        size_gb: 18.0,
        min_memory_gb: 19.0,
        backends: &[Cuda],
        implemented: true,
    },
    // ── Community quantized (popular picks) ──────────────────────────────
    CatalogEntry {
        hf_id: "mlx-community/Qwen3-4B-4bit",
        display_name: "Qwen3 4B",
        family: "Qwen3",
        param_count: "4B",
        quantization: Some("4-bit"),
        size_gb: 2.8,
        min_memory_gb: 4.0,
        backends: &[Metal],
        implemented: true,
    },
    CatalogEntry {
        hf_id: "mlx-community/Qwen3-8B-4bit",
        display_name: "Qwen3 8B",
        family: "Qwen3",
        param_count: "8B",
        quantization: Some("4-bit"),
        size_gb: 5.0,
        min_memory_gb: 6.0,
        backends: &[Metal],
        implemented: true,
    },
    // ── Qwen3.5 / Qwen3.6 Mixture-of-Experts (Metal-only, Phase 1). ──────
    CatalogEntry {
        hf_id: "mlx-community/Qwen3.6-35B-A3B-4bit",
        display_name: "Qwen3.6 35B-A3B",
        family: "Qwen3.5-MoE",
        param_count: "35B",
        quantization: Some("4-bit"),
        size_gb: 20.4,
        min_memory_gb: 24.0,
        backends: &[Metal],
        implemented: true,
    },
];

/// Return catalog entries that can run on this system, sorted by quality
/// (larger models first among those that fit).
pub(crate) fn recommend_models(info: &SystemInfo) -> Vec<&'static CatalogEntry> {
    let mut fits: Vec<&CatalogEntry> = CATALOG.iter().filter(|e| e.fits(info)).collect();
    // Sort: larger param count first (better quality), then prefer quantized
    // (smaller download) when same param count.
    fits.sort_by(|a, b| {
        b.min_memory_gb
            .partial_cmp(&a.min_memory_gb)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{GpuInfo, SystemInfo};

    fn make_info(backend: CompiledBackend, memory_gb: f64) -> SystemInfo {
        let gpu = match backend {
            CompiledBackend::Metal => GpuInfo::Metal {
                chip: "Apple M4".to_string(),
                unified_memory_gb: memory_gb,
            },
            CompiledBackend::Cuda => GpuInfo::Cuda {
                name: "L4".to_string(),
                vram_gb: memory_gb,
            },
            CompiledBackend::Cpu => GpuInfo::None,
            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "cpu")))]
            CompiledBackend::None => GpuInfo::None,
        };
        SystemInfo {
            cpu_name: "test".to_string(),
            cpu_cores: 8,
            total_ram_gb: memory_gb,
            available_ram_gb: memory_gb,
            gpu,
            compiled_backend: backend,
        }
    }

    #[test]
    fn small_metal_system_gets_small_models() {
        let info = make_info(CompiledBackend::Metal, 8.0);
        let recs = recommend_models(&info);
        assert!(!recs.is_empty());
        // Should not recommend 8B full-precision (needs 18 GB)
        assert!(recs.iter().all(|e| e.min_memory_gb <= 8.0 * 0.75));
    }

    #[test]
    fn large_cuda_system_includes_big_models() {
        let info = make_info(CompiledBackend::Cuda, 24.0);
        let recs = recommend_models(&info);
        assert!(recs.iter().any(|e| e.param_count == "8B"));
    }

    #[test]
    fn cpu_backend_excludes_gpu_only_models() {
        let info = make_info(CompiledBackend::Cpu, 32.0);
        let recs = recommend_models(&info);
        // GLM-4 is CUDA-only
        assert!(recs.iter().all(|e| e.family != "GLM4"));
    }

    #[test]
    fn catalog_entries_have_valid_data() {
        for entry in CATALOG {
            assert!(!entry.hf_id.is_empty());
            assert!(!entry.display_name.is_empty());
            assert!(entry.size_gb > 0.0);
            assert!(entry.min_memory_gb > 0.0);
            assert!(!entry.backends.is_empty());
        }
    }
}
