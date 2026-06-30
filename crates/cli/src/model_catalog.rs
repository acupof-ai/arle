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
    pub(crate) quantization: Option<&'static str>,
    /// Approximate download size in GB.
    pub(crate) size_gb: f64,
    /// Minimum memory (RAM/VRAM) needed to load the model.
    pub(crate) min_memory_gb: f64,
    /// Which backends can run this model.
    pub(crate) backends: &'static [CompiledBackend],
    /// Whether ARLE has a working implementation for this arch.
    pub(crate) implemented: bool,
    /// Set on the flagship picks ARLE leads with — a one-line reason shown in
    /// the picker (e.g. "best quality · spec decode"). `None` for the rest.
    pub(crate) recommended: Option<&'static str>,
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

/// Canonical DeepSeek-OCR model (MXFP8 MLX). Used by `arle ocr` and the agent
/// `ocr` tool as the default, auto-downloaded VLM. Metal-only.
// Consumed only by the backend-gated `ocr` module — gate to match so a
// no-backend build doesn't flag it dead.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) const DEEPSEEK_OCR_MODEL_ID: &str = "sahilchachra/unlimited-ocr-mxfp8-mlx";

/// Curated catalog of selectable models. Display order is decided by
/// `recommend_models` (flagship picks first), not by position here.
pub(crate) const CATALOG: &[CatalogEntry] = &[
    // ── MLX quantized (Metal-optimized) ──────────────────────────────────
    CatalogEntry {
        hf_id: "mlx-community/Qwen3-0.6B-4bit",
        display_name: "Qwen3 0.6B",
        quantization: Some("4-bit"),
        size_gb: 0.5,
        min_memory_gb: 1.0,
        backends: &[Metal],
        implemented: true,
        recommended: None,
    },
    CatalogEntry {
        hf_id: "mlx-community/Qwen3-0.6B-bf16",
        display_name: "Qwen3 0.6B",
        quantization: Some("bf16"),
        size_gb: 1.2,
        min_memory_gb: 2.0,
        backends: &[Metal],
        implemented: true,
        recommended: None,
    },
    // ── Full-precision (CUDA + CPU) ──────────────────────────────────────
    CatalogEntry {
        hf_id: "Qwen/Qwen3-0.6B",
        display_name: "Qwen3 0.6B",
        quantization: None,
        size_gb: 1.6,
        min_memory_gb: 2.5,
        backends: &[Cuda, Metal, Cpu],
        implemented: true,
        recommended: None,
    },
    CatalogEntry {
        hf_id: "Qwen/Qwen3-4B",
        display_name: "Qwen3 4B",
        quantization: None,
        size_gb: 9.4,
        min_memory_gb: 10.0,
        backends: &[Cuda, Metal, Cpu],
        implemented: true,
        recommended: None,
    },
    CatalogEntry {
        hf_id: "Qwen/Qwen3-8B",
        display_name: "Qwen3 8B",
        quantization: None,
        size_gb: 17.0,
        min_memory_gb: 18.0,
        backends: &[Cuda, Metal],
        implemented: true,
        recommended: None,
    },
    CatalogEntry {
        hf_id: "Qwen/Qwen3.5-4B",
        display_name: "Qwen3.5 4B",
        quantization: None,
        size_gb: 9.8,
        min_memory_gb: 10.5,
        backends: &[Cuda, Metal],
        implemented: true,
        recommended: None,
    },
    // ── Community quantized (popular picks) ──────────────────────────────
    CatalogEntry {
        hf_id: "mlx-community/Qwen3-4B-4bit",
        display_name: "Qwen3 4B",
        quantization: Some("4-bit"),
        size_gb: 2.8,
        min_memory_gb: 4.0,
        backends: &[Metal],
        implemented: true,
        recommended: None,
    },
    CatalogEntry {
        hf_id: "mlx-community/Qwen3-8B-4bit",
        display_name: "Qwen3 8B",
        quantization: Some("4-bit"),
        size_gb: 5.0,
        min_memory_gb: 6.0,
        backends: &[Metal],
        implemented: true,
        recommended: None,
    },
    // ── Qwen3.6 dense (GatedDeltaNet) — highest quality, self-speculative ─
    // OptiQ mixed 4/8-bit: PPL 7.82 (vs uniform-4bit 8.56). Its own NextN-MTP
    // head is auto-enabled for spec decode (~18 tok/s, past the bandwidth floor).
    CatalogEntry {
        hf_id: "mlx-community/Qwen3.6-27B-OptiQ-4bit",
        display_name: "Qwen3.6 27B",
        quantization: Some("OptiQ 4/8-bit"),
        size_gb: 19.0,
        min_memory_gb: 31.0,
        backends: &[Metal],
        implemented: true,
        recommended: Some("best quality · spec decode"),
    },
    // ── Qwen3.5 / Qwen3.6 Mixture-of-Experts (Metal-only, Phase 1). ──────
    CatalogEntry {
        hf_id: "mlx-community/Qwen3.6-35B-A3B-4bit",
        display_name: "Qwen3.6 35B-A3B",
        quantization: Some("4-bit"),
        size_gb: 20.4,
        min_memory_gb: 24.0,
        backends: &[Metal],
        implemented: true,
        recommended: Some("fastest · MoE"),
    },
];

/// Return catalog entries that can run on this system, best first: the
/// flagship `recommended` picks lead (so Enter lands on a great model),
/// then the rest by capacity (larger / higher quality first).
pub(crate) fn recommend_models(info: &SystemInfo) -> Vec<&'static CatalogEntry> {
    let mut fits: Vec<&CatalogEntry> = CATALOG.iter().filter(|e| e.fits(info)).collect();
    fits.sort_by(|a, b| {
        // Flagship picks first; within each group, larger memory (= higher
        // quality) first.
        b.recommended
            .is_some()
            .cmp(&a.recommended.is_some())
            .then_with(|| {
                b.min_memory_gb
                    .partial_cmp(&a.min_memory_gb)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    fits
}

/// Look up a catalog entry by its exact HuggingFace id. Lets the picker
/// surface a flagship `recommended` note on a model the user already has
/// on disk. Only reachable when a backend is compiled — `model_picker`, its
/// sole non-test caller, carries the same gate.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) fn find_by_hf_id(hf_id: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|e| e.hf_id == hf_id)
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
                recommended_working_set_gb: None,
            },
            CompiledBackend::Cuda => GpuInfo::Cuda {
                name: "L4".to_string(),
                vram_gb: memory_gb,
            },
            #[cfg(feature = "hip")]
            CompiledBackend::Hip => GpuInfo::None,
            #[cfg(feature = "vulkan")]
            CompiledBackend::Vulkan => GpuInfo::None,
            CompiledBackend::Cpu => GpuInfo::None,
            #[cfg(not(any(
                feature = "cuda",
                feature = "metal",
                feature = "hip",
                feature = "vulkan",
                feature = "cpu"
            )))]
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
        assert!(recs.iter().any(|e| e.display_name.contains("8B")));
    }

    #[test]
    fn cpu_backend_excludes_gpu_only_models() {
        let info = make_info(CompiledBackend::Cpu, 32.0);
        let recs = recommend_models(&info);
        assert!(
            recs.iter()
                .all(|e| e.backends.contains(&CompiledBackend::Cpu))
        );
    }

    #[test]
    fn catalog_entries_have_valid_data() {
        for entry in CATALOG {
            assert!(!entry.hf_id.is_empty());
            assert!(!entry.display_name.is_empty());
            assert!(entry.size_gb > 0.0);
            assert!(entry.min_memory_gb > 0.0);
            assert!(!entry.backends.is_empty());
            // A recommended note, when present, must be a non-empty one-liner.
            if let Some(note) = entry.recommended {
                assert!(!note.is_empty());
                assert!(!note.contains('\n'));
            }
        }
    }

    #[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
    #[test]
    fn flagship_picks_are_marked_recommended() {
        // The two models ARLE leads with must carry a one-line reason.
        for id in [
            "mlx-community/Qwen3.6-27B-OptiQ-4bit",
            "mlx-community/Qwen3.6-35B-A3B-4bit",
        ] {
            let entry = find_by_hf_id(id).expect("flagship in catalog");
            assert!(
                entry.recommended.is_some(),
                "{id} should be a recommended flagship"
            );
        }
    }

    #[test]
    fn recommend_lists_flagships_before_the_rest() {
        // A big Metal box fits both flagships and the smaller models; the
        // flagships must lead so Enter lands on a great model.
        let info = make_info(CompiledBackend::Metal, 64.0);
        let recs = recommend_models(&info);
        let first = recs.first().expect("at least one recommendation");
        assert!(
            first.recommended.is_some(),
            "first recommendation should be a flagship, got {}",
            first.hf_id
        );
    }
}
