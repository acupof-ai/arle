use crate::hardware::{CompiledBackend, SystemInfo};

#[derive(Debug, Clone)]
pub(crate) struct CatalogEntry {
    pub(crate) hf_id: &'static str,
    pub(crate) display_name: &'static str,
    pub(crate) quantization: Option<&'static str>,
    pub(crate) size_gb: f64,
    pub(crate) min_memory_gb: f64,
    pub(crate) backends: &'static [CompiledBackend],
    pub(crate) implemented: bool,
    /// Set on the flagship picks ARLE leads with — a one-line reason shown in
    /// the picker (e.g. "best quality · spec decode"). `None` for the rest.
    pub(crate) recommended: Option<&'static str>,
}

impl CatalogEntry {
    pub(crate) fn fits(&self, info: &SystemInfo) -> bool {
        self.implemented
            && self.backends.contains(&info.compiled_backend)
            && self.min_memory_gb <= info.effective_memory_gb()
    }
}

use CompiledBackend::{Cpu, Cuda, Metal};

// Consumed only by the backend-gated `ocr` module — gate to match so a
// no-backend build doesn't flag it dead.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) const DEEPSEEK_OCR_MODEL_ID: &str = "sahilchachra/unlimited-ocr-mxfp8-mlx";

/// Display order is decided by `recommend_models` (flagship picks first), not
/// by position here.
pub(crate) const CATALOG: &[CatalogEntry] = &[
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
    // 73% fewer tokens than base Qwen3.6-27B-FP8 at identical agentic reward
    // (5/5 greedy); card claims "50% fewer thinking tokens, preserved quality".
    CatalogEntry {
        hf_id: "bottlecapai/ThinkingCap-Qwen3.6-27B-FP8",
        display_name: "ThinkingCap 27B",
        quantization: Some("FP8"),
        size_gb: 29.0,
        min_memory_gb: 32.0,
        backends: &[Cuda],
        implemented: true,
        recommended: Some("best agentic · 73% fewer tokens"),
    },
];

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

/// Backend-gated to match `model_picker`, its sole non-test caller.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) fn find_by_hf_id(hf_id: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|e| e.hf_id == hf_id)
}
