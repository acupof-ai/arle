//! Shared model bootstrap and runtime factory helpers.
//!
//! This module owns model discovery plus the common "load tokenizer + weights +
//! runtime options" path used by both the server entrypoint and direct engine
//! constructors.

#[cfg(feature = "cuda")]
use std::fmt;
#[cfg(feature = "cuda")]
use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::thread::JoinHandle;

#[cfg(feature = "cuda")]
use anyhow::{Context, Result, bail};
#[cfg(feature = "cuda")]
use log::{info, warn};

#[cfg(feature = "cuda")]
use crate::model::{ModelForward, ModelRuntimeConfig, Qwen3Model, Qwen35Model};
#[cfg(feature = "cuda")]
use crate::model_registry::{ModelArch, detect_arch};
#[cfg(feature = "cuda")]
use crate::model_source::ResolvedModelSource;
#[cfg(feature = "cuda")]
use crate::scheduler::{Scheduler, SchedulerConfig, SchedulerHandle};
#[cfg(feature = "cuda")]
use crate::tokenizer::Tokenizer;

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelType {
    Qwen3,
    Qwen35,
    /// Qwen3.5 Mixture-of-Experts (Qwen3.6-35B-A3B). CUDA path is a
    /// `todo!()` stub until the CUDA MoE kernel lands; Metal path lives
    /// entirely outside this module. See `docs/plans/qwen36-moe-metal.md`.
    Qwen35Moe,
}

#[cfg(feature = "cuda")]
impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen3 => write!(f, "Qwen3"),
            Self::Qwen35 => write!(f, "Qwen3.5"),
            Self::Qwen35Moe => write!(f, "Qwen3.5-MoE"),
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub struct InferenceEngineOptions {
    pub enable_cuda_graph: bool,
}

#[cfg(feature = "cuda")]
impl Default for InferenceEngineOptions {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct ServerRuntimeConfig {
    pub engine: InferenceEngineOptions,
    pub scheduler: SchedulerConfig,
    pub seed: u64,
    pub max_seq_len: Option<usize>,
    /// KV cache quantization dtype for contiguous cache (single-request path).
    pub kv_cache_dtype: crate::model::kv_cache::KVCacheDtype,
    /// KV pool storage format (paged pool). Determines attention dispatch.
    pub kv_pool_format: crate::model::kv_cache::KVFormat,
}

#[cfg(feature = "cuda")]
impl Default for ServerRuntimeConfig {
    fn default() -> Self {
        Self {
            engine: InferenceEngineOptions::default(),
            scheduler: SchedulerConfig::runtime_defaults(4),
            seed: 42,
            max_seq_len: None,
            kv_cache_dtype: crate::model::kv_cache::KVCacheDtype::BF16,
            kv_pool_format: crate::model::kv_cache::KVFormat::BF16,
        }
    }
}

#[cfg(feature = "cuda")]
pub fn model_id_from_path(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string()
}

#[cfg(feature = "cuda")]
pub fn detect_model_type(model_path: &str) -> Result<ModelType> {
    let resolved = resolve_model_path_for_runtime(model_path)?;
    match detect_arch(resolved.to_str().unwrap_or(model_path))? {
        ModelArch::Qwen3 => Ok(ModelType::Qwen3),
        ModelArch::Qwen35 => Ok(ModelType::Qwen35),
        ModelArch::Qwen3_5_Moe => Ok(ModelType::Qwen35Moe),
        arch => bail!("model architecture {arch:?} is not supported by the runtime yet"),
    }
}

#[cfg(feature = "cuda")]
fn resolve_model_path_for_runtime(model_path: &str) -> Result<PathBuf> {
    crate::hf_hub::resolve_model_path(model_path)
        .with_context(|| format!("failed to resolve model '{model_path}'"))
}

#[cfg(feature = "cuda")]
pub struct ModelComponents<M> {
    pub model_id: String,
    pub tokenizer: Tokenizer,
    pub model: M,
}

#[cfg(feature = "cuda")]
pub enum LoadedModelComponents {
    Qwen3(ModelComponents<Qwen3Model>),
    Qwen35(ModelComponents<Qwen35Model>),
    /// Qwen3.5 MoE shares the `Qwen35Model` component type for now; the
    /// MoE-specific dispatch happens at the engine layer. The CUDA loader
    /// for this variant is intentionally a `todo!()` stub.
    Qwen35Moe(ModelComponents<Qwen35Model>),
}

#[cfg(feature = "cuda")]
fn load_model_with<M>(
    model_path: &str,
    options: InferenceEngineOptions,
    load_model: impl FnOnce(&str, InferenceEngineOptions) -> Result<M>,
) -> Result<ModelComponents<M>> {
    let source = ResolvedModelSource::resolve(model_path)?;
    let resolved_str = source.resolved_path().to_str().unwrap_or(model_path);
    let tokenizer = source.load_tokenizer()?;
    let model = load_model(resolved_str, options)?;
    Ok(ModelComponents {
        model_id: model_id_from_path(model_path),
        tokenizer,
        model,
    })
}

#[cfg(feature = "cuda")]
pub fn load_qwen3_components(
    model_path: &str,
    options: InferenceEngineOptions,
) -> Result<ModelComponents<Qwen3Model>> {
    load_model_with(model_path, options, |model_path, options| {
        let model = Qwen3Model::from_safetensors_with_runtime(
            model_path,
            ModelRuntimeConfig {
                enable_cuda_graph: options.enable_cuda_graph,
            },
        )?;
        match std::env::var("INFER_LORA_PATH") {
            Ok(lora_path) if !lora_path.trim().is_empty() => {
                log::info!("Attaching LoRA adapter from {}", lora_path);
                model.load_and_attach_lora(&lora_path)
            }
            _ => Ok(model),
        }
    })
}

#[cfg(feature = "cuda")]
pub fn load_qwen35_components(
    model_path: &str,
    options: InferenceEngineOptions,
) -> Result<ModelComponents<Qwen35Model>> {
    load_model_with(model_path, options, |model_path, options| {
        Qwen35Model::from_safetensors_with_options(model_path, options.enable_cuda_graph)
    })
}

/// Qwen3.5 MoE (Qwen3.6-35B-A3B) CUDA loader stub.
///
/// The CUDA MoE forward path is not yet implemented. Metal has its own
/// code path that does not go through this function. We keep the symbol so
/// the CUDA dispatch table type-checks; attempting to actually load a MoE
/// model under CUDA panics with a clear message.
#[cfg(feature = "cuda")]
pub fn load_qwen35_moe_components(
    _model_path: &str,
    _options: InferenceEngineOptions,
) -> Result<ModelComponents<Qwen35Model>> {
    todo!("GPU required: Qwen3.6 CUDA not yet implemented")
}

#[cfg(feature = "cuda")]
pub fn load_model_components(
    model_path: &str,
    options: InferenceEngineOptions,
) -> Result<LoadedModelComponents> {
    match detect_model_type(model_path)? {
        ModelType::Qwen3 => Ok(LoadedModelComponents::Qwen3(load_qwen3_components(
            model_path, options,
        )?)),
        ModelType::Qwen35 => Ok(LoadedModelComponents::Qwen35(load_qwen35_components(
            model_path, options,
        )?)),
        ModelType::Qwen35Moe => Ok(LoadedModelComponents::Qwen35Moe(
            load_qwen35_moe_components(model_path, options)?,
        )),
    }
}

#[cfg(feature = "cuda")]
pub fn spawn_scheduler_handle(
    components: LoadedModelComponents,
    runtime: ServerRuntimeConfig,
    metrics: crate::metrics::ServerMetrics,
) -> Result<(SchedulerHandle, SchedulerRuntimeGuard)> {
    match components {
        LoadedModelComponents::Qwen3(components) => {
            spawn_scheduler_for_model(components, runtime, metrics)
        }
        LoadedModelComponents::Qwen35(components)
        | LoadedModelComponents::Qwen35Moe(components) => {
            spawn_scheduler_for_model(components, runtime, metrics)
        }
    }
}

#[cfg(feature = "cuda")]
pub fn spawn_scheduler_handle_from_path(
    model_path: &str,
    runtime: ServerRuntimeConfig,
    metrics: crate::metrics::ServerMetrics,
) -> Result<(SchedulerHandle, SchedulerRuntimeGuard)> {
    let components = load_model_components(model_path, runtime.engine)?;
    spawn_scheduler_handle(components, runtime, metrics)
}

#[cfg(feature = "cuda")]
pub struct SchedulerRuntimeGuard {
    model_id: String,
    thread: Option<JoinHandle<()>>,
}

#[cfg(feature = "cuda")]
impl SchedulerRuntimeGuard {
    fn new(model_id: String, thread: JoinHandle<()>) -> Self {
        Self {
            model_id,
            thread: Some(thread),
        }
    }

    pub fn wait(mut self) {
        self.join_inner();
    }

    fn join_inner(&mut self) {
        let Some(thread) = self.thread.take() else {
            return;
        };
        info!(
            "Waiting for scheduler thread to shut down cleanly (model={})",
            self.model_id
        );
        match thread.join() {
            Ok(()) => info!(
                "Scheduler thread shut down cleanly (model={})",
                self.model_id
            ),
            Err(_) => warn!(
                "Scheduler thread panicked during shutdown (model={})",
                self.model_id
            ),
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for SchedulerRuntimeGuard {
    fn drop(&mut self) {
        self.join_inner();
    }
}

#[cfg(feature = "cuda")]
fn spawn_scheduler_for_model<M: ModelForward + 'static>(
    components: ModelComponents<M>,
    runtime: ServerRuntimeConfig,
    metrics: crate::metrics::ServerMetrics,
) -> Result<(SchedulerHandle, SchedulerRuntimeGuard)> {
    let ModelComponents {
        model_id,
        tokenizer,
        model,
    } = components;

    let ServerRuntimeConfig {
        scheduler,
        seed,
        max_seq_len,
        kv_cache_dtype,
        kv_pool_format,
        ..
    } = runtime;

    let (scheduler, handle) = Scheduler::with_config(
        model,
        tokenizer,
        &model_id,
        seed,
        metrics,
        scheduler,
        max_seq_len,
        kv_cache_dtype,
        kv_pool_format,
    )?;
    let thread = std::thread::spawn(move || scheduler.run());
    Ok((handle, SchedulerRuntimeGuard::new(model_id, thread)))
}

#[cfg(all(feature = "cuda", test))]
mod tests {
    use super::SchedulerRuntimeGuard;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    #[test]
    fn scheduler_runtime_guard_joins_thread_on_drop() {
        let joined = Arc::new(AtomicBool::new(false));
        let joined_thread = Arc::clone(&joined);
        let thread = std::thread::spawn(move || {
            joined_thread.store(true, Ordering::SeqCst);
        });

        drop(SchedulerRuntimeGuard::new("test-model".to_string(), thread));

        assert!(joined.load(Ordering::SeqCst));
    }
}
