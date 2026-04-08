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
use anyhow::{Context, Result, bail};

#[cfg(feature = "cuda")]
use crate::model::{GLM4Model, ModelForward, ModelRuntimeConfig, Qwen3Model, Qwen35Model};
#[cfg(feature = "cuda")]
use crate::model_registry::{ModelArch, detect_arch};
#[cfg(feature = "cuda")]
use crate::scheduler::{Scheduler, SchedulerConfig, SchedulerHandle};
#[cfg(feature = "cuda")]
use crate::tokenizer::Tokenizer;

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelType {
    Qwen3,
    Qwen35,
    GLM4,
}

#[cfg(feature = "cuda")]
impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen3 => write!(f, "Qwen3"),
            Self::Qwen35 => write!(f, "Qwen3.5"),
            Self::GLM4 => write!(f, "GLM-4"),
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub struct EngineOptions {
    pub enable_cuda_graph: bool,
}

#[cfg(feature = "cuda")]
impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct ServerRuntimeConfig {
    pub engine: EngineOptions,
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
            engine: EngineOptions::default(),
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
        ModelArch::GLM4 => Ok(ModelType::GLM4),
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
    GLM4(ModelComponents<GLM4Model>),
}

#[cfg(feature = "cuda")]
fn load_model_with<M>(
    model_path: &str,
    options: EngineOptions,
    load_model: impl FnOnce(&str, EngineOptions) -> Result<M>,
) -> Result<ModelComponents<M>> {
    let resolved = resolve_model_path_for_runtime(model_path)?;
    let resolved_str = resolved.to_str().unwrap_or(model_path);
    let tokenizer = Tokenizer::from_file(resolved_str)?;
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
    options: EngineOptions,
) -> Result<ModelComponents<Qwen3Model>> {
    load_model_with(model_path, options, |model_path, options| {
        Qwen3Model::from_safetensors_with_runtime(
            model_path,
            ModelRuntimeConfig {
                enable_cuda_graph: options.enable_cuda_graph,
            },
        )
    })
}

#[cfg(feature = "cuda")]
pub fn load_qwen35_components(
    model_path: &str,
    options: EngineOptions,
) -> Result<ModelComponents<Qwen35Model>> {
    load_model_with(model_path, options, |model_path, options| {
        Qwen35Model::from_safetensors_with_options(model_path, options.enable_cuda_graph)
    })
}

#[cfg(feature = "cuda")]
pub fn load_glm4_components(
    model_path: &str,
    options: EngineOptions,
) -> Result<ModelComponents<GLM4Model>> {
    load_model_with(model_path, options, |model_path, options| {
        GLM4Model::from_safetensors(model_path, options.enable_cuda_graph)
    })
}

#[cfg(feature = "cuda")]
pub fn load_model_components(
    model_path: &str,
    options: EngineOptions,
) -> Result<LoadedModelComponents> {
    match detect_model_type(model_path)? {
        ModelType::Qwen3 => Ok(LoadedModelComponents::Qwen3(load_qwen3_components(
            model_path, options,
        )?)),
        ModelType::Qwen35 => Ok(LoadedModelComponents::Qwen35(load_qwen35_components(
            model_path, options,
        )?)),
        ModelType::GLM4 => Ok(LoadedModelComponents::GLM4(load_glm4_components(
            model_path, options,
        )?)),
    }
}

#[cfg(feature = "cuda")]
pub fn spawn_scheduler_handle(
    components: LoadedModelComponents,
    runtime: ServerRuntimeConfig,
) -> Result<SchedulerHandle> {
    match components {
        LoadedModelComponents::Qwen3(components) => spawn_scheduler_for_model(components, runtime),
        LoadedModelComponents::Qwen35(components) => spawn_scheduler_for_model(components, runtime),
        LoadedModelComponents::GLM4(components) => spawn_scheduler_for_model(components, runtime),
    }
}

#[cfg(feature = "cuda")]
pub fn spawn_scheduler_handle_from_path(
    model_path: &str,
    runtime: ServerRuntimeConfig,
) -> Result<SchedulerHandle> {
    let components = load_model_components(model_path, runtime.engine)?;
    spawn_scheduler_handle(components, runtime)
}

#[cfg(feature = "cuda")]
fn spawn_scheduler_for_model<M: ModelForward + 'static>(
    components: ModelComponents<M>,
    runtime: ServerRuntimeConfig,
) -> Result<SchedulerHandle> {
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
        scheduler,
        max_seq_len,
        kv_cache_dtype,
        kv_pool_format,
    )?;
    std::thread::spawn(move || scheduler.run());
    Ok(handle)
}
