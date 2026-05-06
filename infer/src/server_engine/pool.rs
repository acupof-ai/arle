use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use super::LoadedInferenceEngine;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnginePoolModelType {
    TextGeneration,
    Embedding,
    Reranker,
}

impl EnginePoolModelType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TextGeneration => "text-generation",
            Self::Embedding => "embedding",
            Self::Reranker => "reranker",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "text" | "llm" | "generation" | "text-generation" => Ok(Self::TextGeneration),
            "embedding" | "embed" => Ok(Self::Embedding),
            "reranker" | "rerank" => Ok(Self::Reranker),
            other => Err(format!(
                "unknown engine model type `{other}`; expected text-generation, embedding, or reranker"
            )),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnginePoolModelSpec {
    pub id: String,
    pub model_path: String,
    pub aliases: Vec<String>,
    pub model_type: EnginePoolModelType,
    pub pinned: bool,
    pub ttl: Option<Duration>,
    pub memory_estimate_bytes: Option<u64>,
}

impl EnginePoolModelSpec {
    pub fn text_generation(id: impl Into<String>, model_path: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            model_path: model_path.into(),
            aliases: Vec::new(),
            model_type: EnginePoolModelType::TextGeneration,
            pinned: false,
            ttl: None,
            memory_estimate_bytes: None,
        }
    }

    pub fn parse_cli(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err("pool model spec must not be empty".to_string());
        }

        let mut id: Option<String> = None;
        let mut model_path: Option<String> = None;
        let mut aliases: Vec<String> = Vec::new();
        let mut model_type = EnginePoolModelType::TextGeneration;
        let mut pinned = false;
        let mut ttl: Option<Duration> = None;
        let mut memory_estimate_bytes: Option<u64> = None;

        for (part_idx, part) in trimmed.split(',').map(str::trim).enumerate() {
            if part.is_empty() {
                continue;
            }
            let (key, value) = part.split_once('=').ok_or_else(|| {
                format!("invalid pool model spec segment `{part}`; expected key=value")
            })?;
            let key = key.trim();
            let value = value.trim();
            if value.is_empty() {
                return Err(format!("pool model spec field `{key}` must not be empty"));
            }
            match key {
                "id" | "model" => id = Some(value.to_string()),
                "path" | "model_path" => model_path = Some(value.to_string()),
                "type" | "kind" | "model_type" => model_type = EnginePoolModelType::parse(value)?,
                "alias" | "aliases" => {
                    aliases.extend(
                        value
                            .split(['|', ';'])
                            .map(str::trim)
                            .filter(|alias| !alias.is_empty())
                            .map(ToOwned::to_owned),
                    );
                }
                "pin" | "pinned" => {
                    pinned = match value {
                        "1" | "true" | "yes" | "on" => true,
                        "0" | "false" | "no" | "off" => false,
                        _ => {
                            return Err(format!(
                                "pool model pinned value must be true/false, got `{value}`"
                            ));
                        }
                    };
                }
                "ttl" | "ttl_secs" => {
                    let secs = value
                        .parse::<u64>()
                        .map_err(|_| format!("ttl_secs must be an integer, got `{value}`"))?;
                    ttl = Some(Duration::from_secs(secs));
                }
                "memory" | "memory_bytes" | "estimate_bytes" => {
                    memory_estimate_bytes =
                        Some(value.parse::<u64>().map_err(|_| {
                            format!("memory_bytes must be an integer, got `{value}`")
                        })?);
                }
                _ if part_idx == 0 && id.is_none() && model_path.is_none() => {
                    id = Some(key.to_string());
                    model_path = Some(value.to_string());
                }
                _ => return Err(format!("unknown pool model spec field `{key}`")),
            }
        }

        let model_path = model_path.ok_or_else(|| {
            "pool model spec requires `path=...` or shorthand `id=path`".to_string()
        })?;
        let id = id.unwrap_or_else(|| model_id_from_path(&model_path));
        validate_model_name("id", &id)?;
        for alias in &aliases {
            validate_model_name("alias", alias)?;
        }

        Ok(Self {
            id,
            model_path,
            aliases,
            model_type,
            pinned,
            ttl,
            memory_estimate_bytes,
        })
    }

    pub fn all_names(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.id.as_str()).chain(self.aliases.iter().map(String::as_str))
    }
}

#[derive(Clone, Debug, Default)]
pub struct EnginePoolConfig {
    pub models: Vec<EnginePoolModelSpec>,
    pub memory_budget_bytes: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnginePoolModelInfo {
    pub id: String,
    pub aliases: Vec<String>,
    pub model_type: EnginePoolModelType,
    pub loaded: bool,
    pub pinned: bool,
    pub active_requests: usize,
    pub memory_estimate_bytes: Option<u64>,
}

struct EnginePoolEntry<E> {
    engine: Arc<Mutex<E>>,
    spec: EnginePoolModelSpec,
    last_used: Arc<Mutex<Instant>>,
    active_requests: Arc<AtomicUsize>,
}

pub struct EnginePool<E> {
    specs: HashMap<String, EnginePoolModelSpec>,
    aliases: HashMap<String, String>,
    entries: HashMap<String, EnginePoolEntry<E>>,
    memory_budget_bytes: Option<u64>,
    loaded_estimate_bytes: u64,
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub type LoadedEnginePool = EnginePool<LoadedInferenceEngine>;

pub struct EngineLease<E> {
    model_id: String,
    engine: Arc<Mutex<E>>,
    last_used: Arc<Mutex<Instant>>,
    active_requests: Arc<AtomicUsize>,
}

impl<E> EngineLease<E> {
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn lock(&self) -> MutexGuard<'_, E> {
        self.engine.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<E> Drop for EngineLease<E> {
    fn drop(&mut self) {
        *self
            .last_used
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Instant::now();
        self.active_requests.fetch_sub(1, Ordering::AcqRel);
    }
}

impl<E> EnginePool<E> {
    pub fn new(config: EnginePoolConfig) -> Result<Self> {
        let mut specs = HashMap::new();
        let mut aliases = HashMap::new();
        let mut seen = HashSet::new();
        for spec in config.models {
            for name in spec.all_names() {
                ensure!(
                    seen.insert(name.to_string()),
                    "duplicate engine pool model id or alias `{name}`"
                );
            }
            for alias in &spec.aliases {
                aliases.insert(alias.clone(), spec.id.clone());
            }
            specs.insert(spec.id.clone(), spec);
        }
        Ok(Self {
            specs,
            aliases,
            entries: HashMap::new(),
            memory_budget_bytes: config.memory_budget_bytes,
            loaded_estimate_bytes: 0,
        })
    }

    pub fn configured_models(&self) -> Vec<EnginePoolModelInfo> {
        let mut models: Vec<_> = self
            .specs
            .values()
            .map(|spec| {
                let entry = self.entries.get(&spec.id);
                EnginePoolModelInfo {
                    id: spec.id.clone(),
                    aliases: spec.aliases.clone(),
                    model_type: spec.model_type,
                    loaded: entry.is_some(),
                    pinned: spec.pinned,
                    active_requests: entry
                        .map_or(0, |entry| entry.active_requests.load(Ordering::Acquire)),
                    memory_estimate_bytes: spec.memory_estimate_bytes,
                }
            })
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }

    pub fn resolve_model_id(&self, model: &str) -> Option<&str> {
        let model = model.trim();
        if let Some((id, _)) = self.specs.get_key_value(model) {
            return Some(id.as_str());
        }
        self.aliases.get(model).map(String::as_str)
    }

    pub fn checkout_with_loader<F>(&mut self, model: &str, loader: F) -> Result<EngineLease<E>>
    where
        F: FnOnce(&EnginePoolModelSpec) -> Result<E>,
    {
        let model_id = self
            .resolve_model_id(model)
            .ok_or_else(|| anyhow!("engine pool model `{model}` is not configured"))?
            .to_string();
        let spec = self
            .specs
            .get(&model_id)
            .cloned()
            .context("resolved engine pool model is missing its spec")?;
        ensure!(
            spec.model_type == EnginePoolModelType::TextGeneration,
            "{} model `{}` is a configured control-plane stub, not a text-generation engine",
            spec.model_type.as_str(),
            spec.id
        );

        if !self.entries.contains_key(&model_id) {
            let estimate = spec.memory_estimate_bytes.unwrap_or(0);
            let victims = self.eviction_plan(estimate)?;
            self.apply_evictions(&victims);
            let engine =
                loader(&spec).with_context(|| format!("load engine pool model `{}`", spec.id))?;
            self.loaded_estimate_bytes = self.loaded_estimate_bytes.saturating_add(estimate);
            self.entries.insert(
                model_id.clone(),
                EnginePoolEntry {
                    engine: Arc::new(Mutex::new(engine)),
                    spec,
                    last_used: Arc::new(Mutex::new(Instant::now())),
                    active_requests: Arc::new(AtomicUsize::new(0)),
                },
            );
        }

        let entry = self
            .entries
            .get(&model_id)
            .expect("engine entry inserted or already loaded");
        entry.active_requests.fetch_add(1, Ordering::AcqRel);
        Ok(EngineLease {
            model_id: entry.spec.id.clone(),
            engine: entry.engine.clone(),
            last_used: entry.last_used.clone(),
            active_requests: entry.active_requests.clone(),
        })
    }

    pub fn unload_idle(&mut self, model: &str) -> Result<bool> {
        let Some(model_id) = self.resolve_model_id(model).map(str::to_string) else {
            return Ok(false);
        };
        let Some(entry) = self.entries.get(&model_id) else {
            return Ok(false);
        };
        ensure!(
            entry.active_requests.load(Ordering::Acquire) == 0,
            "cannot unload active engine pool model `{model_id}`"
        );
        ensure!(
            !entry.spec.pinned,
            "cannot unload pinned engine pool model `{model_id}`"
        );
        let entry = self
            .entries
            .remove(&model_id)
            .expect("entry existence checked above");
        self.loaded_estimate_bytes = self
            .loaded_estimate_bytes
            .saturating_sub(entry.spec.memory_estimate_bytes.unwrap_or(0));
        Ok(true)
    }

    pub fn unload_expired(&mut self, now: Instant) -> Vec<String> {
        let expired: Vec<String> = self
            .entries
            .iter()
            .filter_map(|(id, entry)| {
                let ttl = entry.spec.ttl?;
                let last_used = *entry
                    .last_used
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                (entry.active_requests.load(Ordering::Acquire) == 0
                    && !entry.spec.pinned
                    && now.duration_since(last_used) >= ttl)
                    .then(|| id.clone())
            })
            .collect();
        for id in &expired {
            if let Some(entry) = self.entries.remove(id) {
                self.loaded_estimate_bytes = self
                    .loaded_estimate_bytes
                    .saturating_sub(entry.spec.memory_estimate_bytes.unwrap_or(0));
            }
        }
        expired
    }

    fn eviction_plan(&self, incoming_estimate: u64) -> Result<Vec<String>> {
        let Some(budget) = self.memory_budget_bytes else {
            return Ok(Vec::new());
        };
        ensure!(
            incoming_estimate <= budget,
            "engine pool memory budget {} bytes cannot fit incoming {} bytes",
            budget,
            incoming_estimate
        );

        let projected = self.loaded_estimate_bytes.saturating_add(incoming_estimate);
        if projected <= budget {
            return Ok(Vec::new());
        }

        let required = projected - budget;
        let mut candidates: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry.active_requests.load(Ordering::Acquire) == 0 && !entry.spec.pinned
            })
            .map(|(id, entry)| {
                let last_used = *entry
                    .last_used
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                (
                    id.clone(),
                    last_used,
                    entry.spec.memory_estimate_bytes.unwrap_or(0),
                )
            })
            .collect();
        candidates.sort_by_key(|(_, last_used, _)| *last_used);

        let mut victims = Vec::new();
        let mut freed = 0u64;
        for (id, _, estimate) in candidates {
            freed = freed.saturating_add(estimate);
            victims.push(id);
            if freed >= required {
                return Ok(victims);
            }
        }

        bail!(
            "engine pool memory budget {} bytes cannot fit incoming {} bytes without unloading an active or pinned model",
            budget,
            incoming_estimate
        )
    }

    fn apply_evictions(&mut self, victims: &[String]) {
        for victim_id in victims {
            if let Some(entry) = self.entries.remove(victim_id) {
                self.loaded_estimate_bytes = self
                    .loaded_estimate_bytes
                    .saturating_sub(entry.spec.memory_estimate_bytes.unwrap_or(0));
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl LoadedEnginePool {
    pub fn checkout_loaded(
        &mut self,
        model: &str,
        enable_cuda_graph: bool,
    ) -> Result<EngineLease<LoadedInferenceEngine>> {
        self.checkout_with_loader(model, |spec| {
            LoadedInferenceEngine::load(&spec.model_path, enable_cuda_graph)
        })
    }
}

fn validate_model_name(field: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("pool model {field} must not be empty"));
    }
    if value.contains(',') || value.contains('=') {
        return Err(format!(
            "pool model {field} `{value}` must not contain ',' or '='"
        ));
    }
    Ok(())
}

fn model_id_from_path(model_path: &str) -> String {
    model_path
        .trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .find(|segment| !segment.is_empty())
        .unwrap_or(model_path)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: &str, bytes: u64) -> EnginePoolModelSpec {
        EnginePoolModelSpec {
            id: id.to_string(),
            model_path: format!("/models/{id}"),
            aliases: Vec::new(),
            model_type: EnginePoolModelType::TextGeneration,
            pinned: false,
            ttl: None,
            memory_estimate_bytes: Some(bytes),
        }
    }

    fn is_loaded(pool: &EnginePool<usize>, id: &str) -> bool {
        pool.configured_models()
            .iter()
            .any(|model| model.id == id && model.loaded)
    }

    #[test]
    fn pool_eviction_skips_active_requests() {
        let mut pool = EnginePool::<usize>::new(EnginePoolConfig {
            models: vec![spec("a", 60), spec("b", 60)],
            memory_budget_bytes: Some(100),
        })
        .expect("pool");

        let lease = pool
            .checkout_with_loader("a", |_| Ok(1usize))
            .expect("load a");
        assert_eq!(lease.model_id(), "a");
        assert!(pool.checkout_with_loader("b", |_| Ok(2usize)).is_err());
        drop(lease);

        let lease_b = pool
            .checkout_with_loader("b", |_| Ok(2usize))
            .expect("load b after a is idle");
        assert_eq!(lease_b.model_id(), "b");
        drop(lease_b);
        let info = pool.configured_models();
        assert!(!info.iter().find(|model| model.id == "a").expect("a").loaded);
        assert!(info.iter().find(|model| model.id == "b").expect("b").loaded);
    }

    #[test]
    fn pool_rejects_oversized_checkout_without_evicting_idle_model() {
        let mut pool = EnginePool::<usize>::new(EnginePoolConfig {
            models: vec![spec("a", 60), spec("huge", 120)],
            memory_budget_bytes: Some(100),
        })
        .expect("pool");

        let lease = pool
            .checkout_with_loader("a", |_| Ok(1usize))
            .expect("load a");
        drop(lease);

        let mut loader_called = false;
        let err = pool
            .checkout_with_loader("huge", |_| {
                loader_called = true;
                Ok(2usize)
            })
            .err()
            .expect("oversized checkout must fail")
            .to_string();
        assert!(!loader_called);
        assert!(err.contains("cannot fit incoming 120 bytes"));
        assert!(is_loaded(&pool, "a"));
        assert!(!is_loaded(&pool, "huge"));
        assert_eq!(pool.loaded_estimate_bytes, 60);
    }

    #[test]
    fn pool_failed_active_eviction_plan_keeps_idle_entries() {
        let mut pool = EnginePool::<usize>::new(EnginePoolConfig {
            models: vec![spec("a", 60), spec("b", 40), spec("c", 80)],
            memory_budget_bytes: Some(100),
        })
        .expect("pool");

        drop(
            pool.checkout_with_loader("a", |_| Ok(1usize))
                .expect("load a"),
        );
        let lease_b = pool
            .checkout_with_loader("b", |_| Ok(2usize))
            .expect("load b");

        let err = pool
            .checkout_with_loader("c", |_| Ok(3usize))
            .err()
            .expect("active b must block enough eviction")
            .to_string();
        assert!(err.contains("without unloading an active or pinned model"));
        assert!(is_loaded(&pool, "a"));
        assert!(is_loaded(&pool, "b"));
        assert!(!is_loaded(&pool, "c"));
        assert_eq!(pool.loaded_estimate_bytes, 100);
        drop(lease_b);
    }

    #[test]
    fn pool_loader_failure_keeps_existing_entries() {
        let mut pool = EnginePool::<usize>::new(EnginePoolConfig {
            models: vec![spec("a", 60), spec("b", 40)],
            memory_budget_bytes: Some(100),
        })
        .expect("pool");

        drop(
            pool.checkout_with_loader("a", |_| Ok(1usize))
                .expect("load a"),
        );
        let err = pool
            .checkout_with_loader("b", |_| Err(anyhow::anyhow!("boom")))
            .err()
            .expect("loader failure")
            .to_string();
        assert!(err.contains("load engine pool model `b`"));
        assert!(is_loaded(&pool, "a"));
        assert!(!is_loaded(&pool, "b"));
        assert_eq!(pool.loaded_estimate_bytes, 60);
    }

    #[test]
    fn pool_drops_victims_before_loading_replacement() {
        struct DropMarker(std::sync::Arc<std::sync::atomic::AtomicUsize>);

        impl Drop for DropMarker {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut pool = EnginePool::<DropMarker>::new(EnginePoolConfig {
            models: vec![spec("a", 60), spec("b", 60)],
            memory_budget_bytes: Some(100),
        })
        .expect("pool");

        drop(
            pool.checkout_with_loader("a", |_| Ok(DropMarker(drops.clone())))
                .expect("load a"),
        );
        let lease_b = pool
            .checkout_with_loader("b", |_| {
                assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 1);
                Ok(DropMarker(drops.clone()))
            })
            .expect("load b");
        drop(lease_b);
    }

    #[test]
    fn pool_rejects_non_generation_stubs() {
        let mut embedding = spec("embed", 10);
        embedding.model_type = EnginePoolModelType::Embedding;
        let mut pool = EnginePool::<usize>::new(EnginePoolConfig {
            models: vec![embedding],
            memory_budget_bytes: None,
        })
        .expect("pool");

        let err = pool
            .checkout_with_loader("embed", |_| Ok(1usize))
            .err()
            .expect("embedding is not a generation engine")
            .to_string();
        assert!(err.contains("control-plane stub"));
    }

    #[test]
    fn pool_resolves_aliases_and_parses_cli_specs() {
        let spec = EnginePoolModelSpec::parse_cli(
            "coder=mlx-community/Qwen3-4B,type=text-generation,aliases=main|assistant,pinned=true,memory_bytes=42,ttl_secs=9",
        )
        .expect("parse");
        assert_eq!(spec.id, "coder");
        assert_eq!(spec.aliases, ["main", "assistant"]);
        assert!(spec.pinned);
        assert_eq!(spec.memory_estimate_bytes, Some(42));
        assert_eq!(spec.ttl, Some(Duration::from_secs(9)));

        let pool = EnginePool::<usize>::new(EnginePoolConfig {
            models: vec![spec],
            memory_budget_bytes: None,
        })
        .expect("pool");
        assert_eq!(pool.resolve_model_id("main"), Some("coder"));
    }
}
