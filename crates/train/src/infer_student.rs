//! Infer-runtime student rollout (OPD Phase P1 bring-up).
//!
//! Mirrors [`crate::teacher_infer::InferTeacher`]: holds an in-process
//! `LoadedInferenceEngine` and drives rollouts through the serving scheduler's
//! token-id generation path. The student differs from the teacher only in that
//! its LoRA weights update every training step via `sync_lora_from_store`.
//!
//! `generate_rollout` submits one request to infer-core so the backend owns a
//! KV slot and decodes incrementally.

#[cfg(feature = "cuda")]
use std::collections::BTreeMap;
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use anyhow::{Result, anyhow, bail};
use autograd::TensorId;
#[cfg(feature = "cuda")]
use autograd::{Backend, TensorStore};
#[cfg(feature = "cuda")]
use infer_api::{
    LoadedInferenceEngine, LoraHalf, StudentLoraLayer, StudentLoraMatrices, StudentLoraProjection,
    StudentLoraProjectionUpdate, StudentLoraUpdate, parse_student_adapter_name,
};
#[cfg(feature = "cuda")]
use infer_plan::SamplingParams;

#[cfg(feature = "cuda")]
use crate::lora::LoraConfig;

#[cfg(feature = "cuda")]
pub struct InferStudent {
    engine: Arc<Mutex<LoadedInferenceEngine>>,
    train_backend: Arc<dyn Backend>,
    vocab_size: usize,
}

#[cfg(feature = "cuda")]
impl InferStudent {
    pub fn new(
        engine: Arc<Mutex<LoadedInferenceEngine>>,
        train_backend: Arc<dyn Backend>,
        vocab_size: usize,
    ) -> Self {
        Self {
            engine,
            train_backend,
            vocab_size,
        }
    }

    pub fn engine(&self) -> &Arc<Mutex<LoadedInferenceEngine>> {
        &self.engine
    }

    pub fn train_backend(&self) -> &Arc<dyn Backend> {
        &self.train_backend
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn offload_engine_weights(&self) -> Result<usize> {
        let engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.offload_engine_weights()
    }

    pub fn reload_engine_weights(&self) -> Result<()> {
        let engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.reload_engine_weights()
    }

    /// Release the inference forward scratch (no weight offload, no KV eviction)
    /// to free VRAM for the co-resident OPD writeback.
    pub fn release_inference_scratch(&self) -> Result<()> {
        let engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.release_inference_scratch()
    }

    /// Drop the KV pool before the masked-CE writeback. The writeback's forward
    /// is a fresh autograd pass that doesn't read this engine's KV, so the pool
    /// is dead during writeback — freeing it is the agent-OPD headroom lever.
    pub fn release_kv_pool(&self) -> Result<()> {
        let engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.release_kv_pool()
    }

    pub fn ensure_kv_pool(&self) -> Result<()> {
        let engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.ensure_kv_pool()
    }

    pub fn ensure_kv_pool_and_resume_admissions(&self) -> Result<()> {
        let engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.ensure_kv_pool_and_resume_admissions()
    }

    /// Generate `rollout_len` tokens from `prompt_ids` and return prompt+generated.
    /// Forces `ignore_eos=true` and clears `stop_token_ids` for exact-length rollout.
    pub fn generate_rollout(
        &self,
        prompt_ids: &[u32],
        rollout_len: usize,
        sampling: Option<&SamplingParams>,
    ) -> Result<Vec<u32>> {
        if prompt_ids.is_empty() {
            bail!("InferStudent rollout requires a non-empty prompt");
        }
        validate_token_ids("prompt", prompt_ids, self.vocab_size)?;
        if rollout_len == 0 {
            return Ok(prompt_ids.to_vec());
        }

        let mut params = sampling.cloned().unwrap_or_default();
        params.ignore_eos = true;
        params.stop_token_ids.clear();

        let generated = {
            let engine = self
                .engine
                .lock()
                .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
            engine.generate_token_ids(prompt_ids, rollout_len, params)?
        };
        if generated.len() != rollout_len {
            bail!(
                "InferStudent rollout generated {} tokens, expected {rollout_len}. \
                 Hint: verify the infer engine max_total_tokens covers prompt+rollout \
                 and that OPD rollout generation keeps exact-length sampling.",
                generated.len()
            );
        }
        validate_token_ids("generated rollout", &generated, self.vocab_size)?;

        let rollout: Vec<u32> = prompt_ids.iter().copied().chain(generated).collect();
        Ok(rollout)
    }

    /// Generate `n` EOS-respecting completions of up to `max_new_tokens` each,
    /// for rubric-OPD rejection sampling. Returns generated tokens per sample.
    pub fn generate_samples(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        n: usize,
        sampling: Option<&SamplingParams>,
    ) -> Result<Vec<Vec<u32>>> {
        if prompt_ids.is_empty() {
            bail!("InferStudent rollout requires a non-empty prompt");
        }
        validate_token_ids("prompt", prompt_ids, self.vocab_size)?;
        if max_new_tokens == 0 || n == 0 {
            return Ok(Vec::new());
        }

        let base = sampling.cloned().unwrap_or_default();
        let requests: Vec<(Vec<u32>, SamplingParams)> = (0..n)
            .map(|i| {
                let mut params = base.clone();
                if let Some(seed) = base.seed {
                    params.seed = Some(seed.wrapping_add(i as u64));
                }
                (prompt_ids.to_vec(), params)
            })
            .collect();
        self.generate_batch(&requests, max_new_tokens)
    }

    /// Submit all `(prompt, sampling)` requests to the continuous-batching engine
    /// and collect results in order. Batching amortizes weight reads (B=1 decode
    /// is memory-bandwidth-bound).
    pub fn generate_batch(
        &self,
        requests: &[(Vec<u32>, SamplingParams)],
        max_new_tokens: usize,
    ) -> Result<Vec<Vec<u32>>> {
        if requests.is_empty() || max_new_tokens == 0 {
            return Ok(vec![Vec::new(); requests.len()]);
        }
        for (prompt, _) in requests {
            if prompt.is_empty() {
                bail!("InferStudent batch generate requires non-empty prompts");
            }
            validate_token_ids("prompt", prompt, self.vocab_size)?;
        }
        let generated = {
            let engine = self
                .engine
                .lock()
                .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
            engine.generate_token_ids_batch(requests, max_new_tokens)?
        };
        for ids in &generated {
            validate_token_ids("generated sample", ids, self.vocab_size)?;
        }
        Ok(generated)
    }

    /// D2H LoRA A/B from the train store and push into the infer engine, which
    /// re-merges from the pristine base (idempotent — no delta accumulation).
    pub fn sync_lora_from_store(
        &self,
        store: &mut TensorStore,
        adapter_map: &HashMap<&'static str, TensorId>,
        param_name_map: &HashMap<&'static str, TensorId>,
        lora_config: LoraConfig,
    ) -> Result<()> {
        if lora_config.rank == 0 {
            bail!("InferStudent LoRA sync: lora_config.rank must be > 0");
        }

        let mut layers: HashMap<usize, PartialLayer> = HashMap::new();
        let mut unsupported = Vec::new();
        for (&name, &tensor_id) in adapter_map {
            let Some((layer_idx, projection, which)) = parse_student_adapter_name(name) else {
                unsupported.push(name);
                continue;
            };
            let shape = store
                .get(tensor_id)
                .ok_or_else(|| {
                    anyhow!("LoRA sync: tensor id {tensor_id:?} ({name}) missing from store")
                })?
                .shape
                .clone();
            if shape.len() != 2 {
                bail!("LoRA sync: {name} expected rank-2 matrix, got shape {shape:?}");
            }
            let values = store
                .to_host(tensor_id)
                .map_err(|err| anyhow!("LoRA sync: D2H {name} failed: {err}"))?;
            let entry = layers.entry(layer_idx).or_default();
            let slot = entry.projections.entry(projection).or_default();
            match which {
                LoraHalf::A => slot.a = Some((values, shape[0], shape[1])),
                LoraHalf::B => slot.b = Some((values, shape[0], shape[1])),
            }
        }

        if !unsupported.is_empty() {
            unsupported.sort_unstable();
            bail!(
                "LoRA sync: unsupported adapter tensor name(s): {}. \
                 Hint: extend infer_api::parse_student_adapter_name before using infer-engine rollout \
                 for this LoRA target set.",
                unsupported.join(", ")
            );
        }

        if layers.is_empty() {
            bail!(
                "LoRA sync: no supported adapters found in adapter_map ({} entries)",
                adapter_map.len()
            );
        }

        let mut layer_indices: Vec<usize> = layers.keys().copied().collect();
        layer_indices.sort_unstable();

        let out_layers = layer_indices
            .into_iter()
            .map(|layer_idx| {
                let partial = layers.remove(&layer_idx).expect("layer present");
                let projections = partial
                    .projections
                    .into_iter()
                    .map(|(projection, partial_proj)| {
                        let label = projection.label();
                        let matrices = partial_proj.into_matrices(
                            lora_config.rank,
                            layer_idx,
                            label.as_ref(),
                        )?;
                        Ok(StudentLoraProjectionUpdate {
                            projection,
                            matrices,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(StudentLoraLayer {
                    layer_idx,
                    projections,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let update = StudentLoraUpdate {
            layers: out_layers,
            rank: lora_config.rank,
            alpha: lora_config.alpha,
        };

        let engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.remerge_student_lora(update)?;

        // Re-point the student's frozen-base tensors at the engine's merged BF16
        // bytes. SKIPPED under offload=student|all: the same step's
        // offload_engine_weights frees these buffers, so the trainer must keep
        // its own frozen-base copies (a re-point would read freed memory).
        // Retired FP8 qweight/scales are never freed: they are the pristine
        // source for the idempotent re-merge and the trainer's shared alias.
        if !crate::opd::engine_offload_mode().offloads_student() {
            let bf16_ptrs = engine.frozen_base_bf16_pointers()?;
            drop(engine);

            for entry in &bf16_ptrs {
                let suffix = format!(".layers.{}.{}.weight", entry.layer_idx, entry.proj_suffix);
                let Some((&name, &id)) = param_name_map
                    .iter()
                    .find(|(name, _)| name.ends_with(&suffix))
                else {
                    continue;
                };
                if engine_bytes_are_merged(name, adapter_map) {
                    continue;
                }
                let shape = vec![entry.rows, entry.cols];
                let handle = {
                    let backend = store.backend();
                    backend
                        .import_bf16_device_ptr(entry.data_ptr, &shape)
                        .map_err(|err| {
                            anyhow!("LoRA sync: import BF16 base for {name} failed: {err}")
                        })?
                };
                store.replace_device_handle(id, handle).map_err(|err| {
                    anyhow!("LoRA sync: replace BF16 base for {name} failed: {err}")
                })?;
                store.set_requires_grad(id, false).map_err(|err| {
                    anyhow!("LoRA sync: set requires_grad=false for {name} failed: {err}")
                })?;
            }
        }
        Ok(())
    }
}

/// True when the engine's buffer for `param_name` holds MERGED bytes (base +
/// LoRA delta). The trainer's forward re-adds A·B, so re-pointing the frozen
/// base at such a buffer would double-apply the delta (issue #201).
// Non-cuda builds reach this only from the #[cfg(test)] naming-contract guard.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn engine_bytes_are_merged(
    param_name: &str,
    adapter_map: &HashMap<&'static str, TensorId>,
) -> bool {
    adapter_map.contains_key(format!("{param_name}.lora_a").as_str())
}

/// Save LoRA A/B adapters to a single safetensors file (bf16). The cheap
/// alternative to full-materialize save (which hangs at 27B); adapters are tiny.
#[cfg(feature = "cuda")]
pub fn save_lora_adapters(
    store: &mut TensorStore,
    adapter_map: &HashMap<&'static str, TensorId>,
    out_path: &std::path::Path,
) -> Result<()> {
    use autograd::SafetensorsRegistry;
    if adapter_map.is_empty() {
        bail!("save_lora_adapters: adapter_map is empty (no LoRA tensors to save)");
    }
    let mut registry = SafetensorsRegistry::new();
    for (&name, &tensor_id) in adapter_map {
        registry.insert(name, tensor_id);
    }
    registry
        .save_from_bf16(store, out_path)
        .map_err(|err| anyhow!("save_lora_adapters to {}: {err}", out_path.display()))
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct PartialLayer {
    projections: BTreeMap<StudentLoraProjection, PartialProj>,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct PartialProj {
    a: Option<(Vec<f32>, usize, usize)>,
    b: Option<(Vec<f32>, usize, usize)>,
}

#[cfg(feature = "cuda")]
impl PartialProj {
    fn into_matrices(
        self,
        rank: usize,
        layer_idx: usize,
        label: &str,
    ) -> Result<StudentLoraMatrices> {
        match (self.a, self.b) {
            (Some((a, a_rows, a_cols)), Some((b, b_rows, b_cols))) => {
                if a_rows != rank {
                    bail!(
                        "LoRA sync: layer {layer_idx} {label} lora_A rows {a_rows} != rank {rank}"
                    );
                }
                if b_cols != rank {
                    bail!(
                        "LoRA sync: layer {layer_idx} {label} lora_B cols {b_cols} != rank {rank}"
                    );
                }
                Ok(StudentLoraMatrices {
                    a,
                    b,
                    rank,
                    in_features: a_cols,
                    out_features: b_rows,
                })
            }
            (None, None) => bail!("LoRA sync: layer {layer_idx} {label} has no lora_A/lora_B"),
            (Some(_), None) => {
                bail!("LoRA sync: layer {layer_idx} {label} has lora_A without lora_B")
            }
            (None, Some(_)) => {
                bail!("LoRA sync: layer {layer_idx} {label} has lora_B without lora_A")
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn validate_token_ids(label: &str, tokens: &[u32], vocab_size: usize) -> Result<()> {
    for (idx, &token) in tokens.iter().enumerate() {
        if token as usize >= vocab_size {
            bail!(
                "InferStudent {label} token id {token} at index {idx} is outside vocab_size={vocab_size}"
            );
        }
    }
    Ok(())
}
