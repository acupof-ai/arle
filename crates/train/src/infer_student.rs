//! Infer-runtime student rollout (OPD Phase P1 bring-up).
//!
//! Mirrors [`crate::teacher_infer::InferTeacher`]: holds an in-process
//! `LoadedInferenceEngine` and drives rollouts through the serving scheduler's
//! token-id generation path. The student differs from the teacher only in that
//! its LoRA weights update every training step via `sync_lora_from_store`.
//!
//! `generate_rollout` submits one request to infer-core so the backend owns a
//! KV slot and decodes incrementally. `decode_next_token` remains as the
//! raw-logits parity/debug surface.

#[cfg(feature = "cuda")]
use std::collections::{BTreeMap, HashMap};
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use anyhow::{Result, anyhow, bail};
#[cfg(feature = "cuda")]
use autograd::{Backend, TensorId, TensorStore};
#[cfg(feature = "cuda")]
use infer_api::{
    LoadedInferenceEngine, LoraHalf, StudentLoraLayer, StudentLoraMatrices, StudentLoraProjection,
    StudentLoraProjectionUpdate, StudentLoraUpdate, parse_student_adapter_name,
};
#[cfg(feature = "cuda")]
use infer_plan::{SamplingParams, sample_token};

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

    /// Offload the rollout engine's device weights to host RAM (OPD
    /// time-share), freeing VRAM for the student backward. Returns bytes freed.
    pub fn offload_engine_weights(&self) -> Result<usize> {
        let engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.offload_engine_weights()
    }

    /// Reload the rollout engine's device weights before the next rollout.
    pub fn reload_engine_weights(&self) -> Result<()> {
        let engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.reload_engine_weights()
    }

    /// Release the rollout engine's inference forward scratch WITHOUT offloading
    /// weights or evicting KV, freeing VRAM for the co-resident OPD writeback
    /// (the agent-OPD rollout->writeback path never offloads, so the 24K-shaped
    /// scratch otherwise OOMs the masked-CE writeback).
    pub fn release_inference_scratch(&self) -> Result<()> {
        let engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.release_inference_scratch()
    }

    /// Drop the rollout engine's KV pool before the co-resident masked-CE
    /// writeback. The writeback's `forward_hidden_states` is a FRESH autograd
    /// forward that does NOT read this engine's KV cache, so the pool is DEAD
    /// during the writeback — freeing it (~KV-pool GB) is the agent-OPD writeback
    /// headroom lever. The rollout has finished + synced before this is called, so
    /// the pool holds no resident pages. Re-acquired by [`Self::ensure_kv_pool`]
    /// before the next round's rollout.
    pub fn release_kv_pool(&self) -> Result<()> {
        let engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.release_kv_pool()
    }

    /// Re-acquire the rollout engine's KV pool dropped by
    /// [`Self::release_kv_pool`] before the next round's rollout. Idempotent.
    pub fn ensure_kv_pool(&self) -> Result<()> {
        let engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.ensure_kv_pool()
    }

    /// Re-acquire the KV pool, then resume admission only after success.
    pub fn ensure_kv_pool_and_resume_admissions(&self) -> Result<()> {
        let engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.ensure_kv_pool_and_resume_admissions()
    }

    /// Generate `rollout_len` tokens from `prompt_ids` through the infer
    /// scheduler/KV path and return the full prompt+generated rollout.
    ///
    /// OPD rollout requires an exact-length token sequence. The old raw-logits
    /// loop sampled unconditionally and ignored EOS/stop ids, so the serve-path
    /// request does the same by forcing `ignore_eos=true` and clearing
    /// `stop_token_ids` on the per-rollout sampling copy.
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

    /// Generate `n` **natural, EOS-respecting** completions of up to
    /// `max_new_tokens` tokens each, for rubric-OPD rejection sampling. Returns the
    /// GENERATED tokens per sample (prompt excluded), variable length.
    ///
    /// This is the rubric-OPD counterpart of [`Self::generate_rollout`]: that path
    /// forces exact-length `ignore_eos` decoding for token-KL alignment, whereas the
    /// rubric judge needs *complete* solutions (stop at EOS) and the writeback CE
    /// trains the natural completion. Per-sample seeds (`base + i` when a seed is
    /// set) give reproducible diversity at temperature > 0; with no seed the engine
    /// RNG advances across calls. Samples are independent requests.
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

        // Natural generation: do NOT force ignore_eos (that is the token-KL
        // exact-length path); the rubric judge needs complete answers. The N
        // per-seed samples are submitted together so the continuous-batching
        // rollout engine (num_slots=2) decodes them concurrently instead of one
        // at a time. `generate_batch` validates each result.
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

    /// Batched generation: each `(prompt, sampling)` is an independent request
    /// submitted together to the continuous-batching engine, then collected in
    /// order. Used by rubric-OPD eval to decode all N prompts concurrently —
    /// per-request B=1 decode is memory-bandwidth-bound, so batching amortizes
    /// the weight reads. Engine is locked once for the whole batch.
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

    /// Run a single forward over `input_ids` (with absolute `positions`) and
    /// return the next token over the **last** position's logits.
    ///
    /// The infer engine's `forward_token_logits` is stateless from the caller's
    /// POV: it accepts the full token sequence + contiguous positions each call
    /// (matching how `InferTeacher` drives it). The rollout loop therefore
    /// passes the growing full sequence with `positions = 0..len` each step.
    pub fn decode_next_token(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        sampling: Option<&SamplingParams>,
        position: u64,
    ) -> Result<u32> {
        if input_ids.is_empty() {
            bail!("InferStudent requires a non-empty token sequence");
        }
        if input_ids.len() != positions.len() {
            bail!(
                "InferStudent token/position length mismatch: tokens={} positions={}",
                input_ids.len(),
                positions.len()
            );
        }

        let raw_logits = {
            let engine = self
                .engine
                .lock()
                .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
            engine.forward_token_logits(input_ids, positions)?
        };

        if raw_logits.vocab_size() != self.vocab_size {
            bail!(
                "InferStudent vocab mismatch: raw logits vocab={}, configured vocab={}",
                raw_logits.vocab_size(),
                self.vocab_size
            );
        }
        if raw_logits.seq_len() != input_ids.len() {
            bail!(
                "InferStudent seq_len mismatch: raw logits seq_len={}, input token len={}",
                raw_logits.seq_len(),
                input_ids.len()
            );
        }

        // Host argmax over the last position. `to_host_f32` materializes the
        // full [seq_len, vocab_size] block (mirrors teacher BF16->F32 handling
        // via the engine's own conversion); we only scan the last row.
        let host = raw_logits.to_host_f32()?;
        let vocab = raw_logits.vocab_size();
        let seq_len = raw_logits.seq_len();
        let last_row = &host[(seq_len - 1) * vocab..seq_len * vocab];
        let token = match sampling {
            None => argmax(last_row)? as u32,
            Some(params) => sample_token(last_row, params, position),
        };
        Ok(token)
    }

    /// Per-step student LoRA sync (OPD P2).
    ///
    /// D2H the LoRA A/B adapter tensors from the train `TensorStore` and push
    /// them into the infer student engine, which restores cached base weights
    /// and folds each fresh adapter in-memory (`remerge_student_lora`).
    /// Idempotent across steps: the infer side always re-merges from the same
    /// pristine base, so deltas never accumulate.
    ///
    /// `adapter_map` is the train model's `adapter_name_map()`. Matrices are
    /// exported raw (un-scaled); the infer merge applies `scale = alpha / r`
    /// once. Every recognized projection must provide both A and B.
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

        // Collect per-layer A/B from the train store, keyed by absolute layer
        // index and projection. BTreeMap gives deterministic update ordering.
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
                // lora_A shape = [rank, in_features]
                LoraHalf::A => slot.a = Some((values, shape[0], shape[1])),
                // lora_B shape = [out_features, rank]
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

        // Refresh the train student's frozen base: the merged weights now live
        // in the rollout engine's dense-BF16 `data` buffers. Re-point the
        // student's frozen-base tensors at those BF16 bytes (non-owning view),
        // then free the retired FP8 qweight/scales. This both keeps the
        // student's base in sync with the merged weights AND frees ~27 GB of
        // FP8 keepalive that would otherwise OOM the next round's forward.
        let bf16_ptrs = engine.frozen_base_bf16_pointers()?;
        drop(engine);

        for entry in &bf16_ptrs {
            // Match the train-side parameter name: `*.layers.{layer_idx}.{proj_suffix}.weight`.
            let suffix = format!(".layers.{}.{}.weight", entry.layer_idx, entry.proj_suffix);
            let Some((&name, &id)) = param_name_map
                .iter()
                .find(|(name, _)| name.ends_with(&suffix))
            else {
                continue;
            };
            let shape = vec![entry.rows, entry.cols];
            let handle = {
                let backend = store.backend();
                backend
                    .import_bf16_device_ptr(entry.data_ptr, &shape)
                    .map_err(|err| {
                        anyhow!("LoRA sync: import BF16 base for {name} failed: {err}")
                    })?
            };
            store
                .replace_device_handle(id, handle)
                .map_err(|err| anyhow!("LoRA sync: replace BF16 base for {name} failed: {err}"))?;
            store.set_requires_grad(id, false).map_err(|err| {
                anyhow!("LoRA sync: set requires_grad=false for {name} failed: {err}")
            })?;
        }

        // Now that the student re-aliases the BF16 bytes, free the retired FP8
        // buffers. Re-lock the engine (the borrow above was released).
        let mut engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.free_retired_fp8_buffers();
        Ok(())
    }
}

/// Fast adapter-only (LoRA) checkpoint save. D2H-reads each adapter A/B
/// `TensorId` from the store and writes them to a single safetensors file, keyed
/// by adapter name. This is the cheap alternative to the full-materialize save,
/// which host-loops the merged base+LoRA weights and hangs at 27B; the adapter
/// matrices are tiny, so this is fast. The saved file can be re-merged onto the
/// base model for eval. bf16 so a downstream infer loader can consume it.
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

/// Adapter accumulator for one layer during the store scan.
#[cfg(feature = "cuda")]
#[derive(Default)]
struct PartialLayer {
    projections: BTreeMap<StudentLoraProjection, PartialProj>,
}

/// A single projection's optional A/B host matrices, each as
/// `(values, rows, cols)`.
#[cfg(feature = "cuda")]
#[derive(Default)]
struct PartialProj {
    a: Option<(Vec<f32>, usize, usize)>,
    b: Option<(Vec<f32>, usize, usize)>,
}

#[cfg(feature = "cuda")]
impl PartialProj {
    /// Convert to `StudentLoraMatrices`. A dangling half (A without B or vice
    /// versa) is an error.
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
fn argmax(logits: &[f32]) -> Result<usize> {
    if logits.is_empty() {
        bail!("argmax over empty logits");
    }
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (idx, &val) in logits.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    Ok(best_idx)
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
