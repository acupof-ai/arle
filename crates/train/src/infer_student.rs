//! Infer-runtime student rollout (OPD Phase P1 bring-up).
//!
//! Mirrors [`crate::teacher_infer::InferTeacher`]: holds an in-process
//! `LoadedInferenceEngine` and drives it via `forward_token_logits`. The
//! student differs from the teacher only in that its LoRA weights update every
//! training step — but that per-step sync is **P2**. This module is the
//! zero-LoRA (step-0 == base) bring-up + measurement path only; it contains no
//! LoRA-sync machinery.
//!
//! `decode_next_token` greedily picks the argmax over the **last position**'s
//! logits. Host argmax is sufficient for the bring-up smoke; the device/D2D
//! bridge used by the teacher's KL path is not needed because the rollout loop
//! consumes only the token sequence (see plan
//! `docs/plans/2026-05-29-opd-student-rollout-via-infer.md`).

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use anyhow::{Result, anyhow, bail};
#[cfg(feature = "cuda")]
use autograd::{Backend, TensorId, TensorStore};
#[cfg(feature = "cuda")]
use infer::server_engine::{
    LoadedInferenceEngine, StudentLoraLayer, StudentLoraMatrices, StudentLoraUpdate,
};

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

    /// Run a single greedy forward over `input_ids` (with absolute `positions`)
    /// and return the argmax token over the **last** position's logits.
    ///
    /// The infer engine's `forward_token_logits` is stateless from the caller's
    /// POV: it accepts the full token sequence + contiguous positions each call
    /// (matching how `InferTeacher` drives it). The rollout loop therefore
    /// passes the growing full sequence with `positions = 0..len` each step.
    pub fn decode_next_token(&self, input_ids: &[u32], positions: &[u32]) -> Result<u32> {
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
        let token = argmax(last_row)?;
        Ok(token as u32)
    }

    /// Per-step student LoRA sync (OPD P2).
    ///
    /// D2H the q/v LoRA A/B adapter tensors from the train `TensorStore` and
    /// push them into the infer student engine, which restores its cached base
    /// q/v weights and folds the fresh adapter in-memory (`remerge_student_lora`).
    /// Idempotent across steps: the infer side always re-merges from the same
    /// pristine base, so deltas never accumulate.
    ///
    /// `adapter_map` is the train model's `adapter_name_map()`; only q/v
    /// adapters (full-attention layers) are recognized — the train target set
    /// must be `AttentionQv`. Matrices are exported raw (un-scaled); the infer
    /// merge applies `scale = alpha / r` once.
    pub fn sync_lora_from_store(
        &self,
        store: &mut TensorStore,
        adapter_map: &HashMap<&'static str, TensorId>,
        lora_config: LoraConfig,
    ) -> Result<()> {
        if lora_config.rank == 0 {
            bail!("InferStudent LoRA sync: lora_config.rank must be > 0");
        }

        // Collect per-layer A/B from the train store, keyed by absolute layer
        // index. Each entry is (q_a, q_b, v_a, v_b) slots filled as found.
        let mut layers: HashMap<usize, PartialLayer> = HashMap::new();
        for (&name, &tensor_id) in adapter_map {
            let Some((layer_idx, module, which)) = parse_adapter_name(name) else {
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
            let slot = match module {
                AdapterModule::Q => &mut entry.q,
                AdapterModule::V => &mut entry.v,
            };
            match which {
                Which::A => {
                    // lora_A shape = [rank, in_features]
                    slot.a = Some((values, shape[0], shape[1]));
                }
                Which::B => {
                    // lora_B shape = [out_features, rank]
                    slot.b = Some((values, shape[0], shape[1]));
                }
            }
        }

        if layers.is_empty() {
            bail!(
                "LoRA sync: no q/v adapters found in adapter_map ({} entries); \
                 train target set must be AttentionQv",
                adapter_map.len()
            );
        }

        let mut layer_indices: Vec<usize> = layers.keys().copied().collect();
        layer_indices.sort_unstable();

        let mut out_layers: Vec<StudentLoraLayer> = Vec::with_capacity(layer_indices.len());
        for layer_idx in layer_indices {
            let partial = layers.remove(&layer_idx).expect("layer present");
            let q_proj = partial
                .q
                .into_matrices(lora_config.rank, layer_idx, "q_proj")?;
            let v_proj = partial
                .v
                .into_matrices(lora_config.rank, layer_idx, "v_proj")?;
            out_layers.push(StudentLoraLayer {
                layer_idx,
                q_proj,
                v_proj,
            });
        }

        let update = StudentLoraUpdate {
            layers: out_layers,
            rank: lora_config.rank,
            alpha: lora_config.alpha,
        };

        let engine = self
            .engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
        engine.remerge_student_lora(update)
    }
}

/// q/v adapter accumulator for one layer during the store scan.
#[cfg(feature = "cuda")]
#[derive(Default)]
struct PartialLayer {
    q: PartialProj,
    v: PartialProj,
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
    /// Convert to `StudentLoraMatrices`, or `None` if this projection had no
    /// adapter. A dangling half (A without B or vice versa) is an error.
    fn into_matrices(
        self,
        rank: usize,
        layer_idx: usize,
        label: &str,
    ) -> Result<Option<StudentLoraMatrices>> {
        match (self.a, self.b) {
            (None, None) => Ok(None),
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
                Ok(Some(StudentLoraMatrices {
                    a,
                    b,
                    rank,
                    in_features: a_cols,
                    out_features: b_rows,
                }))
            }
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
#[derive(Copy, Clone)]
enum AdapterModule {
    Q,
    V,
}

#[cfg(feature = "cuda")]
#[derive(Copy, Clone)]
enum Which {
    A,
    B,
}

/// Parse a train adapter tensor name like
/// `model.language_model.layers.7.self_attn.q_proj.weight.lora_a` into
/// `(layer_idx, module, which)`. Returns `None` for non-q/v adapters (e.g.
/// MLP adapters under an `AllLinear` target set are ignored).
#[cfg(feature = "cuda")]
fn parse_adapter_name(name: &str) -> Option<(usize, AdapterModule, Which)> {
    let which = if name.ends_with(".lora_a") {
        Which::A
    } else if name.ends_with(".lora_b") {
        Which::B
    } else {
        return None;
    };
    let parts: Vec<&str> = name.split('.').collect();
    let layers_pos = parts.iter().position(|part| *part == "layers")?;
    let layer_idx: usize = parts.get(layers_pos + 1)?.parse().ok()?;
    if *parts.get(layers_pos + 2)? != "self_attn" {
        return None;
    }
    let module = match *parts.get(layers_pos + 3)? {
        "q_proj" => AdapterModule::Q,
        "v_proj" => AdapterModule::V,
        _ => return None,
    };
    Some((layer_idx, module, which))
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
