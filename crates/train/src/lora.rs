use std::{collections::HashMap, f32::consts::TAU};

use autograd::{
    AutogradError, Result, Tape, Tensor, TensorId, TensorStore,
    ops::{add, matmul_bt_with_site, mul_scalar, reshape},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f32,
}

impl LoraConfig {
    pub fn scale(self) -> f32 {
        self.alpha / self.rank as f32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraTargetSet {
    AllLinear,
    AttentionQv,
    /// DAPO-style: full attention q/k/v/o + linear attention in_proj_qkv/out_proj.
    /// Covers all attention projections across both layer types; skips MLP entirely.
    AttentionFull,
}

impl LoraTargetSet {
    pub fn label(self) -> &'static str {
        match self {
            Self::AllLinear => "all-linear",
            Self::AttentionQv => "attention-qv",
            Self::AttentionFull => "attention-full",
        }
    }

    pub fn includes(self, base_name: &str) -> bool {
        match self {
            Self::AllLinear => true,
            Self::AttentionQv => {
                base_name.ends_with(".self_attn.q_proj.weight")
                    || base_name.ends_with(".self_attn.v_proj.weight")
            }
            Self::AttentionFull => {
                base_name.ends_with(".self_attn.q_proj.weight")
                    || base_name.ends_with(".self_attn.k_proj.weight")
                    || base_name.ends_with(".self_attn.v_proj.weight")
                    || base_name.ends_with(".self_attn.o_proj.weight")
                    || base_name.ends_with(".linear_attn.in_proj_qkv.weight")
                    || base_name.ends_with(".linear_attn.out_proj.weight")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraAdapterConfig {
    pub base_model_name_or_path: String,
    pub bias: String,
    pub fan_in_fan_out: bool,
    pub inference_mode: bool,
    pub lora_alpha: f32,
    pub lora_dropout: f32,
    pub peft_type: String,
    pub r: usize,
    pub revision: Option<String>,
    pub target_modules: Vec<String>,
    pub task_type: String,
    pub model_family: String,
}

impl LoraAdapterConfig {
    pub fn new(
        base_model_name_or_path: impl Into<String>,
        model_family: &str,
        lora: LoraConfig,
    ) -> Self {
        Self {
            base_model_name_or_path: base_model_name_or_path.into(),
            bias: "none".to_string(),
            fan_in_fan_out: false,
            inference_mode: true,
            lora_alpha: lora.alpha,
            lora_dropout: 0.0,
            peft_type: "LORA".to_string(),
            r: lora.rank,
            revision: None,
            target_modules: vec!["all-linear".to_string()],
            task_type: "CAUSAL_LM".to_string(),
            model_family: model_family.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearWithLora {
    base_name: &'static str,
    weight: TensorId,
    lora: Option<LoraWeights>,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearLoraParts {
    pub weight: TensorId,
    pub lora_a: Option<TensorId>,
    pub lora_b: Option<TensorId>,
    pub lora_scale: f32,
}

#[derive(Debug, Clone)]
struct LoraWeights {
    lora_a_name: &'static str,
    lora_b_name: &'static str,
    lora_a: TensorId,
    lora_b: TensorId,
    rank: usize,
    scale: f32,
}

impl LinearWithLora {
    pub fn new(
        base_name: &'static str,
        in_features: usize,
        out_features: usize,
        base_requires_grad: bool,
        lora: Option<LoraConfig>,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_internal(
            base_name,
            in_features,
            out_features,
            base_requires_grad,
            lora,
            true,
            store,
        )
    }

    pub(crate) fn new_with_unmaterialized_base(
        base_name: &'static str,
        in_features: usize,
        out_features: usize,
        base_requires_grad: bool,
        lora: Option<LoraConfig>,
        store: &mut TensorStore,
    ) -> Result<Self> {
        Self::new_internal(
            base_name,
            in_features,
            out_features,
            base_requires_grad,
            lora,
            false,
            store,
        )
    }

    fn new_internal(
        base_name: &'static str,
        in_features: usize,
        out_features: usize,
        base_requires_grad: bool,
        lora: Option<LoraConfig>,
        materialize_frozen_base: bool,
        store: &mut TensorStore,
    ) -> Result<Self> {
        let weight = base_parameter(
            base_name,
            &[out_features, in_features],
            0.02,
            base_requires_grad,
            materialize_frozen_base,
            store,
        )?;
        let lora = match lora {
            Some(cfg) => {
                if cfg.rank == 0 {
                    return Err(tape_invariant("LoRA rank must be > 0".into()));
                }
                let lora_a_name = leak_name(format!("{base_name}.lora_a"));
                let lora_b_name = leak_name(format!("{base_name}.lora_b"));
                let lora_a =
                    normal_parameter(lora_a_name, &[cfg.rank, in_features], 0.02, true, store)?;
                let lora_b = zeros_parameter(lora_b_name, &[out_features, cfg.rank], true, store)?;
                Some(LoraWeights {
                    lora_a_name,
                    lora_b_name,
                    lora_a,
                    lora_b,
                    rank: cfg.rank,
                    scale: cfg.scale(),
                })
            }
            None => None,
        };

        Ok(Self {
            base_name,
            weight,
            lora,
        })
    }

    pub fn forward(
        &self,
        x: TensorId,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let x_shape = store
            .get(x)
            .ok_or(AutogradError::InvalidTensorId(x))?
            .shape
            .clone();
        let weight_shape = store
            .get(self.weight)
            .ok_or(AutogradError::InvalidTensorId(self.weight))?
            .shape
            .clone();
        if weight_shape.len() != 2 {
            return Err(AutogradError::InvalidRank {
                expected: "2",
                got: weight_shape.len(),
            });
        }

        let input_dim = *x_shape.last().ok_or(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        })?;
        if input_dim != weight_shape[1] {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![weight_shape[1]],
                got: vec![input_dim],
            });
        }

        let prefix_elems = x_shape.iter().product::<usize>() / input_dim;
        let flat_x = reshape(x, &[prefix_elems, input_dim], store, tape)?;
        let mut projected = matmul_bt_with_site(flat_x, self.weight, store, tape, self.base_name)?;
        if let Some(lora) = &self.lora {
            let low_rank = matmul_bt_with_site(flat_x, lora.lora_a, store, tape, lora.lora_a_name)?;
            let delta_raw =
                matmul_bt_with_site(low_rank, lora.lora_b, store, tape, lora.lora_b_name)?;
            let delta = mul_scalar(delta_raw, lora.scale, store, tape)?;
            let base = projected;
            projected = add(base, delta, store, tape)?;
            // tape-disabled checkpoint forward: free the base + delta ring now.
            if !tape.enabled {
                for id in [base, low_rank, delta_raw, delta] {
                    store.free(id)?;
                }
            }
        }

        let mut output_shape = x_shape[..x_shape.len() - 1].to_vec();
        output_shape.push(weight_shape[0]);
        reshape(projected, &output_shape, store, tape)
    }

    pub fn parts(&self) -> LinearLoraParts {
        LinearLoraParts {
            weight: self.weight,
            lora_a: self.lora.as_ref().map(|lora| lora.lora_a),
            lora_b: self.lora.as_ref().map(|lora| lora.lora_b),
            lora_scale: self.lora.as_ref().map(|lora| lora.scale).unwrap_or(0.0),
        }
    }

    pub fn base_weight(&self) -> TensorId {
        self.weight
    }

    pub fn set_base_weight(&mut self, weight: TensorId) {
        self.weight = weight;
    }

    pub fn parameter_name_map(&self) -> HashMap<&'static str, TensorId> {
        HashMap::from([(self.base_name, self.weight)])
    }

    pub fn adapter_name_map(&self) -> HashMap<&'static str, TensorId> {
        match &self.lora {
            Some(lora) => HashMap::from([
                (lora.lora_a_name, lora.lora_a),
                (lora.lora_b_name, lora.lora_b),
            ]),
            None => HashMap::new(),
        }
    }

    /// LoRA adapter `(name, id)` pairs in a fixed A-then-B order. Param
    /// registration order must be identical on every process: CP/TP grad
    /// all-reduce issues one collective per param by position, and
    /// `adapter_name_map`'s HashMap iteration order is per-process randomized,
    /// so two ranks would pair mismatched shapes into the same collective and
    /// wedge NCCL (no size rendezvous).
    pub fn adapter_ordered(&self) -> Vec<(&'static str, TensorId)> {
        match &self.lora {
            Some(lora) => vec![
                (lora.lora_a_name, lora.lora_a),
                (lora.lora_b_name, lora.lora_b),
            ],
            None => Vec::new(),
        }
    }

    pub fn merged_tensor(&self, store: &mut TensorStore) -> Result<Tensor> {
        let shape = store
            .get(self.weight)
            .ok_or(AutogradError::InvalidTensorId(self.weight))?
            .shape
            .clone();
        let data = store.to_host(self.weight)?;
        let mut merged = Tensor::new(data, shape, false)?;
        if let Some(lora) = &self.lora {
            let out_features = merged.shape[0];
            let in_features = merged.shape[1];
            let a = store.to_host(lora.lora_a)?;
            let b = store.to_host(lora.lora_b)?;
            for out_idx in 0..out_features {
                for in_idx in 0..in_features {
                    let mut delta = 0.0_f32;
                    for rank_idx in 0..lora.rank {
                        delta +=
                            b[out_idx * lora.rank + rank_idx] * a[rank_idx * in_features + in_idx];
                    }
                    merged.data[out_idx * in_features + in_idx] += lora.scale * delta;
                }
            }
        }
        Ok(merged)
    }
}

fn normal_parameter(
    name: &'static str,
    shape: &[usize],
    std: f32,
    requires_grad: bool,
    store: &mut TensorStore,
) -> Result<TensorId> {
    let mut state = seed_from_name(name);
    let size = shape.iter().product();
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let u1 = next_uniform(&mut state).max(f32::MIN_POSITIVE);
        let u2 = next_uniform(&mut state);
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = TAU * u2;
        data.push(std * radius * theta.cos());
        if data.len() < size {
            data.push(std * radius * theta.sin());
        }
    }
    Ok(store.alloc(Tensor::new(data, shape.to_vec(), requires_grad)?))
}

fn base_parameter(
    name: &'static str,
    shape: &[usize],
    std: f32,
    requires_grad: bool,
    materialize_frozen_base: bool,
    store: &mut TensorStore,
) -> Result<TensorId> {
    if requires_grad || materialize_frozen_base {
        return normal_parameter(name, shape, std, requires_grad, store);
    }
    let _ = name;
    Ok(store.alloc(Tensor::unmaterialized(shape.to_vec(), false)?))
}

fn zeros_parameter(
    name: &'static str,
    shape: &[usize],
    requires_grad: bool,
    store: &mut TensorStore,
) -> Result<TensorId> {
    let _ = name;
    Ok(store.alloc(Tensor::new(
        vec![0.0; shape.iter().product()],
        shape.to_vec(),
        requires_grad,
    )?))
}

pub(crate) fn seed_from_name(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub(crate) fn next_uniform(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    bits as f32 / (u32::MAX >> 8) as f32
}

pub(crate) fn leak_name(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

fn tape_invariant(message: String) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(message.into_boxed_str()))
}

/// Cadence for LoRA consolidation: fires every K steps, or when the adapter
/// norm crosses a threshold. Pure host logic — the consolidation itself
/// (merge + re-quant) is the caller's job.
pub struct ConsolidationCadence {
    every_k: Option<usize>,
    norm_threshold: Option<f32>,
    step: usize,
}

impl ConsolidationCadence {
    #[must_use]
    pub fn new(every_k: Option<usize>, norm_threshold: Option<f32>) -> Self {
        Self {
            every_k,
            norm_threshold,
            step: 0,
        }
    }

    /// Advance the step counter; returns true when consolidation should fire.
    pub fn should_consolidate(&mut self, adapter_norm: f32) -> bool {
        self.step += 1;
        let k_fire = self.every_k.is_some_and(|k| self.step.is_multiple_of(k));
        let norm_fire = self.norm_threshold.is_some_and(|t| adapter_norm >= t);
        k_fire || norm_fire
    }
}

/// Snapshot of a base weight tensor's host data, for consolidation rollback.
/// The merge folds the adapter into the base; if the post-merge gate fails,
/// `restore` rewinds the base to its pre-merge state.
pub struct BaseWeightSnapshot {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl BaseWeightSnapshot {
    /// Capture the base weight's host data. The tensor must be host-resident.
    pub fn capture(weight: TensorId, store: &mut TensorStore) -> Result<Self> {
        let tensor = store
            .get(weight)
            .ok_or(AutogradError::InvalidTensorId(weight))?;
        let shape = tensor.shape.clone();
        let data = store.to_host(weight)?;
        Ok(Self { data, shape })
    }

    /// Restore the base weight to its snapshotted state. Marks the tensor
    /// dirty so the next device sync refreshes the device copy.
    pub fn restore(&self, weight: TensorId, store: &mut TensorStore) -> Result<()> {
        let tensor = store
            .get_mut(weight)
            .ok_or(AutogradError::InvalidTensorId(weight))?;
        if tensor.shape != self.shape {
            return Err(AutogradError::TapeInvariant(Box::leak(
                format!(
                    "BaseWeightSnapshot shape mismatch: live {:?} vs snapshot {:?}",
                    tensor.shape, self.shape
                )
                .into_boxed_str(),
            )));
        }
        tensor.data.copy_from_slice(&self.data);
        tensor.dirty = autograd::tensor::Dirty::Host;
        tensor.device_handle = None;
        Ok(())
    }
}

#[cfg(test)]
mod consolidation_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn cadence_fires_every_k() {
        let mut c = ConsolidationCadence::new(Some(3), None);
        assert!(!c.should_consolidate(0.0));
        assert!(!c.should_consolidate(0.0));
        assert!(c.should_consolidate(0.0));
        assert!(!c.should_consolidate(0.0));
        assert!(!c.should_consolidate(0.0));
        assert!(c.should_consolidate(0.0));
    }

    #[test]
    fn cadence_fires_on_norm_threshold() {
        let mut c = ConsolidationCadence::new(None, Some(1.0));
        assert!(!c.should_consolidate(0.5));
        assert!(c.should_consolidate(1.0));
        assert!(c.should_consolidate(2.0));
    }

    #[test]
    fn cadence_fires_on_either_condition() {
        let mut c = ConsolidationCadence::new(Some(10), Some(1.0));
        assert!(!c.should_consolidate(0.0));
        assert!(c.should_consolidate(1.5)); // norm fires before K
        assert!(!c.should_consolidate(0.0));
    }

    #[test]
    fn base_weight_snapshot_restore_roundtrip() {
        let mut store = TensorStore::with_backend(Arc::new(autograd::CpuBackend));
        let id = store.alloc(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap());
        let snap = BaseWeightSnapshot::capture(id, &mut store).unwrap();
        // Mutate the base (simulating a merge).
        store.get_mut(id).unwrap().data = vec![10.0, 20.0, 30.0, 40.0];
        snap.restore(id, &mut store).unwrap();
        assert_eq!(store.to_host(id).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
