use std::{collections::HashMap, f32::consts::TAU};

use autograd::{
    AutogradError, Result, Tape, Tensor, TensorId, TensorStore,
    ops::{add, matmul, mul_scalar, reshape, transpose},
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
        let weight = normal_parameter(
            base_name,
            &[out_features, in_features],
            0.02,
            base_requires_grad,
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
        let weight_t = transpose(self.weight, 0, 1, store, tape)?;
        let mut projected = matmul(flat_x, weight_t, store, tape)?;
        if let Some(lora) = &self.lora {
            let a_t = transpose(lora.lora_a, 0, 1, store, tape)?;
            let b_t = transpose(lora.lora_b, 0, 1, store, tape)?;
            let low_rank = matmul(flat_x, a_t, store, tape)?;
            let delta = matmul(low_rank, b_t, store, tape)?;
            let delta = mul_scalar(delta, lora.scale, store, tape)?;
            projected = add(projected, delta, store, tape)?;
        }

        let mut output_shape = x_shape[..x_shape.len() - 1].to_vec();
        output_shape.push(weight_shape[0]);
        reshape(projected, &output_shape, store, tape)
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

fn seed_from_name(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn next_uniform(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    bits as f32 / (u32::MAX >> 8) as f32
}

fn leak_name(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

fn tape_invariant(message: String) -> AutogradError {
    AutogradError::TapeInvariant(Box::leak(message.into_boxed_str()))
}
