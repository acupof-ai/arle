use autograd::{
    AutogradError, Tensor, Result, Tape, TensorId, TensorStore,
    module::{Linear, Module},
    ops::{add, matmul, mul_scalar, reshape, transpose},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f32,
}

#[derive(Debug, Clone)]
pub struct LoraLinear {
    base: Linear,
    lora_a: TensorId,
    lora_b: TensorId,
    scale: f32,
    in_features: usize,
    out_features: usize,
    rank: usize,
}

impl LoraLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        with_bias: bool,
        cfg: &LoraConfig,
        store: &mut TensorStore,
    ) -> Self {
        assert!(cfg.rank > 0, "LoRA rank must be positive");

        let base = Linear::new(in_features, out_features, with_bias, store);
        let bound = 1.0 / (in_features as f32).sqrt();
        let mut state = 0x4C4F_5241_u32 ^ ((cfg.rank as u32) << 16) ^ out_features as u32;
        let lora_a_data = (0..cfg.rank * in_features)
            .map(|_| sample_uniform(&mut state, bound))
            .collect::<Vec<_>>();
        let lora_a = store.alloc(
            Tensor::new(lora_a_data, vec![cfg.rank, in_features], true)
                .expect("LoRA A init shape is internally consistent"),
        );
        let lora_b = store.alloc(
            Tensor::new(
                vec![0.0; out_features * cfg.rank],
                vec![out_features, cfg.rank],
                true,
            )
            .expect("LoRA B init shape is internally consistent"),
        );

        Self {
            base,
            lora_a,
            lora_b,
            scale: cfg.alpha / cfg.rank as f32,
            in_features,
            out_features,
            rank: cfg.rank,
        }
    }

    pub fn freeze_base(&self, store: &mut TensorStore) {
        self.base.freeze(store);
    }

    pub fn forward(
        &self,
        x: TensorId,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let base_out = self.base.forward(x, store, tape)?;
        let a_out = project_last_dim(x, self.lora_a, self.in_features, self.rank, store, tape)?;
        let b_out = project_last_dim(
            a_out,
            self.lora_b,
            self.rank,
            self.out_features,
            store,
            tape,
        )?;
        let scaled = mul_scalar(b_out, self.scale, store, tape)?;
        add(base_out, scaled, store, tape)
    }

    pub fn trainable_parameters(&self) -> Vec<TensorId> {
        vec![self.lora_a, self.lora_b]
    }

    pub fn base_parameters(&self) -> Vec<TensorId> {
        self.base.parameters()
    }
}

fn project_last_dim(
    x: TensorId,
    weight: TensorId,
    in_features: usize,
    out_features: usize,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let input_shape = store
        .get(x)
        .ok_or(AutogradError::InvalidTensorId(x))?
        .shape
        .clone();
    let input_dim = *input_shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if input_dim != in_features {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![in_features],
            got: vec![input_dim],
        });
    }

    let weight_shape = store
        .get(weight)
        .ok_or(AutogradError::InvalidTensorId(weight))?
        .shape
        .clone();
    if weight_shape != vec![out_features, in_features] {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![out_features, in_features],
            got: weight_shape,
        });
    }

    let prefix_elems = input_shape.iter().product::<usize>() / in_features;
    let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
    output_shape.push(out_features);
    let flat_x = reshape(x, &[prefix_elems, in_features], store, tape)?;
    let weight_t = transpose(weight, 0, 1, store, tape)?;
    let flat_y = matmul(flat_x, weight_t, store, tape)?;
    reshape(flat_y, &output_shape, store, tape)
}

fn sample_uniform(state: &mut u32, bound: f32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let unit = (*state >> 8) as f32 / (u32::MAX >> 8) as f32;
    ((unit * 2.0) - 1.0) * bound
}
