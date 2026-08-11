use crate::{
    AutogradError, Result,
    ops::{add_broadcast, matmul, reshape, transpose},
    tensor::{Tensor, TensorId, TensorStore},
};

pub trait Parameter {
    fn id(&self) -> TensorId;

    fn requires_grad(&self) -> bool {
        true
    }
}

impl Parameter for TensorId {
    fn id(&self) -> TensorId {
        *self
    }
}

pub trait Module {
    fn parameters(&self) -> Vec<TensorId>;
}

#[derive(Debug, Clone)]
pub struct Linear {
    w: TensorId,
    b: Option<TensorId>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        with_bias: bool,
        store: &mut TensorStore,
    ) -> Self {
        let bound = 1.0 / (in_features as f32).sqrt();
        let mut state = 0x9E37_79B9_u32 ^ ((in_features as u32) << 16) ^ out_features as u32;
        let weight_data = (0..out_features * in_features)
            .map(|_| sample_uniform(&mut state, bound))
            .collect::<Vec<_>>();
        let weight = Tensor::new(weight_data, vec![out_features, in_features], true)
            .expect("linear weight init shape is internally consistent");
        let w = store.alloc(weight);

        let b = if with_bias {
            let bias = Tensor::new(vec![0.0; out_features], vec![out_features], true)
                .expect("linear bias init shape is internally consistent");
            Some(store.alloc(bias))
        } else {
            None
        };

        Self {
            w,
            b,
            in_features,
            out_features,
        }
    }

    pub fn forward(
        &self,
        x: TensorId,
        store: &mut TensorStore,
        tape: &mut crate::Tape,
    ) -> Result<TensorId> {
        let x_shape = store.tensor(x)?.shape.clone();
        let input_dim = *x_shape.last().ok_or(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        })?;
        if input_dim != self.in_features {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![self.in_features],
                got: vec![input_dim],
            });
        }

        let weight_shape = store.tensor(self.w)?.shape.clone();
        if weight_shape != vec![self.out_features, self.in_features] {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![self.out_features, self.in_features],
                got: weight_shape,
            });
        }
        if let Some(bias_id) = self.b {
            let bias_shape = store.tensor(bias_id)?.shape.clone();
            if bias_shape != vec![self.out_features] {
                return Err(AutogradError::ShapeMismatch {
                    expected: vec![self.out_features],
                    got: bias_shape,
                });
            }
        }

        let prefix_elems = x_shape.iter().product::<usize>() / self.in_features;
        let mut output_shape = x_shape[..x_shape.len() - 1].to_vec();
        output_shape.push(self.out_features);
        let flat_x = reshape(x, &[prefix_elems, self.in_features], store, tape)?;
        let weight_t = transpose(self.w, 0, 1, store, tape)?;
        let output = matmul(flat_x, weight_t, store, tape)?;
        let output = reshape(output, &output_shape, store, tape)?;
        if let Some(bias_id) = self.b {
            add_broadcast(output, bias_id, store, tape)
        } else {
            Ok(output)
        }
    }

    pub fn freeze(&self, store: &mut TensorStore) {
        store
            .get_mut(self.w)
            .expect("linear weight must exist while freezing")
            .requires_grad = false;
        if let Some(bias_id) = self.b {
            store
                .get_mut(bias_id)
                .expect("linear bias must exist while freezing")
                .requires_grad = false;
        }
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<TensorId> {
        let mut params = vec![self.w];
        if let Some(bias_id) = self.b {
            params.push(bias_id);
        }
        params
    }
}

fn sample_uniform(state: &mut u32, bound: f32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let unit = (*state >> 8) as f32 / (u32::MAX >> 8) as f32;
    ((unit * 2.0) - 1.0) * bound
}
