use crate::{
    AutogradError, Result,
    ops::{add_broadcast, matmul, transpose},
    tensor::{GpuTensor, TensorId, TensorStore},
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
        let weight = GpuTensor::new(weight_data, vec![out_features, in_features], true)
            .expect("linear weight init shape is internally consistent");
        let w = store.alloc(weight);

        let b = if with_bias {
            let bias = GpuTensor::new(vec![0.0; out_features], vec![out_features], true)
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

        let weight_t = transpose(self.w, 0, 1, store, tape)?;
        let output = matmul(x, weight_t, store, tape)?;
        if let Some(bias_id) = self.b {
            add_broadcast(output, bias_id, store, tape)
        } else {
            Ok(output)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;

    #[test]
    fn linear_forward_matches_hand_computed() -> Result<()> {
        let mut store = TensorStore::default();
        let mut tape = Tape::new();
        let linear = Linear::new(2, 2, true, &mut store);

        store.tensor_mut(linear.w)?.data = vec![0.5, -1.0, 1.5, 2.0];
        let bias_id = linear.b.expect("bias exists");
        store.tensor_mut(bias_id)?.data = vec![0.25, -0.5];

        let x = store.from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let y = linear.forward(x, &mut store, &mut tape)?;

        assert_eq!(store.to_host(y)?, vec![-1.25, 5.0, -2.25, 12.0]);
        Ok(())
    }
}
