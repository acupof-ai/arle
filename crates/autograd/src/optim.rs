use std::collections::HashMap;

use crate::{TensorId, tensor::TensorStore};

#[derive(Debug, Clone)]
pub struct AdamW {
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    wd: f32,
    step: i32,
    state: HashMap<TensorId, (Vec<f32>, Vec<f32>)>,
}

impl AdamW {
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, wd: f32) -> Self {
        Self {
            lr,
            betas,
            eps,
            wd,
            step: 0,
            state: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &[TensorId], store: &mut TensorStore) {
        self.step += 1;
        let (beta1, beta2) = self.betas;
        let bc1 = 1.0 - beta1.powi(self.step);
        let bc2 = 1.0 - beta2.powi(self.step);

        for &param_id in params {
            let Some(param_snapshot) = store.get(param_id) else {
                panic!("adamw parameter {param_id} does not exist");
            };
            let Some(grad_id) = param_snapshot.grad else {
                continue;
            };

            let grad = store
                .to_host(grad_id)
                .expect("gradient tensor should be readable from the store");
            let param_len = param_snapshot.data.len();
            let state = self
                .state
                .entry(param_id)
                .or_insert_with(|| (vec![0.0; param_len], vec![0.0; param_len]));
            let (m, v) = state;
            let param = store
                .get_mut(param_id)
                .expect("parameter tensor should still exist when stepping");

            if self.wd > 0.0 {
                let decay = 1.0 - (self.lr * self.wd);
                for value in &mut param.data {
                    *value *= decay;
                }
            }

            for index in 0..param.data.len() {
                let g = grad[index];
                m[index] = (beta1 * m[index]) + ((1.0 - beta1) * g);
                v[index] = (beta2 * v[index]) + ((1.0 - beta2) * g * g);
                let m_hat = m[index] / bc1;
                let v_hat = v[index] / bc2;
                param.data[index] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }

    pub fn zero_grad(&mut self, params: &[TensorId], store: &mut TensorStore) {
        for &param_id in params {
            let grad_id = store.get(param_id).and_then(|tensor| tensor.grad);
            if let Some(grad_id) = grad_id
                && let Some(grad) = store.get_mut(grad_id)
            {
                grad.data.fill(0.0);
            }
        }
    }
}
