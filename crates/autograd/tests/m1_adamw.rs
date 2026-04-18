use autograd::{GpuTensor, TensorStore, optim::AdamW};

#[test]
fn adamw_matches_reference_update_for_ten_steps() {
    let mut store = TensorStore::default();
    let param = store.alloc(
        GpuTensor::new(vec![0.25, -0.5, 1.5], vec![3], true)
            .expect("parameter shape is internally consistent"),
    );
    let grad = store.alloc(
        GpuTensor::new(vec![0.1, -0.2, 0.3], vec![3], false)
            .expect("gradient shape is internally consistent"),
    );
    store.get_mut(param).expect("parameter exists").grad = Some(grad);

    let mut optim = AdamW::new(0.01, (0.9, 0.999), 1e-8, 0.05);

    let mut reference = [0.25_f32, -0.5, 1.5];
    let grad_values = [0.1_f32, -0.2, 0.3];
    let mut m = [0.0_f32; 3];
    let mut v = [0.0_f32; 3];

    for step in 1..=10 {
        optim.step(&[param], &mut store);

        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let lr = 0.01_f32;
        let wd = 0.05_f32;
        let eps = 1e-8_f32;
        let bc1 = 1.0 - beta1.powi(step);
        let bc2 = 1.0 - beta2.powi(step);
        for index in 0..reference.len() {
            let g = grad_values[index];
            if wd > 0.0 {
                reference[index] *= 1.0 - (lr * wd);
            }
            m[index] = (beta1 * m[index]) + ((1.0 - beta1) * g);
            v[index] = (beta2 * v[index]) + ((1.0 - beta2) * g * g);
            let m_hat = m[index] / bc1;
            let v_hat = v[index] / bc2;
            reference[index] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    let actual = store.to_host(param).expect("parameter host copy");
    for (lhs, rhs) in actual.iter().zip(reference.iter()) {
        assert!((lhs - rhs).abs() < 1e-5, "{lhs} vs {rhs}");
    }
}
