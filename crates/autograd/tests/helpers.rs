#![allow(dead_code)]

pub fn num_grad<F: Fn(&[f32]) -> f32>(f: F, x: &mut [f32], eps: f32) -> Vec<f32> {
    let mut grads = Vec::with_capacity(x.len());
    for index in 0..x.len() {
        let original = x[index];

        x[index] = original + eps;
        let plus = f(x);

        x[index] = original - eps;
        let minus = f(x);

        x[index] = original;
        grads.push((plus - minus) / (2.0 * eps));
    }
    grads
}

pub fn max_abs_err(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}
