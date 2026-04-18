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

pub fn random_vec(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (u32::MAX >> 8) as f32;
            (unit * 2.0) - 1.0
        })
        .collect()
}

pub fn max_abs_err(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}
