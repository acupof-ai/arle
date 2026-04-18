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
