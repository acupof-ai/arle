use std::collections::{HashMap, HashSet};

use autograd::{
    Result, Tape, Tensor, TensorId, TensorStore,
    ops::{causal_sdpa_recompute, mul, sum},
};

#[test]
fn cpu_causal_sdpa_recompute_gradients_match_finite_difference() -> Result<()> {
    run_causal_sdpa_recompute_gradients_match_finite_difference(TensorStore::default(), "cpu")
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_causal_sdpa_recompute_gradients_match_finite_difference() -> Result<()> {
    use std::sync::Arc;

    use autograd::{backend::Backend, backend_cuda::CudaBackend};

    let ordinal = std::env::var("ARLE_CUDA_TEST_DEVICE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let backend: Arc<dyn Backend> = Arc::new(CudaBackend::new(ordinal)?);
    run_causal_sdpa_recompute_gradients_match_finite_difference(
        TensorStore::with_backend(backend),
        "cuda",
    )
}

fn run_causal_sdpa_recompute_gradients_match_finite_difference(
    mut store: TensorStore,
    backend_label: &'static str,
) -> Result<()> {
    const BATCH: usize = 1;
    const HEADS: usize = 2;
    const SEQ_LEN: usize = 4;
    const HEAD_DIM: usize = 256;
    let shape = vec![BATCH, HEADS, SEQ_LEN, HEAD_DIM];
    let size = shape.iter().product();

    let q = store.alloc(Tensor::new(
        deterministic_vec(size, 0x4d59_5df4, 0.05),
        shape.clone(),
        true,
    )?);
    let k = store.alloc(Tensor::new(
        deterministic_vec(size, 0x7a35_2b19, 0.05),
        shape.clone(),
        true,
    )?);
    let v = store.alloc(Tensor::new(
        deterministic_vec(size, 0x1f12_d9e7, 0.25),
        shape.clone(),
        true,
    )?);
    let probe = store.alloc(Tensor::new(
        deterministic_vec(size, 0x29c6_a413, 0.40),
        shape.clone(),
        false,
    )?);

    let mut tape = Tape::new();
    let loss = loss(q, k, v, probe, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let checked = [("q", q), ("k", k), ("v", v)];
    let analytic: HashMap<TensorId, Vec<f32>> = checked
        .iter()
        .map(|(_, id)| {
            let len = store.get(*id).expect("tensor exists").size;
            let grad = match grads.get(id).copied() {
                Some(grad_id) => store.to_host(grad_id).expect("grad host"),
                None => vec![0.0; len],
            };
            (*id, grad)
        })
        .collect();
    let probe_data = store.to_host(probe)?;
    let keep: HashSet<TensorId> = [q, k, v, probe].into_iter().collect();
    store.retain_ids(&keep);

    let eps = std::env::var("ARLE_A1_FD_EPS")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(1.0e-3_f32);
    let max_per_tensor = std::env::var("ARLE_A1_FD_MAX_PER_TENSOR")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(if backend_label == "cuda" { 128 } else { size });
    let rel_tol = 1.0e-2_f64;
    let relative_floor = 2.0e-4_f64;
    let tiny_abs_tol = 2.0e-5_f64;
    let mut max_abs_at_worst_rel = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut max_tiny_abs = 0.0_f64;
    let mut checked_values = 0usize;
    let mut relative_values = 0usize;
    let mut tiny_values = 0usize;
    let mut tiny_abs_failures = 0usize;
    let mut worst_name = "";
    let mut worst_index = 0usize;
    let mut worst_analytic = 0.0_f64;
    let mut worst_numeric = 0.0_f64;

    for (name, tensor_id) in checked {
        let indices = checked_indices(size, max_per_tensor);
        let analytic_values = &analytic[&tensor_id];
        for index in indices {
            let original = store.get(tensor_id).expect("tensor exists").data[index];
            set_value(&mut store, tensor_id, index, original + eps);
            let plus = forward_output(q, k, v, &mut store)?;
            store.retain_ids(&keep);

            set_value(&mut store, tensor_id, index, original - eps);
            let minus = forward_output(q, k, v, &mut store)?;
            store.retain_ids(&keep);

            set_value(&mut store, tensor_id, index, original);

            let numeric = plus
                .iter()
                .zip(minus.iter())
                .zip(probe_data.iter())
                .map(|((&plus_value, &minus_value), &probe_value)| {
                    f64::from(plus_value - minus_value) * f64::from(probe_value)
                })
                .sum::<f64>()
                / (2.0 * f64::from(eps));
            let analytic_value = f64::from(analytic_values[index]);
            let abs = (analytic_value - numeric).abs();
            let denom = analytic_value.abs().max(numeric.abs());
            if denom < relative_floor {
                tiny_values += 1;
                max_tiny_abs = max_tiny_abs.max(abs);
                if abs > tiny_abs_tol {
                    tiny_abs_failures += 1;
                }
            } else {
                relative_values += 1;
                let rel = abs / denom;
                if rel > max_rel {
                    max_abs_at_worst_rel = abs;
                    max_rel = rel;
                    worst_name = name;
                    worst_index = index;
                    worst_analytic = analytic_value;
                    worst_numeric = numeric;
                }
            }
            checked_values += 1;
        }
    }

    eprintln!(
        "a1_attention_finite_diff backend={backend_label} eps={eps:.1e} \
         max_per_tensor={max_per_tensor} checked_values={checked_values} \
         relative_values={relative_values} tiny_values={tiny_values} \
         max_abs_at_worst_rel={max_abs_at_worst_rel:.6e} \
         max_rel={max_rel:.6e} worst={worst_name}[{worst_index}] \
         analytic={worst_analytic:.6e} numeric={worst_numeric:.6e} \
         max_tiny_abs={max_tiny_abs:.6e} tiny_abs_failures={tiny_abs_failures}"
    );
    assert!(
        max_rel < rel_tol && tiny_abs_failures == 0,
        "A1 attention finite diff failed: backend={backend_label} eps={eps:.1e} \
         checked={checked_values} relative={relative_values} tiny={tiny_values} \
         max_abs_at_worst_rel={max_abs_at_worst_rel:.6e} \
         max_rel={max_rel:.6e} worst={worst_name}[{worst_index}] \
         analytic={worst_analytic:.6e} numeric={worst_numeric:.6e} \
         max_tiny_abs={max_tiny_abs:.6e} tiny_abs_failures={tiny_abs_failures}"
    );
    Ok(())
}

fn loss(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    probe: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let out = causal_sdpa_recompute(q, k, v, store, tape)?;
    let weighted = mul(out, probe, store, tape)?;
    sum(weighted, store, tape)
}

fn forward_output(
    q: TensorId,
    k: TensorId,
    v: TensorId,
    store: &mut TensorStore,
) -> Result<Vec<f32>> {
    let mut tape = Tape::new();
    tape.set_enabled(false);
    let out = causal_sdpa_recompute(q, k, v, store, &mut tape)?;
    store.to_host(out)
}

fn set_value(store: &mut TensorStore, tensor_id: TensorId, index: usize, value: f32) {
    store.get_mut(tensor_id).expect("tensor exists").data[index] = value;
}

fn checked_indices(len: usize, max_count: usize) -> Vec<usize> {
    if max_count >= len {
        return (0..len).collect();
    }
    let mut out = Vec::with_capacity(max_count + 1);
    for i in 0..max_count {
        let index = i * len / max_count;
        if out.last().copied() != Some(index) {
            out.push(index);
        }
    }
    if out.last().copied() != Some(len - 1) {
        out.push(len - 1);
    }
    out
}

fn deterministic_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((state >> 40) as f32) / ((1u64 << 24) as f32);
        out.push((unit - 0.5) * scale);
    }
    out
}
