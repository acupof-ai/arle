use std::collections::HashSet;

use autograd::{
    Result, Tape, Tensor, TensorId, TensorStore,
    ops::{mul, sum},
};
use train::{LinearWithLora, LoraConfig};

#[test]
fn cpu_fp8_frozen_base_lora_gradients_match_finite_difference() -> Result<()> {
    run_fp8_frozen_base_lora_gradients_match_finite_difference(TensorStore::default(), "cpu")
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_fp8_frozen_base_lora_gradients_match_finite_difference() -> Result<()> {
    use std::sync::Arc;

    use autograd::{backend::Backend, backend_cuda::CudaBackend};

    let ordinal = std::env::var("ARLE_CUDA_TEST_DEVICE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let backend: Arc<dyn Backend> = Arc::new(CudaBackend::new(ordinal)?);
    run_fp8_frozen_base_lora_gradients_match_finite_difference(
        TensorStore::with_backend(backend),
        "cuda",
    )
}

fn run_fp8_frozen_base_lora_gradients_match_finite_difference(
    mut store: TensorStore,
    backend_label: &'static str,
) -> Result<()> {
    const M: usize = 3;
    const K: usize = 4;
    const N: usize = 5;
    const R: usize = 2;
    const LORA_B_INDEX: usize = 3;
    const INPUT_INDEX: usize = 5;

    let mut linear = LinearWithLora::new(
        "qlora_fp8.proj.weight",
        K,
        N,
        false,
        Some(LoraConfig {
            rank: R,
            alpha: 4.0,
        }),
        &mut store,
    )?;
    linear.set_base_weight_to_fp8_block_scaled(
        &fp8_weight_bytes(),
        &fp8_block_scales(),
        2,
        2,
        &mut store,
    )?;
    let parts = linear.parts();
    let lora_b = parts.lora_b.expect("LoRA B exists");
    let input = store.alloc(Tensor::new(
        deterministic_vec(M * K, 0x58f1_a73d, 0.35),
        vec![M, K],
        true,
    )?);
    let probe = store.alloc(Tensor::new(
        deterministic_vec(M * N, 0x91c3_4e2d, 0.40),
        vec![M, N],
        false,
    )?);

    let mut tape = Tape::new();
    let loss = loss_value_id(&linear, input, probe, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;
    let lora_b_grad = grad_host(&mut store, &grads, lora_b)?;
    let input_grad = grad_host(&mut store, &grads, input)?;

    let keep: HashSet<TensorId> = [
        parts.weight,
        parts.lora_a.expect("LoRA A exists"),
        lora_b,
        input,
        probe,
    ]
    .into_iter()
    .collect();
    store.retain_ids(&keep);

    let eps = 1.0e-3_f32;
    let lora_numeric = central_diff(&linear, lora_b, LORA_B_INDEX, input, probe, eps, &mut store)?;
    store.retain_ids(&keep);
    let input_numeric = central_diff(&linear, input, INPUT_INDEX, input, probe, eps, &mut store)?;

    let lora_rel = rel_err(lora_b_grad[LORA_B_INDEX], lora_numeric);
    let input_rel = rel_err(input_grad[INPUT_INDEX], input_numeric);
    eprintln!(
        "qlora_fp8_fd backend={backend_label} eps={eps:.1e} \
         lora_b[{LORA_B_INDEX}] analytic={:.9e} numeric={:.9e} rel_err={:.3e} \
         input[{INPUT_INDEX}] analytic={:.9e} numeric={:.9e} rel_err={:.3e}",
        lora_b_grad[LORA_B_INDEX],
        lora_numeric,
        lora_rel,
        input_grad[INPUT_INDEX],
        input_numeric,
        input_rel
    );
    assert!(
        lora_rel <= 1.0e-2,
        "FP8 frozen-base LoRA-B finite diff failed on {backend_label}: analytic={} numeric={} rel={}",
        lora_b_grad[LORA_B_INDEX],
        lora_numeric,
        lora_rel
    );
    assert!(
        input_rel <= 1.0e-2,
        "FP8 frozen-base input-gradient finite diff failed on {backend_label}: analytic={} numeric={} rel={}",
        input_grad[INPUT_INDEX],
        input_numeric,
        input_rel
    );
    Ok(())
}

fn loss_value_id(
    linear: &LinearWithLora,
    input: TensorId,
    probe: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let output = linear.forward(input, store, tape)?;
    let weighted = mul(output, probe, store, tape)?;
    sum(weighted, store, tape)
}

fn loss_value(
    linear: &LinearWithLora,
    input: TensorId,
    probe: TensorId,
    store: &mut TensorStore,
) -> Result<f32> {
    let mut tape = Tape::new();
    let loss = loss_value_id(linear, input, probe, store, &mut tape)?;
    Ok(store.to_host(loss)?[0])
}

fn central_diff(
    linear: &LinearWithLora,
    param: TensorId,
    index: usize,
    input: TensorId,
    probe: TensorId,
    eps: f32,
    store: &mut TensorStore,
) -> Result<f32> {
    let original = store.to_host(param)?[index];
    set_value(store, param, index, original + eps);
    let plus = loss_value(linear, input, probe, store)?;
    set_value(store, param, index, original - eps);
    let minus = loss_value(linear, input, probe, store)?;
    set_value(store, param, index, original);
    Ok((plus - minus) / (2.0 * eps))
}

fn grad_host(
    store: &mut TensorStore,
    grads: &std::collections::HashMap<TensorId, TensorId>,
    id: TensorId,
) -> Result<Vec<f32>> {
    match grads.get(&id).copied() {
        Some(grad_id) => store.to_host(grad_id),
        None => Ok(vec![0.0; store.get(id).expect("tensor exists").size]),
    }
}

fn set_value(store: &mut TensorStore, id: TensorId, index: usize, value: f32) {
    store.get_mut(id).expect("tensor exists").data[index] = value;
}

fn rel_err(analytic: f32, numeric: f32) -> f64 {
    let analytic = f64::from(analytic);
    let numeric = f64::from(numeric);
    (analytic - numeric).abs() / analytic.abs().max(numeric.abs()).max(1.0e-12)
}

fn fp8_weight_bytes() -> Vec<u8> {
    vec![
        0x38, 0xb8, 0x30, 0x40, 0x34, 0xbc, 0x3c, 0xb0, 0x3a, 0xba, 0x32, 0x42, 0xbe, 0x36, 0xb4,
        0x3e, 0x28, 0xa8, 0x48, 0xc0,
    ]
}

fn fp8_block_scales() -> Vec<f32> {
    vec![1.0, 0.5, 1.25, 0.75, 1.5, 0.625]
}

fn deterministic_vec(len: usize, mut state: u64, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = ((state >> 32) as u32) as f32 / u32::MAX as f32;
            (bits * 2.0 - 1.0) * scale
        })
        .collect()
}
