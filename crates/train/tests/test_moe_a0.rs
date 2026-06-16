use std::collections::{HashMap, HashSet};

use autograd::{
    Result, Tape, Tensor, TensorId, TensorStore,
    ops::{add, mul, mul_scalar, sum},
};
use train::{LoraConfig, MoeConfig, MoeWithLora};

#[test]
fn cpu_moe_lora_and_router_gradients_match_finite_difference() -> Result<()> {
    run_moe_lora_and_router_gradients_match_finite_difference(TensorStore::default(), "cpu")
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn cuda_moe_lora_and_router_gradients_match_finite_difference() -> Result<()> {
    use std::sync::Arc;

    use autograd::{backend::Backend, backend_cuda::CudaBackend};

    let ordinal = std::env::var("ARLE_CUDA_TEST_DEVICE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let backend: Arc<dyn Backend> = Arc::new(CudaBackend::new(ordinal)?);
    run_moe_lora_and_router_gradients_match_finite_difference(
        TensorStore::with_backend(backend),
        "cuda",
    )
}

fn run_moe_lora_and_router_gradients_match_finite_difference(
    mut store: TensorStore,
    backend_label: &'static str,
) -> Result<()> {
    const TOKENS: usize = 5;
    const HIDDEN: usize = 64;
    const EXPERTS: usize = 4;
    const TOP_K: usize = 2;
    const INTERMEDIATE: usize = 128;

    let cfg = MoeConfig {
        hidden_size: HIDDEN,
        intermediate_size: INTERMEDIATE,
        num_experts: EXPERTS,
        top_k: TOP_K,
        lora: LoraConfig {
            rank: 2,
            alpha: 4.0,
        },
    };
    let moe = MoeWithLora::new("a0_moe", cfg, &mut store)?;
    let params = moe.parameter_name_map();
    let adapters = moe.adapter_name_map();

    set_stable_router(&mut store, &params, HIDDEN, EXPERTS);
    set_nonzero_adapters(&mut store, &adapters);

    let input = store.alloc(Tensor::new(
        stable_input(TOKENS, HIDDEN),
        vec![TOKENS, HIDDEN],
        true,
    )?);
    let probe = store.alloc(Tensor::new(
        deterministic_vec(TOKENS * HIDDEN, 0x51d3_7a91, 0.50),
        vec![TOKENS, HIDDEN],
        false,
    )?);

    let mut tape = Tape::new();
    let loss = loss(&moe, input, probe, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;

    let mut checked: Vec<(&'static str, TensorId)> = Vec::new();
    checked.push((
        "a0_moe.router.weight",
        *params
            .get("a0_moe.router.weight")
            .expect("router weight exists"),
    ));
    let mut adapter_entries: Vec<_> = adapters.iter().map(|(&name, &id)| (name, id)).collect();
    adapter_entries.sort_by_key(|(name, _)| *name);
    checked.extend(adapter_entries);

    let analytic: HashMap<TensorId, Vec<f32>> = checked
        .iter()
        .map(|(_, id)| {
            let size = store.get(*id).expect("param exists").size;
            let grad = match grads.get(id).copied() {
                Some(grad_id) => store.to_host(grad_id).expect("grad host"),
                None => vec![0.0; size],
            };
            (*id, grad)
        })
        .collect();
    let probe_data = store.to_host(probe)?;

    let keep: HashSet<TensorId> = params
        .values()
        .chain(adapters.values())
        .copied()
        .chain([input, probe])
        .collect();

    let eps = std::env::var("ARLE_A0_FD_EPS")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(1.0e-3_f32);
    let rel_tol = 1.0e-2_f64;
    let relative_floor = 2.0e-4_f64;
    let tiny_abs_tol = 3.0e-6_f64;
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

    for (name, param_id) in checked {
        let len = store.get(param_id).expect("param exists").size;
        let analytic_values = &analytic[&param_id];
        for (index, &analytic_f32) in analytic_values.iter().enumerate().take(len) {
            let original = store.get(param_id).expect("param exists").data[index];
            set_param_value(&mut store, param_id, index, original + eps);
            let plus = forward_output(&moe, input, &mut store)?;
            store.retain_ids(&keep);

            set_param_value(&mut store, param_id, index, original - eps);
            let minus = forward_output(&moe, input, &mut store)?;
            store.retain_ids(&keep);

            set_param_value(&mut store, param_id, index, original);

            let numeric = plus
                .iter()
                .zip(minus.iter())
                .zip(probe_data.iter())
                .map(|((&plus_value, &minus_value), &probe_value)| {
                    f64::from(plus_value - minus_value) * f64::from(probe_value)
                })
                .sum::<f64>()
                / (2.0 * f64::from(eps));
            let analytic_value = f64::from(analytic_f32);
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
        "a0_moe_finite_diff backend={backend_label} eps={eps:.1e} checked_values={checked_values} \
         relative_values={relative_values} tiny_values={tiny_values} \
         max_abs_at_worst_rel={max_abs_at_worst_rel:.6e} \
         max_rel={max_rel:.6e} worst={worst_name}[{worst_index}] \
         analytic={worst_analytic:.6e} numeric={worst_numeric:.6e} \
         max_tiny_abs={max_tiny_abs:.6e} tiny_abs_failures={tiny_abs_failures}"
    );
    assert!(
        max_rel < rel_tol && tiny_abs_failures == 0,
        "A0 MoE finite diff failed: checked={checked_values} relative={relative_values} \
         tiny={tiny_values} backend={backend_label} eps={eps:.1e} \
         max_abs_at_worst_rel={max_abs_at_worst_rel:.6e} \
         max_rel={max_rel:.6e} worst={worst_name}[{worst_index}] \
         analytic={worst_analytic:.6e} numeric={worst_numeric:.6e} \
         max_tiny_abs={max_tiny_abs:.6e} tiny_abs_failures={tiny_abs_failures}"
    );
    Ok(())
}

fn loss(
    moe: &MoeWithLora,
    input: TensorId,
    probe: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let out = moe.forward(input, store, tape)?;
    let weighted = mul(out, probe, store, tape)?;
    let anchor = mul_scalar(input, 0.0, store, tape)?;
    let weighted = add(weighted, anchor, store, tape)?;
    sum(weighted, store, tape)
}

fn forward_output(moe: &MoeWithLora, input: TensorId, store: &mut TensorStore) -> Result<Vec<f32>> {
    let mut tape = Tape::new();
    let out = moe.forward(input, store, &mut tape)?;
    store.to_host(out)
}

fn set_param_value(store: &mut TensorStore, param_id: TensorId, index: usize, value: f32) {
    store.get_mut(param_id).expect("param exists").data[index] = value;
}

fn set_stable_router(
    store: &mut TensorStore,
    params: &HashMap<&'static str, TensorId>,
    hidden: usize,
    experts: usize,
) {
    let router = *params
        .get("a0_moe.router.weight")
        .expect("router weight exists");
    let tensor = store.get_mut(router).expect("router tensor exists");
    tensor.data.fill(0.0);
    for expert in 0..experts {
        tensor.data[expert * hidden + expert] = 1.0;
    }
}

fn set_nonzero_adapters(store: &mut TensorStore, adapters: &HashMap<&'static str, TensorId>) {
    let mut entries: Vec<_> = adapters.iter().map(|(&name, &id)| (name, id)).collect();
    entries.sort_by_key(|(name, _)| *name);
    for (name, id) in entries {
        let scale = if name.ends_with(".lora_b") {
            0.020
        } else {
            0.015
        };
        let seed = fnv1a(name);
        let tensor = store.get_mut(id).expect("adapter tensor exists");
        tensor.data = deterministic_vec(tensor.size, seed, scale);
    }
}

fn stable_input(tokens: usize, hidden: usize) -> Vec<f32> {
    let route_scores = [
        [5.0, 3.0, 1.0, -1.0],
        [1.0, 5.0, 3.0, -1.0],
        [-1.0, 1.0, 5.0, 3.0],
        [3.0, -1.0, 1.0, 5.0],
        [5.0, -1.0, 3.0, 1.0],
    ];
    let mut data = deterministic_vec(tokens * hidden, 0x7a31_9b25, 0.20);
    for token in 0..tokens {
        for expert in 0..4 {
            data[token * hidden + expert] = route_scores[token][expert];
        }
    }
    data
}

fn deterministic_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 40) as u32) as f32 / ((1_u32 << 24) - 1) as f32;
            (unit * 2.0 - 1.0) * scale
        })
        .collect()
}

fn fnv1a(value: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in value.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
