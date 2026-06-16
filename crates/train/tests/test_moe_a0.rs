use std::collections::{HashMap, HashSet};

use autograd::{
    Result, Tape, Tensor, TensorId, TensorStore,
    ops::{add, mul, mul_scalar, sum},
};
use train::{LoraConfig, MoeConfig, MoeWithLora};

#[test]
fn moe_lora_and_router_gradients_match_finite_difference() -> Result<()> {
    const TOKENS: usize = 5;
    const HIDDEN: usize = 64;
    const EXPERTS: usize = 4;
    const TOP_K: usize = 2;
    const INTERMEDIATE: usize = 128;

    let mut store = TensorStore::default();
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
    let target = store.alloc(Tensor::new(
        deterministic_vec(TOKENS * HIDDEN, 0x51d3_7a91, 0.10),
        vec![TOKENS, HIDDEN],
        false,
    )?);

    let mut tape = Tape::new();
    let loss = loss(&moe, input, target, &mut store, &mut tape)?;
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

    let keep: HashSet<TensorId> = params
        .values()
        .chain(adapters.values())
        .copied()
        .chain([input, target])
        .collect();

    let eps = 1.0e-2_f32;
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut checked_values = 0usize;
    let mut worst_name = "";
    let mut worst_index = 0usize;
    let mut worst_analytic = 0.0_f32;
    let mut worst_numeric = 0.0_f32;

    for (name, param_id) in checked {
        let len = store.get(param_id).expect("param exists").size;
        for index in 0..len {
            let original = store.get(param_id).expect("param exists").data[index];
            set_param_value(&mut store, param_id, index, original + eps);
            let plus = scalar_loss(&moe, input, target, &mut store)?;
            store.retain_ids(&keep);

            set_param_value(&mut store, param_id, index, original - eps);
            let minus = scalar_loss(&moe, input, target, &mut store)?;
            store.retain_ids(&keep);

            set_param_value(&mut store, param_id, index, original);

            let numeric = (plus - minus) / (2.0 * eps);
            let analytic_value = analytic[&param_id][index];
            let abs = (analytic_value - numeric).abs();
            let rel = abs / analytic_value.abs().max(numeric.abs()).max(1.0e-6);
            if abs > max_abs {
                max_abs = abs;
                max_rel = rel;
                worst_name = name;
                worst_index = index;
                worst_analytic = analytic_value;
                worst_numeric = numeric;
            }
            checked_values += 1;
        }
    }

    eprintln!(
        "a0_moe_finite_diff checked_values={checked_values} max_abs={max_abs:.6e} \
         max_rel={max_rel:.6e} worst={worst_name}[{worst_index}] \
         analytic={worst_analytic:.6e} numeric={worst_numeric:.6e}"
    );
    assert!(
        max_abs < 1.0e-3,
        "A0 MoE finite diff failed: checked={checked_values} max_abs={max_abs:.6e} \
         max_rel={max_rel:.6e} worst={worst_name}[{worst_index}] \
         analytic={worst_analytic:.6e} numeric={worst_numeric:.6e}"
    );
    Ok(())
}

fn loss(
    moe: &MoeWithLora,
    input: TensorId,
    target: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let out = moe.forward(input, store, tape)?;
    let neg_target = mul_scalar(target, -1.0, store, tape)?;
    let diff = add(out, neg_target, store, tape)?;
    let sq = mul(diff, diff, store, tape)?;
    sum(sq, store, tape)
}

fn scalar_loss(
    moe: &MoeWithLora,
    input: TensorId,
    target: TensorId,
    store: &mut TensorStore,
) -> Result<f32> {
    let mut tape = Tape::new();
    let loss = loss(moe, input, target, store, &mut tape)?;
    Ok(store.to_host(loss)?[0])
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
    let mut data = deterministic_vec(tokens * hidden, 0x7a31_9b25, 0.005);
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
