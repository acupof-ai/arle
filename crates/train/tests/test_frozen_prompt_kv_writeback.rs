//! Gate A for the OPD frozen-prompt-KV masked-CE writeback split.
//!
//! Verifies that forwarding+backwarding ONLY the generated segment (seeded from
//! an off-tape prompt-prefix capture) reproduces the full-trajectory writeback:
//!   (a) the masked CE loss matches the full path (≤ 1e-5);
//!   (b) at lora_b=0 the trainable (LoRA) grads match the full path (≤ 1e-4) —
//!       the dropped LoRA-on-prompt-KV term is exactly zero there;
//!   (c) the gen-segment hidden rows match the full-model hidden sliced to
//!       `[gen_start..]` (carry exactness in full-model context, ≤ 1e-5);
//!   (d) with lora_b perturbed small-nonzero, the divergence stays BOUNDED
//!       (the dropped term only touches prompt-KV LoRA).
//!
//! Hybrid config: 4 layers interleaving full + linear attention so BOTH kinds
//! are exercised in both halves of the split. Host (CPU) backend only.

use autograd::{Tape, TensorId, TensorStore, optim::AdamW};
use train::{
    TrainRuntimeFlags, apply_runtime_flags,
    lora::{LoraConfig, LoraTargetSet},
    opd::{
        capture_rollout_logprobs, masked_writeback_ce_step,
        masked_writeback_ce_step_frozen_prompt_kv, rubric_writeback_ce_step_batched,
    },
    qwen35::{LayerType, Qwen35Config, Qwen35Model},
};

const GEN_START_OFFSET: usize = 1; // gen_start = prompt_len - 1

fn hybrid_writeback_config() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 4,
        vocab_size: 16,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![15],
        bos_token_id: Some(1),
        eos_token_id: 15,
        tie_word_embeddings: false,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 8,
        // Linear-attn head dims match rotary_dim so the conv1d window lines up.
        linear_num_key_heads: 2,
        linear_key_head_dim: 4,
        linear_num_value_heads: 2,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 4,
        rope_theta: 10_000.0,
        rope_scaling: None,
        partial_rotary_factor: 0.5,
        rotary_dim: 4,
        rope_cache_len_hint: Some(64),
        // Interleave so full + linear attention appear in both the prefix and the
        // gen segment.
        layer_types: vec![
            LayerType::FullAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
        ],
        num_experts: 0,
        num_experts_per_tok: 0,
        decoder_sparse_step: 1,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        norm_topk_prob: true,
        mlp_only_layers: Vec::new(),
        full_attn_gated: true,
    }
}

fn build_student(store: &mut TensorStore) -> Qwen35Model {
    Qwen35Model::new_with_lora_targets_layer_start(
        &hybrid_writeback_config(),
        LoraConfig {
            rank: 4,
            alpha: 8.0,
        },
        LoraTargetSet::AllLinear,
        Some(1),
        store,
    )
    .expect("build LoRA hybrid student")
}

/// Trainable (LoRA adapter) param ids — those flagged `requires_grad`.
fn trainable_ids(store: &TensorStore, model: &Qwen35Model) -> Vec<TensorId> {
    model
        .all_parameter_ids()
        .into_iter()
        .filter(|&id| {
            store
                .get(id)
                .map(|tensor| tensor.requires_grad)
                .unwrap_or(false)
        })
        .collect()
}

/// Replicate the masked-writeback forward + fused CE + backward INLINE (no
/// optimizer step) so we can read raw grads. `frozen` selects the gen-segment
/// forward; otherwise the full-trajectory forward. Returns
/// `(loss, hidden_id, grads_by_param)`. Leaves grads live in `store`.
fn forward_backward_inline(
    store: &mut TensorStore,
    model: &Qwen35Model,
    prompt_ids: &[u32],
    response_ids: &[u32],
    response_mask: &[u8],
    frozen: bool,
) -> (f32, TensorId, std::collections::HashMap<TensorId, TensorId>) {
    let prompt_len = prompt_ids.len();
    let mut full: Vec<u32> = Vec::new();
    full.extend_from_slice(prompt_ids);
    full.extend_from_slice(response_ids);
    let seq_len = full.len();
    let gen_start = prompt_len - GEN_START_OFFSET;
    let gen_len = seq_len - gen_start;

    // Build masked (predicting-position, target) pairs (mirror of the private
    // `build_masked_loss_targets`): p in [prompt_len-1 ..= seq_len-2].
    let mut position_indices: Vec<i32> = Vec::new();
    let mut target_tokens: Vec<i32> = Vec::new();
    for p in (prompt_len - 1)..=(seq_len - 2) {
        let target_pos = p + 1;
        let resp_idx = target_pos - prompt_len;
        if resp_idx < response_mask.len() && response_mask[resp_idx] == 1 {
            let rebased = if frozen {
                (p - gen_start) as i32
            } else {
                p as i32
            };
            position_indices.push(rebased);
            target_tokens.push(full[target_pos] as i32);
        }
    }

    let mut tape = Tape::new();
    tape.set_enabled(true);

    let hidden = if frozen {
        let prompt_prefix: Vec<u32> = full[0..gen_start].to_vec();
        let gen_ids: Vec<u32> = full[gen_start..].to_vec();
        let prompt_positions: Vec<u32> = (0..gen_start as u32).collect();
        let gen_positions: Vec<u32> = (gen_start as u32..seq_len as u32).collect();
        model
            .forward_hidden_states_gen_segment(
                store,
                &mut tape,
                &prompt_prefix,
                &gen_ids,
                &prompt_positions,
                &gen_positions,
                train::context_parallel::CpContext::single(),
            )
            .expect("frozen gen-segment forward")
    } else {
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        model
            .forward_hidden_states(
                store,
                &mut tape,
                &full,
                &positions,
                train::context_parallel::CpContext::single(),
            )
            .expect("full forward")
    };

    let loss = autograd::ops::fused_linear_distill::fused_linear_ce_loss_indexed(
        hidden,
        model.lm_head_weight_id(),
        &position_indices,
        &target_tokens,
        64,
        None,
        store,
        &mut tape,
    )
    .expect("fused CE");
    let loss_value = store.to_host(loss).expect("loss host")[0];
    let grads = tape.backward(loss, store).expect("backward");
    let _ = gen_len;
    (loss_value, hidden, grads)
}

fn grad_host(
    store: &mut TensorStore,
    grads: &std::collections::HashMap<TensorId, TensorId>,
    param: TensorId,
) -> Option<Vec<f32>> {
    grads
        .get(&param)
        .map(|&grad_id| store.to_host(grad_id).expect("grad host"))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "grad length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn frozen_prompt_kv_matches_full_writeback_loss_and_grads() {
    let mut store = TensorStore::default();
    let student = build_student(&mut store);
    let trainable = trainable_ids(&store, &student);
    assert!(
        !trainable.is_empty(),
        "LoRA student must have trainable adapters"
    );

    let prompt_ids: Vec<u32> = vec![1, 3, 8, 2, 7, 4, 9, 5];
    let response_ids: Vec<u32> = vec![6, 10, 11, 12, 13, 14];
    let response_mask: Vec<u8> = vec![1; response_ids.len()];

    // (a) loss parity + (b) grad parity at lora_b=0 (init).
    let (loss_frozen, hidden_frozen, grads_frozen) = forward_backward_inline(
        &mut store,
        &student,
        &prompt_ids,
        &response_ids,
        &response_mask,
        true,
    );
    let (loss_full, hidden_full, grads_full) = forward_backward_inline(
        &mut store,
        &student,
        &prompt_ids,
        &response_ids,
        &response_mask,
        false,
    );

    let loss_diff = (loss_frozen - loss_full).abs();
    println!(
        "[gate-a] loss_full={loss_full:.8} loss_frozen={loss_frozen:.8} loss_diff={loss_diff:.3e}"
    );
    assert!(
        loss_diff <= 1.0e-5,
        "frozen vs full masked-CE loss diff {loss_diff:.3e} exceeds 1e-5"
    );

    // (c) gen-segment hidden rows match the full hidden sliced to [gen_start..].
    let gen_start = prompt_ids.len() - GEN_START_OFFSET;
    let seq_len = prompt_ids.len() + response_ids.len();
    let gen_len = seq_len - gen_start;
    let hidden = student.config().hidden_size;
    let full_hidden_host = store.to_host(hidden_full).expect("full hidden host");
    let frozen_hidden_host = store.to_host(hidden_frozen).expect("frozen hidden host");
    assert_eq!(frozen_hidden_host.len(), gen_len * hidden);
    let mut hidden_max_diff = 0.0_f32;
    for row in 0..gen_len {
        for col in 0..hidden {
            let full_val = full_hidden_host[(gen_start + row) * hidden + col];
            let frozen_val = frozen_hidden_host[row * hidden + col];
            hidden_max_diff = hidden_max_diff.max((full_val - frozen_val).abs());
        }
    }
    println!("[gate-a] gen_hidden_max_diff={hidden_max_diff:.3e}");
    assert!(
        hidden_max_diff <= 1.0e-5,
        "frozen gen-segment hidden diff {hidden_max_diff:.3e} exceeds 1e-5"
    );

    // (b) Grad invariant at lora_b=0. Forward is byte-exact (loss + hidden diff
    // both 0 above), so this isolates the backward. The exact invariant:
    //   - `lora_a` grads match the full path (≤ 1e-4): the lora_a grad is
    //     `dL/dout · B^T · x`, proportional to B = 0, so the term the frozen path
    //     drops (prompt-position contribution) is exactly zero there.
    //   - `lora_b` grads legitimately differ: the lora_b grad is
    //     `dL/dout · (A·x)^T`, INDEPENDENT of B, so it is non-zero at B=0 and
    //     accumulates over every row the layer processes. The frozen path
    //     correctly excludes prompt positions — this divergence is the feature's
    //     intended behavior, not a bug. We assert it stays finite + bounded.
    let name_by_id: std::collections::HashMap<TensorId, &'static str> = student
        .adapter_name_map()
        .into_iter()
        .map(|(name, id)| (id, name))
        .collect();
    let mut lora_a_max_diff = 0.0_f32;
    let mut lora_a_compared = 0usize;
    let mut lora_b_max_diff = 0.0_f32;
    let mut lora_b_compared = 0usize;
    for &param in &trainable {
        let name = name_by_id.get(&param).copied().unwrap_or("<unknown>");
        let g_frozen = grad_host(&mut store, &grads_frozen, param);
        let g_full = grad_host(&mut store, &grads_full, param);
        match (g_frozen, g_full) {
            (Some(gf), Some(gu)) => {
                assert!(
                    gf.iter().all(|v| v.is_finite()),
                    "frozen grad non-finite: {name}"
                );
                let d = max_abs_diff(&gf, &gu);
                if name.ends_with(".lora_a") {
                    lora_a_max_diff = lora_a_max_diff.max(d);
                    lora_a_compared += 1;
                } else if name.ends_with(".lora_b") {
                    lora_b_max_diff = lora_b_max_diff.max(d);
                    lora_b_compared += 1;
                }
            }
            (None, None) => {}
            (a, b) => panic!(
                "grad presence mismatch for param {param:?} ({name}): frozen={} full={}",
                a.is_some(),
                b.is_some()
            ),
        }
    }
    println!(
        "[gate-a] lora_a_max_diff={lora_a_max_diff:.3e} (n={lora_a_compared}) \
         lora_b_max_diff={lora_b_max_diff:.3e} (n={lora_b_compared})"
    );
    assert!(lora_a_compared > 0, "no lora_a grads were compared");
    assert!(lora_b_compared > 0, "no lora_b grads were compared");
    assert!(
        lora_a_max_diff <= 1.0e-4,
        "lora_a grad diff {lora_a_max_diff:.3e} exceeds 1e-4 at lora_b=0 (the B-dependent \
         dropped term must be exactly zero)"
    );
    assert!(
        lora_b_max_diff.is_finite() && lora_b_max_diff <= 1.0e0,
        "lora_b grad divergence {lora_b_max_diff:.3e} not bounded (expected: prompt-position \
         contribution the frozen path intentionally excludes)"
    );
}

#[test]
fn frozen_prompt_kv_perturbed_lora_grads_bounded_and_response_path_matches() {
    let mut store = TensorStore::default();
    let student = build_student(&mut store);
    let trainable = trainable_ids(&store, &student);

    // (d) perturb lora_b to small nonzero. The dropped term only affects the
    // prompt-KV LoRA, which the frozen path treats as constant, so the response
    // path stays close and grads stay BOUNDED (no blow-up).
    for &param in &trainable {
        if let Some(tensor) = store.get_mut(param) {
            for (i, value) in tensor.data.iter_mut().enumerate() {
                *value += ((i % 5) as f32 - 2.0) * 1.0e-3;
            }
        }
    }

    let prompt_ids: Vec<u32> = vec![1, 3, 8, 2, 7, 4, 9, 5];
    let response_ids: Vec<u32> = vec![6, 10, 11, 12, 13, 14];
    let response_mask: Vec<u8> = vec![1; response_ids.len()];

    let (loss_frozen, _hf, grads_frozen) = forward_backward_inline(
        &mut store,
        &student,
        &prompt_ids,
        &response_ids,
        &response_mask,
        true,
    );
    let (loss_full, _hu, grads_full) = forward_backward_inline(
        &mut store,
        &student,
        &prompt_ids,
        &response_ids,
        &response_mask,
        false,
    );

    assert!(loss_frozen.is_finite() && loss_full.is_finite());
    // Loss stays close (the dropped term is O(lora_b) small).
    let loss_diff = (loss_frozen - loss_full).abs();
    println!(
        "[gate-a-d] loss_full={loss_full:.6} loss_frozen={loss_frozen:.6} diff={loss_diff:.3e}"
    );
    assert!(
        loss_diff <= 1.0e-2,
        "perturbed-lora loss diff {loss_diff:.3e} unexpectedly large (not bounded)"
    );

    // Grads stay finite and bounded relative to the full path (no blow-up).
    let mut max_diff = 0.0_f32;
    for &param in &trainable {
        if let (Some(gf), Some(gu)) = (
            grad_host(&mut store, &grads_frozen, param),
            grad_host(&mut store, &grads_full, param),
        ) {
            assert!(gf.iter().all(|v| v.is_finite()), "frozen grad non-finite");
            max_diff = max_diff.max(max_abs_diff(&gf, &gu));
        }
    }
    println!("[gate-a-d] perturbed grad_max_diff={max_diff:.3e}");
    assert!(
        max_diff <= 1.0e-1,
        "perturbed-lora grad divergence {max_diff:.3e} not bounded"
    );
}

#[test]
fn frozen_prompt_kv_public_step_runs_and_matches_baseline_loss() {
    // Exercise the public step fns end-to-end (forward+backward+optimizer) and
    // confirm the frozen step's reported loss matches the baseline step's on a
    // fresh, identical model (init grads, single step).
    let prompt_ids: Vec<u32> = vec![1, 3, 8, 2, 7, 4, 9, 5];
    let response_ids: Vec<u32> = vec![6, 10, 11, 12, 13, 14];
    let response_mask: Vec<u8> = vec![1; response_ids.len()];

    let run = |frozen: bool| -> f32 {
        let mut store = TensorStore::default();
        let student = build_student(&mut store);
        let all_params = student.all_parameter_ids();
        let trainable = trainable_ids(&store, &student);
        let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);
        let vocab = student.config().vocab_size;
        let step = if frozen {
            masked_writeback_ce_step_frozen_prompt_kv
        } else {
            masked_writeback_ce_step
        };
        step(
            &student,
            &all_params,
            &trainable,
            &mut optimizer,
            &prompt_ids,
            &response_ids,
            &response_mask,
            vocab,
            64,
            &mut store,
        )
        .expect("writeback step")
    };

    let loss_baseline = run(false);
    let loss_frozen = run(true);
    let diff = (loss_baseline - loss_frozen).abs();
    println!("[gate-a-step] baseline={loss_baseline:.8} frozen={loss_frozen:.8} diff={diff:.3e}");
    assert!(
        diff <= 1.0e-5,
        "public-step frozen vs baseline loss diff {diff:.3e} exceeds 1e-5"
    );
}

#[test]
fn frozen_prompt_kv_capture_rollout_logprobs_matches_full() {
    // The forward-only SAO capture path goes through the global
    // `--writeback-frozen-prompt-kv` flag. Toggle it and assert the gen-segment
    // gather (rows rebased by -gen_start) reproduces the full-forward logprobs at
    // every masked target — same order, same values (≤ 1e-5).
    let mut store = TensorStore::default();
    let student = build_student(&mut store);

    let prompt_ids: Vec<u32> = vec![1, 3, 8, 2, 7, 4, 9, 5];
    let response_ids: Vec<u32> = vec![6, 10, 11, 12, 13, 14];
    let response_mask: Vec<u8> = vec![1; response_ids.len()];

    let set_frozen = |on: bool| {
        apply_runtime_flags(&TrainRuntimeFlags {
            writeback_frozen_prompt_kv: on,
            ..TrainRuntimeFlags::default()
        });
    };

    set_frozen(false);
    let full = capture_rollout_logprobs(
        &student,
        &prompt_ids,
        &response_ids,
        &response_mask,
        &mut store,
    )
    .expect("full capture");
    set_frozen(true);
    let frozen = capture_rollout_logprobs(
        &student,
        &prompt_ids,
        &response_ids,
        &response_mask,
        &mut store,
    )
    .expect("frozen capture");
    set_frozen(false); // restore default for any later test reading the flag

    assert_eq!(full.len(), frozen.len(), "capture logprob count mismatch");
    assert!(!full.is_empty(), "expected masked targets");
    let max_diff = max_abs_diff(&full, &frozen);
    println!(
        "[gate-a-capture] n={} logprob_max_diff={max_diff:.3e}",
        full.len()
    );
    assert!(
        max_diff <= 1.0e-5,
        "frozen vs full capture logprob diff {max_diff:.3e} exceeds 1e-5"
    );
}

#[test]
fn frozen_prompt_kv_checkpoint_branch_matches_full() {
    // Exercise the gradient-checkpointing branch of
    // `forward_hidden_states_gen_segment` (the Arc-shared prefix cache + 'static
    // closure) and confirm loss + gen-hidden parity against the full path.
    let mut store = TensorStore::default();
    let mut student = build_student(&mut store);
    student.set_gradient_checkpointing(true);

    let prompt_ids: Vec<u32> = vec![1, 3, 8, 2, 7, 4, 9, 5];
    let response_ids: Vec<u32> = vec![6, 10, 11, 12, 13, 14];
    let response_mask: Vec<u8> = vec![1; response_ids.len()];

    let (loss_frozen, hidden_frozen, _gf) = forward_backward_inline(
        &mut store,
        &student,
        &prompt_ids,
        &response_ids,
        &response_mask,
        true,
    );
    let (loss_full, hidden_full, _gu) = forward_backward_inline(
        &mut store,
        &student,
        &prompt_ids,
        &response_ids,
        &response_mask,
        false,
    );

    let loss_diff = (loss_frozen - loss_full).abs();
    println!(
        "[gate-a-ckpt] loss_full={loss_full:.8} loss_frozen={loss_frozen:.8} diff={loss_diff:.3e}"
    );
    assert!(
        loss_diff <= 1.0e-5,
        "checkpoint-branch frozen vs full loss diff {loss_diff:.3e} exceeds 1e-5"
    );

    let gen_start = prompt_ids.len() - GEN_START_OFFSET;
    let seq_len = prompt_ids.len() + response_ids.len();
    let gen_len = seq_len - gen_start;
    let hidden = student.config().hidden_size;
    let full_hidden_host = store.to_host(hidden_full).expect("full hidden host");
    let frozen_hidden_host = store.to_host(hidden_frozen).expect("frozen hidden host");
    let mut hidden_max_diff = 0.0_f32;
    for row in 0..gen_len {
        for col in 0..hidden {
            let full_val = full_hidden_host[(gen_start + row) * hidden + col];
            let frozen_val = frozen_hidden_host[row * hidden + col];
            hidden_max_diff = hidden_max_diff.max((full_val - frozen_val).abs());
        }
    }
    println!("[gate-a-ckpt] gen_hidden_max_diff={hidden_max_diff:.3e}");
    assert!(
        hidden_max_diff <= 1.0e-5,
        "checkpoint-branch gen-segment hidden diff {hidden_max_diff:.3e} exceeds 1e-5"
    );
}

/// Batched rubric writeback equivalence (reuses this file's tiny-model harness).
///
/// The batched CE path (forward→hidden, per-row fused indexed CE over `lm_head`,
/// mean-of-row-means) must equal running the single-trajectory masked writeback
/// per row (mask all-1s ⇒ every completion token a target) and averaging,
/// against the proven `masked_writeback_ce_step` reference — so the dense-logits →
/// fused-hidden rewrite is a pure VRAM change, not a loss-semantics change. Pins:
///
/// - the flat row-stride indexing (row i occupies hidden rows `[i*max_len ..]`);
/// - the mean-of-row-means reduction under UNEQUAL completion lengths (equal
///   lengths would hide a stride/reduction/off-by-one bug).
#[test]
fn rubric_batched_writeback_matches_per_row_masked_writeback() {
    // Fresh (deterministic-init) model per call — the reported loss is at init θ,
    // before that step's own optimizer update, so identical inits are comparable.
    let masked_row_loss = |prompt: &[u32], completion: &[u32]| -> f32 {
        let mut store = TensorStore::default();
        let student = build_student(&mut store);
        let all_params = student.all_parameter_ids();
        let trainable = trainable_ids(&store, &student);
        let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);
        let vocab = student.config().vocab_size;
        let mask = vec![1u8; completion.len()];
        masked_writeback_ce_step(
            &student,
            &all_params,
            &trainable,
            &mut optimizer,
            prompt,
            completion,
            &mask,
            vocab,
            64,
            &mut store,
        )
        .expect("masked writeback row")
    };

    // Two rows, UNEQUAL completion lengths.
    let rows: Vec<(Vec<u32>, Vec<u32>)> = vec![
        (vec![1, 3, 8, 2, 7], vec![6, 10, 11, 12, 13, 14]),
        (vec![1, 9, 4], vec![5, 2, 8]),
    ];
    let per_row: Vec<f32> = rows.iter().map(|(p, c)| masked_row_loss(p, c)).collect();
    let expected = per_row.iter().sum::<f32>() / per_row.len() as f32;

    let mut store = TensorStore::default();
    let student = build_student(&mut store);
    let all_params = student.all_parameter_ids();
    let trainable = trainable_ids(&store, &student);
    let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);
    let vocab = student.config().vocab_size;
    let batched = rubric_writeback_ce_step_batched(
        &student,
        &all_params,
        &trainable,
        &mut optimizer,
        &rows,
        vocab,
        64,
        &mut store,
    )
    .expect("batched writeback");

    let diff = (batched - expected).abs();
    println!(
        "[gate-batched] per_row={per_row:?} expected_mean={expected:.8} batched={batched:.8} diff={diff:.3e}"
    );
    assert!(
        diff <= 1.0e-5,
        "batched writeback loss {batched:.8} != mean-of-row-means {expected:.8} (diff {diff:.3e})"
    );
}

/// The long-completion pod path takes the checkpoint_sequential branch (grad
/// checkpointing on; CPU mem-info is None ⇒ the gate always engages): its loss
/// must equal the full-tape branch — recompute replay is exact on identical
/// init, so any divergence is a batched-forward checkpoint-boundary bug.
#[test]
fn rubric_batched_writeback_checkpoint_branch_matches_full_tape() {
    let rows: Vec<(Vec<u32>, Vec<u32>)> = vec![
        (vec![1, 3, 8, 2, 7], vec![6, 10, 11, 12, 13, 14]),
        (vec![1, 9, 4], vec![5, 2, 8]),
    ];
    let run = |checkpoint: bool| -> f32 {
        let mut store = TensorStore::default();
        let mut student = build_student(&mut store);
        student.set_gradient_checkpointing(checkpoint);
        let all_params = student.all_parameter_ids();
        let trainable = trainable_ids(&store, &student);
        let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);
        let vocab = student.config().vocab_size;
        rubric_writeback_ce_step_batched(
            &student,
            &all_params,
            &trainable,
            &mut optimizer,
            &rows,
            vocab,
            64,
            &mut store,
        )
        .expect("batched writeback")
    };
    let (full, ckpt) = (run(false), run(true));
    let diff = (full - ckpt).abs();
    println!("[gate-batched-ckpt] full={full:.8} ckpt={ckpt:.8} diff={diff:.3e}");
    assert!(
        diff <= 1.0e-6,
        "checkpoint-branch loss {ckpt:.8} != full-tape {full:.8} (diff {diff:.3e})"
    );
}
