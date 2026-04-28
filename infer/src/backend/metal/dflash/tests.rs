use std::{env, path::PathBuf};

use super::*;
use crate::backend::metal::{
    config::{MetalModelArch, load_metal_config},
    mlx::{Dtype, as_dtype, eval, expand_dims, reshape, slice},
};
use crate::test_support::metal_test_guard;

fn dflash_fc_input_dim(weight: &WeightTensor) -> i32 {
    match weight {
        WeightTensor::Dense(w) => w.shape()[0],
        WeightTensor::Quantized {
            scales, group_size, ..
        } => scales.shape()[1] * *group_size,
        WeightTensor::GgufPacked { cols, .. }
        | WeightTensor::GgufPackedInputReordered { cols, .. } => *cols,
    }
}

#[test]
fn draft_forward_batched_matches_forward_for_b1() -> Result<()> {
    let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
        eprintln!("QWEN35_MODEL_PATH unset; skipping DFlash draft batched B=1 equivalence test");
        return Ok(());
    };
    eprintln!("draft_forward_batched_matches_forward_for_b1 env_ready");
    let _guard = metal_test_guard();
    eprintln!("draft_forward_batched_matches_forward_for_b1 guard_ready");

    let target_config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(_) = &target_config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    eprintln!("draft_forward_batched_matches_forward_for_b1 config_ready");

    let runtime = match MetalDflashRuntime::load(
        &MetalDflashOptions {
            draft_model: "z-lab/Qwen3.5-4B-DFlash".to_string(),
            speculative_tokens: None,
        },
        &target_config,
    ) {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!(
                "DFlash draft model unavailable ({err:#}); skipping draft_forward_batched_matches_forward_for_b1. Set `QWEN35_DFLASH_DRAFT_PATH` to a local checkpoint to enable."
            );
            return Ok(());
        }
    };
    eprintln!("draft_forward_batched_matches_forward_for_b1 runtime_loaded");
    let cpp_model = runtime
        .draft_cpp_model
        .as_ref()
        .context("DFlash draft C++ model unavailable")?;

    let seq = 3_i32;
    let context_len = 2_i32;
    let hidden_size = runtime.draft_config.hidden_size as i32;
    let target_hidden_width = dflash_fc_input_dim(&runtime.draft_weights.fc);

    let noise_data: Vec<f32> = (0..(seq * hidden_size))
        .map(|idx| idx as f32 / 128.0)
        .collect();
    let target_data: Vec<f32> = (0..(context_len * target_hidden_width))
        .map(|idx| (idx as f32 - 17.0) / 256.0)
        .collect();
    let noise_embedding = MlxArray::from_slice_f32(&noise_data, &[seq, hidden_size]);
    let target_hidden = MlxArray::from_slice_f32(&target_data, &[context_len, target_hidden_width]);

    let initial_tokens = usize::try_from(context_len + seq + 4).unwrap_or_default();
    let mut scalar_state = ContiguousKvState::new(
        runtime.draft_config.num_hidden_layers,
        runtime.draft_config.num_key_value_heads as i32,
        runtime.draft_config.head_dim as i32,
        initial_tokens,
    );
    let scalar_hidden = dflash_draft_forward_cpp(
        cpp_model,
        &noise_embedding,
        &target_hidden,
        &mut scalar_state,
    )?;
    eprintln!("draft_forward_batched_matches_forward_for_b1 scalar_done");

    let noise_embedding_batched = expand_dims(&noise_embedding, 0);
    let target_hidden_batched = expand_dims(&target_hidden, 0);
    let q_offsets = MlxArray::from_slice_i32(&[context_len], &[1]);
    let k_offsets = MlxArray::from_slice_i32(&[0], &[1]);

    let batched_state = ContiguousKvState::new(
        runtime.draft_config.num_hidden_layers,
        runtime.draft_config.num_key_value_heads as i32,
        runtime.draft_config.head_dim as i32,
        initial_tokens,
    );
    let batched_kv = batched_state.active_kv_flat();
    let (batched_hidden, _) = cpp_model.forward_batched(
        &noise_embedding_batched,
        &target_hidden_batched,
        1,
        &q_offsets,
        &k_offsets,
        &batched_kv,
        None,
    )?;
    eprintln!("draft_forward_batched_matches_forward_for_b1 batched_done");

    let batched_row0 = slice(
        &batched_hidden,
        &[0, 0, 0],
        &[1, seq, hidden_size],
        &[1, 1, 1],
    );
    let batched_row0 = reshape(&batched_row0, &[seq, hidden_size]);

    let scalar_hidden_f32 = as_dtype(&scalar_hidden, Dtype::Float32);
    let batched_hidden_f32 = as_dtype(&batched_row0, Dtype::Float32);
    eval(&[&scalar_hidden_f32, &batched_hidden_f32]);

    assert_eq!(scalar_hidden_f32.shape(), batched_hidden_f32.shape());
    let mut max_abs_delta = 0.0_f32;
    for (idx, (lhs, rhs)) in scalar_hidden_f32
        .as_slice_f32()
        .iter()
        .zip(batched_hidden_f32.as_slice_f32().iter())
        .enumerate()
    {
        let delta = (lhs - rhs).abs();
        max_abs_delta = max_abs_delta.max(delta);
        assert!(
            delta < 1e-3,
            "hidden[{idx}] mismatch: {lhs} vs {rhs} (|delta|={delta})"
        );
    }
    eprintln!("draft_forward_batched_matches_forward_for_b1 max_abs_delta={max_abs_delta}");

    Ok(())
}

#[test]
fn qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1() -> Result<()> {
    use super::super::gdr::MetalRecurrentState;
    use super::super::mlx::zeros;
    use super::super::qwen35::load_qwen35_metal_weights;

    let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
        eprintln!("QWEN35_MODEL_PATH unset; skipping DFlash rollback varlen B=1 equivalence test");
        return Ok(());
    };
    eprintln!("qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1 env_ready");
    let _guard = metal_test_guard();

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    let weights = load_qwen35_metal_weights(&model_path, &config)?;
    let cpp_model = weights
        .cpp_model
        .as_ref()
        .context("Qwen3.5 compiled C++ model unavailable")?;

    let prompt_tokens = [1_i32, 2, 3, 4];
    let block_tokens = [5_i32, 6, 7, 8];
    let prompt_len = prompt_tokens.len() as i32;
    let block_size = block_tokens.len() as i32;
    let kv_capacity = prompt_len + block_size + 4;
    let cache_shape = [
        1_i32,
        config.num_key_value_heads as i32,
        kv_capacity,
        config.head_dim as i32,
    ];

    let num_full_layers = arch.num_full_attention_layers();
    let mut kv_flat: Vec<MlxArray> = (0..num_full_layers)
        .flat_map(|_| {
            [
                zeros(&cache_shape, Dtype::Bfloat16),
                zeros(&cache_shape, Dtype::Bfloat16),
            ]
        })
        .collect();
    let recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
    let mut gdr_flat: Vec<MlxArray> = recurrent
        .states
        .iter()
        .zip(recurrent.conv_states.iter())
        .flat_map(|(state, conv)| [state.clone(), conv.clone()])
        .collect();

    // Prefill to warm KV and GDR state.
    let prompt_arr = MlxArray::from_slice_i32(&prompt_tokens, &[prompt_len]);
    let prompt_logits =
        cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kv_flat, &mut gdr_flat)?;
    let mut prompt_refs: Vec<&MlxArray> = Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
    prompt_refs.push(&prompt_logits);
    prompt_refs.extend(kv_flat.iter());
    prompt_refs.extend(gdr_flat.iter());
    eval(&prompt_refs);
    eprintln!("qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1 prefill_done");

    // Snapshot GDR state *before* verify — rollback paths will restore
    // from this and replay the accepted prefix through the captured tapes.
    let gdr_snapshot: Vec<MlxArray> = gdr_flat.to_vec();

    // Enable tape mode + run verify_block_batched with B=1.
    unsafe { mlx_sys::qwen35_set_tape_mode(cpp_model.as_raw(), true) };
    let _tape_guard = Qwen35VerifyStateGuard {
        raw: cpp_model.as_raw(),
    };

    let cache_pos = prompt_len;
    let mut post_verify_kv = kv_flat.clone();
    let mut post_verify_gdr = gdr_flat.clone();
    let batched_tokens = MlxArray::from_slice_i32(&block_tokens, &[1, block_size]);
    let rope_offsets = MlxArray::from_slice_i32(&[cache_pos], &[1]);
    let verify_logits = cpp_model.verify_block_batched(
        &batched_tokens,
        1,
        block_size,
        &[cache_pos],
        &mut post_verify_kv,
        &mut post_verify_gdr,
        None,
        &rope_offsets,
    )?;
    eval(&[&verify_logits]);
    eprintln!("qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1 verify_done");

    let expected_tape_count = gdr_snapshot.len() / 2;
    let tapes = drain_current_qwen35_gdr_tapes(cpp_model, expected_tape_count)?;
    ensure!(
        !tapes.is_empty(),
        "expected at least one GDR tape from verify_block_batched"
    );
    eprintln!(
        "qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1 tapes={} T={:?}",
        tapes.len(),
        tapes[0].innovation_tape.shape()
    );

    // Vary the accepted prefix length to exercise several T_padded values,
    // including full acceptance and full rejection.
    let ks: [i32; 4] = [0, 1, 2, block_size];
    let mut max_abs_delta_overall = 0.0_f32;
    for &k in &ks {
        let mut scalar_gdr = gdr_snapshot.clone();
        let mut varlen_gdr = gdr_snapshot.clone();

        qwen35_rollback_to_accepted(&mut scalar_gdr, &gdr_snapshot, &tapes, k as usize)?;
        qwen35_rollback_to_accepted_varlen(&mut varlen_gdr, &gdr_snapshot, &tapes, &[k])?;

        let mut eval_refs: Vec<&MlxArray> = Vec::with_capacity(scalar_gdr.len() + varlen_gdr.len());
        eval_refs.extend(scalar_gdr.iter());
        eval_refs.extend(varlen_gdr.iter());
        eval(&eval_refs);

        assert_eq!(
            scalar_gdr.len(),
            varlen_gdr.len(),
            "rollback output count mismatch at k={k}"
        );
        for (idx, (lhs, rhs)) in scalar_gdr.iter().zip(varlen_gdr.iter()).enumerate() {
            assert_eq!(
                lhs.shape(),
                rhs.shape(),
                "rollback[{idx}] shape mismatch at k={k}: {:?} vs {:?}",
                lhs.shape(),
                rhs.shape()
            );
            let lhs_f32 = as_dtype(lhs, Dtype::Float32);
            let rhs_f32 = as_dtype(rhs, Dtype::Float32);
            eval(&[&lhs_f32, &rhs_f32]);
            let mut max_abs_delta = 0.0_f32;
            for (lv, rv) in lhs_f32
                .as_slice_f32()
                .iter()
                .zip(rhs_f32.as_slice_f32().iter())
            {
                let delta = (lv - rv).abs();
                max_abs_delta = max_abs_delta.max(delta);
            }
            assert!(
                max_abs_delta < 1e-3,
                "rollback[{idx}] mismatch at k={k}: max_abs_delta={max_abs_delta}"
            );
            max_abs_delta_overall = max_abs_delta_overall.max(max_abs_delta);
        }
        eprintln!(
            "qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1 k={k} max_abs_delta={max_abs_delta_overall}"
        );
    }
    eprintln!(
        "qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1 overall_max_abs_delta={max_abs_delta_overall}"
    );

    Ok(())
}

/// Phase-1 bit-ident test for `qwen35_dflash_speculative_block_batched`.
///
/// Build two synthetic DFlash rows with distinct prompts. Run two
/// sequential `qwen35_dflash_speculative_block` calls to capture the
/// scalar baseline (per-row `accepted_inputs`, `accepted_tokens`,
/// `updated_target_hidden`). Then re-run the same inputs through
/// `qwen35_dflash_speculative_block_batched` (B=2, equal cache_lens →
/// `left_padding = [0, 0]`, `batch_cache_len = prompt_len`) and assert
/// per-row identity.
#[test]
fn dflash_qwen35_verify_batched_matches_two_single_row_runs() -> Result<()> {
    use super::super::gdr::MetalRecurrentState;
    use super::super::mlx::zeros;
    use super::super::qwen35::load_qwen35_metal_weights;

    let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
        eprintln!("QWEN35_MODEL_PATH unset; skipping DFlash batched verify B=2 equivalence test");
        return Ok(());
    };
    eprintln!("dflash_qwen35_verify_batched_matches_two_single_row_runs env_ready");
    let _guard = metal_test_guard();

    let target_config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(arch) = &target_config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };
    let weights = load_qwen35_metal_weights(&model_path, &target_config)?;
    let cpp_model = weights
        .cpp_model
        .as_ref()
        .context("Qwen3.5 compiled C++ model unavailable")?;

    let runtime = match MetalDflashRuntime::load(
        &MetalDflashOptions {
            draft_model: "z-lab/Qwen3.5-4B-DFlash".to_string(),
            speculative_tokens: None,
        },
        &target_config,
    ) {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!(
                "DFlash draft model unavailable ({err:#}); skipping dflash_qwen35_verify_batched_matches_two_single_row_runs. Set `QWEN35_DFLASH_DRAFT_PATH` to a local checkpoint to enable."
            );
            return Ok(());
        }
    };
    eprintln!("dflash_qwen35_verify_batched_matches_two_single_row_runs runtime_loaded");

    // Two distinct prompts of equal length so both rows share the same
    // pre-verify cache_len (no left-padding plumbing is exercised at the
    // Phase-1 layer; that's Phase 2's job at the scheduler boundary).
    let row_prompts: [[u32; 4]; 2] = [[1, 2, 3, 4], [5, 6, 7, 8]];
    let row_currents: [u32; 2] = [11, 13]; // distinct first-block tokens
    let prompt_len = row_prompts[0].len() as i32;
    let block_size = runtime.block_size as i32;
    let kv_capacity = prompt_len + block_size + KV_CACHE_CHUNK;
    let cache_shape = [
        1_i32,
        target_config.num_key_value_heads as i32,
        kv_capacity,
        target_config.head_dim as i32,
    ];

    let num_full_layers = arch.num_full_attention_layers();
    let n_capture_layers = runtime.target_layer_ids.len();
    let target_hidden_width = i32::try_from(n_capture_layers * target_config.hidden_size)
        .context("target_hidden width does not fit i32")?;
    let ctx_len = 1_i32; // synthetic single-row warmup hidden

    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    // Synthesize a deterministic, distinct target_hidden per row so the
    // draft branches actually diverge. Shape `[ctx_len, n_cap*hidden]`.
    let make_target_hidden = |row_idx: usize| -> MlxArray {
        let n = (ctx_len as usize) * (target_hidden_width as usize);
        let data: Vec<f32> = (0..n)
            .map(|i| ((row_idx as f32) + 1.0) * (i as f32) / (n as f32 * 64.0))
            .collect();
        MlxArray::from_slice_f32(&data, &[ctx_len, target_hidden_width])
    };

    // Helper: build a fresh per-row target state by running prefill on
    // the row's prompt — populates `kv_flat` + `gdr_flat` consistent with
    // the C++ verify path. Returns post-prefill state.
    let build_row_state = |prompt: &[u32]| -> Result<(Vec<MlxArray>, Vec<MlxArray>)> {
        let mut kv_flat: Vec<MlxArray> = (0..num_full_layers)
            .flat_map(|_| {
                [
                    zeros(&cache_shape, super::super::mlx::Dtype::Bfloat16),
                    zeros(&cache_shape, super::super::mlx::Dtype::Bfloat16),
                ]
            })
            .collect();
        let recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
        let mut gdr_flat: Vec<MlxArray> = recurrent
            .states
            .iter()
            .zip(recurrent.conv_states.iter())
            .flat_map(|(state, conv)| [state.clone(), conv.clone()])
            .collect();

        let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        let prompt_arr = MlxArray::from_slice_i32(&prompt_i32, &[prompt_len]);
        let prompt_logits =
            cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kv_flat, &mut gdr_flat)?;
        let mut refs: Vec<&MlxArray> = Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
        refs.push(&prompt_logits);
        refs.extend(kv_flat.iter());
        refs.extend(gdr_flat.iter());
        eval(&refs);
        Ok((kv_flat, gdr_flat))
    };

    let (row0_kv, row0_gdr) = build_row_state(&row_prompts[0])?;
    let (row1_kv, row1_gdr) = build_row_state(&row_prompts[1])?;

    let target_hidden_per_row: Vec<MlxArray> = (0..2).map(make_target_hidden).collect();

    // ── Scalar baseline: run two sequential single-row blocks. ──
    eprintln!("dflash_qwen35_verify_batched_matches_two_single_row_runs scalar_start");
    let mut scalar_results: Vec<DFlashBlockResult> = Vec::with_capacity(2);
    let mut scalar_post_kv: Vec<Vec<MlxArray>> = Vec::with_capacity(2);
    let mut scalar_post_gdr: Vec<Vec<MlxArray>> = Vec::with_capacity(2);
    let mut scalar_post_cache_lens: Vec<i32> = Vec::with_capacity(2);
    for (row_idx, (kv_in, gdr_in)) in [
        (row0_kv.clone(), row0_gdr.clone()),
        (row1_kv.clone(), row1_gdr.clone()),
    ]
    .into_iter()
    .enumerate()
    {
        let mut kv_flat = kv_in;
        let mut gdr_flat = gdr_in;
        let mut cache_len = prompt_len;
        let mut draft_state = ContiguousKvState::new(
            runtime.draft_config.num_hidden_layers,
            runtime.draft_config.num_key_value_heads as i32,
            runtime.draft_config.head_dim as i32,
            64,
        );
        let block = qwen35_dflash_speculative_block(
            &runtime,
            row_currents[row_idx],
            &target_hidden_per_row[row_idx],
            weights
                .embedding
                .dense()
                .context("Qwen3.5/Qwen3.6 DFlash requires dense target embeddings")?,
            &weights.lm_head,
            &target_config,
            cpp_model,
            &params,
            &mut kv_flat,
            &mut gdr_flat,
            &mut cache_len,
            &mut draft_state,
            None,
        )?;
        // Materialize the result tensor so its values are independent of
        // any post-call mutation (defensive — should already be evaluated
        // by the function's terminal `eval`).
        eval(&[&block.updated_target_hidden]);
        scalar_results.push(block);
        scalar_post_kv.push(kv_flat);
        scalar_post_gdr.push(gdr_flat);
        scalar_post_cache_lens.push(cache_len);
    }
    eprintln!(
        "dflash_qwen35_verify_batched_matches_two_single_row_runs scalar_done accepted_inputs=[{}, {}]",
        scalar_results[0].accepted_inputs, scalar_results[1].accepted_inputs
    );

    // ── Batched run: stack pre-verify per-row state + one batched call. ──
    let n_kv = row0_kv.len();
    let n_gdr = row0_gdr.len();
    let mut packed_kv: Vec<MlxArray> = Vec::with_capacity(n_kv);
    for kv_idx in 0..n_kv {
        let stacked = vec![row0_kv[kv_idx].clone(), row1_kv[kv_idx].clone()];
        packed_kv.push(concatenate_axis(&stacked, 0));
    }
    let mut packed_gdr: Vec<MlxArray> = Vec::with_capacity(n_gdr);
    for gdr_idx in 0..n_gdr {
        let stacked = vec![row0_gdr[gdr_idx].clone(), row1_gdr[gdr_idx].clone()];
        packed_gdr.push(concatenate_axis(&stacked, 0));
    }

    let mut target_cache_lens: [i32; 2] = [prompt_len, prompt_len];
    let left_padding: [i32; 2] = [0, 0];
    let batch_cache_len = prompt_len;

    let mut draft_states: Vec<ContiguousKvState> = (0..2)
        .map(|_| {
            ContiguousKvState::new(
                runtime.draft_config.num_hidden_layers,
                runtime.draft_config.num_key_value_heads as i32,
                runtime.draft_config.head_dim as i32,
                64,
            )
        })
        .collect();

    let params_per_row = vec![params.clone(), params.clone()];
    let current_tokens = row_currents.to_vec();

    eprintln!("dflash_qwen35_verify_batched_matches_two_single_row_runs batched_start");
    let batched_results = qwen35_dflash_speculative_block_batched(
        &runtime,
        weights
            .embedding
            .dense()
            .context("Qwen3.5/Qwen3.6 DFlash requires dense target embeddings")?,
        &weights.lm_head,
        &target_config,
        cpp_model,
        &params_per_row,
        &current_tokens,
        &target_hidden_per_row,
        &mut packed_kv,
        &mut packed_gdr,
        &mut target_cache_lens,
        &left_padding,
        batch_cache_len,
        &mut draft_states,
    )?;
    eprintln!(
        "dflash_qwen35_verify_batched_matches_two_single_row_runs batched_done accepted_inputs=[{}, {}]",
        batched_results[0].accepted_inputs, batched_results[1].accepted_inputs
    );

    assert_eq!(batched_results.len(), 2, "batched returned wrong row count");

    // ── Predicates: per-row equality of accepted_inputs + accepted_tokens
    //                + updated_target_hidden (max_abs_delta == 0.0). ──
    let mut overall_max_abs_delta = 0.0_f32;
    for b in 0..2 {
        let scalar = &scalar_results[b];
        let batched = &batched_results[b];
        assert_eq!(
            batched.accepted_inputs, scalar.accepted_inputs,
            "row {b}: accepted_inputs mismatch (batched={}, scalar={})",
            batched.accepted_inputs, scalar.accepted_inputs
        );
        assert_eq!(
            batched.accepted_tokens, scalar.accepted_tokens,
            "row {b}: accepted_tokens mismatch (batched={:?}, scalar={:?})",
            batched.accepted_tokens, scalar.accepted_tokens
        );

        let scalar_f32 = as_dtype(&scalar.updated_target_hidden, Dtype::Float32);
        let batched_f32 = as_dtype(&batched.updated_target_hidden, Dtype::Float32);
        eval(&[&scalar_f32, &batched_f32]);
        assert_eq!(
            scalar_f32.shape(),
            batched_f32.shape(),
            "row {b}: updated_target_hidden shape mismatch (scalar={:?}, batched={:?})",
            scalar_f32.shape(),
            batched_f32.shape()
        );

        let mut max_abs_delta = 0.0_f32;
        for (idx, (lhs, rhs)) in scalar_f32
            .as_slice_f32()
            .iter()
            .zip(batched_f32.as_slice_f32().iter())
            .enumerate()
        {
            let delta = (lhs - rhs).abs();
            if delta > 0.0 {
                panic!(
                    "row {b}: updated_target_hidden[{idx}] mismatch: scalar={lhs} batched={rhs} (|delta|={delta})"
                );
            }
            max_abs_delta = max_abs_delta.max(delta);
        }
        overall_max_abs_delta = overall_max_abs_delta.max(max_abs_delta);
        eprintln!(
            "dflash_qwen35_verify_batched_matches_two_single_row_runs row {b} max_abs_delta={max_abs_delta}"
        );
    }
    // Suppress unused warnings for state we set up but don't compare here
    // (Phase 2 covers KV/GDR per-row unstacking + cache_len consistency).
    let _ = (
        scalar_post_kv,
        scalar_post_gdr,
        scalar_post_cache_lens,
        target_cache_lens,
    );

    eprintln!(
        "dflash_qwen35_verify_batched_matches_two_single_row_runs overall_max_abs_delta={overall_max_abs_delta}"
    );
    assert_eq!(overall_max_abs_delta, 0.0);

    Ok(())
}

/// Phase-2B bit-ident test for `MetalRequestState::try_decode_qwen35_dflash_speculative_batch`.
///
/// Constructs three Qwen3.5 DFlash `MetalRequestState` instances that share
/// the same prompt, sampling params (greedy, temperature=0.0) and model
/// handles. Drives each through prefill to the Decode phase (which
/// captures `target_hidden` + a committed `last_token` per the Qwen3.5
/// prefill path), then:
///
///   * Runs a single scalar `decode_step` on state C to snapshot the
///     expected first token — the scalar DFlash `decode_token` routes
///     through `qwen35_dflash_speculative_block`.
///   * Runs `MetalRequestState::try_decode_qwen35_dflash_speculative_batch`
///     on `[&mut A, &mut B]` — exercises stacking, the batched kernel,
///     unstacking, draft-state reinstallation, and `record_sampled_token`.
///
/// Because all three rows start from identical state with greedy sampling,
/// the returned first tokens must match bit-identically
/// (`sampled[0] == sampled[1] == scalar_first_token`). This proves the
/// wrapper's stack/unstack + scheduler-state glue preserves scalar
/// semantics; the kernel-level numerics (updated_target_hidden,
/// accepted_tokens) are already covered by
/// `dflash_qwen35_verify_batched_matches_two_single_row_runs`.
#[test]
fn qwen35_dflash_packed_batch_b2_matches_scalar_runs() -> Result<()> {
    use super::super::qwen35::load_qwen35_metal_weights;
    use super::super::request_state::MetalRequestState;
    use super::super::weights::MetalWeights;

    let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
        eprintln!(
            "QWEN35_MODEL_PATH unset; skipping DFlash packed batch B=2 wrapper equivalence test"
        );
        return Ok(());
    };
    eprintln!("qwen35_dflash_packed_batch_b2_matches_scalar_runs env_ready");
    let _guard = metal_test_guard();

    // Force both the scalar and batched paths to use the C++ draft forward
    // so their numerics are apples-to-apples. The batched path gates on
    // `DFLASH_DRAFT_CPP=1` (see `MetalDflashRuntime::batched_draft_path_eligible`);
    // the scalar path gates on the same env var in `dflash_draft_forward`
    // (see ~line 1253). Without this, scalar would use Rust and batched
    // would be rejected at the eligibility gate — not a bit-ident test.
    // SAFETY: set_var in tests is a thread-unsafe stdlib API; metal_test_guard
    // holds a global lock that serializes Metal tests, so this is sound here.
    unsafe {
        env::set_var("DFLASH_DRAFT_CPP", "1");
    }

    let config = load_metal_config(&model_path)?;
    let MetalModelArch::Qwen35(_) = &config.arch else {
        anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
    };

    // Runtime + target_config must outlive every request state; the
    // `MetalRequestState::new` dflash arg wants `&'static` handles. Leak
    // owned values (the test process exits immediately after, so this
    // is a bounded ≤1MB leak per run). Weights + config similarly leak
    // for the `&'a` constructor bounds.
    let runtime = match MetalDflashRuntime::load(
        &MetalDflashOptions {
            draft_model: "z-lab/Qwen3.5-4B-DFlash".to_string(),
            speculative_tokens: None,
        },
        &config,
    ) {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!(
                "DFlash draft model unavailable ({err:#}); skipping qwen35_dflash_packed_batch_b2_matches_scalar_runs. Set `QWEN35_DFLASH_DRAFT_PATH` to a local checkpoint to enable."
            );
            return Ok(());
        }
    };
    eprintln!("qwen35_dflash_packed_batch_b2_matches_scalar_runs runtime_loaded");

    let runtime_static: &'static MetalDflashRuntime = Box::leak(Box::new(runtime));

    // Load weights once and share the same `&MetalWeights` across all three
    // states. We Box::leak the MetalWeights enum since MetalRequestState is
    // parameterized by `'a` tied to weights/config references.
    let weights_inner = load_qwen35_metal_weights(&model_path, &config)?;
    let weights_leaked: &'static MetalWeights =
        Box::leak(Box::new(MetalWeights::Qwen35(weights_inner)));
    let config_leaked: &'static MetalModelConfig = Box::leak(Box::new(config));

    // DFlash target_config is the same MetalModelConfig we already leaked.
    let dflash_tuple: Option<(&'static MetalDflashRuntime, &'static MetalModelConfig)> =
        Some((runtime_static, config_leaked));

    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    // Short prompt that's trivially tokenized via raw IDs (no tokenizer
    // dependency in this test). Any valid token sequence with ≥2 tokens
    // exercises the prefill → Decode transition.
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let max_new_tokens = 4_usize;

    let build_state = || -> Result<MetalRequestState<'static>> {
        MetalRequestState::new(
            weights_leaked,
            config_leaked,
            prompt.clone(),
            &params,
            false, // use_kv_pool=false (DFlash disables pool anyway)
            max_new_tokens,
            dflash_tuple,
        )
    };

    // Prefill a state to reach Decode phase with target_hidden captured
    // and last_token committed. Budget ≥ prompt_len guarantees terminal
    // prefill step in one call.
    let prefill_to_decode = |state: &mut MetalRequestState<'_>| -> Result<()> {
        let budget = prompt.len() + 1;
        let result = state.prefill_chunk(budget)?;
        ensure!(
            result.emitted_token.is_some(),
            "terminal prefill did not emit a token"
        );
        ensure!(
            state.phase() == super::super::request_state::MetalRequestPhase::Decode,
            "expected Decode phase after prefill, got {:?}",
            state.phase()
        );
        Ok(())
    };

    // ── Scalar baseline: state C drives one decode_step through the
    //     single-row DFlash path. ──
    let mut state_c = build_state()?;
    prefill_to_decode(&mut state_c)?;
    eprintln!("qwen35_dflash_packed_batch_b2_matches_scalar_runs scalar_prefill_done");
    let scalar_first_token = state_c
        .decode_step()?
        .context("scalar decode_step returned None in Decode phase")?;
    eprintln!(
        "qwen35_dflash_packed_batch_b2_matches_scalar_runs scalar_first_token={scalar_first_token}"
    );

    // ── Batched run: states A and B exercise the wrapper. ──
    let mut state_a = build_state()?;
    let mut state_b = build_state()?;
    prefill_to_decode(&mut state_a)?;
    prefill_to_decode(&mut state_b)?;
    eprintln!("qwen35_dflash_packed_batch_b2_matches_scalar_runs batched_prefill_done");

    let mut states: Vec<&mut MetalRequestState<'static>> = vec![&mut state_a, &mut state_b];
    let outcome = MetalRequestState::try_decode_qwen35_dflash_speculative_batch(&mut states)?
        .context("wrapper returned Ok(None) despite satisfying eligibility preconditions")?;
    ensure!(
        outcome.tokens.len() == 2,
        "wrapper returned {} first tokens, expected 2",
        outcome.tokens.len()
    );
    ensure!(
        outcome.ready_indices == vec![0, 1],
        "wrapper routed rows {:?}, expected [0, 1]",
        outcome.ready_indices
    );
    let sampled = outcome.tokens;
    eprintln!(
        "qwen35_dflash_packed_batch_b2_matches_scalar_runs batched_first_tokens=[{}, {}]",
        sampled[0], sampled[1]
    );

    assert_eq!(
        sampled[0], scalar_first_token,
        "row 0 first token mismatch: batched={} scalar={}",
        sampled[0], scalar_first_token
    );
    assert_eq!(
        sampled[1], scalar_first_token,
        "row 1 first token mismatch: batched={} scalar={}",
        sampled[1], scalar_first_token
    );

    Ok(())
}

// ── Item 2: compatibility-fallback unit tests ────────────────────────────
//
// These exercise the pure-logic `check_compatibility` helper with
// synthetic configs — no GPU, no weights, no network. They pin the
// contract that shape/arch mismatches produce a named `FieldMismatch`
// with both values and a "Fix:" suggestion instead of an opaque
// FFI crash at weight-load time.

fn synthetic_target_config() -> super::super::config::MetalModelConfig {
    use super::super::config::{MetalModelArch, MetalModelConfig, MetalNormWeightMode};
    MetalModelConfig {
        hidden_size: 2048,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        num_hidden_layers: 36,
        vocab_size: 151_936,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        head_dim: 128,
        eos_token_id: 151_643,
        stop_token_ids: vec![151_643],
        quantization: None,
        norm_weight_mode: MetalNormWeightMode::AddUnitOffset,
        arch: MetalModelArch::Qwen3,
    }
}

fn synthetic_draft_config() -> super::DFlashDraftConfig {
    super::DFlashDraftConfig {
        hidden_size: 2048,
        num_hidden_layers: 1,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        head_dim: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        block_size: 4,
        mask_token_id: 0,
        target_layer_ids: vec![35],
        quantization: None,
    }
}

#[test]
fn compat_check_matching_configs_pass() {
    let target = synthetic_target_config();
    let draft = synthetic_draft_config();
    super::check_compatibility(&target, &draft, "synthetic/draft").expect("should accept");
}

#[test]
fn compat_check_hidden_size_mismatch_names_field_and_values() {
    let target = synthetic_target_config();
    let mut draft = synthetic_draft_config();
    draft.hidden_size = 4096;
    let err = super::check_compatibility(&target, &draft, "synthetic/draft")
        .expect_err("should reject mismatched hidden_size");
    let msg = err.to_string();
    assert!(
        msg.contains("hidden_size"),
        "error should name the field: {msg}"
    );
    assert!(msg.contains("2048"), "error should include target: {msg}");
    assert!(msg.contains("4096"), "error should include draft: {msg}");
    assert!(msg.contains("Fix:"), "error should suggest a fix: {msg}");
}

#[test]
fn compat_check_kv_projection_width_mismatch_named() {
    let target = synthetic_target_config();
    let mut draft = synthetic_draft_config();
    draft.num_key_value_heads = 4;
    let err = super::check_compatibility(&target, &draft, "synthetic/draft")
        .expect_err("should reject mismatched kv projection width");
    assert!(err.to_string().contains("kv_proj_width"));
}

#[test]
fn compat_check_rebucketed_heads_are_accepted_when_widths_match() {
    let target = synthetic_target_config();
    let mut draft = synthetic_draft_config();
    draft.num_attention_heads = 32;
    draft.num_key_value_heads = 16;
    draft.head_dim = 64;
    super::check_compatibility(&target, &draft, "synthetic/draft")
        .expect("same q/kv projection widths should be accepted");
}

#[test]
fn compat_check_target_layer_oob_named() {
    let target = synthetic_target_config();
    let mut draft = synthetic_draft_config();
    draft.target_layer_ids = vec![99]; // target has 36 layers
    let err = super::check_compatibility(&target, &draft, "synthetic/draft")
        .expect_err("should reject out-of-range target_layer_ids");
    assert!(err.to_string().contains("target_layer_ids"));
}

#[test]
fn metal_dflash_options_rejects_zero_speculative_tokens() {
    let options = super::MetalDflashOptions {
        draft_model: "z-lab/Qwen3.5-4B-DFlash".to_string(),
        speculative_tokens: Some(0),
    };
    let err = options
        .validate()
        .expect_err("validate() must reject speculative_tokens=0");
    let msg = err.to_string();
    assert!(
        msg.contains(">= 1") || msg.contains("must be"),
        "error should explain the >= 1 requirement: {msg}"
    );
}

#[test]
fn metal_dflash_options_accepts_unset_and_positive_speculative_tokens() {
    let unset = super::MetalDflashOptions {
        draft_model: "z-lab/Qwen3.5-4B-DFlash".to_string(),
        speculative_tokens: None,
    };
    unset
        .validate()
        .expect("unset speculative_tokens must fall through to draft default");
    let positive = super::MetalDflashOptions {
        draft_model: "z-lab/Qwen3.5-4B-DFlash".to_string(),
        speculative_tokens: Some(4),
    };
    positive
        .validate()
        .expect("positive speculative_tokens must validate");
}

#[test]
fn metal_dflash_options_rejects_empty_draft_model() {
    let options = super::MetalDflashOptions {
        draft_model: "   ".to_string(),
        speculative_tokens: Some(4),
    };
    options
        .validate()
        .expect_err("empty draft model must be rejected before load");
}

#[test]
fn packed_verify_needs_attn_mask_skips_zero_left_padding() {
    assert!(!super::packed_verify_needs_attn_mask(&[0, 0, 0]));
}

#[test]
fn packed_verify_needs_attn_mask_detects_varlen_rows() {
    assert!(super::packed_verify_needs_attn_mask(&[0, 2, 0]));
}
