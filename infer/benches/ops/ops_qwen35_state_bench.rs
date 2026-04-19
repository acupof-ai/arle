use criterion::{BenchmarkId, Criterion, Throughput};
use infer::backend::cuda::tensor::DeviceContext;
use infer::model::qwen35::prefill_buffers::GdrChunkwiseScratch35;
use infer::ops;

use super::common::{
    QWEN35_4B_LINEAR_K_DIM, QWEN35_4B_LINEAR_K_HEADS, QWEN35_4B_LINEAR_V_DIM,
    QWEN35_4B_LINEAR_V_HEADS, configure_group, f32_slice, hidden_states, iter_sync,
    positive_device_vec, zero_f32_slice,
};

pub(crate) fn bench_qwen35_state_ops(c: &mut Criterion) {
    // Qwen3.5-4B linear attention: q=16×128, k=16×128, v=32×128
    let conv_channels = QWEN35_4B_LINEAR_K_HEADS * QWEN35_4B_LINEAR_K_DIM * 2
        + QWEN35_4B_LINEAR_V_HEADS * QWEN35_4B_LINEAR_V_DIM;

    let mut group = c.benchmark_group("ops_qwen35_state");
    configure_group(&mut group);

    for &seq_len in &[128usize, 512, 2048] {
        group.throughput(Throughput::Elements(
            (QWEN35_4B_LINEAR_V_HEADS * QWEN35_4B_LINEAR_V_DIM * seq_len) as u64,
        ));
        group.bench_function(
            BenchmarkId::new("gated_delta_rule_prefill_chunkwise_into", seq_len),
            |b| {
                let ctx = DeviceContext::new().expect("failed to create CUDA context");
                let qkv =
                    hidden_states(&ctx, conv_channels, seq_len).expect("failed to allocate qkv");
                let b_proj = hidden_states(&ctx, QWEN35_4B_LINEAR_V_HEADS, seq_len)
                    .expect("failed to allocate b_proj");
                let a_proj = hidden_states(&ctx, QWEN35_4B_LINEAR_V_HEADS, seq_len)
                    .expect("failed to allocate a_proj");
                let dt_bias = positive_device_vec(&ctx, QWEN35_4B_LINEAR_V_HEADS)
                    .expect("failed to allocate dt_bias");
                let a_log =
                    f32_slice(&ctx, QWEN35_4B_LINEAR_V_HEADS).expect("failed to allocate a_log");
                let mut state = zero_f32_slice(
                    &ctx,
                    QWEN35_4B_LINEAR_V_HEADS * QWEN35_4B_LINEAR_K_DIM * QWEN35_4B_LINEAR_V_DIM,
                )
                .expect("failed to allocate recurrent state");
                let mut recurrent_out = infer::backend::cuda::tensor::HiddenStates::zeros(
                    &ctx,
                    QWEN35_4B_LINEAR_V_HEADS * QWEN35_4B_LINEAR_V_DIM,
                    seq_len,
                )
                .expect("failed to allocate recurrent out");
                let mut scratch = GdrChunkwiseScratch35::from_dims(
                    &ctx,
                    QWEN35_4B_LINEAR_V_HEADS,
                    QWEN35_4B_LINEAR_K_DIM,
                    QWEN35_4B_LINEAR_V_DIM,
                    seq_len,
                )
                .expect("failed to allocate chunkwise scratch");
                iter_sync(b, &ctx, || {
                    ops::gated_delta_rule_prefill_chunkwise_into(
                        &ctx,
                        &qkv,
                        &b_proj,
                        &a_proj,
                        &ops::GdrWeights {
                            dt_bias: &dt_bias,
                            a_log: &a_log,
                        },
                        &mut state,
                        &mut scratch,
                        &mut recurrent_out,
                        &ops::GdrHeadConfig {
                            num_key_heads: QWEN35_4B_LINEAR_K_HEADS,
                            num_value_heads: QWEN35_4B_LINEAR_V_HEADS,
                            key_dim: QWEN35_4B_LINEAR_K_DIM,
                            val_dim: QWEN35_4B_LINEAR_V_DIM,
                        },
                    )
                    .expect("gated_delta_rule_prefill_chunkwise_into failed");
                });
            },
        );
    }

    group.finish();
}

// NOTE: `bench_qwen35_prefill_attn_ops` removed 2026-04-19. It called
// `ops::prefill_attention_hd256_batch` which was narrowed to `pub(crate)` in
// 0ab2cd1 (kernel-crate routing) and its arg list packed into
// NormRopeParams/HeadConfig structs in b9d7d0e. Re-exposing those pub would
// re-widen the surface 0ab2cd1 minimized. Re-add as an in-crate #[bench] or
// a dedicated internal harness if this microbench is needed.
