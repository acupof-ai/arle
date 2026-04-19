use criterion::{BenchmarkId, Criterion, Throughput};
use infer::backend::cuda::tensor::{DeviceContext, DeviceVec};
use infer::ops;

use super::common::{
    ATTN_SEQ_LEN, HEAD_DIM_128, KV_HEADS_128, MAX_SEQ_LEN, Q_HEADS_128, ROPE_THETA_QWEN3,
    configure_group, decode_meta, device_vec, iter_sync, positive_device_vec, rope_cache,
    zero_f32_slice,
};

pub(crate) fn bench_attention_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_attention");
    configure_group(&mut group);

    // NOTE: `prefill_attention_batch` bench removed 2026-04-19. The function
    // was narrowed to `pub(crate)` in 0ab2cd1 (kernel-crate routing) and its
    // arg list packed into NormRopeParams/HeadConfig structs in b9d7d0e — both
    // `pub(crate)` too. Exposing them to this external bench would re-widen
    // the surface 0ab2cd1 explicitly minimized. Re-add as an in-crate
    // #[bench] or a dedicated internal harness if this microbench is needed.

    group.throughput(Throughput::Elements((Q_HEADS_128 * HEAD_DIM_128) as u64));
    group.bench_function(
        BenchmarkId::new("fused_attention_decode_into", ATTN_SEQ_LEN),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let q_dim = Q_HEADS_128 * HEAD_DIM_128;
            let kv_dim = KV_HEADS_128 * HEAD_DIM_128;
            let q_full = device_vec(&ctx, q_dim).expect("failed to allocate q_full");
            let k_full = device_vec(&ctx, kv_dim).expect("failed to allocate k_full");
            let v_full = device_vec(&ctx, kv_dim).expect("failed to allocate v_full");
            let q_norm =
                positive_device_vec(&ctx, HEAD_DIM_128).expect("failed to allocate q_norm");
            let k_norm =
                positive_device_vec(&ctx, HEAD_DIM_128).expect("failed to allocate k_norm");
            let (cos_cache, sin_cache) =
                rope_cache(&ctx, MAX_SEQ_LEN, HEAD_DIM_128, ROPE_THETA_QWEN3)
                    .expect("failed to create rope cache");
            let current_pos = ATTN_SEQ_LEN - 1;
            let decode_meta = decode_meta(&ctx, 13, current_pos, ATTN_SEQ_LEN)
                .expect("failed to allocate decode meta");
            let cache_len = KV_HEADS_128 * MAX_SEQ_LEN * HEAD_DIM_128;
            let mut k_cache =
                DeviceVec::zeros(&ctx, cache_len).expect("failed to allocate k cache");
            let mut v_cache =
                DeviceVec::zeros(&ctx, cache_len).expect("failed to allocate v cache");
            let mut fused_out =
                DeviceVec::zeros(&ctx, q_dim).expect("failed to allocate fused out");
            let num_kv_splits = 4usize;
            let mut partial_out = zero_f32_slice(&ctx, Q_HEADS_128 * num_kv_splits * HEAD_DIM_128)
                .expect("partial_out");
            let mut partial_m =
                zero_f32_slice(&ctx, Q_HEADS_128 * num_kv_splits).expect("partial_m");
            let mut partial_l =
                zero_f32_slice(&ctx, Q_HEADS_128 * num_kv_splits).expect("partial_l");
            iter_sync(b, &ctx, || {
                ops::fused_attention_decode_into(
                    &ctx,
                    &q_full,
                    &k_full,
                    &v_full,
                    &q_norm,
                    &k_norm,
                    &cos_cache,
                    &sin_cache,
                    &decode_meta,
                    &mut k_cache,
                    &mut v_cache,
                    &mut fused_out,
                    &mut partial_out,
                    &mut partial_m,
                    &mut partial_l,
                    Q_HEADS_128,
                    KV_HEADS_128,
                )
                .expect("fused_attention_decode_into failed");
            });
        },
    );

    group.finish();
}
