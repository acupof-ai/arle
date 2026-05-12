use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use cuda_kernels::kv_quant;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use infer::backend::cuda::tensor::{DeviceContext, DeviceVec, HiddenStates};
use infer::ops;

use super::common::{
    ATTN_SEQ_LEN, BATCH_SEQ_LEN, HEAD_DIM_128, KV_HEADS_128, MAX_SEQ_LEN, Q_HEADS_128,
    QWEN35_4B_HEAD_DIM, QWEN35_4B_KV_HEADS, QWEN35_4B_Q_HEADS, ROPE_THETA_QWEN3, VECTOR_DIM,
    VOCAB_SIZE, configure_group, decode_meta, device_vec, embedding_matrix, hidden_states,
    iter_sync, positive_device_vec, rope_cache, token_ids, zero_f32_slice,
};

pub(crate) fn bench_cuda_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_cuda");
    configure_group(&mut group);

    group.throughput(Throughput::Elements((VECTOR_DIM * BATCH_SEQ_LEN) as u64));
    group.bench_function(BenchmarkId::new("add_batch", BATCH_SEQ_LEN), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let add_batch_a = hidden_states(&ctx, VECTOR_DIM, BATCH_SEQ_LEN)
            .expect("failed to allocate add batch lhs");
        let add_batch_b = hidden_states(&ctx, VECTOR_DIM, BATCH_SEQ_LEN)
            .expect("failed to allocate add batch rhs");
        iter_sync(b, &ctx, || {
            let out = ops::add_batch(&ctx, &add_batch_a, &add_batch_b).expect("add_batch failed");
            black_box(out);
        });
    });

    group.throughput(Throughput::Elements((VECTOR_DIM * BATCH_SEQ_LEN) as u64));
    group.bench_function(BenchmarkId::new("silu_mul_batch", BATCH_SEQ_LEN), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let gate = hidden_states(&ctx, VECTOR_DIM, BATCH_SEQ_LEN).expect("failed to allocate gate");
        let up = hidden_states(&ctx, VECTOR_DIM, BATCH_SEQ_LEN).expect("failed to allocate up");
        iter_sync(b, &ctx, || {
            let out = ops::silu_mul_batch(&ctx, &gate, &up).expect("silu_mul_batch failed");
            black_box(out);
        });
    });

    group.throughput(Throughput::Elements(VECTOR_DIM as u64));
    group.bench_function(BenchmarkId::new("embedding_decode_into", VECTOR_DIM), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let embed = embedding_matrix(&ctx, VOCAB_SIZE, VECTOR_DIM)
            .expect("failed to allocate embedding matrix");
        let token_id = 17_u32;
        let mut embed_out =
            DeviceVec::zeros(&ctx, VECTOR_DIM).expect("failed to allocate embedding out");
        let decode_meta_embed =
            decode_meta(&ctx, token_id as i32, 0, 1).expect("failed to allocate decode meta");
        iter_sync(b, &ctx, || {
            ops::embedding_decode_into(&ctx, &embed, &decode_meta_embed, &mut embed_out)
                .expect("embedding_decode_into failed");
        });
    });

    group.throughput(Throughput::Elements((VECTOR_DIM * BATCH_SEQ_LEN) as u64));
    group.bench_function(BenchmarkId::new("embedding_batch", BATCH_SEQ_LEN), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let embed = embedding_matrix(&ctx, VOCAB_SIZE, VECTOR_DIM)
            .expect("failed to allocate embedding matrix");
        let token_ids_gpu =
            token_ids(&ctx, BATCH_SEQ_LEN, VOCAB_SIZE).expect("failed to allocate token ids");
        let mut embed_batch_out = HiddenStates::zeros(&ctx, VECTOR_DIM, BATCH_SEQ_LEN)
            .expect("failed to allocate batched embedding out");
        iter_sync(b, &ctx, || {
            ops::embedding_batch(&ctx, &embed, &token_ids_gpu, &mut embed_batch_out)
                .expect("embedding_batch failed");
        });
    });

    group.throughput(Throughput::Elements((4 * 1024 * 256) as u64));
    group.bench_function(
        BenchmarkId::new("dequantize_paged_kv_fp8_to_hnd", 1024),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let num_kv_heads = 4usize;
            let head_dim = 256usize;
            let total_tokens = 1024usize;
            let kv_dim = num_kv_heads * head_dim;
            let elem_count = total_tokens * kv_dim;

            let fp8_host: Vec<u8> = (0..elem_count)
                .map(|idx| 0x20u8.saturating_add((idx % 31) as u8))
                .collect();
            let scale_host: Vec<f32> = (0..total_tokens * num_kv_heads)
                .map(|idx| 0.001 + (idx % 17) as f32 * 0.000_25)
                .collect();
            let token_rows_host: Vec<i32> = (0..total_tokens).map(|idx| idx as i32).collect();

            let kv_fp8 = ctx
                .stream
                .clone_htod(&fp8_host)
                .expect("failed to allocate fp8 kv");
            let scales = ctx
                .stream
                .clone_htod(&scale_host)
                .expect("failed to allocate fp8 scales");
            let token_rows = ctx
                .stream
                .clone_htod(&token_rows_host)
                .expect("failed to allocate token rows");
            let mut kv_bf16_hnd = ctx
                .stream
                .alloc_zeros::<u16>(elem_count)
                .expect("failed to allocate bf16 hnd output");
            let (fp8_ptr, _fp8_guard) = kv_fp8.device_ptr(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales.device_ptr(&ctx.stream);
            let (out_ptr, _out_guard) = kv_bf16_hnd.device_ptr_mut(&ctx.stream);

            iter_sync(b, &ctx, || {
                kv_quant::dequantize_paged_kv_fp8_to_hnd(
                    &ctx,
                    fp8_ptr,
                    scales_ptr,
                    out_ptr,
                    &token_rows,
                    num_kv_heads,
                    head_dim,
                    kv_dim,
                    total_tokens,
                )
                .expect("dequantize_paged_kv_fp8_to_hnd failed");
            });
        },
    );

    group.throughput(Throughput::Elements(
        (4 * 4096 * QWEN35_4B_Q_HEADS * QWEN35_4B_HEAD_DIM) as u64,
    ));
    group.bench_function(BenchmarkId::new("decode_attention_fp8_qwen35", 4096), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let batch_size = 4usize;
        let seq_len = 4096usize;
        let page_size = 16usize;
        let pages_per_request = seq_len / page_size;
        let total_pages = batch_size * pages_per_request;
        let total_rows = total_pages * page_size;
        let q_dim = QWEN35_4B_Q_HEADS * QWEN35_4B_HEAD_DIM;
        let kv_dim = QWEN35_4B_KV_HEADS * QWEN35_4B_HEAD_DIM;
        let data_len = total_rows * kv_dim;
        let scales_len = total_rows * QWEN35_4B_KV_HEADS;

        let q = hidden_states(&ctx, q_dim, batch_size).expect("failed to allocate q");
        let fp8_pattern = [0x00u8, 0x38, 0xb8, 0x40, 0xc0, 0x30, 0xb0, 0x34];
        let k_host: Vec<u8> = (0..data_len)
            .map(|idx| fp8_pattern[(idx * 3 + 1) % fp8_pattern.len()])
            .collect();
        let v_host: Vec<u8> = (0..data_len)
            .map(|idx| fp8_pattern[(idx * 5 + 2) % fp8_pattern.len()])
            .collect();
        let scale_host: Vec<f32> = (0..scales_len)
            .map(|idx| 0.001 + (idx % 19) as f32 * 0.000_25)
            .collect();
        let kv_indices_host: Vec<i32> = (0..total_pages).map(|idx| idx as i32).collect();
        let mut kv_meta_host = Vec::with_capacity(batch_size + 1 + batch_size);
        for req in 0..=batch_size {
            kv_meta_host.push((req * pages_per_request) as i32);
        }
        kv_meta_host.extend(std::iter::repeat_n(page_size as i32, batch_size));

        let k_data = ctx.stream.clone_htod(&k_host).expect("failed to H2D k");
        let v_data = ctx.stream.clone_htod(&v_host).expect("failed to H2D v");
        let k_scales = ctx
            .stream
            .clone_htod(&scale_host)
            .expect("failed to H2D k scales");
        let v_scales = ctx
            .stream
            .clone_htod(&scale_host)
            .expect("failed to H2D v scales");
        let kv_indices = ctx
            .stream
            .clone_htod(&kv_indices_host)
            .expect("failed to H2D kv indices");
        let kv_meta = ctx
            .stream
            .clone_htod(&kv_meta_host)
            .expect("failed to H2D kv meta");
        let mut out =
            HiddenStates::zeros(&ctx, q_dim, batch_size).expect("failed to allocate attention out");
        let workspace_bytes = kv_quant::decode_attention_int8_workspace_bytes(
            batch_size,
            QWEN35_4B_Q_HEADS,
            QWEN35_4B_HEAD_DIM,
            32,
        );
        let workspace = ctx
            .stream
            .alloc_zeros::<u8>(workspace_bytes)
            .expect("failed to allocate attention workspace");
        let (k_ptr, _k_guard) = k_data.device_ptr(&ctx.stream);
        let (v_ptr, _v_guard) = v_data.device_ptr(&ctx.stream);
        let (k_scale_ptr, _ks_guard) = k_scales.device_ptr(&ctx.stream);
        let (v_scale_ptr, _vs_guard) = v_scales.device_ptr(&ctx.stream);

        iter_sync(b, &ctx, || {
            kv_quant::decode_attention_fp8(
                &ctx,
                &q,
                k_ptr,
                v_ptr,
                k_scale_ptr,
                v_scale_ptr,
                &kv_indices,
                &kv_meta,
                &mut out,
                batch_size,
                QWEN35_4B_Q_HEADS,
                QWEN35_4B_KV_HEADS,
                QWEN35_4B_HEAD_DIM,
                kv_dim,
                1.0 / (QWEN35_4B_HEAD_DIM as f32).sqrt(),
                &workspace,
                workspace_bytes,
            )
            .expect("decode_attention_fp8 failed");
        });
    });

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
            let decode_meta_attn = decode_meta(&ctx, 13, current_pos, ATTN_SEQ_LEN)
                .expect("failed to allocate attention decode meta");
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
                    &decode_meta_attn,
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
