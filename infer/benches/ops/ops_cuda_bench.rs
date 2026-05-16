use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use cuda_kernels::{ffi, kv_quant};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use infer::backend::cuda::tensor::{DeviceContext, DeviceVec, HiddenStates};
use infer::ops;

use super::common::{
    ATTN_SEQ_LEN, BATCH_SEQ_LEN, HEAD_DIM_128, KV_HEADS_128, MAX_SEQ_LEN, Q_HEADS_128,
    QWEN35_4B_HEAD_DIM, QWEN35_4B_HIDDEN, QWEN35_4B_INTERMEDIATE, QWEN35_4B_KV_HEADS,
    QWEN35_4B_Q_HEADS, ROPE_THETA_QWEN3, VECTOR_DIM, VOCAB_SIZE, bf16_data_scaled, configure_group,
    decode_meta, device_vec, device_vec_scaled, embedding_matrix, hidden_states, iter_sync,
    positive_device_vec, rope_cache, token_ids, zero_f32_slice,
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

    for &(label, rows, cols) in &[
        ("qwen35_hidden_2048x2560", 2048usize, QWEN35_4B_HIDDEN),
        (
            "qwen35_intermediate_2048x9216",
            2048usize,
            QWEN35_4B_INTERMEDIATE,
        ),
    ] {
        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_function(BenchmarkId::new("quantize_bf16_rows_to_int8", label), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let input = device_vec_scaled(&ctx, rows * cols, 0.015_625)
                .expect("failed to allocate int8 activation input");
            let mut output = ctx
                .stream
                .alloc_zeros::<i8>(rows * cols)
                .expect("failed to allocate int8 activation output");
            let mut scales = ctx
                .stream
                .alloc_zeros::<f32>(rows)
                .expect("failed to allocate int8 activation scales");
            let (input_ptr, _input_guard) = input.data.device_ptr(&ctx.stream);
            let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales.device_ptr_mut(&ctx.stream);

            iter_sync(b, &ctx, || unsafe {
                ffi::quantize_bf16_rows_to_int8_cuda(
                    input_ptr as *const ffi::Half,
                    output_ptr as *mut i8,
                    scales_ptr as *mut f32,
                    rows as i32,
                    cols as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("quantize_bf16_rows_to_int8_cuda failed");
            });
        });
    }

    for &(label, rows, cols) in &[
        ("qwen35_hidden_2048x2560", 2048usize, QWEN35_4B_HIDDEN),
        (
            "qwen35_intermediate_2048x9216",
            2048usize,
            QWEN35_4B_INTERMEDIATE,
        ),
    ] {
        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_function(
            BenchmarkId::new("quantize_bf16_rows_to_fp8_e4m3", label),
            |b| {
                let ctx = DeviceContext::new().expect("failed to create CUDA context");
                let input = device_vec_scaled(&ctx, rows * cols, 0.015_625)
                    .expect("failed to allocate fp8 activation input");
                let mut output = ctx
                    .stream
                    .alloc_zeros::<u8>(rows * cols)
                    .expect("failed to allocate fp8 activation output");
                let mut scales = ctx
                    .stream
                    .alloc_zeros::<f32>(rows)
                    .expect("failed to allocate fp8 activation scales");
                let (input_ptr, _input_guard) = input.data.device_ptr(&ctx.stream);
                let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
                let (scales_ptr, _scales_guard) = scales.device_ptr_mut(&ctx.stream);

                iter_sync(b, &ctx, || unsafe {
                    ffi::quantize_bf16_rows_to_fp8_e4m3_cuda(
                        input_ptr as *const ffi::Half,
                        output_ptr as *mut u8,
                        scales_ptr as *mut f32,
                        rows as i32,
                        cols as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .expect("quantize_bf16_rows_to_fp8_e4m3_cuda failed");
                });
            },
        );
    }

    for &(label, rows, cols, scale_rows, scale_cols) in &[
        (
            "dsv4_mini_hidden_1024x1024",
            1024usize,
            1024usize,
            8usize,
            8usize,
        ),
        (
            "dsv4_mini_moe_512x1024",
            512usize,
            1024usize,
            4usize,
            8usize,
        ),
    ] {
        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_function(BenchmarkId::new("dsv4_fp8_gemv", label), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let fp8_pattern = [0x38u8, 0xb8, 0x40, 0xc0, 0x30, 0xb0, 0x34, 0xb4];
            let weight_host: Vec<u8> = (0..rows * cols)
                .map(|idx| fp8_pattern[(idx * 7 + 3) % fp8_pattern.len()])
                .collect();
            let scale_host = vec![127u8; scale_rows * scale_cols];
            let weight = ctx
                .stream
                .clone_htod(&weight_host)
                .expect("failed to H2D dsv4 fp8 weights");
            let scales = ctx
                .stream
                .clone_htod(&scale_host)
                .expect("failed to H2D dsv4 scales");
            let input =
                device_vec_scaled(&ctx, cols, 0.015_625).expect("failed to allocate dsv4 input");
            let mut output = DeviceVec::zeros(&ctx, rows).expect("failed to allocate dsv4 output");
            let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales.device_ptr(&ctx.stream);
            let (input_ptr, _input_guard) = input.data.device_ptr(&ctx.stream);
            let (output_ptr, _output_guard) = output.data.device_ptr_mut(&ctx.stream);

            iter_sync(b, &ctx, || unsafe {
                ffi::dsv4_fp8_gemv_cuda(
                    weight_ptr as *const u8,
                    scales_ptr as *const u8,
                    input_ptr as *const ffi::Half,
                    output_ptr as *mut ffi::Half,
                    rows as i32,
                    cols as i32,
                    scale_rows as i32,
                    scale_cols as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("dsv4_fp8_gemv_cuda failed");
            });
        });

        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_function(BenchmarkId::new("dsv4_fp4_gemv", label), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let fp4_pattern = [0x21u8, 0xb3, 0x64, 0x9a, 0x52, 0xc1, 0x73, 0x8b];
            let weight_host: Vec<u8> = (0..rows * cols / 2)
                .map(|idx| fp4_pattern[(idx * 5 + 1) % fp4_pattern.len()])
                .collect();
            let scale_host = vec![127u8; scale_rows * scale_cols];
            let weight = ctx
                .stream
                .clone_htod(&weight_host)
                .expect("failed to H2D dsv4 fp4 weights");
            let scales = ctx
                .stream
                .clone_htod(&scale_host)
                .expect("failed to H2D dsv4 scales");
            let input =
                device_vec_scaled(&ctx, cols, 0.015_625).expect("failed to allocate dsv4 input");
            let mut output = DeviceVec::zeros(&ctx, rows).expect("failed to allocate dsv4 output");
            let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales.device_ptr(&ctx.stream);
            let (input_ptr, _input_guard) = input.data.device_ptr(&ctx.stream);
            let (output_ptr, _output_guard) = output.data.device_ptr_mut(&ctx.stream);

            iter_sync(b, &ctx, || unsafe {
                ffi::dsv4_fp4_gemv_cuda(
                    weight_ptr as *const u8,
                    scales_ptr as *const u8,
                    input_ptr as *const ffi::Half,
                    output_ptr as *mut ffi::Half,
                    rows as i32,
                    cols as i32,
                    scale_rows as i32,
                    scale_cols as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("dsv4_fp4_gemv_cuda failed");
            });
        });

        let batch = 4usize;

        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_function(BenchmarkId::new("dsv4_fp8_gemv_batch_b1", label), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let fp8_pattern = [0x38u8, 0xb8, 0x40, 0xc0, 0x30, 0xb0, 0x34, 0xb4];
            let weight_host: Vec<u8> = (0..rows * cols)
                .map(|idx| fp8_pattern[(idx * 7 + 3) % fp8_pattern.len()])
                .collect();
            let scale_host = vec![127u8; scale_rows * scale_cols];
            let weight = ctx
                .stream
                .clone_htod(&weight_host)
                .expect("failed to H2D dsv4 fp8 weights");
            let scales = ctx
                .stream
                .clone_htod(&scale_host)
                .expect("failed to H2D dsv4 scales");
            let input = hidden_states(&ctx, cols, 1).expect("failed to allocate dsv4 input");
            let mut output =
                HiddenStates::zeros(&ctx, rows, 1).expect("failed to allocate dsv4 output");
            let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales.device_ptr(&ctx.stream);
            let (input_ptr, _input_guard) = input.data.device_ptr(&ctx.stream);
            let (output_ptr, _output_guard) = output.data.device_ptr_mut(&ctx.stream);

            iter_sync(b, &ctx, || unsafe {
                ffi::dsv4_fp8_gemv_batch_cuda(
                    weight_ptr as *const u8,
                    scales_ptr as *const u8,
                    input_ptr as *const ffi::Half,
                    output_ptr as *mut ffi::Half,
                    1,
                    rows as i32,
                    cols as i32,
                    scale_rows as i32,
                    scale_cols as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("dsv4_fp8_gemv_batch_cuda failed");
            });
        });

        group.throughput(Throughput::Elements((batch * rows * cols) as u64));
        group.bench_function(BenchmarkId::new("dsv4_fp8_gemv_batch", label), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let fp8_pattern = [0x38u8, 0xb8, 0x40, 0xc0, 0x30, 0xb0, 0x34, 0xb4];
            let weight_host: Vec<u8> = (0..rows * cols)
                .map(|idx| fp8_pattern[(idx * 7 + 3) % fp8_pattern.len()])
                .collect();
            let scale_host = vec![127u8; scale_rows * scale_cols];
            let weight = ctx
                .stream
                .clone_htod(&weight_host)
                .expect("failed to H2D dsv4 fp8 weights");
            let scales = ctx
                .stream
                .clone_htod(&scale_host)
                .expect("failed to H2D dsv4 scales");
            let input = hidden_states(&ctx, cols, batch).expect("failed to allocate dsv4 input");
            let mut output =
                HiddenStates::zeros(&ctx, rows, batch).expect("failed to allocate dsv4 output");
            let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales.device_ptr(&ctx.stream);
            let (input_ptr, _input_guard) = input.data.device_ptr(&ctx.stream);
            let (output_ptr, _output_guard) = output.data.device_ptr_mut(&ctx.stream);

            iter_sync(b, &ctx, || unsafe {
                ffi::dsv4_fp8_gemv_batch_cuda(
                    weight_ptr as *const u8,
                    scales_ptr as *const u8,
                    input_ptr as *const ffi::Half,
                    output_ptr as *mut ffi::Half,
                    batch as i32,
                    rows as i32,
                    cols as i32,
                    scale_rows as i32,
                    scale_cols as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("dsv4_fp8_gemv_batch_cuda failed");
            });
        });

        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_function(BenchmarkId::new("dsv4_fp4_gemv_batch_b1", label), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let fp4_pattern = [0x21u8, 0xb3, 0x64, 0x9a, 0x52, 0xc1, 0x73, 0x8b];
            let weight_host: Vec<u8> = (0..rows * cols / 2)
                .map(|idx| fp4_pattern[(idx * 5 + 1) % fp4_pattern.len()])
                .collect();
            let scale_host = vec![127u8; scale_rows * scale_cols];
            let weight = ctx
                .stream
                .clone_htod(&weight_host)
                .expect("failed to H2D dsv4 fp4 weights");
            let scales = ctx
                .stream
                .clone_htod(&scale_host)
                .expect("failed to H2D dsv4 scales");
            let input = hidden_states(&ctx, cols, 1).expect("failed to allocate dsv4 input");
            let mut output =
                HiddenStates::zeros(&ctx, rows, 1).expect("failed to allocate dsv4 output");
            let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales.device_ptr(&ctx.stream);
            let (input_ptr, _input_guard) = input.data.device_ptr(&ctx.stream);
            let (output_ptr, _output_guard) = output.data.device_ptr_mut(&ctx.stream);

            iter_sync(b, &ctx, || unsafe {
                ffi::dsv4_fp4_gemv_batch_cuda(
                    weight_ptr as *const u8,
                    scales_ptr as *const u8,
                    input_ptr as *const ffi::Half,
                    output_ptr as *mut ffi::Half,
                    1,
                    rows as i32,
                    cols as i32,
                    scale_rows as i32,
                    scale_cols as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("dsv4_fp4_gemv_batch_cuda failed");
            });
        });

        group.throughput(Throughput::Elements((batch * rows * cols) as u64));
        group.bench_function(BenchmarkId::new("dsv4_fp4_gemv_batch", label), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let fp4_pattern = [0x21u8, 0xb3, 0x64, 0x9a, 0x52, 0xc1, 0x73, 0x8b];
            let weight_host: Vec<u8> = (0..rows * cols / 2)
                .map(|idx| fp4_pattern[(idx * 5 + 1) % fp4_pattern.len()])
                .collect();
            let scale_host = vec![127u8; scale_rows * scale_cols];
            let weight = ctx
                .stream
                .clone_htod(&weight_host)
                .expect("failed to H2D dsv4 fp4 weights");
            let scales = ctx
                .stream
                .clone_htod(&scale_host)
                .expect("failed to H2D dsv4 scales");
            let input = hidden_states(&ctx, cols, batch).expect("failed to allocate dsv4 input");
            let mut output =
                HiddenStates::zeros(&ctx, rows, batch).expect("failed to allocate dsv4 output");
            let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
            let (scales_ptr, _scales_guard) = scales.device_ptr(&ctx.stream);
            let (input_ptr, _input_guard) = input.data.device_ptr(&ctx.stream);
            let (output_ptr, _output_guard) = output.data.device_ptr_mut(&ctx.stream);

            iter_sync(b, &ctx, || unsafe {
                ffi::dsv4_fp4_gemv_batch_cuda(
                    weight_ptr as *const u8,
                    scales_ptr as *const u8,
                    input_ptr as *const ffi::Half,
                    output_ptr as *mut ffi::Half,
                    batch as i32,
                    rows as i32,
                    cols as i32,
                    scale_rows as i32,
                    scale_cols as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("dsv4_fp4_gemv_batch_cuda failed");
            });
        });
    }

    for &(label, num_experts, routes_per_expert, rows, cols, scale_rows, scale_cols) in &[
        (
            "dsv4_mini_t4_e4_512x1024",
            4usize,
            1usize,
            512usize,
            1024usize,
            4usize,
            8usize,
        ),
        (
            "dsv4_mini_t64_e4_512x1024",
            4usize,
            16usize,
            512usize,
            1024usize,
            4usize,
            8usize,
        ),
    ] {
        let total_routes = num_experts * routes_per_expert;
        group.throughput(Throughput::Elements((total_routes * rows * cols) as u64));
        group.bench_function(BenchmarkId::new("dsv4_fp4_grouped_gemv", label), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let fp4_pattern = [0x21u8, 0xb3, 0x64, 0x9a, 0x52, 0xc1, 0x73, 0x8b];
            let scale_host = vec![127u8; scale_rows * scale_cols];

            let mut weight = Vec::with_capacity(num_experts);
            let mut scales = Vec::with_capacity(num_experts);
            for expert in 0..num_experts {
                let weight_host: Vec<u8> = (0..rows * cols / 2)
                    .map(|idx| fp4_pattern[(idx * 5 + expert + 1) % fp4_pattern.len()])
                    .collect();
                weight.push(
                    ctx.stream
                        .clone_htod(&weight_host)
                        .expect("failed to H2D dsv4 grouped fp4 weight"),
                );
                scales.push(
                    ctx.stream
                        .clone_htod(&scale_host)
                        .expect("failed to H2D dsv4 grouped fp4 scales"),
                );
            }

            let mut weight_guards = Vec::with_capacity(num_experts);
            let mut scale_guards = Vec::with_capacity(num_experts);
            let mut weight_ptrs_host = Vec::with_capacity(num_experts);
            let mut scale_ptrs_host = Vec::with_capacity(num_experts);
            for idx in 0..num_experts {
                let (ptr, guard) = weight[idx].device_ptr(&ctx.stream);
                weight_ptrs_host.push(ptr as u64);
                weight_guards.push(guard);
                let (ptr, guard) = scales[idx].device_ptr(&ctx.stream);
                scale_ptrs_host.push(ptr as u64);
                scale_guards.push(guard);
            }

            let weight_ptrs = ctx
                .stream
                .clone_htod(&weight_ptrs_host)
                .expect("failed to H2D dsv4 grouped weight ptrs");
            let scale_ptrs = ctx
                .stream
                .clone_htod(&scale_ptrs_host)
                .expect("failed to H2D dsv4 grouped scale ptrs");
            let offsets_host: Vec<i32> = (0..num_experts)
                .map(|expert| (expert * routes_per_expert) as i32)
                .collect();
            let counts_host = vec![routes_per_expert as i32; num_experts];
            let offsets = ctx
                .stream
                .clone_htod(&offsets_host)
                .expect("failed to H2D dsv4 grouped offsets");
            let counts = ctx
                .stream
                .clone_htod(&counts_host)
                .expect("failed to H2D dsv4 grouped counts");
            let input = hidden_states(&ctx, cols, total_routes)
                .expect("failed to allocate dsv4 grouped input");
            let mut output = HiddenStates::zeros(&ctx, rows, total_routes)
                .expect("failed to allocate dsv4 grouped output");
            let (weight_ptrs, _weight_ptrs_guard) = weight_ptrs.device_ptr(&ctx.stream);
            let (scale_ptrs, _scale_ptrs_guard) = scale_ptrs.device_ptr(&ctx.stream);
            let (offsets_ptr, _offsets_guard) = offsets.device_ptr(&ctx.stream);
            let (counts_ptr, _counts_guard) = counts.device_ptr(&ctx.stream);
            let (input_ptr, _input_guard) = input.data.device_ptr(&ctx.stream);
            let (output_ptr, _output_guard) = output.data.device_ptr_mut(&ctx.stream);

            iter_sync(b, &ctx, || unsafe {
                ffi::dsv4_fp4_grouped_gemv_batch_cuda(
                    weight_ptrs as *const u64,
                    scale_ptrs as *const u64,
                    input_ptr as *const ffi::Half,
                    output_ptr as *mut ffi::Half,
                    offsets_ptr as *const i32,
                    counts_ptr as *const i32,
                    std::ptr::null(),
                    num_experts as i32,
                    routes_per_expert as i32,
                    rows as i32,
                    cols as i32,
                    scale_rows as i32,
                    scale_cols as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("dsv4_fp4_grouped_gemv_batch_cuda failed");
            });
        });

        group.bench_function(BenchmarkId::new("dsv4_fp4_grouped_gemv_pair", label), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let fp4_pattern_a = [0x21u8, 0xb3, 0x64, 0x9a, 0x52, 0xc1, 0x73, 0x8b];
            let fp4_pattern_b = [0x34u8, 0xa2, 0x17, 0xc5, 0x69, 0x8d, 0x42, 0xb1];
            let scale_host = vec![127u8; scale_rows * scale_cols];

            let mut weight_a = Vec::with_capacity(num_experts);
            let mut weight_b = Vec::with_capacity(num_experts);
            let mut scales_a = Vec::with_capacity(num_experts);
            let mut scales_b = Vec::with_capacity(num_experts);
            for expert in 0..num_experts {
                let weight_a_host: Vec<u8> = (0..rows * cols / 2)
                    .map(|idx| fp4_pattern_a[(idx * 5 + expert + 1) % fp4_pattern_a.len()])
                    .collect();
                let weight_b_host: Vec<u8> = (0..rows * cols / 2)
                    .map(|idx| fp4_pattern_b[(idx * 7 + expert + 3) % fp4_pattern_b.len()])
                    .collect();
                weight_a.push(
                    ctx.stream
                        .clone_htod(&weight_a_host)
                        .expect("failed to H2D dsv4 grouped fp4 weight a"),
                );
                weight_b.push(
                    ctx.stream
                        .clone_htod(&weight_b_host)
                        .expect("failed to H2D dsv4 grouped fp4 weight b"),
                );
                scales_a.push(
                    ctx.stream
                        .clone_htod(&scale_host)
                        .expect("failed to H2D dsv4 grouped fp4 scales a"),
                );
                scales_b.push(
                    ctx.stream
                        .clone_htod(&scale_host)
                        .expect("failed to H2D dsv4 grouped fp4 scales b"),
                );
            }

            let mut weight_a_guards = Vec::with_capacity(num_experts);
            let mut weight_b_guards = Vec::with_capacity(num_experts);
            let mut scale_a_guards = Vec::with_capacity(num_experts);
            let mut scale_b_guards = Vec::with_capacity(num_experts);
            let mut weight_a_ptrs_host = Vec::with_capacity(num_experts);
            let mut weight_b_ptrs_host = Vec::with_capacity(num_experts);
            let mut scale_a_ptrs_host = Vec::with_capacity(num_experts);
            let mut scale_b_ptrs_host = Vec::with_capacity(num_experts);
            for idx in 0..num_experts {
                let (ptr, guard) = weight_a[idx].device_ptr(&ctx.stream);
                weight_a_ptrs_host.push(ptr as u64);
                weight_a_guards.push(guard);
                let (ptr, guard) = weight_b[idx].device_ptr(&ctx.stream);
                weight_b_ptrs_host.push(ptr as u64);
                weight_b_guards.push(guard);
                let (ptr, guard) = scales_a[idx].device_ptr(&ctx.stream);
                scale_a_ptrs_host.push(ptr as u64);
                scale_a_guards.push(guard);
                let (ptr, guard) = scales_b[idx].device_ptr(&ctx.stream);
                scale_b_ptrs_host.push(ptr as u64);
                scale_b_guards.push(guard);
            }

            let weight_a_ptrs = ctx
                .stream
                .clone_htod(&weight_a_ptrs_host)
                .expect("failed to H2D dsv4 grouped weight a ptrs");
            let weight_b_ptrs = ctx
                .stream
                .clone_htod(&weight_b_ptrs_host)
                .expect("failed to H2D dsv4 grouped weight b ptrs");
            let scale_a_ptrs = ctx
                .stream
                .clone_htod(&scale_a_ptrs_host)
                .expect("failed to H2D dsv4 grouped scale a ptrs");
            let scale_b_ptrs = ctx
                .stream
                .clone_htod(&scale_b_ptrs_host)
                .expect("failed to H2D dsv4 grouped scale b ptrs");
            let offsets_host: Vec<i32> = (0..num_experts)
                .map(|expert| (expert * routes_per_expert) as i32)
                .collect();
            let counts_host = vec![routes_per_expert as i32; num_experts];
            let offsets = ctx
                .stream
                .clone_htod(&offsets_host)
                .expect("failed to H2D dsv4 grouped offsets");
            let counts = ctx
                .stream
                .clone_htod(&counts_host)
                .expect("failed to H2D dsv4 grouped counts");
            let input = hidden_states(&ctx, cols, total_routes)
                .expect("failed to allocate dsv4 grouped input");
            let mut output_a = HiddenStates::zeros(&ctx, rows, total_routes)
                .expect("failed to allocate dsv4 grouped output a");
            let mut output_b = HiddenStates::zeros(&ctx, rows, total_routes)
                .expect("failed to allocate dsv4 grouped output b");
            let (weight_a_ptrs, _weight_a_ptrs_guard) = weight_a_ptrs.device_ptr(&ctx.stream);
            let (weight_b_ptrs, _weight_b_ptrs_guard) = weight_b_ptrs.device_ptr(&ctx.stream);
            let (scale_a_ptrs, _scale_a_ptrs_guard) = scale_a_ptrs.device_ptr(&ctx.stream);
            let (scale_b_ptrs, _scale_b_ptrs_guard) = scale_b_ptrs.device_ptr(&ctx.stream);
            let (offsets_ptr, _offsets_guard) = offsets.device_ptr(&ctx.stream);
            let (counts_ptr, _counts_guard) = counts.device_ptr(&ctx.stream);
            let (input_ptr, _input_guard) = input.data.device_ptr(&ctx.stream);
            let (output_a_ptr, _output_a_guard) = output_a.data.device_ptr_mut(&ctx.stream);
            let (output_b_ptr, _output_b_guard) = output_b.data.device_ptr_mut(&ctx.stream);

            iter_sync(b, &ctx, || unsafe {
                ffi::dsv4_fp4_grouped_gemv_pair_batch_cuda(
                    weight_a_ptrs as *const u64,
                    scale_a_ptrs as *const u64,
                    weight_b_ptrs as *const u64,
                    scale_b_ptrs as *const u64,
                    input_ptr as *const ffi::Half,
                    output_a_ptr as *mut ffi::Half,
                    output_b_ptr as *mut ffi::Half,
                    offsets_ptr as *const i32,
                    counts_ptr as *const i32,
                    std::ptr::null(),
                    num_experts as i32,
                    routes_per_expert as i32,
                    rows as i32,
                    cols as i32,
                    scale_rows as i32,
                    scale_cols as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("dsv4_fp4_grouped_gemv_pair_batch_cuda failed");
            });
        });
    }

    for &(label, num_tokens, n_experts, topk) in &[
        ("dsv4_mini_decode_t1_e16_top2", 1usize, 16usize, 2usize),
        ("dsv4_mini_batch_t64_e16_top2", 64usize, 16usize, 2usize),
    ] {
        group.throughput(Throughput::Elements((num_tokens * n_experts) as u64));
        group.bench_function(BenchmarkId::new("dsv4_route", label), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let logits_host = bf16_data_scaled(num_tokens * n_experts, 0.125);
            let bias_host = bf16_data_scaled(n_experts, 0.031_25);
            let token_ids_host: Vec<u32> = (0..num_tokens).map(|idx| idx as u32).collect();
            let logits = ctx
                .stream
                .clone_htod(&logits_host)
                .expect("failed to H2D dsv4 route logits");
            let bias = ctx
                .stream
                .clone_htod(&bias_host)
                .expect("failed to H2D dsv4 route bias");
            let token_ids = ctx
                .stream
                .clone_htod(&token_ids_host)
                .expect("failed to H2D dsv4 route token ids");
            let mut indices = ctx
                .stream
                .alloc_zeros::<i32>(num_tokens * topk)
                .expect("failed to allocate dsv4 route indices");
            let mut weights = ctx
                .stream
                .alloc_zeros::<f32>(num_tokens * topk)
                .expect("failed to allocate dsv4 route weights");
            let (logits_ptr, _logits_guard) = logits.device_ptr(&ctx.stream);
            let (bias_ptr, _bias_guard) = bias.device_ptr(&ctx.stream);
            let (token_ptr, _token_guard) = token_ids.device_ptr(&ctx.stream);
            let (idx_ptr, _idx_guard) = indices.device_ptr_mut(&ctx.stream);
            let (weight_ptr, _weight_guard) = weights.device_ptr_mut(&ctx.stream);

            iter_sync(b, &ctx, || unsafe {
                ffi::dsv4_route_cuda(
                    logits_ptr as *const ffi::Half,
                    bias_ptr as *const ffi::Half,
                    std::ptr::null(),
                    token_ptr as *const u32,
                    idx_ptr as *mut i32,
                    weight_ptr as *mut f32,
                    num_tokens as i32,
                    n_experts as i32,
                    topk as i32,
                    1,
                    2,
                    1.5,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("dsv4_route_cuda failed");
            });
        });
    }

    for &(label, num_tokens, residual_hidden_dim, mix_dim, hc_mult, sinkhorn_iters) in &[
        (
            "dsv4_mini_decode_t1_h4096_m24_hc4",
            1usize,
            4096usize,
            24usize,
            4usize,
            20usize,
        ),
        (
            "dsv4_mini_batch_t64_h4096_m24_hc4",
            64usize,
            4096usize,
            24usize,
            4usize,
            20usize,
        ),
    ] {
        group.throughput(Throughput::Elements(
            (num_tokens * residual_hidden_dim) as u64,
        ));
        group.bench_function(BenchmarkId::new("dsv4_mhc_params", label), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let residual = hidden_states(&ctx, residual_hidden_dim, num_tokens)
                .expect("failed to allocate dsv4 mhc residual");
            let mixes_host = bf16_data_scaled(num_tokens * mix_dim, 0.0625);
            let base_host = bf16_data_scaled(mix_dim, 0.015_625);
            let scale_host = bf16_data_scaled(3, 0.125);
            let mixes = ctx
                .stream
                .clone_htod(&mixes_host)
                .expect("failed to H2D dsv4 mhc mixes");
            let base = ctx
                .stream
                .clone_htod(&base_host)
                .expect("failed to H2D dsv4 mhc base");
            let scale = ctx
                .stream
                .clone_htod(&scale_host)
                .expect("failed to H2D dsv4 mhc scale");
            let mut pre = ctx
                .stream
                .alloc_zeros::<f32>(num_tokens * hc_mult)
                .expect("failed to allocate dsv4 mhc pre");
            let mut post = ctx
                .stream
                .alloc_zeros::<f32>(num_tokens * hc_mult)
                .expect("failed to allocate dsv4 mhc post");
            let mut comb = ctx
                .stream
                .alloc_zeros::<f32>(num_tokens * hc_mult * hc_mult)
                .expect("failed to allocate dsv4 mhc comb");
            let (residual_ptr, _residual_guard) = residual.data.device_ptr(&ctx.stream);
            let (mixes_ptr, _mixes_guard) = mixes.device_ptr(&ctx.stream);
            let (base_ptr, _base_guard) = base.device_ptr(&ctx.stream);
            let (scale_ptr, _scale_guard) = scale.device_ptr(&ctx.stream);
            let (pre_ptr, _pre_guard) = pre.device_ptr_mut(&ctx.stream);
            let (post_ptr, _post_guard) = post.device_ptr_mut(&ctx.stream);
            let (comb_ptr, _comb_guard) = comb.device_ptr_mut(&ctx.stream);

            iter_sync(b, &ctx, || unsafe {
                ffi::dsv4_mhc_params_cuda(
                    residual_ptr as *const ffi::Half,
                    mixes_ptr as *const ffi::Half,
                    base_ptr as *const ffi::Half,
                    scale_ptr as *const ffi::Half,
                    pre_ptr as *mut f32,
                    post_ptr as *mut f32,
                    comb_ptr as *mut f32,
                    num_tokens as i32,
                    residual_hidden_dim as i32,
                    mix_dim as i32,
                    hc_mult as i32,
                    1.0e-6,
                    sinkhorn_iters as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("dsv4_mhc_params_cuda failed");
            });
        });
    }

    for &(
        label,
        head_dim,
        ratio,
        overlap,
        apply_rope,
        num_tokens,
        start_pos,
        pending_tokens,
        compressed_base,
        has_prev_overlap,
    ) in &[
        (
            "dsv4_mini_csa_first_r4_h64_overlap_rope",
            64usize,
            4usize,
            true,
            true,
            4usize,
            0usize,
            0usize,
            0usize,
            false,
        ),
        (
            "dsv4_mini_csa_decode_r4_h64_overlap_rope",
            64usize,
            4usize,
            true,
            true,
            1usize,
            7usize,
            3usize,
            1usize,
            true,
        ),
        (
            "dsv4_mini_indexer_decode_r4_h64_overlap_no_rope",
            64usize,
            4usize,
            true,
            false,
            1usize,
            7usize,
            3usize,
            1usize,
            true,
        ),
        (
            "dsv4_mini_csa_pending_r4_h64_overlap_rope",
            64usize,
            4usize,
            true,
            true,
            1usize,
            2usize,
            2usize,
            1usize,
            true,
        ),
        (
            "dsv4_mini_indexer_pending_r4_h64_overlap_no_rope",
            64usize,
            4usize,
            true,
            false,
            1usize,
            2usize,
            2usize,
            1usize,
            true,
        ),
        (
            "dsv4_mini_hca_decode_r96_h64_rope",
            64usize,
            96usize,
            false,
            true,
            1usize,
            191usize,
            95usize,
            1usize,
            false,
        ),
        (
            "dsv4_mini_hca_pending_r96_h64_rope",
            64usize,
            96usize,
            false,
            true,
            1usize,
            94usize,
            94usize,
            1usize,
            false,
        ),
    ] {
        let width = if overlap { 2 * head_dim } else { head_dim };
        let completed = (pending_tokens + num_tokens) / ratio;
        group.throughput(Throughput::Elements(
            ((pending_tokens + num_tokens) * width) as u64,
        ));
        group.bench_function(BenchmarkId::new("dsv4_compressor_update", label), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let kv_raw = ctx
                .stream
                .clone_htod(&bf16_data_scaled(num_tokens * width, 0.015_625))
                .expect("failed to H2D dsv4 compressor kv raw");
            let score_raw = ctx
                .stream
                .clone_htod(&bf16_data_scaled(num_tokens * width, 0.031_25))
                .expect("failed to H2D dsv4 compressor score raw");
            let ape = ctx
                .stream
                .clone_htod(&bf16_data_scaled(ratio * width, 0.003_906_25))
                .expect("failed to H2D dsv4 compressor ape");
            let norm = ctx
                .stream
                .clone_htod(&bf16_data_scaled(head_dim, 0.007_812_5))
                .expect("failed to H2D dsv4 compressor norm");
            let mut pending_kv = ctx
                .stream
                .clone_htod(&bf16_data_scaled(ratio * width, 0.011_718_75))
                .expect("failed to H2D dsv4 compressor pending kv");
            let mut pending_score = ctx
                .stream
                .clone_htod(&bf16_data_scaled(ratio * width, 0.019_531_25))
                .expect("failed to H2D dsv4 compressor pending score");
            let mut prev_overlap_kv = ctx
                .stream
                .clone_htod(&bf16_data_scaled(ratio * head_dim, 0.013_671_875))
                .expect("failed to H2D dsv4 compressor previous overlap kv");
            let mut prev_overlap_score = ctx
                .stream
                .clone_htod(&bf16_data_scaled(ratio * head_dim, 0.017_578_125))
                .expect("failed to H2D dsv4 compressor previous overlap score");
            let mut compressed = ctx
                .stream
                .alloc_zeros::<u16>((compressed_base + completed.max(1)) * head_dim)
                .expect("failed to allocate dsv4 compressor compressed rows");

            let (kv_raw_ptr, _kv_raw_guard) = kv_raw.device_ptr(&ctx.stream);
            let (score_raw_ptr, _score_raw_guard) = score_raw.device_ptr(&ctx.stream);
            let (ape_ptr, _ape_guard) = ape.device_ptr(&ctx.stream);
            let (norm_ptr, _norm_guard) = norm.device_ptr(&ctx.stream);
            let (pending_kv_ptr, _pending_kv_guard) = pending_kv.device_ptr_mut(&ctx.stream);
            let (pending_score_ptr, _pending_score_guard) =
                pending_score.device_ptr_mut(&ctx.stream);
            let (prev_overlap_kv_ptr, _prev_overlap_kv_guard) =
                prev_overlap_kv.device_ptr_mut(&ctx.stream);
            let (prev_overlap_score_ptr, _prev_overlap_score_guard) =
                prev_overlap_score.device_ptr_mut(&ctx.stream);
            let (compressed_ptr, _compressed_guard) = compressed.device_ptr_mut(&ctx.stream);

            let rope_dim = if apply_rope { 32usize } else { 0usize };
            iter_sync(b, &ctx, || unsafe {
                ffi::dsv4_compressor_update_cuda(
                    kv_raw_ptr as *const ffi::Half,
                    score_raw_ptr as *const ffi::Half,
                    ape_ptr as *const ffi::Half,
                    norm_ptr as *const ffi::Half,
                    pending_kv_ptr as *mut ffi::Half,
                    pending_score_ptr as *mut ffi::Half,
                    prev_overlap_kv_ptr as *mut ffi::Half,
                    prev_overlap_score_ptr as *mut ffi::Half,
                    compressed_ptr as *mut ffi::Half,
                    num_tokens as i32,
                    start_pos as i32,
                    pending_tokens as i32,
                    compressed_base as i32,
                    head_dim as i32,
                    ratio as i32,
                    width as i32,
                    i32::from(overlap),
                    i32::from(has_prev_overlap),
                    1.0e-6,
                    rope_dim as i32,
                    160_000.0,
                    65_536,
                    16.0,
                    32.0,
                    1.0,
                    ctx.stream.cu_stream(),
                )
                .result()
                .expect("dsv4_compressor_update_cuda failed");
            });
        });
    }

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
        (8 * QWEN35_4B_KV_HEADS * QWEN35_4B_HEAD_DIM) as u64,
    ));
    group.bench_function(BenchmarkId::new("quantize_paged_kv_fp8_qwen35", 8), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let batch_size = 8usize;
        let page_size = 16usize;
        let num_kv_heads = QWEN35_4B_KV_HEADS;
        let head_dim = QWEN35_4B_HEAD_DIM;
        let kv_dim = num_kv_heads * head_dim;
        let elem_count = page_size * kv_dim;

        let kv_bf16 = device_vec(&ctx, elem_count).expect("failed to allocate bf16 kv work");
        let mut kv_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(elem_count)
            .expect("failed to allocate fp8 kv pool");
        let mut scales = ctx
            .stream
            .alloc_zeros::<f32>(page_size * num_kv_heads)
            .expect("failed to allocate fp8 kv scales");
        let new_token_indices_host: Vec<i32> = (0..batch_size).map(|idx| idx as i32).collect();
        let new_token_indices = ctx
            .stream
            .clone_htod(&new_token_indices_host)
            .expect("failed to H2D fp8 kv token rows");
        let (src_ptr, _src_guard) = kv_bf16.data.device_ptr(&ctx.stream);
        let (dst_ptr, _dst_guard) = kv_fp8.device_ptr_mut(&ctx.stream);
        let (scale_ptr, _scale_guard) = scales.device_ptr_mut(&ctx.stream);

        iter_sync(b, &ctx, || {
            kv_quant::quantize_paged_kv_fp8(
                &ctx,
                src_ptr,
                dst_ptr,
                scale_ptr,
                &new_token_indices,
                num_kv_heads,
                head_dim,
                kv_dim,
                batch_size,
            )
            .expect("quantize_paged_kv_fp8 failed");
        });
    });

    group.throughput(Throughput::Elements(
        (2 * 8 * QWEN35_4B_KV_HEADS * QWEN35_4B_HEAD_DIM) as u64,
    ));
    group.bench_function(
        BenchmarkId::new("quantize_paged_kv_fp8_qwen35_pair", 8),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let batch_size = 8usize;
            let page_size = 16usize;
            let num_kv_heads = QWEN35_4B_KV_HEADS;
            let head_dim = QWEN35_4B_HEAD_DIM;
            let kv_dim = num_kv_heads * head_dim;
            let elem_count = page_size * kv_dim;
            let scale_count = page_size * num_kv_heads;

            let k_bf16 = device_vec(&ctx, elem_count).expect("failed to allocate k work");
            let v_bf16 = device_vec(&ctx, elem_count).expect("failed to allocate v work");
            let mut k_fp8 = ctx
                .stream
                .alloc_zeros::<u8>(elem_count)
                .expect("failed to allocate k fp8 pool");
            let mut v_fp8 = ctx
                .stream
                .alloc_zeros::<u8>(elem_count)
                .expect("failed to allocate v fp8 pool");
            let mut k_scales = ctx
                .stream
                .alloc_zeros::<f32>(scale_count)
                .expect("failed to allocate k fp8 scales");
            let mut v_scales = ctx
                .stream
                .alloc_zeros::<f32>(scale_count)
                .expect("failed to allocate v fp8 scales");
            let new_token_indices_host: Vec<i32> = (0..batch_size).map(|idx| idx as i32).collect();
            let new_token_indices = ctx
                .stream
                .clone_htod(&new_token_indices_host)
                .expect("failed to H2D fp8 pair token rows");
            let (k_src_ptr, _k_src_guard) = k_bf16.data.device_ptr(&ctx.stream);
            let (v_src_ptr, _v_src_guard) = v_bf16.data.device_ptr(&ctx.stream);
            let (k_dst_ptr, _k_dst_guard) = k_fp8.device_ptr_mut(&ctx.stream);
            let (v_dst_ptr, _v_dst_guard) = v_fp8.device_ptr_mut(&ctx.stream);
            let (k_scale_ptr, _k_scale_guard) = k_scales.device_ptr_mut(&ctx.stream);
            let (v_scale_ptr, _v_scale_guard) = v_scales.device_ptr_mut(&ctx.stream);

            iter_sync(b, &ctx, || {
                kv_quant::quantize_paged_kv_fp8(
                    &ctx,
                    k_src_ptr,
                    k_dst_ptr,
                    k_scale_ptr,
                    &new_token_indices,
                    num_kv_heads,
                    head_dim,
                    kv_dim,
                    batch_size,
                )
                .expect("quantize_paged_kv_fp8 k failed");
                kv_quant::quantize_paged_kv_fp8(
                    &ctx,
                    v_src_ptr,
                    v_dst_ptr,
                    v_scale_ptr,
                    &new_token_indices,
                    num_kv_heads,
                    head_dim,
                    kv_dim,
                    batch_size,
                )
                .expect("quantize_paged_kv_fp8 v failed");
            });
        },
    );

    group.throughput(Throughput::Elements(
        (2048 * QWEN35_4B_KV_HEADS * QWEN35_4B_HEAD_DIM) as u64,
    ));
    group.bench_function(
        BenchmarkId::new("quantize_scatter_kv_fp8_qwen35", 2048),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let max_seq_len = 4096usize;
            let token_count = 2048usize;
            let num_kv_heads = QWEN35_4B_KV_HEADS;
            let head_dim = QWEN35_4B_HEAD_DIM;
            let kv_dim = num_kv_heads * head_dim;
            let cont_elems = max_seq_len * kv_dim;
            let paged_elems = token_count * kv_dim;

            let kv_cont = device_vec(&ctx, cont_elems).expect("failed to allocate fp8 kv cont");
            let mut kv_fp8 = ctx
                .stream
                .alloc_zeros::<u8>(paged_elems)
                .expect("failed to allocate fp8 scatter pool");
            let mut scales = ctx
                .stream
                .alloc_zeros::<f32>(token_count * num_kv_heads)
                .expect("failed to allocate fp8 scatter scales");
            let page_indices_host: Vec<i32> = (0..token_count).map(|idx| idx as i32).collect();
            let page_indices = ctx
                .stream
                .clone_htod(&page_indices_host)
                .expect("failed to H2D fp8 scatter rows");
            let (dst_ptr, _dst_guard) = kv_fp8.device_ptr_mut(&ctx.stream);
            let (scale_ptr, _scale_guard) = scales.device_ptr_mut(&ctx.stream);

            iter_sync(b, &ctx, || {
                kv_quant::quantize_scatter_kv_fp8_range(
                    &ctx,
                    &kv_cont,
                    dst_ptr,
                    scale_ptr,
                    &page_indices,
                    0,
                    max_seq_len,
                    token_count,
                    num_kv_heads,
                    head_dim,
                    kv_dim,
                )
                .expect("quantize_scatter_kv_fp8_range failed");
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

    group.throughput(Throughput::Elements(2048_u64 * 2048 * 16 * 64));
    group.bench_function(
        BenchmarkId::new("tilelang_prefill_hd64_dsv4mini", 2048),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let batch_size = 1usize;
            let q_len = 2048usize;
            let page_size = 16usize;
            let num_q_heads = 16usize;
            let num_kv_heads = 1usize;
            let head_dim = 64usize;
            let pages_per_request = q_len.div_ceil(page_size);
            let total_pages = batch_size * pages_per_request;
            let total_rows = total_pages * page_size;
            let total_q_tokens = batch_size * q_len;
            let q_dim = num_q_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            let q = hidden_states(&ctx, q_dim, total_q_tokens).expect("failed to allocate hd64 q");
            let k_pool = device_vec(&ctx, total_rows * kv_dim).expect("failed to allocate hd64 k");
            let v_pool = device_vec(&ctx, total_rows * kv_dim).expect("failed to allocate hd64 v");
            let q_indptr_host: Vec<i32> =
                (0..=batch_size).map(|idx| (idx * q_len) as i32).collect();
            let kv_indptr_host: Vec<i32> = (0..=batch_size)
                .map(|idx| (idx * pages_per_request) as i32)
                .collect();
            let kv_indices_host: Vec<i32> = (0..total_pages).map(|idx| idx as i32).collect();
            let last_page_len_host = vec![page_size as i32; batch_size];
            let q_indptr = ctx
                .stream
                .clone_htod(&q_indptr_host)
                .expect("failed to H2D hd64 prefill q indptr");
            let kv_indptr = ctx
                .stream
                .clone_htod(&kv_indptr_host)
                .expect("failed to H2D hd64 prefill kv indptr");
            let kv_indices = ctx
                .stream
                .clone_htod(&kv_indices_host)
                .expect("failed to H2D hd64 prefill kv indices");
            let last_page_len = ctx
                .stream
                .clone_htod(&last_page_len_host)
                .expect("failed to H2D hd64 prefill last page len");
            let mut out = HiddenStates::zeros(&ctx, q_dim, total_q_tokens)
                .expect("failed to allocate hd64 prefill out");

            let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
            let (k_ptr, _gk) = k_pool.data.device_ptr(&ctx.stream);
            let (v_ptr, _gv) = v_pool.data.device_ptr(&ctx.stream);
            let (qoi_ptr, _gqoi) = q_indptr.device_ptr(&ctx.stream);
            let (ind_ptr, _gind) = kv_indptr.device_ptr(&ctx.stream);
            let (idx_ptr, _gidx) = kv_indices.device_ptr(&ctx.stream);
            let (lp_ptr, _glp) = last_page_len.device_ptr(&ctx.stream);

            iter_sync(b, &ctx, || {
                let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
                unsafe {
                    ffi::tilelang_batch_prefill_paged_hd64_q16_kv1_run_cuda(
                        q_ptr as *mut ffi::Half,
                        qoi_ptr as *const i32,
                        k_ptr as *mut ffi::Half,
                        v_ptr as *mut ffi::Half,
                        ind_ptr as *const i32,
                        idx_ptr as *const i32,
                        lp_ptr as *const i32,
                        out_ptr as *mut ffi::Half,
                        batch_size as i32,
                        total_q_tokens as i32,
                        q_len as i32,
                        total_pages as i32,
                        total_pages as i32,
                        num_q_heads as i32,
                        num_kv_heads as i32,
                        page_size as i32,
                        1.0 / (head_dim as f32).sqrt(),
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .expect("tilelang_batch_prefill_paged_hd64_q16_kv1 failed");
                }
            });
        },
    );

    group.throughput(Throughput::Elements((4 * 4096 * 16 * 64) as u64));
    group.bench_function(
        BenchmarkId::new("tilelang_decode_hd64_dsv4mini", 4096),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let batch_size = 4usize;
            let seq_len = 4096usize;
            let page_size = 16usize;
            let num_q_heads = 16usize;
            let num_kv_heads = 1usize;
            let head_dim = 64usize;
            let pages_per_request = seq_len / page_size;
            let total_pages = batch_size * pages_per_request;
            let total_rows = total_pages * page_size;
            let q_dim = num_q_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            let q = hidden_states(&ctx, q_dim, batch_size).expect("failed to allocate hd64 q");
            let k_pool = device_vec(&ctx, total_rows * kv_dim).expect("failed to allocate hd64 k");
            let v_pool = device_vec(&ctx, total_rows * kv_dim).expect("failed to allocate hd64 v");
            let q_indptr_host: Vec<i32> = (0..=batch_size).map(|idx| idx as i32).collect();
            let kv_indptr_host: Vec<i32> = (0..=batch_size)
                .map(|idx| (idx * pages_per_request) as i32)
                .collect();
            let kv_indices_host: Vec<i32> = (0..total_pages).map(|idx| idx as i32).collect();
            let last_page_len_host = vec![page_size as i32; batch_size];
            let q_indptr = ctx
                .stream
                .clone_htod(&q_indptr_host)
                .expect("failed to H2D hd64 q indptr");
            let kv_indptr = ctx
                .stream
                .clone_htod(&kv_indptr_host)
                .expect("failed to H2D hd64 kv indptr");
            let kv_indices = ctx
                .stream
                .clone_htod(&kv_indices_host)
                .expect("failed to H2D hd64 kv indices");
            let last_page_len = ctx
                .stream
                .clone_htod(&last_page_len_host)
                .expect("failed to H2D hd64 last page len");
            let mut out =
                HiddenStates::zeros(&ctx, q_dim, batch_size).expect("failed to allocate hd64 out");

            let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
            let (k_ptr, _gk) = k_pool.data.device_ptr(&ctx.stream);
            let (v_ptr, _gv) = v_pool.data.device_ptr(&ctx.stream);
            let (qoi_ptr, _gqoi) = q_indptr.device_ptr(&ctx.stream);
            let (ind_ptr, _gind) = kv_indptr.device_ptr(&ctx.stream);
            let (idx_ptr, _gidx) = kv_indices.device_ptr(&ctx.stream);
            let (lp_ptr, _glp) = last_page_len.device_ptr(&ctx.stream);

            iter_sync(b, &ctx, || {
                let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
                unsafe {
                    ffi::tilelang_batch_decode_paged_hd64_q16_kv1_run_cuda(
                        q_ptr as *mut ffi::Half,
                        qoi_ptr as *const i32,
                        k_ptr as *mut ffi::Half,
                        v_ptr as *mut ffi::Half,
                        ind_ptr as *const i32,
                        idx_ptr as *const i32,
                        lp_ptr as *const i32,
                        out_ptr as *mut ffi::Half,
                        batch_size as i32,
                        batch_size as i32,
                        1,
                        total_pages as i32,
                        total_pages as i32,
                        num_q_heads as i32,
                        num_kv_heads as i32,
                        page_size as i32,
                        1.0 / (head_dim as f32).sqrt(),
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .expect("tilelang_batch_decode_paged_hd64_q16_kv1 failed");
                }
            });
        },
    );

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
