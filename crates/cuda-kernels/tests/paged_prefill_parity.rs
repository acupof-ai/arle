#[cfg(feature = "cuda")]
mod cuda_tests {
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    use anyhow::{Result, anyhow};
    use cuda_kernels::ffi;
    use cuda_kernels::flashinfer::BatchPrefillPagedPlan;
    use cuda_kernels::tensor::{DeviceContext, DeviceVec, HiddenStates};
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
    use half::bf16;

    // HD128 matches bitwise — same FlashInfer kernel template as reference.
    // HD256 has a small drift because BatchPrefillWithPagedKVCacheDispatched
    // is a different template instantiation than SinglePrefillWithKVCacheDispatched,
    // plus the paged HD256 prep kernel fuses norm+RoPE+pool-write into one kernel
    // vs separate kernels on the non-paged path. Drift observed: <=0.0013 abs on
    // bf16 outputs in the ~±0.04 range, well within one bf16 ULP (~0.0019 at that
    // magnitude). Tolerance sized to accept bf16 rounding, not to hide bugs.
    const ATOL_HD128: f32 = 1e-3;
    const RTOL_HD128: f32 = 5e-3;
    const ATOL_HD256: f32 = 2e-3;
    const RTOL_HD256: f32 = 5e-3;

    fn cuda_test_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("cuda test mutex poisoned")
    }

    fn host_bf16_from_seed(len: usize, seed: u32, scale: f32) -> Vec<bf16> {
        let mut state = seed;
        (0..len)
            .map(|idx| {
                state = state
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223u32.wrapping_add(idx as u32));
                let unit = ((state >> 8) as f32) / ((u32::MAX >> 8) as f32);
                bf16::from_f32((unit * 2.0 - 1.0) * scale)
            })
            .collect()
    }

    fn rope_cache(head_dim: usize, max_pos: usize, theta: f32) -> (Vec<bf16>, Vec<bf16>) {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let mut cos = vec![bf16::ZERO; max_pos * head_dim];
        let mut sin = vec![bf16::ZERO; max_pos * head_dim];
        for pos in 0..max_pos {
            for i in 0..half {
                let freq = pos as f32 * inv_freq[i];
                let cos_v = bf16::from_f32(freq.cos());
                let sin_v = bf16::from_f32(freq.sin());
                cos[pos * head_dim + i] = cos_v;
                cos[pos * head_dim + i + half] = cos_v;
                sin[pos * head_dim + i] = sin_v;
                sin[pos * head_dim + i + half] = sin_v;
            }
        }
        (cos, sin)
    }

    fn rope_cache_partial(rotary_dim: usize, max_pos: usize, theta: f32) -> (Vec<bf16>, Vec<bf16>) {
        let half = rotary_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / rotary_dim as f32))
            .collect();
        let mut cos = vec![bf16::ZERO; max_pos * rotary_dim];
        let mut sin = vec![bf16::ZERO; max_pos * rotary_dim];
        for pos in 0..max_pos {
            for i in 0..half {
                let freq = pos as f32 * inv_freq[i];
                let cos_v = bf16::from_f32(freq.cos());
                let sin_v = bf16::from_f32(freq.sin());
                cos[pos * rotary_dim + i] = cos_v;
                cos[pos * rotary_dim + i + half] = cos_v;
                sin[pos * rotary_dim + i] = sin_v;
                sin[pos * rotary_dim + i + half] = sin_v;
            }
        }
        (cos, sin)
    }

    fn hidden_states_from_host(
        ctx: &DeviceContext,
        data: &[bf16],
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        Ok(HiddenStates {
            data: ctx
                .stream
                .clone_htod(data)
                .map_err(|e| anyhow!("H2D copy failed: {e}"))?,
            hidden_dim,
            seq_len,
        })
    }

    fn assert_all_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
        assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut worst_idx = 0usize;
        for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
            let abs = (got - want).abs();
            let rel = abs / want.abs().max(1e-6);
            if abs > max_abs {
                max_abs = abs;
                max_rel = rel;
                worst_idx = idx;
            }
            let allowed = atol + rtol * want.abs();
            assert!(
                abs <= allowed,
                "{name}: idx={idx} got={got} want={want} abs={abs} allowed={allowed} atol={atol} rtol={rtol}",
            );
        }
        println!(
            "{name}: atol={atol} rtol={rtol} max_abs={max_abs} max_rel={max_rel} worst_idx={worst_idx}"
        );
    }

    fn output_to_host(ctx: &DeviceContext, output: &HiddenStates) -> Result<Vec<f32>> {
        let host = ctx
            .stream
            .clone_dtoh(&output.data)
            .map_err(|e| anyhow!("D2H output failed: {e}"))?;
        ctx.sync()?;
        Ok(host.into_iter().map(|value| value.to_f32()).collect())
    }

    pub fn test_paged_prefill_hd128_matches_single_prefill() -> Result<()> {
        let _guard = cuda_test_guard();
        let ctx = DeviceContext::new()?;
        let batch_size = 1usize;
        let num_q_heads = 8usize;
        let num_kv_heads = 8usize;
        let head_dim = 128usize;
        let seq_len = 17usize;
        let page_size = 16usize;
        let num_pages = 2usize;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q_host = host_bf16_from_seed(q_dim * seq_len, 1, 0.75);
        let k_host = host_bf16_from_seed(kv_dim * seq_len, 2, 0.5);
        let v_host = host_bf16_from_seed(kv_dim * seq_len, 3, 0.5);
        let q_weight = host_bf16_from_seed(head_dim, 4, 0.25);
        let k_weight = host_bf16_from_seed(head_dim, 5, 0.25);
        let (cos_host, sin_host) = rope_cache(head_dim, 64, 10_000.0);

        let q_norm = DeviceVec::from_host(&ctx, &q_weight)?;
        let k_norm = DeviceVec::from_host(&ctx, &k_weight)?;
        let cos_cache = DeviceVec::from_host(&ctx, &cos_host)?;
        let sin_cache = DeviceVec::from_host(&ctx, &sin_host)?;

        let mut ref_q = hidden_states_from_host(&ctx, &q_host, q_dim, seq_len)?;
        let mut ref_k = hidden_states_from_host(&ctx, &k_host, kv_dim, seq_len)?;
        let ref_v = hidden_states_from_host(&ctx, &v_host, kv_dim, seq_len)?;
        let mut ref_k_cache = DeviceVec::zeros(&ctx, num_kv_heads * seq_len * head_dim)?;
        let mut ref_v_cache = DeviceVec::zeros(&ctx, num_kv_heads * seq_len * head_dim)?;
        let mut ref_output = HiddenStates::zeros(&ctx, q_dim, seq_len)?;

        unsafe {
            let (q_ptr, _gq) = ref_q.data.device_ptr_mut(&ctx.stream);
            let (k_ptr, _gk) = ref_k.data.device_ptr_mut(&ctx.stream);
            let (v_ptr, _gv) = ref_v.data.device_ptr(&ctx.stream);
            let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
            let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
            let (cos_ptr, _gcos) = cos_cache.data.device_ptr(&ctx.stream);
            let (sin_ptr, _gsin) = sin_cache.data.device_ptr(&ctx.stream);
            let (kc_ptr, _gkc) = ref_k_cache.data.device_ptr_mut(&ctx.stream);
            let (vc_ptr, _gvc) = ref_v_cache.data.device_ptr_mut(&ctx.stream);
            ffi::prefill_attention_prep_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *mut ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                kc_ptr as *mut ffi::Half,
                vc_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                seq_len as i32,
                0,
                seq_len as i32,
                1e-6,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }

        let reference_kernel = Instant::now();
        unsafe {
            let (q_ptr, _gq) = ref_q.data.device_ptr_mut(&ctx.stream);
            let (kc_ptr, _gkc) = ref_k_cache.data.device_ptr_mut(&ctx.stream);
            let (vc_ptr, _gvc) = ref_v_cache.data.device_ptr_mut(&ctx.stream);
            let (o_ptr, _go) = ref_output.data.device_ptr_mut(&ctx.stream);
            let ret = ffi::flashinfer_single_prefill(
                q_ptr as *mut ffi::Half,
                kc_ptr as *mut ffi::Half,
                vc_ptr as *mut ffi::Half,
                o_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                seq_len as i32,
                seq_len as i32,
                seq_len as i32,
                1.0f32 / (head_dim as f32).sqrt(),
                std::ptr::null_mut(),
                ctx.stream.cu_stream(),
            );
            if ret != 0 {
                return Err(anyhow!(
                    "flashinfer_single_prefill failed with CUDA error {ret}"
                ));
            }
        }
        ctx.sync()?;
        let reference_elapsed = reference_kernel.elapsed();

        let mut paged_q = hidden_states_from_host(&ctx, &q_host, q_dim, seq_len)?;
        let mut paged_k = hidden_states_from_host(&ctx, &k_host, kv_dim, seq_len)?;
        let paged_v = hidden_states_from_host(&ctx, &v_host, kv_dim, seq_len)?;
        let mut paged_output = HiddenStates::zeros(&ctx, q_dim, seq_len)?;
        let mut k_pool: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(num_pages * num_kv_heads * page_size * head_dim)
            .map_err(|e| anyhow!("Alloc k_pool failed: {e}"))?;
        let mut v_pool: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(num_pages * num_kv_heads * page_size * head_dim)
            .map_err(|e| anyhow!("Alloc v_pool failed: {e}"))?;

        let q_indptr_host = vec![0_i32, seq_len as i32];
        let kv_indptr_host = vec![0_i32, num_pages as i32];
        let kv_indices_host = vec![1_i32, 0_i32];
        let kv_last_page_len_host = vec![1_i32];
        let q_indptr_gpu = ctx.stream.clone_htod(&q_indptr_host)?;
        let kv_indptr_gpu = ctx.stream.clone_htod(&kv_indptr_host)?;
        let kv_indices_gpu = ctx.stream.clone_htod(&kv_indices_host)?;
        let kv_last_page_len_gpu = ctx.stream.clone_htod(&kv_last_page_len_host)?;
        let mut lse: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(seq_len * num_q_heads)
            .map_err(|e| anyhow!("Alloc lse failed: {e}"))?;

        unsafe {
            let (q_ptr, _gq) = paged_q.data.device_ptr_mut(&ctx.stream);
            let (k_ptr, _gk) = paged_k.data.device_ptr_mut(&ctx.stream);
            let (v_ptr, _gv) = paged_v.data.device_ptr(&ctx.stream);
            let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
            let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
            let (cos_ptr, _gcos) = cos_cache.data.device_ptr(&ctx.stream);
            let (sin_ptr, _gsin) = sin_cache.data.device_ptr(&ctx.stream);
            let (page_ptr, _gpage) = kv_indices_gpu.device_ptr(&ctx.stream);
            let (kp_ptr, _gkp) = k_pool.device_ptr_mut(&ctx.stream);
            let (vp_ptr, _gvp) = v_pool.device_ptr_mut(&ctx.stream);
            ffi::prefill_attention_paged_prep_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *mut ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                page_ptr as *const i32,
                page_size as i32,
                kp_ptr as *mut ffi::Half,
                vp_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                seq_len as i32,
                0,
                1e-6,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }

        let mut plan = BatchPrefillPagedPlan::new(&ctx, seq_len, num_q_heads)?;
        plan.plan_hd128(
            &ctx,
            &q_indptr_host,
            &kv_indptr_host,
            batch_size,
            num_q_heads,
            num_kv_heads,
            page_size,
        )?;
        ctx.sync()?;

        let paged_kernel = Instant::now();
        {
            let (q_ptr, _gq) = paged_q.data.device_ptr_mut(&ctx.stream);
            let (q_indptr_ptr, _gqi) = q_indptr_gpu.device_ptr(&ctx.stream);
            let (kp_ptr, _gkp) = k_pool.device_ptr_mut(&ctx.stream);
            let (vp_ptr, _gvp) = v_pool.device_ptr_mut(&ctx.stream);
            let (kv_indptr_ptr, _gkvi) = kv_indptr_gpu.device_ptr(&ctx.stream);
            let (kv_indices_ptr, _gkvs) = kv_indices_gpu.device_ptr(&ctx.stream);
            let (last_ptr, _glast) = kv_last_page_len_gpu.device_ptr(&ctx.stream);
            let (out_ptr, _go) = paged_output.data.device_ptr_mut(&ctx.stream);
            let (lse_ptr, _glse) = lse.device_ptr_mut(&ctx.stream);
            plan.run_hd128(
                &ctx,
                q_ptr,
                q_indptr_ptr,
                kp_ptr,
                vp_ptr,
                kv_indptr_ptr,
                kv_indices_ptr,
                last_ptr,
                out_ptr,
                Some(lse_ptr),
                batch_size,
                num_q_heads,
                num_kv_heads,
                page_size,
            )?;
        }
        ctx.sync()?;
        let paged_elapsed = paged_kernel.elapsed();

        println!(
            "test_paged_prefill_hd128_matches_single_prefill: reference_run={:.3}ms paged_run={:.3}ms",
            reference_elapsed.as_secs_f64() * 1_000.0,
            paged_elapsed.as_secs_f64() * 1_000.0
        );

        let ref_host = output_to_host(&ctx, &ref_output)?;
        let paged_host = output_to_host(&ctx, &paged_output)?;
        assert_all_close(
            "hd128 output",
            &paged_host,
            &ref_host,
            ATOL_HD128,
            RTOL_HD128,
        );
        Ok(())
    }

    pub fn test_paged_prefill_hd256_matches_single_prefill() -> Result<()> {
        let _guard = cuda_test_guard();
        let ctx = DeviceContext::new()?;
        let batch_size = 1usize;
        let num_q_heads = 16usize;
        let num_kv_heads = 16usize;
        let head_dim = 256usize;
        let rotary_dim = 64usize;
        let seq_len = 17usize;
        let page_size = 16usize;
        let num_pages = 2usize;
        let q_dim = num_q_heads * head_dim;
        let q_full_dim = q_dim * 2;
        let kv_dim = num_kv_heads * head_dim;

        let q_full_host = host_bf16_from_seed(q_full_dim * seq_len, 11, 0.5);
        let k_host = host_bf16_from_seed(kv_dim * seq_len, 12, 0.5);
        let v_host = host_bf16_from_seed(kv_dim * seq_len, 13, 0.5);
        let q_weight = host_bf16_from_seed(head_dim, 14, 0.2);
        let k_weight = host_bf16_from_seed(head_dim, 15, 0.2);
        let (cos_host, sin_host) = rope_cache_partial(rotary_dim, 64, 10_000_000.0);

        let q_norm = DeviceVec::from_host(&ctx, &q_weight)?;
        let k_norm = DeviceVec::from_host(&ctx, &k_weight)?;
        let cos_cache = DeviceVec::from_host(&ctx, &cos_host)?;
        let sin_cache = DeviceVec::from_host(&ctx, &sin_host)?;

        let ref_q_full = hidden_states_from_host(&ctx, &q_full_host, q_full_dim, seq_len)?;
        let ref_k = hidden_states_from_host(&ctx, &k_host, kv_dim, seq_len)?;
        let ref_v = hidden_states_from_host(&ctx, &v_host, kv_dim, seq_len)?;
        let mut ref_q = HiddenStates::zeros(&ctx, q_dim, seq_len)?;
        let mut ref_k_cache = DeviceVec::zeros(&ctx, num_kv_heads * seq_len * head_dim)?;
        let mut ref_v_cache = DeviceVec::zeros(&ctx, num_kv_heads * seq_len * head_dim)?;
        let mut ref_output = HiddenStates::zeros(&ctx, q_dim, seq_len)?;
        let start_pos_gpu = ctx.stream.clone_htod(&[0_i32])?;

        unsafe {
            let (qf_ptr, _gqf) = ref_q_full.data.device_ptr(&ctx.stream);
            let (k_ptr, _gk) = ref_k.data.device_ptr(&ctx.stream);
            let (v_ptr, _gv) = ref_v.data.device_ptr(&ctx.stream);
            let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
            let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
            let (cos_ptr, _gcos) = cos_cache.data.device_ptr(&ctx.stream);
            let (sin_ptr, _gsin) = sin_cache.data.device_ptr(&ctx.stream);
            let (q_ptr, _gq) = ref_q.data.device_ptr_mut(&ctx.stream);
            let (kc_ptr, _gkc) = ref_k_cache.data.device_ptr_mut(&ctx.stream);
            let (vc_ptr, _gvc) = ref_v_cache.data.device_ptr_mut(&ctx.stream);
            let (start_ptr, _gstart) = start_pos_gpu.device_ptr(&ctx.stream);
            ffi::prefill_attention_hd256_prep_cuda(
                qf_ptr as *const ffi::Half,
                k_ptr as *const ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                q_ptr as *mut ffi::Half,
                kc_ptr as *mut ffi::Half,
                vc_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                seq_len as i32,
                start_ptr as *const i32,
                rotary_dim as i32,
                1e-6,
                seq_len as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }

        let reference_kernel = Instant::now();
        unsafe {
            let (q_ptr, _gq) = ref_q.data.device_ptr_mut(&ctx.stream);
            let (kc_ptr, _gkc) = ref_k_cache.data.device_ptr_mut(&ctx.stream);
            let (vc_ptr, _gvc) = ref_v_cache.data.device_ptr_mut(&ctx.stream);
            let (o_ptr, _go) = ref_output.data.device_ptr_mut(&ctx.stream);
            let ret = ffi::flashinfer_single_prefill_hd256(
                q_ptr as *mut ffi::Half,
                kc_ptr as *mut ffi::Half,
                vc_ptr as *mut ffi::Half,
                o_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                seq_len as i32,
                seq_len as i32,
                seq_len as i32,
                1.0f32 / (head_dim as f32).sqrt(),
                std::ptr::null_mut(),
                ctx.stream.cu_stream(),
            );
            if ret != 0 {
                return Err(anyhow!(
                    "flashinfer_single_prefill_hd256 failed with CUDA error {ret}"
                ));
            }
        }
        ctx.sync()?;
        let reference_elapsed = reference_kernel.elapsed();

        let paged_q_full = hidden_states_from_host(&ctx, &q_full_host, q_full_dim, seq_len)?;
        let paged_k = hidden_states_from_host(&ctx, &k_host, kv_dim, seq_len)?;
        let paged_v = hidden_states_from_host(&ctx, &v_host, kv_dim, seq_len)?;
        let mut paged_q = HiddenStates::zeros(&ctx, q_dim, seq_len)?;
        let mut paged_output = HiddenStates::zeros(&ctx, q_dim, seq_len)?;
        let mut k_pool: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(num_pages * num_kv_heads * page_size * head_dim)
            .map_err(|e| anyhow!("Alloc k_pool failed: {e}"))?;
        let mut v_pool: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(num_pages * num_kv_heads * page_size * head_dim)
            .map_err(|e| anyhow!("Alloc v_pool failed: {e}"))?;

        let q_indptr_host = vec![0_i32, seq_len as i32];
        let kv_indptr_host = vec![0_i32, num_pages as i32];
        let kv_indices_host = vec![1_i32, 0_i32];
        let kv_last_page_len_host = vec![1_i32];
        let q_indptr_gpu = ctx.stream.clone_htod(&q_indptr_host)?;
        let kv_indptr_gpu = ctx.stream.clone_htod(&kv_indptr_host)?;
        let kv_indices_gpu = ctx.stream.clone_htod(&kv_indices_host)?;
        let kv_last_page_len_gpu = ctx.stream.clone_htod(&kv_last_page_len_host)?;
        let mut lse: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(seq_len * num_q_heads)
            .map_err(|e| anyhow!("Alloc lse failed: {e}"))?;

        unsafe {
            let (qf_ptr, _gqf) = paged_q_full.data.device_ptr(&ctx.stream);
            let (q_ptr, _gq) = paged_q.data.device_ptr_mut(&ctx.stream);
            let (k_ptr, _gk) = paged_k.data.device_ptr(&ctx.stream);
            let (v_ptr, _gv) = paged_v.data.device_ptr(&ctx.stream);
            let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
            let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
            let (cos_ptr, _gcos) = cos_cache.data.device_ptr(&ctx.stream);
            let (sin_ptr, _gsin) = sin_cache.data.device_ptr(&ctx.stream);
            let (page_ptr, _gpage) = kv_indices_gpu.device_ptr(&ctx.stream);
            let (kp_ptr, _gkp) = k_pool.device_ptr_mut(&ctx.stream);
            let (vp_ptr, _gvp) = v_pool.device_ptr_mut(&ctx.stream);
            ffi::prefill_attention_paged_prep_hd256_cuda(
                qf_ptr as *const ffi::Half,
                q_ptr as *mut ffi::Half,
                k_ptr as *const ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                page_ptr as *const i32,
                page_size as i32,
                kp_ptr as *mut ffi::Half,
                vp_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                seq_len as i32,
                0,
                rotary_dim as i32,
                1e-6,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }

        let mut plan = BatchPrefillPagedPlan::new(&ctx, seq_len, num_q_heads)?;
        plan.plan_hd256(
            &ctx,
            &q_indptr_host,
            &kv_indptr_host,
            batch_size,
            num_q_heads,
            num_kv_heads,
            page_size,
        )?;
        ctx.sync()?;

        let paged_kernel = Instant::now();
        {
            let (q_ptr, _gq) = paged_q.data.device_ptr_mut(&ctx.stream);
            let (q_indptr_ptr, _gqi) = q_indptr_gpu.device_ptr(&ctx.stream);
            let (kp_ptr, _gkp) = k_pool.device_ptr_mut(&ctx.stream);
            let (vp_ptr, _gvp) = v_pool.device_ptr_mut(&ctx.stream);
            let (kv_indptr_ptr, _gkvi) = kv_indptr_gpu.device_ptr(&ctx.stream);
            let (kv_indices_ptr, _gkvs) = kv_indices_gpu.device_ptr(&ctx.stream);
            let (last_ptr, _glast) = kv_last_page_len_gpu.device_ptr(&ctx.stream);
            let (out_ptr, _go) = paged_output.data.device_ptr_mut(&ctx.stream);
            let (lse_ptr, _glse) = lse.device_ptr_mut(&ctx.stream);
            plan.run_hd256(
                &ctx,
                q_ptr,
                q_indptr_ptr,
                kp_ptr,
                vp_ptr,
                kv_indptr_ptr,
                kv_indices_ptr,
                last_ptr,
                out_ptr,
                Some(lse_ptr),
                batch_size,
                num_q_heads,
                num_kv_heads,
                page_size,
            )?;
        }
        ctx.sync()?;
        let paged_elapsed = paged_kernel.elapsed();

        println!(
            "test_paged_prefill_hd256_matches_single_prefill: reference_run={:.3}ms paged_run={:.3}ms",
            reference_elapsed.as_secs_f64() * 1_000.0,
            paged_elapsed.as_secs_f64() * 1_000.0
        );

        let ref_host = output_to_host(&ctx, &ref_output)?;
        let paged_host = output_to_host(&ctx, &paged_output)?;
        assert_all_close(
            "hd256 output",
            &paged_host,
            &ref_host,
            ATOL_HD256,
            RTOL_HD256,
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn main() {
    run_named(
        "test_paged_prefill_hd128_matches_single_prefill",
        cuda_tests::test_paged_prefill_hd128_matches_single_prefill,
    );
    run_named(
        "test_paged_prefill_hd256_matches_single_prefill",
        cuda_tests::test_paged_prefill_hd256_matches_single_prefill,
    );
}

#[cfg(feature = "cuda")]
#[allow(clippy::exit)] // harness-style binary test; exit signals CI failure
fn run_named(name: &str, test_fn: fn() -> anyhow::Result<()>) {
    match test_fn() {
        Ok(()) => {
            println!("{name} ... ok");
        }
        Err(err) => {
            eprintln!("{name} ... FAILED");
            eprintln!("Error: {err:?}");
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {}
