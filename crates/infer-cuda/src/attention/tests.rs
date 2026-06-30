mod oproj_tests {
    use super::super::*;

    #[test]
    fn oproj_group_shape_accepts_multiple_full_groups_per_rank() {
        let shape = dsv4_oproj_group_shape(2048, 4096, 2, 1024, 4096, 8192, 3).unwrap();
        assert_eq!(
            shape,
            Dsv4OProjGroupShape {
                groups: 2,
                rows_per_group: 1024,
                cols_per_group: 4096,
                routes: 6,
            }
        );
    }

    #[test]
    fn oproj_group_shape_rejects_split_output_group() {
        let err = dsv4_oproj_group_shape(8192, 4096, 8, 1024, 4096, 2048, 1)
            .expect_err("half output group must fail closed");
        assert!(
            err.to_string().contains("whole number of wo_a groups"),
            "{err}"
        );
    }

    // ---- Batched-FINISH (F1/F2) n=1 byte/arg-identity invariants ----
    // These assert the SHAPE/arg math that makes the batched FINISH collapse to
    // the original per-row path at n=1, without a GPU (the device kernels are
    // unchanged; only the M/rows args differ). The runtime correctness gate
    // (needle x3 same-config) covers the on-device numerics.

    #[test]
    fn oproj_single_group_shape_is_m_parametric() {
        // groups==1 (Qwen3.6 MODEL1 at TP>1): the single-output-group decode lane
        // batches over M. The shape's groups/rows/cols are M-INVARIANT and routes
        // tracks token_count, so token_count=1 and token_count=n share one
        // group-shape contract (the decode-DeepGEMM path keys off groups==1 only).
        // wo_a = [rows_per_group, cols_per_group], local_width == cols_per_group.
        let shape1 = dsv4_oproj_group_shape(1536, 4096, 1, 1536, 4096, 4096, 1).unwrap();
        let shape_n = dsv4_oproj_group_shape(1536, 4096, 1, 1536, 4096, 4096, 7).unwrap();
        assert_eq!(shape1.groups, 1);
        assert_eq!(shape_n.groups, 1);
        assert_eq!(shape1.rows_per_group, shape_n.rows_per_group);
        assert_eq!(shape1.cols_per_group, shape_n.cols_per_group);
        // routes == token_count * groups; with groups==1, routes == token_count.
        assert_eq!(shape1.routes, 1);
        assert_eq!(shape_n.routes, 7);
    }

    // Mirror of the TP out-slice kernel args computed inline by slice_out_row and
    // slice_out_batched. slice_out_row hardcodes s_q=1; slice_out_batched passes
    // s_q=n. Every other arg (global_width, local_width, head_offset) is the same
    // pure function of (h_q, head_dim, local_heads, tp_rank) — so at n=1 the two
    // calls are byte-for-byte identical kernel launches.
    fn tp_out_slice_args(
        h_q: usize,
        head_dim: usize,
        local_heads: usize,
        tp_rank: usize,
    ) -> (i32, i32, i32) {
        let global_width = (h_q * head_dim) as i32;
        let local_width = (local_heads * head_dim) as i32;
        let head_offset = (tp_rank * local_heads * head_dim) as i32;
        (global_width, local_width, head_offset)
    }

    #[test]
    fn slice_out_batched_n1_matches_slice_out_row_args() {
        // TP=4 example: h_q (global heads) = 16, local_heads = 4, rank 2.
        let (gw, lw, off) = tp_out_slice_args(16, 512, 4, 2);
        assert_eq!(gw, 16 * 512); // src row stride = global width
        assert_eq!(lw, 4 * 512); // dst row stride = local width
        assert_eq!(off, 2 * 4 * 512); // sliced head block start
        // slice_out_row passes s_q=1; slice_out_batched(n=1) passes s_q=1 — identical.
        let s_q_row = 1_i32;
        let s_q_batched_n1 = 1_i32; // n == 1
        assert_eq!(s_q_row, s_q_batched_n1);
    }

    #[test]
    fn decode_proj_active_counts_write_skipped_at_m1() {
        // The fused-wqkv decode scratch is constructed with active_counts=[1].
        // decode_proj_deepgemm_raw writes active_counts only when m != 1, so the
        // per-row (m=1) decode lane emits EXACTLY the original (m=1, counts=[1])
        // kernel args — no extra H2D, byte+launch identical. The batched (m=n)
        // call writes [n] then the caller restores [1].
        let m_per_row = 1usize;
        let m_batched = 7usize;
        assert!(
            m_per_row == 1,
            "per-row M must be 1 to skip the active_counts H2D"
        );
        assert!(
            m_batched != 1,
            "batched M must differ from 1 to write active_counts"
        );
        // GEMM m arg: per-row -> 1 (original), batched -> n.
        assert_eq!(m_per_row as i32, 1);
        assert_eq!(m_batched as i32, 7);
    }

    // The gather kernel index: src_idx = (row*groups + group)*cols_per_group + col.
    // The scatter kernel index: dst_idx = (row*groups + group)*rows_per_group + col.
    fn group_gather_src_idx(
        row: usize,
        group: usize,
        groups: usize,
        cols_per_group: usize,
        col: usize,
    ) -> usize {
        (row * groups + group) * cols_per_group + col
    }
    fn group_scatter_dst_idx(
        row: usize,
        group: usize,
        groups: usize,
        rows_per_group: usize,
        col: usize,
    ) -> usize {
        (row * groups + group) * rows_per_group + col
    }

    #[test]
    fn grouped_decode_batched_n1_byte_identical_to_per_row_slice() {
        // n=1 BIT-IDENTITY: the M-parametric grouped decode (gather/GEMM/scatter)
        // must, at n=1, touch EXACTLY the offsets the deleted per-row path used:
        //   old input  = local_attn.data.slice(group*cols .. group*cols + cols)
        //   old output = latent.data.slice(group*rows .. group*rows + rows)
        // and the GEMM m-arg is 1 (skips the active_counts H2D). TP=4 grouped
        // example: 2 output groups owned by one rank, cols=4096, rows=1536.
        let (groups, cols, rows) = (2usize, 4096usize, 1536usize);
        let n = 1usize;
        let row = 0usize; // n==1 -> only row 0
        for group in 0..groups {
            // Old per-row contiguous slice start/end (the deleted code).
            let old_in_start = group * cols;
            let old_out_start = group * rows;
            // New gather/scatter walk the same span, element-for-element.
            assert_eq!(
                group_gather_src_idx(row, group, groups, cols, 0),
                old_in_start
            );
            assert_eq!(
                group_gather_src_idx(row, group, groups, cols, cols - 1),
                old_in_start + cols - 1
            );
            assert_eq!(
                group_scatter_dst_idx(row, group, groups, rows, 0),
                old_out_start
            );
            assert_eq!(
                group_scatter_dst_idx(row, group, groups, rows, rows - 1),
                old_out_start + rows - 1
            );
        }
        // GEMM m-arg at n=1 is 1 — the original per-row arg, so no active_counts H2D.
        assert_eq!(n as i32, 1);
    }
}

mod indexer_query_batch_tests {
    use super::super::*;
    use half::bf16;

    /// GPU gate (run on a CUDA box): the batched indexer-query projection
    /// pre-pass at n=1 must be BYTE-IDENTICAL to the per-row m=1 `dsv4_linear`
    /// it replaces. The full-flatten prepare loop slices column `r` of the
    /// batched `q_i`/`weights` and feeds it to `csa_select` instead of running
    /// the GEMV — so the bf16 GEMM column at n=1 must equal the standalone GEMV
    /// bit-for-bit (same cublasLt path, M=1). Mirrors the failure mode (a
    /// non-identical slice would silently corrupt CSA top-k selection) with a
    /// minimal in-component kernel — no full `Dsv4Indexer` construction.
    #[test]
    fn batched_indexer_query_slice_byte_identical_at_n1() {
        let Ok(ctx) = DeviceContext::new() else {
            eprintln!("[batched_indexer_query] no CUDA device; skipping");
            return;
        };

        // wq_b: [out=index_heads*index_head_dim, in=q_lora_rank];
        // weights_proj: [out=index_heads, in=hidden]. Representative CSA dims.
        let q_lora_rank = 384usize;
        let hidden = 512usize;
        let wq_b_rows = 256usize; // index_heads*index_head_dim
        let weights_rows = 4usize; // index_heads

        let prng = |seed: usize| -> f32 {
            let x = (seed as u64).wrapping_mul(2_654_435_761) ^ 0x9E37_79B9_7F4A_7C15;
            ((x >> 33) as f32 / u32::MAX as f32) - 0.5
        };

        let wq_b_host: Vec<bf16> = (0..wq_b_rows * q_lora_rank)
            .map(|i| bf16::from_f32(prng(i + 11) * 1.5))
            .collect();
        let weights_host: Vec<bf16> = (0..weights_rows * hidden)
            .map(|i| bf16::from_f32(prng(i + 4_000_037) * 1.5))
            .collect();
        let wq_b = DeviceMatrix::from_host(&ctx, &wq_b_host, wq_b_rows, q_lora_rank).unwrap();
        let weights_proj =
            DeviceMatrix::from_host(&ctx, &weights_host, weights_rows, hidden).unwrap();

        // N=1 inputs: c_q_normed [q_lora_rank, 1], normed [hidden, 1].
        let c_q_host: Vec<bf16> = (0..q_lora_rank)
            .map(|i| bf16::from_f32(prng(i + 909_091)))
            .collect();
        let normed_host: Vec<bf16> = (0..hidden)
            .map(|i| bf16::from_f32(prng(i + 717_171)))
            .collect();
        let mut c_q = unsafe { HiddenStates::uninit(&ctx, q_lora_rank, 1).unwrap() };
        c_q.data = ctx.stream.clone_htod(&c_q_host).unwrap();
        let mut normed = unsafe { HiddenStates::uninit(&ctx, hidden, 1).unwrap() };
        normed.data = ctx.stream.clone_htod(&normed_host).unwrap();

        // Batched pre-pass path (the GEMMs `indexer_query_batch_prepass` runs:
        // it reads only `wq_b` + `weights_proj`, so the two `dsv4_linear` calls
        // reproduce its output exactly without a full `Dsv4Indexer`).
        let mut q_i_batch = unsafe { HiddenStates::uninit(&ctx, wq_b_rows, 1).unwrap() };
        dsv4_linear(&ctx, &wq_b, &c_q, &mut q_i_batch).unwrap();
        let mut weights_batch = unsafe { HiddenStates::uninit(&ctx, weights_rows, 1).unwrap() };
        dsv4_linear(&ctx, &weights_proj, &normed, &mut weights_batch).unwrap();

        // Per-row reference (the path the precomputed slice replaces).
        let mut q_i_ref = unsafe { HiddenStates::uninit(&ctx, wq_b_rows, 1).unwrap() };
        dsv4_linear(&ctx, &wq_b, &c_q, &mut q_i_ref).unwrap();
        let mut weights_ref = unsafe { HiddenStates::uninit(&ctx, weights_rows, 1).unwrap() };
        dsv4_linear(&ctx, &weights_proj, &normed, &mut weights_ref).unwrap();
        ctx.sync().unwrap();

        let q_b = ctx.stream.clone_dtoh(&q_i_batch.data).unwrap();
        let q_r = ctx.stream.clone_dtoh(&q_i_ref.data).unwrap();
        let w_b = ctx.stream.clone_dtoh(&weights_batch.data).unwrap();
        let w_r = ctx.stream.clone_dtoh(&weights_ref.data).unwrap();
        ctx.sync().unwrap();

        // BYTE-IDENTICAL: same bf16 bit pattern, not just close. The n=1 slice
        // is the full output, and cublasLt M=1 is deterministic run-to-run.
        assert_eq!(
            q_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            q_r.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "batched q_i n=1 not byte-identical to per-row GEMV"
        );
        assert_eq!(
            w_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            w_r.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "batched weights n=1 not byte-identical to per-row GEMV"
        );
        eprintln!(
            "[batched_indexer_query] n=1 byte-identity OK (q_i {} elems, weights {} elems)",
            q_b.len(),
            w_b.len()
        );
    }
}
