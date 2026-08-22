//! The merge path supports two base-weight formats:
//! - **FP8 block-scaled**: the FP8 qweight/scales are kept alive in the matrix
//!   after BF16 promotion (for the `--share-frozen-base` student alias), and
//!   the pristine base is recovered by dequantizing on the fly — no separate
//!   BF16 base cache.
//! - **Native BF16** (e.g. linear-attn `in_proj_ba`): the base row window is
//!   cached once before the first merge so the all-zero-adapter restore path
//!   can put it back.

use super::*;
use StudentLoraProjection::*;
use cuda_kernels::quant_linear as cuda_ql;
use cuda_kernels::tensor::cache_ptr;

/// Block-scale grid metadata of an FP8 LoRA target, as the dequant/requant
/// launchers consume it.
fn lora_fp8_scale_shape(matrix: &DeviceMatrix) -> cuda_ql::Fp8ScaleShape {
    cuda_ql::Fp8ScaleShape {
        scale_rows: matrix.quant_scale_rows as i32,
        scale_cols: matrix.quant_scale_cols as i32,
        block_m: matrix.quant_block_m as i32,
        block_k: matrix.quant_block_k as i32,
    }
}

#[derive(Debug, Clone)]
pub struct StudentLoraMatrices {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub rank: usize,
    pub in_features: usize,
    pub out_features: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StudentLoraProjection {
    FullQ,
    FullK,
    FullV,
    FullO,
    LinearQkv,
    LinearZ,
    LinearB,
    LinearA,
    LinearOut,
    MlpGate,
    MlpUp,
    MlpDown,
    MoeRouter,
    MoeSharedGate,
    MoeSharedUp,
    MoeSharedDown,
    MoeSharedExpertGate,
    MoeExpertGate { expert_idx: usize },
    MoeExpertUp { expert_idx: usize },
    MoeExpertDown { expert_idx: usize },
}

impl StudentLoraProjection {
    pub fn label(self) -> Cow<'static, str> {
        match self {
            Self::FullQ => Cow::Borrowed("self_attn.q_proj"),
            Self::FullK => Cow::Borrowed("self_attn.k_proj"),
            Self::FullV => Cow::Borrowed("self_attn.v_proj"),
            Self::FullO => Cow::Borrowed("self_attn.o_proj"),
            Self::LinearQkv => Cow::Borrowed("self_attn.in_proj_qkv"),
            Self::LinearZ => Cow::Borrowed("self_attn.in_proj_z"),
            Self::LinearB => Cow::Borrowed("self_attn.in_proj_b"),
            Self::LinearA => Cow::Borrowed("self_attn.in_proj_a"),
            Self::LinearOut => Cow::Borrowed("self_attn.out_proj"),
            Self::MlpGate => Cow::Borrowed("mlp.gate_proj"),
            Self::MlpUp => Cow::Borrowed("mlp.up_proj"),
            Self::MlpDown => Cow::Borrowed("mlp.down_proj"),
            Self::MoeRouter => Cow::Borrowed("mlp.gate"),
            Self::MoeSharedGate => Cow::Borrowed("mlp.shared_expert.gate_proj"),
            Self::MoeSharedUp => Cow::Borrowed("mlp.shared_expert.up_proj"),
            Self::MoeSharedDown => Cow::Borrowed("mlp.shared_expert.down_proj"),
            Self::MoeSharedExpertGate => Cow::Borrowed("mlp.shared_expert_gate"),
            Self::MoeExpertGate { expert_idx } => {
                Cow::Owned(format!("mlp.experts.{expert_idx}.gate_proj"))
            }
            Self::MoeExpertUp { expert_idx } => {
                Cow::Owned(format!("mlp.experts.{expert_idx}.up_proj"))
            }
            Self::MoeExpertDown { expert_idx } => {
                Cow::Owned(format!("mlp.experts.{expert_idx}.down_proj"))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct StudentLoraProjectionUpdate {
    pub projection: StudentLoraProjection,
    pub matrices: StudentLoraMatrices,
}

/// `layer_idx` is the absolute model-layer index.
#[derive(Debug, Clone)]
pub struct StudentLoraLayer {
    pub layer_idx: usize,
    pub projections: Vec<StudentLoraProjectionUpdate>,
}

/// Carries raw A/B per full-attention layer plus `rank`/`alpha`; the merge
/// path applies `scale = alpha / rank` once.
#[derive(Debug, Clone)]
pub struct StudentLoraUpdate {
    pub layers: Vec<StudentLoraLayer>,
    pub rank: usize,
    pub alpha: f32,
    /// Requantize merged weights back to FP8 (2× base residency instead of 3×).
    pub requant_fp8: bool,
}

/// Resident FP8 block-scaled device pointers, exposed read-only for the
/// train-infer weight-sharing path (`--share-frozen-base`). The autograd
/// student's frozen base layers import these pointers as a NON-OWNING view
/// instead of allocating their own ~27 GB copy.
#[derive(Debug, Clone)]
pub struct SharedFp8BaseProjection {
    pub layer_idx: usize,
    pub proj_suffix: String,
    pub weight_ptr: u64,
    pub scale_ptr: u64,
    pub rows: usize,
    pub cols: usize,
    pub block_m: usize,
    pub block_k: usize,
}

/// Resident NVFP4 device pointers in the Marlin layout, the FP4 twin of
/// [`SharedFp8BaseProjection`]. The repack frees the group bytes, so this is
/// the only form a shared NVFP4 base can be borrowed in. `full_rows` is the
/// packed matrix's own N: the Marlin tile walk is in full-N coordinates, so a
/// fused matrix's row slice is a filter on that walk rather than a pointer
/// offset, and the importer needs both.
#[derive(Debug, Clone)]
pub struct SharedFp4BaseProjection {
    pub layer_idx: usize,
    pub proj_suffix: String,
    pub weight_ptr: u64,
    pub scale_tail_ptr: u64,
    pub global_scale: f32,
    pub full_rows: usize,
    pub row_offset: usize,
    pub rows: usize,
    pub cols: usize,
}

/// Non-owning view of a resident dense-BF16 base projection's device pointer,
/// for refreshing the train student's frozen base AFTER a LoRA re-merge (the
/// merged weights live in the BF16 `data` buffer; the retired FP8 buffers are
/// freed once the student re-aliases the BF16 bytes).
#[derive(Debug, Clone)]
pub struct SharedBf16BaseProjection {
    pub layer_idx: usize,
    pub proj_suffix: String,
    pub data_ptr: u64,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LoraBaseKey {
    layer_idx: usize,
    projection: StudentLoraProjection,
}

impl Qwen35Model {
    /// Walk every per-matrix base projection once — dense attention, dense MLP,
    /// the MoE shared expert, and the per-expert `DeviceMatrix` vecs — handing
    /// each to `f` as `(layer_idx, suffix, matrix, row_offset, sub_rows)`. The
    /// row window is the fused-matrix slice (q out of qkv); it is `(0, rows)`
    /// for anything unfused. The three `frozen_base_*_pointers` collectors
    /// differ only in what they do per matrix.
    fn for_each_base_projection<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(usize, String, &DeviceMatrix, usize, usize) -> Result<()>,
    {
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let dense: &[StudentLoraProjection] = match &layer.attn {
                Qwen35Attn::Full(_) => &[FullQ, FullK, FullV, FullO],
                Qwen35Attn::Linear(_) => &[LinearQkv, LinearZ, LinearB, LinearA, LinearOut],
            };
            let windowed = |projections: &[StudentLoraProjection], f: &mut F| -> Result<()> {
                for &proj in projections {
                    let m = self.lora_matrix(layer_idx, proj)?;
                    let (off, n) = self.lora_row_window(layer_idx, proj, m.rows);
                    f(layer_idx, proj.label().into_owned(), m, off, n)?;
                }
                Ok(())
            };
            windowed(dense, &mut f)?;
            if layer.mlp.is_some() {
                windowed(&[MlpGate, MlpUp, MlpDown], &mut f)?;
            }
            if let Some(moe) = &layer.moe {
                for &proj in &[MoeSharedGate, MoeSharedUp, MoeSharedDown] {
                    let m = self.lora_matrix(layer_idx, proj)?;
                    f(layer_idx, proj.label().into_owned(), m, 0, m.rows)?;
                }
                for e in 0..moe.gate.len() {
                    for &proj in &[
                        MoeExpertGate { expert_idx: e },
                        MoeExpertUp { expert_idx: e },
                        MoeExpertDown { expert_idx: e },
                    ] {
                        let m = self.lora_matrix(layer_idx, proj)?;
                        f(layer_idx, proj.label().into_owned(), m, 0, m.rows)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Read-only borrow of every resident FP8 block-scaled base projection's
    /// device pointers, for train-infer weight sharing (`--share-frozen-base`).
    ///
    /// in_proj_a/b are tiny per-head BF16 (fp8_block_scaled_ptrs skips them via
    /// the format filter). MoE routed experts come from either the per-expert
    /// `DeviceMatrix` vecs (DeepGEMM disabled) or sliced out of the fused
    /// `w13`/`down` grouped FP8 buffers (default). The train side picks the
    /// subset it actually shares by matching `(layer_idx, proj_suffix)`.
    /// Single-GPU only — TP/EP shards would split the base, so group index
    /// equals global expert index.
    pub(crate) fn frozen_base_fp8_pointers(&self) -> Result<Vec<SharedFp8BaseProjection>> {
        ensure!(
            self.tp.is_single(),
            "frozen-base FP8 sharing is single-GPU only; got TP world_size={}",
            self.tp.config().world_size
        );
        // FP8→BF16 promotion must retire (not free) the FP8 buffers: the
        // importer holds non-owning views of these pointers.
        self.frozen_base_ptrs_exported
            .store(true, Ordering::Relaxed);
        let ctx = &self.ctx;
        // `row_offset` must be a multiple of `block_m` (FP8 block-scaled invariant).
        fn push_row_slice(
            out: &mut Vec<SharedFp8BaseProjection>,
            ctx: &DeviceContext,
            layer_idx: usize,
            suffix: String,
            m: &DeviceMatrix,
            row_offset: usize,
            sub_rows: usize,
        ) -> Result<()> {
            // `fp8_block_scaled_ptrs` returns None both for a weight that was
            // never FP8 (in_proj_a/b are tiny BF16 — skip those) and for one
            // whose FP8 source the Marlin repack released. Skipping the second
            // silently drops it from the shared table, and the importer answers
            // the gap with a private full-size copy instead of an error.
            ensure!(
                !m.quant_source_freed(),
                "layer {layer_idx} {suffix}: base weight was Marlin-repacked and its FP8 source \
                 released at load, so there is nothing to share. Load without the Marlin repack \
                 for a frozen-base-sharing engine."
            );
            let Some((weight_ptr, scale_ptr, rows, cols, block_m, block_k)) =
                m.fp8_block_scaled_ptrs(ctx)
            else {
                return Ok(());
            };
            debug_assert!(row_offset.is_multiple_of(block_m));
            debug_assert!(row_offset + sub_rows <= rows);
            let scale_row_offset = (row_offset / block_m) * m.quant_scale_cols;
            out.push(SharedFp8BaseProjection {
                layer_idx,
                proj_suffix: suffix,
                weight_ptr: weight_ptr + (row_offset * cols) as u64,
                scale_ptr: scale_ptr + (scale_row_offset * std::mem::size_of::<f32>()) as u64,
                rows: sub_rows,
                cols,
                block_m,
                block_k,
            });
            Ok(())
        }
        let mut out = Vec::new();
        self.for_each_base_projection(|layer_idx, suffix, m, off, n| {
            push_row_slice(&mut out, ctx, layer_idx, suffix, m, off, n)
        })?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if let Some(moe) = &layer.moe
                && moe.gate.is_empty()
            {
                {
                    // Grouped FP8 buffers: slice per-expert ptrs directly.
                    if let Some(w13) = &moe.w13_fp8_grouped {
                        let mi = w13.rows / 2;
                        for e in 0..w13.groups {
                            if let Some(p) = w13.expert_slice_fp8_ptrs(ctx, e, 0, mi) {
                                out.push(SharedFp8BaseProjection {
                                    layer_idx,
                                    proj_suffix: format!("mlp.experts.{e}.gate_proj"),
                                    weight_ptr: p.0,
                                    scale_ptr: p.1,
                                    rows: p.2,
                                    cols: p.3,
                                    block_m: p.4,
                                    block_k: p.5,
                                });
                            }
                            if let Some(p) = w13.expert_slice_fp8_ptrs(ctx, e, mi, mi) {
                                out.push(SharedFp8BaseProjection {
                                    layer_idx,
                                    proj_suffix: format!("mlp.experts.{e}.up_proj"),
                                    weight_ptr: p.0,
                                    scale_ptr: p.1,
                                    rows: p.2,
                                    cols: p.3,
                                    block_m: p.4,
                                    block_k: p.5,
                                });
                            }
                        }
                    }
                    if let Some(down) = &moe.down_fp8_grouped {
                        for e in 0..down.groups {
                            if let Some(p) = down.expert_slice_fp8_ptrs(ctx, e, 0, down.rows) {
                                out.push(SharedFp8BaseProjection {
                                    layer_idx,
                                    proj_suffix: format!("mlp.experts.{e}.down_proj"),
                                    weight_ptr: p.0,
                                    scale_ptr: p.1,
                                    rows: p.2,
                                    cols: p.3,
                                    block_m: p.4,
                                    block_k: p.5,
                                });
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// The NVFP4 twin of [`Qwen35Model::frozen_base_fp8_pointers`]. Sharing an
    /// NVFP4 base costs ~21 GB resident against FP8's ~30 GB, and it is the
    /// only way to share one at all: `repack_for_marlin_fp4` releases the group
    /// bytes at load, so there is no group-layout source left to borrow.
    /// Single-GPU only, for the same reason as the FP8 path.
    pub(crate) fn frozen_base_fp4_pointers(&self) -> Result<Vec<SharedFp4BaseProjection>> {
        ensure!(
            self.tp.is_single(),
            "frozen-base NVFP4 sharing is single-GPU only; got TP world_size={}",
            self.tp.config().world_size
        );
        self.frozen_base_ptrs_exported
            .store(true, Ordering::Relaxed);
        let ctx = &self.ctx;
        let mut out = Vec::new();
        self.for_each_base_projection(|layer_idx, suffix, m, row_offset, sub_rows| {
            // `None` here means "not NVFP4" (in_proj_a/b are tiny BF16) — a
            // repacked NVFP4 weight always has both buffers.
            let Some((weight_ptr, scale_tail_ptr, global_scale, full_rows, cols)) =
                m.marlin_fp4_ptrs(ctx)
            else {
                return Ok(());
            };
            ensure!(
                row_offset + sub_rows <= full_rows,
                "layer {layer_idx} {suffix}: row window {row_offset}+{sub_rows} runs past N={full_rows}"
            );
            out.push(SharedFp4BaseProjection {
                layer_idx,
                proj_suffix: suffix,
                weight_ptr,
                scale_tail_ptr,
                global_scale,
                full_rows,
                row_offset,
                rows: sub_rows,
                cols,
            });
            Ok(())
        })?;
        Ok(out)
    }

    /// Non-owning views of every resident dense-BF16 base projection's device
    /// pointer, for refreshing the train student's frozen base AFTER a LoRA
    /// re-merge. Projections still stored as FP8 block-scaled are skipped (the
    /// student keeps its existing FP8 alias for those).
    pub(crate) fn frozen_base_bf16_pointers(&self) -> Result<Vec<SharedBf16BaseProjection>> {
        ensure!(
            self.tp.is_single(),
            "frozen-base BF16 sharing is single-GPU only; got TP world_size={}",
            self.tp.config().world_size
        );
        // The trainer holds non-owning views of these buffers from here on;
        // offload_engine_weights refuses to free them (see its ensure!).
        self.frozen_base_ptrs_exported
            .store(true, Ordering::Relaxed);
        let ctx = &self.ctx;
        fn push_row_slice(
            out: &mut Vec<SharedBf16BaseProjection>,
            ctx: &DeviceContext,
            layer_idx: usize,
            suffix: String,
            m: &DeviceMatrix,
            row_offset: usize,
            sub_rows: usize,
        ) -> Result<()> {
            if m.weight_format() != WeightFormat::DenseBf16 {
                return Ok(());
            }
            debug_assert!(row_offset + sub_rows <= m.rows);
            let (ptr, _g) = m.data.device_ptr(ctx.stream.as_ref());
            out.push(SharedBf16BaseProjection {
                layer_idx,
                proj_suffix: suffix,
                data_ptr: ptr + (row_offset * m.cols * std::mem::size_of::<bf16>()) as u64,
                rows: sub_rows,
                cols: m.cols,
            });
            Ok(())
        }
        let mut out = Vec::new();
        self.for_each_base_projection(|layer_idx, suffix, m, off, n| {
            push_row_slice(&mut out, ctx, layer_idx, suffix, m, off, n)
        })?;
        Ok(out)
    }

    /// `A` is `[rank, in]`, `B` is `[out, rank]`, matching the train-side
    /// `LinearWithLora` contract.
    pub(crate) fn remerge_student_lora(&mut self, update: StudentLoraUpdate) -> Result<()> {
        ensure!(update.rank > 0, "student LoRA update has rank=0");
        ensure!(
            self.tp.is_single(),
            "student LoRA re-merge is currently single-GPU only; got TP world_size={}",
            self.tp.config().world_size
        );
        let scale = update.alpha / update.rank as f32;
        let num_layers = self.config.num_hidden_layers;

        for layer in &update.layers {
            let layer_idx = layer.layer_idx;
            ensure!(
                layer_idx < num_layers,
                "student LoRA references layer {layer_idx} but model has {num_layers} layers"
            );
            ensure!(
                !layer.projections.is_empty(),
                "student LoRA layer {layer_idx} carries no projection updates"
            );

            if update.requant_fp8 {
                // Requant hands the next merge a pristine FP8 base, so that
                // merge restores the WHOLE matrix and re-applies only the
                // windows it carries: a dirty row-fused sibling left out of
                // this update would lose its delta silently.
                let present: Vec<StudentLoraProjection> =
                    layer.projections.iter().map(|p| p.projection).collect();
                for key in self.lora_dirty.iter() {
                    ensure!(
                        key.layer_idx != layer_idx || present.contains(&key.projection),
                        "layer {layer_idx} {}: merged under --lora-merge-fp8 but absent from this \
                         update; requant needs every dirty projection of a layer in one update",
                        key.projection.label()
                    );
                }
            }
            for projection in &layer.projections {
                self.merge_lora_proj(
                    layer_idx,
                    projection.projection,
                    &projection.matrices,
                    scale,
                )?;
            }
            // Per layer, not per update: the dense promotions of one layer are
            // the peak, and row-fused siblings share a layer.
            if update.requant_fp8 {
                for projection in &layer.projections {
                    self.requant_merged_matrix(layer_idx, projection.projection)?;
                }
            }
        }
        self.ctx.sync()?;
        Ok(())
    }

    /// Quantize the merged dense back into the FP8 serving slots; the pristine
    /// pair moves to `pristine_fp8` (device addresses unchanged, aliases valid).
    fn requant_merged_matrix(
        &mut self,
        layer_idx: usize,
        projection: StudentLoraProjection,
    ) -> Result<()> {
        let label = projection.label();
        let ctx = self.ctx.clone();
        let matrix = self.lora_matrix_mut(layer_idx, projection)?;
        if !matrix.is_dense_bf16() {
            return Ok(());
        }
        if matrix.pristine_fp8.is_none() {
            if matrix.qweight_u8.is_none() || matrix.scale_f32.is_none() {
                return Ok(());
            }
            ensure!(
                matrix.quant_block_m > 0
                    && matrix.quant_block_k > 0
                    && matrix.quant_scale_rows > 0
                    && matrix.quant_scale_cols > 0,
                "layer {layer_idx} {label}: merge-requant missing block-scale metadata"
            );
            // Both allocations first: a failure between them would leave a
            // half-split state that no retry can repair.
            let merged_qweight = ctx
                .stream
                .alloc_zeros::<u8>(matrix.rows * matrix.cols)
                .map_err(|e| {
                    anyhow!("layer {layer_idx} {label}: merged qweight alloc failed: {e}")
                })?;
            let merged_scales = ctx
                .stream
                .alloc_zeros::<f32>(matrix.quant_scale_rows * matrix.quant_scale_cols)
                .map_err(|e| {
                    anyhow!("layer {layer_idx} {label}: merged scales alloc failed: {e}")
                })?;
            matrix.pristine_fp8 = Some((
                matrix.qweight_u8.take().expect("checked above"),
                matrix.scale_f32.take().expect("checked above"),
            ));
            matrix.qweight_u8 = Some(merged_qweight);
            matrix.scale_f32 = Some(merged_scales);
        }
        {
            let (data, qweight, scales) = (
                &matrix.data,
                matrix.qweight_u8.as_mut().expect("split out above"),
                matrix.scale_f32.as_mut().expect("split out above"),
            );
            cuda_ql::quantize_bf16_to_fp8_block_scaled(
                &ctx,
                data,
                qweight,
                scales,
                matrix.rows,
                matrix.cols,
                matrix.quant_block_m,
                matrix.quant_block_k,
            )
            .map_err(|e| anyhow!("layer {layer_idx} {label}: merge-requant failed: {e}"))?;
        }
        matrix.weight_format = WeightFormat::Fp8BlockScaled;
        matrix.data = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("layer {layer_idx} {label}: dense placeholder alloc: {e}"))?;
        Ok(())
    }

    /// Promote FP8-block-scaled LoRA targets to dense BF16 on first touch.
    /// Replaces the former host remerge lane (O(rows·cols·rank) triple loop +
    /// re-quant + full-W upload, 60-83s/round) with a one-time kernel; every
    /// later re-merge rides the on-device dense lane.
    ///
    /// VRAM: trades FP8→BF16 storage (2×) for touched projections only. If
    /// `--share-frozen-base` exported the FP8 pointers, the retired buffers are
    /// kept alive (aliased non-owningly by the autograd student); otherwise
    /// they are freed.
    fn promote_lora_target_to_bf16(
        &mut self,
        layer_idx: usize,
        projection: StudentLoraProjection,
    ) -> Result<()> {
        let label = projection.label();
        let ctx = self.ctx.clone();
        let matrix = self.lora_matrix_mut(layer_idx, projection)?;
        if matrix.is_dense_bf16() {
            return Ok(());
        }
        // NVFP4's only resident form is the Marlin layout (the repack releases
        // the group bytes at load), so it promotes straight from those tiles.
        // The pristine Marlin buffer is kept, not freed: a share-frozen-base
        // student aliases it, and `weight_format = DenseBf16` already routes the
        // forward to `data`, so nothing reads the stale tiles.
        if matrix.weight_format() == WeightFormat::Fp4E2M1Group {
            let dense = ctx
                .stream
                .alloc_zeros::<bf16>(matrix.rows * matrix.cols)
                .map_err(|e| {
                    anyhow!("layer {layer_idx} {label}: NVFP4 BF16 promotion alloc failed: {e}")
                })?;
            let packed = matrix.marlin_packed.as_ref().ok_or_else(|| {
                anyhow!(
                    "layer {layer_idx} {label}: NVFP4 LoRA target has no Marlin layout to promote \
                     from and its group source was released at load"
                )
            })?;
            let global = matrix.scale_f32.as_ref().ok_or_else(|| {
                anyhow!("layer {layer_idx} {label}: NVFP4 LoRA target missing its global scale")
            })?;
            let mut dense = dense;
            cuda_ql::dequantize_fp4_marlin_to_bf16(
                &ctx,
                packed,
                global,
                matrix.fp4_marlin_scale_lift_inv,
                &mut dense,
                matrix.rows,
                matrix.cols,
                matrix.group_size,
            )
            .map_err(|e| {
                anyhow!("layer {layer_idx} {label}: NVFP4→BF16 promotion dequant failed: {e}")
            })?;
            // Give the matrix the FP8 slots the merge lane expects. Without
            // them `requant_merged_matrix` finds no `qweight_u8`, returns
            // early, and the weight stays dense BF16 forever -- 4x the NVFP4
            // bytes per touched projection, which OOMs a 27B all-linear merge.
            // From here every re-merge rides the proven FP8 lane; the engine
            // serves FP8 rather than NVFP4 once a LoRA has been merged in.
            const FP8_BLOCK: usize = 128;
            let scale_rows = matrix.rows.div_ceil(FP8_BLOCK);
            let scale_cols = matrix.cols.div_ceil(FP8_BLOCK);
            let mut qweight = ctx
                .stream
                .alloc_zeros::<u8>(matrix.rows * matrix.cols)
                .map_err(|e| anyhow!("layer {layer_idx} {label}: NVFP4 fp8 slot alloc: {e}"))?;
            let mut scales = ctx
                .stream
                .alloc_zeros::<f32>(scale_rows * scale_cols)
                .map_err(|e| anyhow!("layer {layer_idx} {label}: NVFP4 fp8 scale alloc: {e}"))?;
            cuda_ql::quantize_bf16_to_fp8_block_scaled(
                &ctx,
                &dense,
                &mut qweight,
                &mut scales,
                matrix.rows,
                matrix.cols,
                FP8_BLOCK,
                FP8_BLOCK,
            )
            .map_err(|e| anyhow!("layer {layer_idx} {label}: NVFP4→FP8 pristine requant: {e}"))?;
            matrix.qweight_u8 = Some(qweight);
            matrix.scale_f32 = Some(scales);
            matrix.quant_block_m = FP8_BLOCK;
            matrix.quant_block_k = FP8_BLOCK;
            matrix.quant_scale_rows = scale_rows;
            matrix.quant_scale_cols = scale_cols;
            matrix.data = dense;
            matrix.weight_format = WeightFormat::DenseBf16;
            // Park the tiles instead of freeing them: a share-frozen-base
            // student aliases the packed bytes. Out of `marlin_packed` the FP8
            // arm this weight requants into can no longer pick them up and read
            // FP4 tiles as FP8.
            matrix.retired_marlin = match (matrix.marlin_packed.take(), matrix.marlin_scales.take())
            {
                (Some(packed), Some(global)) => Some((packed, global)),
                _ => None,
            };
            return Ok(());
        }
        ensure!(
            matrix.weight_format() == WeightFormat::Fp8BlockScaled,
            "layer {layer_idx} {label}: LoRA merge supports dense BF16, FP8 block-scaled, or \
             Marlin-repacked NVFP4 weights; got {:?}",
            matrix.weight_format()
        );
        ensure!(
            matrix.quant_block_m > 0
                && matrix.quant_block_k > 0
                && matrix.quant_scale_rows > 0
                && matrix.quant_scale_cols > 0,
            "layer {layer_idx} {label}: FP8 LoRA target missing block-scale metadata"
        );
        // The merge rewrites `qweight_u8`/`scale_f32` and nothing else, so a
        // target that also carries a Marlin layout would keep serving the
        // un-merged bytes on every arm that reads it. Refuse instead: this used
        // to be unreachable because the repack released the source, and holding
        // the source for the DeepGEMM prefill arm must not turn a loud refusal
        // into a silent disagreement between prefill and decode.
        ensure!(
            matrix.marlin_packed.is_none(),
            "layer {layer_idx} {label}: LoRA merge cannot rewrite the Marlin layout this weight \
             was repacked into at load, so decode would keep serving the un-merged base. Load a \
             LoRA-merging engine on a build/card where the repack does not run."
        );
        let dense = ctx
            .stream
            .alloc_zeros::<bf16>(matrix.rows * matrix.cols)
            .map_err(|e| anyhow!("layer {layer_idx} {label}: BF16 promotion alloc failed: {e}"))?;
        {
            let (qweight, scales) = matrix.merge_base_fp8().ok_or_else(|| {
                anyhow!("layer {layer_idx} {label}: FP8 LoRA target missing qweight/scales")
            })?;
            ensure!(
                qweight.len() == matrix.rows * matrix.cols,
                "layer {layer_idx} {label}: FP8 qweight len {} != rows*cols {}",
                qweight.len(),
                matrix.rows * matrix.cols
            );
            // SAFETY: `dense` covers rows*cols and lives across the launch.
            unsafe {
                cuda_ql::dequantize_fp8_block_scaled_to_bf16(
                    &ctx,
                    qweight,
                    scales,
                    cache_ptr(&dense, &ctx),
                    matrix.rows,
                    matrix.cols,
                    lora_fp8_scale_shape(matrix),
                )
            }
            .map_err(|e| {
                anyhow!("layer {layer_idx} {label}: FP8→BF16 promotion dequant failed: {e}")
            })?;
        }
        // Keep the retired FP8 qweight/scales in the matrix: the
        // share-frozen-base student aliases these device pointers, and the
        // per-step LoRA merge dequantizes them on the fly to recover the
        // pristine BF16 base — avoiding a separate ~2×base BF16 base cache
        // that would OOM the 27B sync (FP8 keepalive + BF16 matrix + BF16
        // base cache ≈ 3× base bytes). The FP8 buffers are never freed here
        // (the student owns the alias lifetime); `weight_format = DenseBf16`
        // keeps the forward path on `data`.
        matrix.data = dense;
        matrix.weight_format = WeightFormat::DenseBf16;
        Ok(())
    }

    /// Projections that live inside a row-fused matrix (MlpUp shares
    /// `gate_up_proj` with MlpGate) map to one canonical key so the pristine
    /// device base is cached once per underlying buffer.
    // Each projection caches its OWN row window: collapsing row-fused siblings
    // (e.g. FullK/FullV onto FullQ) made a restore return the other
    // projection's bytes.
    fn lora_base_cache_key(layer_idx: usize, projection: StudentLoraProjection) -> LoraBaseKey {
        LoraBaseKey {
            layer_idx,
            projection,
        }
    }

    /// `(row_offset, rows)` of a projection inside its (possibly row-fused)
    /// resident matrix: e.g. MlpUp occupies `[inter_dim, inter_dim)` of
    /// `gate_up_proj`, FullK the `[q_gated, kv)` window of `qkv_proj`.
    fn lora_row_window(
        &self,
        layer_idx: usize,
        projection: StudentLoraProjection,
        matrix_rows: usize,
    ) -> (usize, usize) {
        let layer = &self.layers[layer_idx];
        let inter = || layer.mlp.as_ref().map(DenseMlp::inter_dim).unwrap_or(0);
        let vh = || match &layer.attn {
            Qwen35Attn::Linear(lin) => lin.in_proj_ba.rows / 2,
            _ => 0,
        };
        let q_gated = self.local_full_attn_q_proj_dim();
        let kv = self.local_kv_heads * self.config.head_dim;
        let qkv = self.local_linear_qkv_dim();
        match projection {
            MlpGate => (0, inter()),
            MlpUp => (inter(), inter()),
            LinearB => (0, vh()),
            LinearA => (vh(), vh()),
            FullQ => (0, q_gated),
            FullK => (q_gated, kv),
            FullV => (q_gated + kv, kv),
            LinearQkv => (0, qkv),
            LinearZ => (qkv, self.local_linear_z_dim()),
            _ => (0, matrix_rows),
        }
    }

    /// Merge `W = base + scale·(B·A)` for one projection, entirely on device.
    /// FP8-stored targets are promoted to dense BF16 on first touch, so every
    /// projection rides one lane.
    fn merge_lora_proj(
        &mut self,
        layer_idx: usize,
        projection: StudentLoraProjection,
        adapter: &StudentLoraMatrices,
        scale: f32,
    ) -> Result<()> {
        let label = projection.label();
        let key = LoraBaseKey {
            layer_idx,
            projection,
        };

        let adapter_is_zero = adapter.b.iter().all(|&value| value == 0.0);
        if adapter_is_zero && !self.lora_dirty.contains(&key) {
            return Ok(());
        }

        let rows = adapter.out_features;
        let cols = adapter.in_features;
        ensure!(
            adapter.a.len() == adapter.rank * cols,
            "layer {layer_idx} {label}: lora_A len {} != rank*in {}",
            adapter.a.len(),
            adapter.rank * cols
        );
        ensure!(
            adapter.b.len() == rows * adapter.rank,
            "layer {layer_idx} {label}: lora_B len {} != out*rank {}",
            adapter.b.len(),
            rows * adapter.rank
        );

        self.promote_lora_target_to_bf16(layer_idx, projection)?;

        if adapter_is_zero {
            // Restore the pristine *device* base (no host round-trip).
            if self.lora_dirty.remove(&key) {
                let cache_key = Self::lora_base_cache_key(layer_idx, projection);
                self.restore_lora_base_dev(layer_idx, projection, &cache_key)?;
            }
            return Ok(());
        }
        self.merge_lora_proj_device(layer_idx, projection, adapter, scale, &key)
    }

    /// The pristine base is recovered on the fly by dequantizing the FP8
    /// qweight/scales kept in the matrix (kept alive for the share-frozen-base
    /// student alias), so no separate BF16 base cache is needed — saves
    /// ~2×base bytes during the 27B sync.
    fn merge_lora_proj_device(
        &mut self,
        layer_idx: usize,
        projection: StudentLoraProjection,
        adapter: &StudentLoraMatrices,
        scale: f32,
        key: &LoraBaseKey,
    ) -> Result<()> {
        let label = projection.label();
        let rows = adapter.out_features;
        let cols = adapter.in_features;
        let rank = adapter.rank;

        let (row_offset, _) = self.lora_row_window(layer_idx, projection, rows);

        // `a_t` is A transposed to [cols, rank] row-major
        // (== W[c,k]=A[k,c] for `lora_device_gemm`); `b` uploads as-is
        // ([rows, rank] row-major == col-major X[k,r]=B[r,k]).
        let mut a_t = vec![bf16::ZERO; cols * rank];
        for k in 0..rank {
            let a_row = &adapter.a[k * cols..k * cols + cols];
            for (c, &a_kc) in a_row.iter().enumerate() {
                a_t[c * rank + k] = bf16::from_f32(a_kc);
            }
        }
        let b_host: Vec<bf16> = adapter.b.iter().map(|&v| bf16::from_f32(v)).collect();
        let a_t_dev = DeviceVec::from_host(&self.ctx, &a_t)?;
        let b_dev = DeviceVec::from_host(&self.ctx, &b_host)?;

        // Reusable delta scratch, grown to the largest dense matrix seen.
        let needed = rows * cols;
        if self
            .lora_delta_scratch
            .as_ref()
            .map(|s| s.len < needed)
            .unwrap_or(true)
        {
            self.lora_delta_scratch = Some(DeviceVec::zeros(&self.ctx, needed)?);
        }
        let ctx = self.ctx.clone();

        {
            let scratch = self
                .lora_delta_scratch
                .as_mut()
                .expect("scratch allocated above");
            crate::ops::lora_device_gemm(
                &ctx,
                &a_t_dev.data,
                &b_dev.data,
                &mut scratch.data,
                rows,
                cols,
                rank,
            )?;
        }

        // FP8-stored targets dequantize their kept-alive qweight/scales on the
        // fly (no BF16 base cache). Native-BF16 targets (e.g. linear-attn
        // `in_proj_ba`) already hold their base in `data`; we cache that row
        // window once (before the first merge) so the all-zero-adapter restore
        // path can put it back.
        let window = row_offset * cols..row_offset * cols + needed;
        {
            let matrix = self.lora_matrix(layer_idx, projection)?;
            ensure!(
                matrix.is_dense_bf16() && row_offset + rows <= matrix.rows && matrix.cols == cols,
                "layer {layer_idx} {label}: dense device merge shape/format mismatch \
                 ({}x{} {:?} vs window [{row_offset}..{}]x{cols})",
                matrix.rows,
                matrix.cols,
                matrix.weight_format(),
                row_offset + rows
            );
            let cache_key = Self::lora_base_cache_key(layer_idx, projection);
            if let Some((qweight, scales)) = matrix.merge_base_fp8() {
                let base_scratch = DeviceVec::zeros(&ctx, matrix.rows * matrix.cols)?;
                // SAFETY: `base_scratch` covers rows*cols and lives across
                // the launch.
                unsafe {
                    cuda_ql::dequantize_fp8_block_scaled_to_bf16(
                        &ctx,
                        qweight,
                        scales,
                        cache_ptr(&base_scratch.data, &ctx),
                        matrix.rows,
                        matrix.cols,
                        lora_fp8_scale_shape(matrix),
                    )
                }
                .map_err(|e| {
                    anyhow!(
                        "layer {layer_idx} {label}: FP8→BF16 base dequant for merge failed: {e}"
                    )
                })?;
                let src = base_scratch.data.slice(window.clone());
                let matrix = self.lora_matrix_mut(layer_idx, projection)?;
                let mut dst = matrix.data.slice_mut(window.clone());
                ctx.stream.memcpy_dtod(&src, &mut dst).map_err(|e| {
                    anyhow!("layer {layer_idx} {label}: base window D2D copy failed: {e}")
                })?;
            } else if let Some(cache) = self.lora_base_dev.remove(&cache_key) {
                // Dirty re-merge: restore the pristine window first so
                // W = base + Δ stays idempotent (no Δ accumulation across syncs).
                let copied = {
                    let src = cache.data.slice(0..needed);
                    let matrix = self.lora_matrix_mut(layer_idx, projection)?;
                    let mut dst = matrix.data.slice_mut(window.clone());
                    ctx.stream.memcpy_dtod(&src, &mut dst)
                };
                self.lora_base_dev.insert(cache_key, cache);
                copied.map_err(|e| {
                    anyhow!("layer {layer_idx} {label}: BF16 base window restore failed: {e}")
                })?;
            } else {
                // Native-BF16 base: cache the row window once (before the first
                // merge) so restore can put it back.
                let mut cache = DeviceVec::zeros(&ctx, needed)?;
                {
                    let matrix = self.lora_matrix(layer_idx, projection)?;
                    let src = matrix.data.slice(window.clone());
                    let mut dst = cache.data.slice_mut(0..needed);
                    ctx.stream.memcpy_dtod(&src, &mut dst).map_err(|e| {
                        anyhow!("layer {layer_idx} {label}: BF16 base cache D2D copy failed: {e}")
                    })?;
                }
                self.lora_base_dev.insert(cache_key, cache);
            }
        }

        let delta_data = self
            .lora_delta_scratch
            .as_ref()
            .expect("scratch allocated above")
            .data
            .clone();
        let delta_view = delta_data.slice(0..needed);
        {
            let matrix = self.lora_matrix_mut(layer_idx, projection)?;
            let mut out_view = matrix.data.slice_mut(window);
            cuda_kernels::tensor_ops::add_scaled_row(
                &ctx,
                &delta_view,
                &mut out_view,
                needed,
                0,
                scale,
            )
            .map_err(|e| anyhow!("layer {layer_idx} {label}: LoRA scaled add failed: {e}"))?;
        }

        self.lora_dirty.insert(*key);
        Ok(())
    }

    /// FP8 targets dequantize their kept-alive qweight/scales; native-BF16
    /// targets copy from the one-shot row-window cache captured at first merge.
    /// Only the projection's own row window is restored, so the other half of a
    /// row-fused matrix keeps its (possibly merged) state.
    fn restore_lora_base_dev(
        &mut self,
        layer_idx: usize,
        projection: StudentLoraProjection,
        key: &LoraBaseKey,
    ) -> Result<()> {
        let label = projection.label();
        let ctx = self.ctx.clone();
        let matrix_rows = self.lora_matrix(layer_idx, projection)?.rows;
        let (row_offset, rows) = self.lora_row_window(layer_idx, projection, matrix_rows);
        let matrix = self.lora_matrix(layer_idx, projection)?;
        ensure!(
            matrix.is_dense_bf16(),
            "layer {layer_idx} {label}: restore requires dense BF16; got {:?}",
            matrix.weight_format()
        );
        let cols = matrix.cols;
        let window = row_offset * cols..(row_offset + rows) * cols;
        if let Some((qweight, scales)) = matrix.merge_base_fp8() {
            let scratch = DeviceVec::zeros(&ctx, matrix.rows * matrix.cols)?;
            // SAFETY: `scratch` covers rows*cols and lives across the launch.
            unsafe {
                cuda_ql::dequantize_fp8_block_scaled_to_bf16(
                    &ctx,
                    qweight,
                    scales,
                    cache_ptr(&scratch.data, &ctx),
                    matrix.rows,
                    matrix.cols,
                    lora_fp8_scale_shape(matrix),
                )
            }
            .map_err(|e| {
                anyhow!("layer {layer_idx} {label}: FP8→BF16 base dequant for restore failed: {e}")
            })?;
            let src = scratch.data.slice(window.clone());
            let matrix = self.lora_matrix_mut(layer_idx, projection)?;
            let mut dst = matrix.data.slice_mut(window);
            ctx.stream.memcpy_dtod(&src, &mut dst).map_err(|e| {
                anyhow!("layer {layer_idx} {label}: device base restore D2D failed: {e}")
            })?;
        } else {
            let cache_data = self
                .lora_base_dev
                .get(key)
                .ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {label}: BF16 base cache missing for restore \
                         (merge never cached it?)"
                    )
                })?
                .data
                .clone();
            let src = cache_data.slice(0..rows * cols);
            let matrix = self.lora_matrix_mut(layer_idx, projection)?;
            let mut dst = matrix.data.slice_mut(window);
            ctx.stream.memcpy_dtod(&src, &mut dst).map_err(|e| {
                anyhow!("layer {layer_idx} {label}: BF16 base restore D2D failed: {e}")
            })?;
        }
        Ok(())
    }

    fn local_expert_idx(&self, global_expert: usize) -> Result<usize> {
        ensure!(
            self.expert_split.owns(global_expert),
            "Qwen3.6 LoRA sync expert {global_expert} is not local to this rank \
             (local range {}..{})",
            self.expert_split.local_expert_start,
            self.expert_split.local_expert_end()
        );
        Ok(global_expert - self.expert_split.local_expert_start)
    }

    fn lora_matrix(
        &self,
        layer_idx: usize,
        projection: StudentLoraProjection,
    ) -> Result<&DeviceMatrix> {
        let layer = &self.layers[layer_idx];
        match projection {
            FullQ | FullK | FullV | FullO => {
                let Qwen35Attn::Full(full) = &layer.attn else {
                    return Err(anyhow!(
                        "layer {layer_idx} {} requires a full-attention layer",
                        projection.label()
                    ));
                };
                Ok(match projection {
                    // q/k/v live in the row-fused `qkv_proj`; callers address
                    // their window via `lora_row_window`.
                    FullQ | FullK | FullV => &full.qkv_proj,
                    FullO => &full.o_proj,
                    _ => unreachable!("full projection arm checked above"),
                })
            }
            LinearQkv | LinearZ | LinearB | LinearA | LinearOut => {
                let Qwen35Attn::Linear(lin) = &layer.attn else {
                    return Err(anyhow!(
                        "layer {layer_idx} {} requires a linear-attention layer",
                        projection.label()
                    ));
                };
                Ok(match projection {
                    LinearQkv | LinearZ => &lin.in_proj_qkvz,
                    LinearB | LinearA => &lin.in_proj_ba,
                    LinearOut => &lin.out_proj,
                    _ => unreachable!("linear projection arm checked above"),
                })
            }
            MlpGate | MlpUp | MlpDown => {
                let dense = layer.mlp.as_ref().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a dense MLP layer; MoE student LoRA sync is not supported",
                        projection.label()
                    )
                })?;
                Ok(match projection {
                    // Gate/up live in the row-fused `gate_up_proj`; callers
                    // address their half via `lora_row_window`.
                    MlpGate | MlpUp => &dense.gate_up_proj,
                    MlpDown => &dense.down_proj,
                    _ => unreachable!("mlp projection arm checked above"),
                })
            }
            MoeRouter | MoeSharedGate | MoeSharedUp | MoeSharedDown | MoeSharedExpertGate => {
                let moe = layer.moe.as_ref().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a Qwen3.6 MoE layer",
                        projection.label()
                    )
                })?;
                Ok(match projection {
                    MoeRouter => &moe.router_gate,
                    MoeSharedGate => &moe.shared_gate,
                    MoeSharedUp => &moe.shared_up,
                    MoeSharedDown => &moe.shared_down,
                    MoeSharedExpertGate => &moe.shared_gate_router,
                    _ => unreachable!("shared MoE projection arm checked above"),
                })
            }
            MoeExpertGate { expert_idx }
            | MoeExpertUp { expert_idx }
            | MoeExpertDown { expert_idx } => {
                let local_idx = self.local_expert_idx(expert_idx)?;
                let moe = layer.moe.as_ref().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a Qwen3.6 MoE layer",
                        projection.label()
                    )
                })?;
                let experts = match projection {
                    MoeExpertGate { .. } => &moe.gate,
                    MoeExpertUp { .. } => &moe.up,
                    MoeExpertDown { .. } => &moe.down,
                    _ => unreachable!("expert MoE projection arm checked above"),
                };
                experts.get(local_idx).ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} expert matrix is not resident as a per-expert \
                         BF16 DeviceMatrix; grouped/FP8 MoE LoRA sync is not supported by this \
                         re-merge path",
                        projection.label()
                    )
                })
            }
        }
    }

    fn lora_matrix_mut(
        &mut self,
        layer_idx: usize,
        projection: StudentLoraProjection,
    ) -> Result<&mut DeviceMatrix> {
        let layer = &mut self.layers[layer_idx];
        match projection {
            FullQ | FullK | FullV | FullO => {
                let Qwen35Attn::Full(full) = &mut layer.attn else {
                    return Err(anyhow!(
                        "layer {layer_idx} {} requires a full-attention layer",
                        projection.label()
                    ));
                };
                Ok(match projection {
                    FullQ | FullK | FullV => &mut full.qkv_proj,
                    FullO => &mut full.o_proj,
                    _ => unreachable!("full projection arm checked above"),
                })
            }
            LinearQkv | LinearZ | LinearB | LinearA | LinearOut => {
                let Qwen35Attn::Linear(lin) = &mut layer.attn else {
                    return Err(anyhow!(
                        "layer {layer_idx} {} requires a linear-attention layer",
                        projection.label()
                    ));
                };
                Ok(match projection {
                    LinearQkv | LinearZ => &mut lin.in_proj_qkvz,
                    LinearB | LinearA => &mut lin.in_proj_ba,
                    LinearOut => &mut lin.out_proj,
                    _ => unreachable!("linear projection arm checked above"),
                })
            }
            MlpGate | MlpUp | MlpDown => {
                let dense = layer.mlp.as_mut().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a dense MLP layer; MoE student LoRA sync is not supported",
                        projection.label()
                    )
                })?;
                Ok(match projection {
                    MlpGate | MlpUp => &mut dense.gate_up_proj,
                    MlpDown => &mut dense.down_proj,
                    _ => unreachable!("mlp projection arm checked above"),
                })
            }
            MoeRouter | MoeSharedGate | MoeSharedUp | MoeSharedDown | MoeSharedExpertGate => {
                let moe = layer.moe.as_mut().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a Qwen3.6 MoE layer",
                        projection.label()
                    )
                })?;
                Ok(match projection {
                    MoeRouter => &mut moe.router_gate,
                    MoeSharedGate => &mut moe.shared_gate,
                    MoeSharedUp => &mut moe.shared_up,
                    MoeSharedDown => &mut moe.shared_down,
                    MoeSharedExpertGate => &mut moe.shared_gate_router,
                    _ => unreachable!("shared MoE projection arm checked above"),
                })
            }
            MoeExpertGate { expert_idx }
            | MoeExpertUp { expert_idx }
            | MoeExpertDown { expert_idx } => {
                let local_start = self.expert_split.local_expert_start;
                let local_end = self.expert_split.local_expert_end();
                ensure!(
                    (local_start..local_end).contains(&expert_idx),
                    "Qwen3.6 LoRA sync expert {expert_idx} is not local to this rank \
                     (local range {local_start}..{local_end})"
                );
                let local_idx = expert_idx - local_start;
                let moe = layer.moe.as_mut().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a Qwen3.6 MoE layer",
                        projection.label()
                    )
                })?;
                let experts = match projection {
                    MoeExpertGate { .. } => &mut moe.gate,
                    MoeExpertUp { .. } => &mut moe.up,
                    MoeExpertDown { .. } => &mut moe.down,
                    _ => unreachable!("expert MoE projection arm checked above"),
                };
                experts.get_mut(local_idx).ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} expert matrix is not resident as a per-expert \
                         BF16 DeviceMatrix; grouped/FP8 MoE LoRA sync is not supported by this \
                         re-merge path",
                        projection.label()
                    )
                })
            }
        }
    }
}
