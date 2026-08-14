//! LoRA adapter types and the on-device merge/restore path for Qwen3.5/3.6.
//!
//! The merge path supports two base-weight formats:
//! - **FP8 block-scaled**: the FP8 qweight/scales are kept alive in the matrix
//!   after BF16 promotion (for the `--share-frozen-base` student alias), and
//!   the pristine base is recovered by dequantizing on the fly — no separate
//!   BF16 base cache.
//! - **Native BF16** (e.g. linear-attn `in_proj_ba`): the base row window is
//!   cached once before the first merge so the all-zero-adapter restore path
//!   can put it back.

use super::*;

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
        // Plain helper (not a closure) so the grouped-buffer `out.push` calls
        // below don't conflict with a captured `&mut out` borrow.
        fn push(
            out: &mut Vec<SharedFp8BaseProjection>,
            ctx: &DeviceContext,
            layer_idx: usize,
            suffix: String,
            m: &DeviceMatrix,
        ) {
            if let Some((weight_ptr, scale_ptr, rows, cols, block_m, block_k)) =
                m.fp8_block_scaled_ptrs(ctx)
            {
                out.push(SharedFp8BaseProjection {
                    layer_idx,
                    proj_suffix: suffix,
                    weight_ptr,
                    scale_ptr,
                    rows,
                    cols,
                    block_m,
                    block_k,
                });
            }
        }
        let mut out = Vec::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if let Qwen35Attn::Full(full) = &layer.attn {
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "self_attn.qkv_proj".to_string(),
                    &full.qkv_proj,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "self_attn.o_proj".to_string(),
                    &full.o_proj,
                );
            }
            if let Qwen35Attn::Linear(lin) = &layer.attn {
                // in_proj_qkv/z + out_proj ship as resident FP8 in the
                // Qwen3.5/3.6 hybrid checkpoint; in_proj_a/b are tiny per-head
                // BF16 (skipped by the format filter). The train student names
                // these `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`.
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "linear_attn.in_proj_qkvz".to_string(),
                    &lin.in_proj_qkvz,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "linear_attn.in_proj_ba".to_string(),
                    &lin.in_proj_ba,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "linear_attn.out_proj".to_string(),
                    &lin.out_proj,
                );
            }
            if let Some(mlp) = &layer.mlp {
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.gate_up_proj".to_string(),
                    &mlp.gate_up_proj,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.down_proj".to_string(),
                    &mlp.down_proj,
                );
            }
            if let Some(moe) = &layer.moe {
                // Shared expert: always individual FP8 DeviceMatrix (mirror dense).
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.shared_expert.gate_proj".to_string(),
                    &moe.shared_gate,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.shared_expert.up_proj".to_string(),
                    &moe.shared_up,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.shared_expert.down_proj".to_string(),
                    &moe.shared_down,
                );

                // Two storage layouts:
                //  (A) per-expert Vec<DeviceMatrix> (DeepGEMM disabled) -> borrow each directly.
                //  (B) fused FP8 grouped buffers (default) -> slice per-expert ptrs into the group.
                if !moe.gate.is_empty() {
                    for (e, m) in moe.gate.iter().enumerate() {
                        push(
                            &mut out,
                            ctx,
                            layer_idx,
                            format!("mlp.experts.{e}.gate_proj"),
                            m,
                        );
                    }
                    for (e, m) in moe.up.iter().enumerate() {
                        push(
                            &mut out,
                            ctx,
                            layer_idx,
                            format!("mlp.experts.{e}.up_proj"),
                            m,
                        );
                    }
                    for (e, m) in moe.down.iter().enumerate() {
                        push(
                            &mut out,
                            ctx,
                            layer_idx,
                            format!("mlp.experts.{e}.down_proj"),
                            m,
                        );
                    }
                } else {
                    // Single-GPU: group index == global expert index.
                    if let Some(w13) = &moe.w13_fp8_grouped {
                        // rows = 2*moe_intermediate; gate = rows[0..mi], up = rows[mi..2*mi].
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
        fn push(
            out: &mut Vec<SharedBf16BaseProjection>,
            ctx: &DeviceContext,
            layer_idx: usize,
            suffix: String,
            m: &DeviceMatrix,
        ) {
            if m.weight_format() != WeightFormat::DenseBf16 {
                return;
            }
            let (ptr, _g) = m.data.device_ptr(ctx.stream.as_ref());
            out.push(SharedBf16BaseProjection {
                layer_idx,
                proj_suffix: suffix,
                data_ptr: ptr,
                rows: m.rows,
                cols: m.cols,
            });
        }
        let mut out = Vec::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if let Qwen35Attn::Full(full) = &layer.attn {
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "self_attn.qkv_proj".to_string(),
                    &full.qkv_proj,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "self_attn.o_proj".to_string(),
                    &full.o_proj,
                );
            }
            if let Qwen35Attn::Linear(lin) = &layer.attn {
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "linear_attn.in_proj_qkvz".to_string(),
                    &lin.in_proj_qkvz,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "linear_attn.in_proj_ba".to_string(),
                    &lin.in_proj_ba,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "linear_attn.out_proj".to_string(),
                    &lin.out_proj,
                );
            }
            if let Some(mlp) = &layer.mlp {
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.gate_up_proj".to_string(),
                    &mlp.gate_up_proj,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.down_proj".to_string(),
                    &mlp.down_proj,
                );
            }
            if let Some(moe) = &layer.moe {
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.shared_expert.gate_proj".to_string(),
                    &moe.shared_gate,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.shared_expert.up_proj".to_string(),
                    &moe.shared_up,
                );
                push(
                    &mut out,
                    ctx,
                    layer_idx,
                    "mlp.shared_expert.down_proj".to_string(),
                    &moe.shared_down,
                );
                for (e, m) in moe.gate.iter().enumerate() {
                    push(
                        &mut out,
                        ctx,
                        layer_idx,
                        format!("mlp.experts.{e}.gate_proj"),
                        m,
                    );
                }
                for (e, m) in moe.up.iter().enumerate() {
                    push(
                        &mut out,
                        ctx,
                        layer_idx,
                        format!("mlp.experts.{e}.up_proj"),
                        m,
                    );
                }
                for (e, m) in moe.down.iter().enumerate() {
                    push(
                        &mut out,
                        ctx,
                        layer_idx,
                        format!("mlp.experts.{e}.down_proj"),
                        m,
                    );
                }
            }
        }
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

            for projection in &layer.projections {
                self.merge_lora_proj(
                    layer_idx,
                    projection.projection,
                    &projection.matrices,
                    scale,
                )?;
            }
        }
        self.ctx.sync()?;
        Ok(())
    }

    /// Free the retired FP8 qweight/scales buffers for every projection that has
    /// been promoted to dense BF16. Call ONLY after the train student has
    /// re-aliased its frozen base to the BF16 `data` pointer (see
    /// [`Qwen35Model::frozen_base_bf16_pointers`]) — otherwise the student's
    /// FP8 alias dangles. Frees ~27 GB on the 27B model.
    pub(crate) fn free_retired_fp8_buffers(&mut self) {
        fn clear(m: &mut DeviceMatrix) {
            if m.weight_format() == WeightFormat::DenseBf16 {
                m.qweight_u8 = None;
                m.scale_f32 = None;
                m.quant_block_m = 0;
                m.quant_block_k = 0;
                m.quant_scale_rows = 0;
                m.quant_scale_cols = 0;
            }
        }
        for layer in self.layers.iter_mut() {
            if let Qwen35Attn::Full(full) = &mut layer.attn {
                clear(&mut full.qkv_proj);
                clear(&mut full.o_proj);
            }
            if let Qwen35Attn::Linear(lin) = &mut layer.attn {
                clear(&mut lin.in_proj_qkvz);
                clear(&mut lin.in_proj_ba);
                clear(&mut lin.out_proj);
            }
            if let Some(mlp) = &mut layer.mlp {
                clear(&mut mlp.gate_up_proj);
                clear(&mut mlp.down_proj);
            }
            if let Some(moe) = &mut layer.moe {
                clear(&mut moe.shared_gate);
                clear(&mut moe.shared_up);
                clear(&mut moe.shared_down);
                for m in moe.gate.iter_mut() {
                    clear(m);
                }
                for m in moe.up.iter_mut() {
                    clear(m);
                }
                for m in moe.down.iter_mut() {
                    clear(m);
                }
            }
        }
        // The frozen base now IS the merged BF16 `data` (the student re-aliases
        // it). Reset the per-merge dirty set and the one-shot BF16 base cache:
        // the next round's first merge treats the current `data` as the pristine
        // base, and a zero-adapter round is a no-op rather than a
        // restore-from-FP8.
        self.lora_dirty.clear();
        self.lora_base_dev.clear();
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
        ensure!(
            matrix.weight_format() == WeightFormat::Fp8BlockScaled,
            "layer {layer_idx} {label}: LoRA merge supports dense BF16 or FP8 block-scaled \
             weights; got {:?}",
            matrix.weight_format()
        );
        ensure!(
            matrix.quant_block_m > 0
                && matrix.quant_block_k > 0
                && matrix.quant_scale_rows > 0
                && matrix.quant_scale_cols > 0,
            "layer {layer_idx} {label}: FP8 LoRA target missing block-scale metadata"
        );
        let mut dense = ctx
            .stream
            .alloc_zeros::<bf16>(matrix.rows * matrix.cols)
            .map_err(|e| anyhow!("layer {layer_idx} {label}: BF16 promotion alloc failed: {e}"))?;
        {
            let qweight = matrix.qweight_u8.as_ref().ok_or_else(|| {
                anyhow!("layer {layer_idx} {label}: FP8 LoRA target missing qweight")
            })?;
            let scales = matrix.scale_f32.as_ref().ok_or_else(|| {
                anyhow!("layer {layer_idx} {label}: FP8 LoRA target missing f32 scales")
            })?;
            ensure!(
                qweight.len() == matrix.rows * matrix.cols,
                "layer {layer_idx} {label}: FP8 qweight len {} != rows*cols {}",
                qweight.len(),
                matrix.rows * matrix.cols
            );
            let (qw_ptr, _gq) = qweight.device_ptr(&ctx.stream);
            let (scale_ptr, _gs) = scales.device_ptr(&ctx.stream);
            let (dense_ptr, _gd) = dense.device_ptr_mut(&ctx.stream);
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                ffi::dequantize_fp8_block_scaled_to_bf16_cuda(
                    qw_ptr as *const u8,
                    scale_ptr as *const f32,
                    dense_ptr as *mut ffi::Half,
                    matrix.rows as i32,
                    matrix.cols as i32,
                    matrix.quant_scale_rows as i32,
                    matrix.quant_scale_cols as i32,
                    matrix.quant_block_m as i32,
                    matrix.quant_block_k as i32,
                    ctx.stream.cu_stream(),
                )
            }
            .result()
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
    fn lora_base_cache_key(layer_idx: usize, projection: StudentLoraProjection) -> LoraBaseKey {
        let projection = match projection {
            StudentLoraProjection::MlpUp => StudentLoraProjection::MlpGate,
            StudentLoraProjection::LinearA => StudentLoraProjection::LinearB,
            StudentLoraProjection::FullK | StudentLoraProjection::FullV => {
                StudentLoraProjection::FullQ
            }
            StudentLoraProjection::LinearZ => StudentLoraProjection::LinearQkv,
            other => other,
        };
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
            StudentLoraProjection::MlpGate => (0, inter()),
            StudentLoraProjection::MlpUp => (inter(), inter()),
            StudentLoraProjection::LinearB => (0, vh()),
            StudentLoraProjection::LinearA => (vh(), vh()),
            StudentLoraProjection::FullQ => (0, q_gated),
            StudentLoraProjection::FullK => (q_gated, kv),
            StudentLoraProjection::FullV => (q_gated + kv, kv),
            StudentLoraProjection::LinearQkv => (0, qkv),
            StudentLoraProjection::LinearZ => (qkv, self.local_linear_z_dim()),
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
            if matrix.qweight_u8.is_some() {
                let qweight = matrix.qweight_u8.as_ref().ok_or_else(|| {
                    anyhow!("layer {layer_idx} {label}: FP8 base missing qweight for merge")
                })?;
                let scales = matrix.scale_f32.as_ref().ok_or_else(|| {
                    anyhow!("layer {layer_idx} {label}: FP8 base missing f32 scales for merge")
                })?;
                let mut base_scratch = DeviceVec::zeros(&ctx, matrix.rows * matrix.cols)?;
                {
                    let (qw_ptr, _gq) = qweight.device_ptr(&ctx.stream);
                    let (scale_ptr, _gs) = scales.device_ptr(&ctx.stream);
                    let (dst_ptr, _gd) = base_scratch.data.device_ptr_mut(&ctx.stream);
                    // SAFETY: ptrs from live device allocations sized to the dims passed.
                    unsafe {
                        ffi::dequantize_fp8_block_scaled_to_bf16_cuda(
                            qw_ptr as *const u8,
                            scale_ptr as *const f32,
                            dst_ptr as *mut ffi::Half,
                            matrix.rows as i32,
                            matrix.cols as i32,
                            matrix.quant_scale_rows as i32,
                            matrix.quant_scale_cols as i32,
                            matrix.quant_block_m as i32,
                            matrix.quant_block_k as i32,
                            ctx.stream.cu_stream(),
                        )
                    }
                    .result()
                    .map_err(|e| {
                        anyhow!(
                            "layer {layer_idx} {label}: FP8→BF16 base dequant for merge failed: {e}"
                        )
                    })?;
                }
                let src = base_scratch.data.slice(window.clone());
                let matrix = self.lora_matrix_mut(layer_idx, projection)?;
                let mut dst = matrix.data.slice_mut(window.clone());
                ctx.stream.memcpy_dtod(&src, &mut dst).map_err(|e| {
                    anyhow!("layer {layer_idx} {label}: base window D2D copy failed: {e}")
                })?;
            } else if !self.lora_base_dev.contains_key(&cache_key) {
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
            let (delta_ptr, _gd) = delta_view.device_ptr(&ctx.stream);
            let (out_ptr, _go) = out_view.device_ptr_mut(&ctx.stream);
            // SAFETY: ptrs from live device allocations sized to `needed`.
            unsafe {
                ffi::add_scaled_row_cuda(
                    delta_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    needed as i32,
                    0,
                    scale,
                    ctx.stream.cu_stream(),
                )
            }
            .result()
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
        if matrix.qweight_u8.is_some() {
            let qweight = matrix.qweight_u8.as_ref().ok_or_else(|| {
                anyhow!("layer {layer_idx} {label}: FP8 base missing qweight for restore")
            })?;
            let scales = matrix.scale_f32.as_ref().ok_or_else(|| {
                anyhow!("layer {layer_idx} {label}: FP8 base missing f32 scales for restore")
            })?;
            let mut scratch = DeviceVec::zeros(&ctx, matrix.rows * matrix.cols)?;
            {
                let (qw_ptr, _gq) = qweight.device_ptr(&ctx.stream);
                let (scale_ptr, _gs) = scales.device_ptr(&ctx.stream);
                let (dst_ptr, _gd) = scratch.data.device_ptr_mut(&ctx.stream);
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    ffi::dequantize_fp8_block_scaled_to_bf16_cuda(
                        qw_ptr as *const u8,
                        scale_ptr as *const f32,
                        dst_ptr as *mut ffi::Half,
                        matrix.rows as i32,
                        matrix.cols as i32,
                        matrix.quant_scale_rows as i32,
                        matrix.quant_scale_cols as i32,
                        matrix.quant_block_m as i32,
                        matrix.quant_block_k as i32,
                        ctx.stream.cu_stream(),
                    )
                }
                .result()
                .map_err(|e| {
                    anyhow!(
                        "layer {layer_idx} {label}: FP8→BF16 base dequant for restore failed: {e}"
                    )
                })?;
            }
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
            StudentLoraProjection::FullQ
            | StudentLoraProjection::FullK
            | StudentLoraProjection::FullV
            | StudentLoraProjection::FullO => {
                let Qwen35Attn::Full(full) = &layer.attn else {
                    return Err(anyhow!(
                        "layer {layer_idx} {} requires a full-attention layer",
                        projection.label()
                    ));
                };
                Ok(match projection {
                    // q/k/v live in the row-fused `qkv_proj`; callers address
                    // their window via `lora_row_window`.
                    StudentLoraProjection::FullQ
                    | StudentLoraProjection::FullK
                    | StudentLoraProjection::FullV => &full.qkv_proj,
                    StudentLoraProjection::FullO => &full.o_proj,
                    _ => unreachable!("full projection arm checked above"),
                })
            }
            StudentLoraProjection::LinearQkv
            | StudentLoraProjection::LinearZ
            | StudentLoraProjection::LinearB
            | StudentLoraProjection::LinearA
            | StudentLoraProjection::LinearOut => {
                let Qwen35Attn::Linear(lin) = &layer.attn else {
                    return Err(anyhow!(
                        "layer {layer_idx} {} requires a linear-attention layer",
                        projection.label()
                    ));
                };
                Ok(match projection {
                    StudentLoraProjection::LinearQkv | StudentLoraProjection::LinearZ => {
                        &lin.in_proj_qkvz
                    }
                    StudentLoraProjection::LinearB | StudentLoraProjection::LinearA => {
                        &lin.in_proj_ba
                    }
                    StudentLoraProjection::LinearOut => &lin.out_proj,
                    _ => unreachable!("linear projection arm checked above"),
                })
            }
            StudentLoraProjection::MlpGate
            | StudentLoraProjection::MlpUp
            | StudentLoraProjection::MlpDown => {
                let dense = layer.mlp.as_ref().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a dense MLP layer; MoE student LoRA sync is not supported",
                        projection.label()
                    )
                })?;
                Ok(match projection {
                    // Gate/up live in the row-fused `gate_up_proj`; callers
                    // address their half via `lora_row_window`.
                    StudentLoraProjection::MlpGate | StudentLoraProjection::MlpUp => {
                        &dense.gate_up_proj
                    }
                    StudentLoraProjection::MlpDown => &dense.down_proj,
                    _ => unreachable!("mlp projection arm checked above"),
                })
            }
            StudentLoraProjection::MoeRouter
            | StudentLoraProjection::MoeSharedGate
            | StudentLoraProjection::MoeSharedUp
            | StudentLoraProjection::MoeSharedDown
            | StudentLoraProjection::MoeSharedExpertGate => {
                let moe = layer.moe.as_ref().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a Qwen3.6 MoE layer",
                        projection.label()
                    )
                })?;
                Ok(match projection {
                    StudentLoraProjection::MoeRouter => &moe.router_gate,
                    StudentLoraProjection::MoeSharedGate => &moe.shared_gate,
                    StudentLoraProjection::MoeSharedUp => &moe.shared_up,
                    StudentLoraProjection::MoeSharedDown => &moe.shared_down,
                    StudentLoraProjection::MoeSharedExpertGate => &moe.shared_gate_router,
                    _ => unreachable!("shared MoE projection arm checked above"),
                })
            }
            StudentLoraProjection::MoeExpertGate { expert_idx }
            | StudentLoraProjection::MoeExpertUp { expert_idx }
            | StudentLoraProjection::MoeExpertDown { expert_idx } => {
                let local_idx = self.local_expert_idx(expert_idx)?;
                let moe = layer.moe.as_ref().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a Qwen3.6 MoE layer",
                        projection.label()
                    )
                })?;
                let experts = match projection {
                    StudentLoraProjection::MoeExpertGate { .. } => &moe.gate,
                    StudentLoraProjection::MoeExpertUp { .. } => &moe.up,
                    StudentLoraProjection::MoeExpertDown { .. } => &moe.down,
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
            StudentLoraProjection::FullQ
            | StudentLoraProjection::FullK
            | StudentLoraProjection::FullV
            | StudentLoraProjection::FullO => {
                let Qwen35Attn::Full(full) = &mut layer.attn else {
                    return Err(anyhow!(
                        "layer {layer_idx} {} requires a full-attention layer",
                        projection.label()
                    ));
                };
                Ok(match projection {
                    StudentLoraProjection::FullQ
                    | StudentLoraProjection::FullK
                    | StudentLoraProjection::FullV => &mut full.qkv_proj,
                    StudentLoraProjection::FullO => &mut full.o_proj,
                    _ => unreachable!("full projection arm checked above"),
                })
            }
            StudentLoraProjection::LinearQkv
            | StudentLoraProjection::LinearZ
            | StudentLoraProjection::LinearB
            | StudentLoraProjection::LinearA
            | StudentLoraProjection::LinearOut => {
                let Qwen35Attn::Linear(lin) = &mut layer.attn else {
                    return Err(anyhow!(
                        "layer {layer_idx} {} requires a linear-attention layer",
                        projection.label()
                    ));
                };
                Ok(match projection {
                    StudentLoraProjection::LinearQkv | StudentLoraProjection::LinearZ => {
                        &mut lin.in_proj_qkvz
                    }
                    StudentLoraProjection::LinearB | StudentLoraProjection::LinearA => {
                        &mut lin.in_proj_ba
                    }
                    StudentLoraProjection::LinearOut => &mut lin.out_proj,
                    _ => unreachable!("linear projection arm checked above"),
                })
            }
            StudentLoraProjection::MlpGate
            | StudentLoraProjection::MlpUp
            | StudentLoraProjection::MlpDown => {
                let dense = layer.mlp.as_mut().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a dense MLP layer; MoE student LoRA sync is not supported",
                        projection.label()
                    )
                })?;
                Ok(match projection {
                    StudentLoraProjection::MlpGate | StudentLoraProjection::MlpUp => {
                        &mut dense.gate_up_proj
                    }
                    StudentLoraProjection::MlpDown => &mut dense.down_proj,
                    _ => unreachable!("mlp projection arm checked above"),
                })
            }
            StudentLoraProjection::MoeRouter
            | StudentLoraProjection::MoeSharedGate
            | StudentLoraProjection::MoeSharedUp
            | StudentLoraProjection::MoeSharedDown
            | StudentLoraProjection::MoeSharedExpertGate => {
                let moe = layer.moe.as_mut().ok_or_else(|| {
                    anyhow!(
                        "layer {layer_idx} {} requires a Qwen3.6 MoE layer",
                        projection.label()
                    )
                })?;
                Ok(match projection {
                    StudentLoraProjection::MoeRouter => &mut moe.router_gate,
                    StudentLoraProjection::MoeSharedGate => &mut moe.shared_gate,
                    StudentLoraProjection::MoeSharedUp => &mut moe.shared_up,
                    StudentLoraProjection::MoeSharedDown => &mut moe.shared_down,
                    StudentLoraProjection::MoeSharedExpertGate => &mut moe.shared_gate_router,
                    _ => unreachable!("shared MoE projection arm checked above"),
                })
            }
            StudentLoraProjection::MoeExpertGate { expert_idx }
            | StudentLoraProjection::MoeExpertUp { expert_idx }
            | StudentLoraProjection::MoeExpertDown { expert_idx } => {
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
                    StudentLoraProjection::MoeExpertGate { .. } => &mut moe.gate,
                    StudentLoraProjection::MoeExpertUp { .. } => &mut moe.up,
                    StudentLoraProjection::MoeExpertDown { .. } => &mut moe.down,
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
