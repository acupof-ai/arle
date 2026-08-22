use super::*;

impl Dsv4CudaExecutor {
    pub(crate) fn from_dsv4_fp8_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        max_seq_len: usize,
        mtp_draft_tokens: Option<usize>,
        mtp_draft_topk: Option<usize>,
        dspark_draft_model: Option<&Path>,
        dspark_sps_bias_ms: f32,
        dspark_sps_row_ms: f32,
    ) -> Result<Self> {
        ensure!(num_slots > 0, "Dsv4CudaExecutor requires at least one slot");
        ensure!(max_seq_len > 0, "Dsv4CudaExecutor requires max_seq_len > 0");
        let mtp_draft_tokens_for_load = mtp_draft_tokens
            .or_else(|| mtp_draft_topk.map(|_| crate::dsv4::DEFAULT_SPEC_DRAFT_DEPTH));
        // The builder reads the draft config to derive `spec_decode_on`, which
        // allocates the per-slot spec-ring snapshots MTP and DSpark share.
        let mut model = crate::dsv4::Dsv4Model::from_dsv4_fp8_safetensors(
            model_path.as_ref(),
            mtp_draft_tokens_for_load,
            dspark_draft_model,
        )?;
        let mem_dbg = |tag: &str| -> Option<usize> {
            match cudarc::driver::result::mem_get_info() {
                Ok((free, total)) => {
                    let used = total - free;
                    log::info!(
                        "[vram-probe] {tag}: used {}MB free {}MB",
                        used >> 20,
                        free >> 20
                    );
                    Some(used)
                }
                Err(_) => None,
            }
        };
        let base_weights_used = mem_dbg("after base model load (weights+experts)");
        let dspark_draft = dspark_draft_model
            .map(|draft_dir| -> Result<_> {
                let loader = crate::loader::SafetensorLoader::new(draft_dir)?;
                loader.prefetch_shards_rank0(&model.ctx, &model.tp)?;
                let draft = crate::dsv4::load_dspark_draft(
                    &loader,
                    &model.ctx,
                    &model.config,
                    &model.split,
                    model.tp.config(),
                )?;
                model.ctx.sync()?;
                let after = mem_dbg("after DSpark draft weights load");
                if let (Some(before), Some(after)) = (base_weights_used, after) {
                    log::info!(
                        "[vram-ledger] DSpark weights: +{}MB ({}MB -> {}MB)",
                        (after as i64 - before as i64) >> 20,
                        before >> 20,
                        after >> 20,
                    );
                }
                Ok(draft)
            })
            .transpose()?;
        // Reclaim the cuMemAllocAsync pool BEFORE measuring free VRAM: it retains
        // freed weight-load scratch, which the budget would count as USED and
        // starve the KV slot count.
        if let Err(e) = model.ctx.trim_memory_pool() {
            log::warn!("pre-KV-budget trim_memory_pool failed (non-fatal): {e}");
        }
        // Graph capture requires the pool to not query events (illegal during
        // stream capture). MAX release threshold caches blocks without queries.
        if let Err(e) = model.ctx.set_pool_retain(true) {
            log::warn!("set_pool_retain(true) failed (non-fatal): {e}");
        }
        let weights_used_at_model_load = mem_dbg("after weight-load pool trim");
        // DSpark per-slot runtime is allocated after kv_budget_plan; count it
        // here so num_slots doesn't over-commit.
        let dspark_per_slot_bytes = if dspark_draft.is_some() {
            let num_stages = model.config.dspark_num_stages();
            let block_size = model.config.dspark_block_size;
            let head_dim = model.config.head_dim;
            // Sliding-window draft latent: fixed `window + block`, no prompt growth.
            let draft_span = model.config.sliding_window + block_size;
            let latent_kv = num_stages
                .saturating_mul(draft_span)
                .saturating_mul(head_dim)
                .saturating_mul(std::mem::size_of::<half::bf16>());
            // Rough estimate; the adapter's per_slot already covers the trunk layers.
            let attn = num_stages
                .saturating_mul(draft_span)
                .saturating_mul(head_dim)
                .saturating_mul(2);
            latent_kv.saturating_add(attn)
        } else {
            0
        };
        let budget = model.kv_budget_plan(num_slots, max_seq_len, dspark_per_slot_bytes)?;
        let num_slots = budget.num_slots;
        let kv_adapter = model.new_kv_adapter(max_seq_len, budget)?;
        mem_dbg("after new_kv_adapter (KV pools)");
        log::info!(
            "[vram-ledger] adapter predicted {}MB; breakdown {:?}",
            kv_adapter.device_bytes() >> 20,
            kv_adapter
                .device_bytes_breakdown()
                .iter()
                .map(|(name, bytes)| (*name, bytes >> 20))
                .collect::<Vec<_>>()
        );
        let mut slots = Vec::with_capacity(num_slots);
        for slot_idx in 0..num_slots {
            slots.push(model.new_slot_state(max_seq_len, slot_idx, &kv_adapter)?);
            if slot_idx == 0 {
                mem_dbg("after slot 0 (per-slot state)");
                log::info!(
                    "[vram-ledger] slot0 predicted {}MB; breakdown {:?}",
                    slots[0].device_bytes() >> 20,
                    slots[0]
                        .device_bytes_breakdown()
                        .iter()
                        .map(|(name, bytes)| (*name, bytes >> 20))
                        .collect::<Vec<_>>()
                );
                log::info!(
                    "[vram-ledger] slot0 attention sub-totals (Σ layers) {:?}",
                    slots[0]
                        .attention_breakdown_total()
                        .iter()
                        .map(|(name, bytes)| (*name, bytes >> 20))
                        .collect::<Vec<_>>()
                );
                // The KV budget divides free VRAM by the STATIC
                // `per_slot_device_bytes`; drift from the real slot alloc
                // mis-clamps num_slots and engine build OOMs. Warn above 5%.
                let predicted = model.per_slot_device_bytes(max_seq_len)?;
                let actual = slots[0].device_bytes();
                let drift = (predicted as i64 - actual as i64).unsigned_abs() as usize;
                if drift.saturating_mul(20) > actual {
                    log::warn!(
                        "[vram-ledger] DSv4 per-slot budget drift {}%: static per_slot_device_bytes {}MB vs \
                         slot0 device_bytes {}MB — reconcile per_slot_device_bytes with Dsv4SlotState::new",
                        drift.saturating_mul(100) / actual.max(1),
                        predicted >> 20,
                        actual >> 20,
                    );
                }
            }
        }
        let measured_used_after_all = mem_dbg("after all slots (build complete)");
        // Residual = measured used - (weights + adapter + Σ slots): everything not
        // in the named-buffer ledger — CUDA context, library reservations, and
        // per-cudaMalloc rounding across the ~258 tiny per-layer allocs/slot.
        let adapter_bytes = kv_adapter.device_bytes();
        let slots_bytes: usize = slots.iter().map(|s| s.device_bytes()).sum();
        log::info!(
            "[vram-ledger] cumulative predicted: weights {}MB + adapter {}MB + Σ {} slots {}MB = {}MB",
            weights_used_at_model_load.map_or(0, |b| b >> 20),
            adapter_bytes >> 20,
            num_slots,
            slots_bytes >> 20,
            (weights_used_at_model_load.unwrap_or(0) + adapter_bytes + slots_bytes) >> 20
        );
        if let (Some(measured), Some(weights)) =
            (measured_used_after_all, weights_used_at_model_load)
        {
            let predicted_total = weights + adapter_bytes + slots_bytes;
            // Signed: measured can dip below predicted on measurement skew.
            let residual_mb = (measured as i64 - predicted_total as i64) >> 20;
            log::info!(
                "[vram-ledger] residual (ctx+libs+cudaMalloc rounding) = {residual_mb}MB \
                 (measured used {}MB - predicted {}MB)",
                measured >> 20,
                predicted_total >> 20
            );
        }
        let spec_slots = (0..num_slots)
            .map(|_| Dsv4SpecSlotState::default())
            .collect();
        let layer_specs: Vec<_> = model
            .layers
            .iter()
            .map(|layer| (layer.mode, layer.compress_ratio))
            .collect();
        let prefix_entry_bytes = crate::attention::dsv4_prefix_entry_max_bytes(
            &model.config,
            &layer_specs,
            model.kv_arena.page_block_size,
        );
        let dspark = dspark_draft
            .map(|draft| {
                Self::load_dspark_exec(
                    &model,
                    &kv_adapter,
                    draft,
                    dspark_sps_bias_ms,
                    dspark_sps_row_ms,
                    max_seq_len,
                    num_slots,
                )
            })
            .transpose()?;
        model.boot_mega_moe(num_slots.max(256))?;
        // Chunked blobs: a DSv4 slot image is far larger than one fixed page.
        let slot_tier = KvTierStore::with_budget(default_t1_budget_per_rank(), BLOB_CHUNK_BYTES);
        let exec = Self {
            model,
            slots,
            kv_adapter,
            spec_slots,
            spec_draft_tokens: mtp_draft_tokens_for_load,
            spec_draft_topk: mtp_draft_topk,
            num_slots,
            mtp_accepts: 0,
            mtp_rejects: 0,
            mtp_chains: 0,
            prefix_state: crate::attention::Dsv4PrefixStatePool::new(
                default_t1_budget_per_rank(),
                prefix_entry_bytes,
            ),
            pending_prefix_captures: VecDeque::new(),
            dspark,
            slot_tier,
            decode_graph: None,
        };
        log::info!(
            "DSv4 prefill chunk capability: {} tokens (deepep per-forward cap {:?})",
            exec.max_prefill_chunk(),
            exec.model.max_tokens_per_step(),
        );
        Ok(exec)
    }

    fn load_dspark_exec(
        model: &crate::dsv4::Dsv4Model,
        kv_adapter: &crate::attention::Dsv4KvAdapter,
        draft: crate::dsv4::Dsv4DsparkDraft,
        sps_bias_ms: f32,
        sps_row_ms: f32,
        max_seq_len: usize,
        num_slots: usize,
    ) -> Result<Dsv4DsparkExec> {
        ensure!(
            model.config.is_dspark(),
            "--spec-type dspark needs a DSpark-capable DSv4 checkpoint (config carries \
             dspark_block_size + dspark_target_layer_ids); this checkpoint is not one"
        );
        let num_stages = model.config.dspark_num_stages();
        let block_size = model.config.dspark_block_size;
        // A draft stage attends its OWN `latent_kv`, not the trunk KV pool, so
        // the pool arg is a bookkeeping handle only (layer 0's).
        let stage_mode = model.config.attention_mode_for_compress_ratio(0);
        let stage_pool = kv_adapter.layer(0)?;
        // Sliding-window draft latent, independent of prompt length: a
        // full-context latent grows linearly and OOMs on long prompts.
        let draft_span = model.config.sliding_window + block_size;
        let slots = (0..num_slots)
            .map(|slot_idx| -> Result<_> {
                let context_capacity = model.config.sliding_window;
                let df = crate::dsv4::dspark::Dsv4DsparkSlotState::new(
                    &model.ctx,
                    &model.config,
                    num_stages,
                    context_capacity,
                    block_size,
                )?;
                let attn_states = draft
                    .stages
                    .iter()
                    .map(|stage| {
                        let local_width = stage.layer.attention.wq_b.rows;
                        crate::attention::Dsv4LayerAttentionState::new(
                            &model.ctx,
                            &model.config,
                            stage_mode,
                            0,
                            draft_span,
                            &model.kv_arena,
                            local_width / model.config.head_dim,
                            model.tp.config().world_size,
                            slot_idx,
                            stage_pool,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Dsv4DsparkRuntime {
                    df,
                    scratch: crate::dsv4::dspark::Dsv4DsparkScratch::default(),
                    attn_states,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let sps = qwen35_spec::DsparkSps {
            bias_ms: sps_bias_ms,
            row_ms: sps_row_ms,
        };
        log::info!(
            "CUDA DSv4 DSpark runtime initialized: stages={num_stages} block={block_size} \
             sps={sps:?} target_layers={:?}",
            model.config.dspark_target_layer_ids,
        );
        Ok(Dsv4DsparkExec {
            draft,
            sps,
            max_seq_len,
            slots,
        })
    }
}
