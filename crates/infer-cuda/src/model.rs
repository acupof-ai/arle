//! Clean dense-BF16 Qwen3 CUDA forward for the R6 seam port.
//!
//! This is deliberately not a relocation of the legacy `infer/src/model/qwen3`
//! tree. It keeps the tested CUDA kernel calls and the transformer dataflow, but
//! deletes quantized variants, GGUF, TP/NCCL, LoRA, CUDA graphs, spec decode,
//! server/scheduler coupling, and multi-shape dispatch.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use cuda_kernels::KVFormat;
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates, PagedKVPool};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use infer_plan::{ForwardPlan, SlotToken, StepOutput};
use infer_seam::KvPool;
use qwen3_spec::Qwen3Config;
use safetensors::{SafeTensors, tensor::Dtype};

const DEFAULT_ROPE_CACHE_LEN: usize = 32_768;
const SUPPORTED_PAGE_SIZE: usize = 16;

pub(crate) struct RealCudaExecutor {
    model: CudaModel,
    kv: PagedKVPool,
    num_slots: usize,
}

impl std::fmt::Debug for RealCudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealCudaExecutor")
            .field("model", &self.model)
            .field("num_slots", &self.num_slots)
            .field("page_size", &self.kv.page_size)
            .field("max_total_pages", &self.kv.max_total_pages)
            .finish()
    }
}

impl RealCudaExecutor {
    pub(crate) fn from_qwen3_bf16_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
    ) -> Result<Self> {
        ensure!(num_slots > 0, "CudaExecutor requires at least one slot");
        ensure!(
            total_pages > 0,
            "CudaExecutor requires at least one KV page"
        );

        let model = CudaModel::from_safetensors(model_path.as_ref())?;
        let token_budget = total_pages * SUPPORTED_PAGE_SIZE;
        let budget_bytes = PagedKVPool::budget_bytes_for_tokens(
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
            model.config.head_dim,
            token_budget,
            KVFormat::BF16,
        );
        let kv = PagedKVPool::with_format(
            &model.ctx,
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
            model.config.head_dim,
            num_slots,
            budget_bytes,
            KVFormat::BF16,
        )?;
        ensure!(
            kv.page_size == SUPPORTED_PAGE_SIZE,
            "R6 BF16 Qwen3 expects cuda-kernels page_size={SUPPORTED_PAGE_SIZE}, got {}",
            kv.page_size
        );

        Ok(Self {
            model,
            kv,
            num_slots,
        })
    }

    pub(crate) fn submit(
        &mut self,
        plan: &ForwardPlan,
        host_kv: &mut dyn KvPool,
    ) -> Result<StepOutput> {
        ensure!(
            host_kv.page_size() == SUPPORTED_PAGE_SIZE,
            "host CudaKvPool page_size={} does not match CUDA BF16 page_size={SUPPORTED_PAGE_SIZE}",
            host_kv.page_size()
        );

        let rows = plan.decode_rows.len() + plan.prefill_rows.len();
        if rows == 0 {
            return Ok(StepOutput { tokens: Vec::new() });
        }
        ensure!(
            rows == 1,
            "R6 clean CUDA forward is single-row only, got {} prefill + {} decode rows",
            plan.prefill_rows.len(),
            plan.decode_rows.len()
        );

        let token = if let Some(row) = plan.prefill_rows.first() {
            ensure!(
                row.slot < self.num_slots,
                "prefill slot {} outside CUDA executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(!row.tokens.is_empty(), "prefill row must carry tokens");
            let expected_len = row.start_pos + row.tokens.len();
            ensure!(
                host_kv.seq_len(row.slot) >= expected_len,
                "host KV length {} is behind prefill materialization end {} for slot {}",
                host_kv.seq_len(row.slot),
                expected_len,
                row.slot
            );
            self.ensure_slot_ready_for_prefill(row.slot, row.start_pos)?;
            self.kv.alloc_tokens(row.slot, row.tokens.len())?;
            self.model
                .forward_tokens(row.slot, &row.tokens, row.start_pos, &mut self.kv)?
        } else {
            let row = &plan.decode_rows[0];
            ensure!(
                row.slot < self.num_slots,
                "decode slot {} outside CUDA executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(
                self.kv.seq_len(row.slot) == row.kv_seq_len,
                "CUDA materialized cache_len {} != DecodeRow.kv_seq_len {} for slot {}",
                self.kv.seq_len(row.slot),
                row.kv_seq_len,
                row.slot
            );
            ensure!(
                host_kv.seq_len(row.slot) >= row.kv_seq_len + 1,
                "host KV length {} is behind decode materialization end {} for slot {}",
                host_kv.seq_len(row.slot),
                row.kv_seq_len + 1,
                row.slot
            );
            self.kv.alloc_tokens(row.slot, 1)?;
            self.model
                .forward_tokens(row.slot, &[row.last_token], row.kv_seq_len, &mut self.kv)?
        };

        Ok(StepOutput {
            tokens: vec![SlotToken {
                slot: plan
                    .prefill_rows
                    .first()
                    .map(|r| r.slot)
                    .unwrap_or_else(|| plan.decode_rows[0].slot),
                token,
                logprob: None,
                finish: None,
            }],
        })
    }

    fn ensure_slot_ready_for_prefill(&mut self, slot: usize, start_pos: usize) -> Result<()> {
        let materialized = self.kv.seq_len(slot);
        if start_pos == 0 {
            if materialized != 0 {
                self.kv.free_slot(slot);
            }
            return Ok(());
        }
        ensure!(
            materialized == start_pos,
            "chunked prefill requires materialized CUDA cache_len == start_pos; got cache_len={materialized}, start_pos={start_pos}"
        );
        Ok(())
    }
}

struct CudaModel {
    ctx: DeviceContext,
    config: Qwen3Config,
    embed_tokens: DeviceMatrix,
    lm_head: Option<DeviceMatrix>,
    layers: Vec<TransformerBlock>,
    norm: DeviceVec,
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
}

impl std::fmt::Debug for CudaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaModel")
            .field("layers", &self.layers.len())
            .field("hidden_size", &self.config.hidden_size)
            .field("heads", &self.config.num_attention_heads)
            .field("kv_heads", &self.config.num_key_value_heads)
            .field("head_dim", &self.config.head_dim)
            .finish()
    }
}

struct TransformerBlock {
    input_layernorm: DeviceVec,
    attention: Attention,
    post_attention_layernorm: DeviceVec,
    mlp: Mlp,
}

struct Attention {
    q_proj: DeviceMatrix,
    k_proj: DeviceMatrix,
    v_proj: DeviceMatrix,
    o_proj: DeviceMatrix,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
}

struct Mlp {
    gate_proj: DeviceMatrix,
    up_proj: DeviceMatrix,
    down_proj: DeviceMatrix,
}

impl CudaModel {
    fn from_safetensors(model_path: &Path) -> Result<Self> {
        let config = Qwen3Config::from_json_file(model_path.join("config.json"))
            .with_context(|| format!("load Qwen3 config from {}", model_path.display()))?;
        validate_clean_bf16_config(&config)?;

        let ctx = DeviceContext::new()?;
        let loader = SafetensorLoader::new(model_path)?;

        let embed_tokens = loader.load_matrix(&ctx, config.embed_tokens_tensor_name())?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(loader.load_matrix(&ctx, config.lm_head_tensor_name())?)
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let names = config.layer_tensor_names(layer_idx);
            layers.push(TransformerBlock {
                input_layernorm: loader.load_vec(&ctx, &names.input_layernorm)?,
                attention: Attention {
                    q_proj: loader.load_matrix(&ctx, &names.q_proj)?,
                    k_proj: loader.load_matrix(&ctx, &names.k_proj)?,
                    v_proj: loader.load_matrix(&ctx, &names.v_proj)?,
                    o_proj: loader.load_matrix(&ctx, &names.o_proj)?,
                    q_norm: loader.load_vec(&ctx, &names.q_norm)?,
                    k_norm: loader.load_vec(&ctx, &names.k_norm)?,
                },
                post_attention_layernorm: loader.load_vec(&ctx, &names.post_attention_layernorm)?,
                mlp: Mlp {
                    gate_proj: loader.load_matrix(&ctx, &names.mlp_gate_proj)?,
                    up_proj: loader.load_matrix(&ctx, &names.mlp_up_proj)?,
                    down_proj: loader.load_matrix(&ctx, &names.mlp_down_proj)?,
                },
            });
        }
        let norm = loader.load_vec(&ctx, config.norm_tensor_name())?;

        let rope_len = config
            .rope_cache_len_hint()
            .unwrap_or(DEFAULT_ROPE_CACHE_LEN)
            .max(DEFAULT_ROPE_CACHE_LEN);
        let (cos_cache, sin_cache) = precompute_rope(
            &ctx,
            config.head_dim,
            rope_len,
            config.rope_theta,
            config.rope_scaling.as_ref(),
        )?;
        ctx.sync()?;

        Ok(Self {
            ctx,
            config,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
        })
    }

    fn output_projection(&self) -> &DeviceMatrix {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }

    fn forward_tokens(
        &self,
        slot: usize,
        tokens: &[u32],
        start_pos: usize,
        pool: &mut PagedKVPool,
    ) -> Result<u32> {
        ensure!(
            !tokens.is_empty(),
            "forward_tokens requires at least one token"
        );
        ensure!(
            self.config.head_dim == 128,
            "R6 clean CUDA path only wires TileLang HD128 kernels, got head_dim={}",
            self.config.head_dim
        );
        ensure!(
            self.config.num_key_value_heads == 8,
            "R6 clean CUDA path only wires kv8 TileLang kernels, got kv_heads={}",
            self.config.num_key_value_heads
        );

        let seq_len = tokens.len();
        let hidden_size = self.config.hidden_size;
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let inter = self.config.intermediate_size;

        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_ids = upload_i32(&self.ctx, &token_ids)?;
        let mut hidden = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        embedding_batch(&self.ctx, &self.embed_tokens, &token_ids, &mut hidden)?;

        let mut normed = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        let mut q_batch = HiddenStates::zeros(&self.ctx, q_dim, seq_len)?;
        let mut k_batch = HiddenStates::zeros(&self.ctx, kv_dim, seq_len)?;
        let mut v_batch = HiddenStates::zeros(&self.ctx, kv_dim, seq_len)?;
        let mut attn_output = HiddenStates::zeros(&self.ctx, q_dim, seq_len)?;
        let mut o_buf = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        let mut hidden_out = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        let mut gate_out = HiddenStates::zeros(&self.ctx, inter, seq_len)?;
        let mut up_out = HiddenStates::zeros(&self.ctx, inter, seq_len)?;
        let mut act_out = HiddenStates::zeros(&self.ctx, inter, seq_len)?;

        let meta = PageMeta::for_slot(&self.ctx, pool, slot, start_pos, seq_len)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            rms_norm_batch(
                &self.ctx,
                &hidden,
                &layer.input_layernorm,
                self.config.rms_norm_eps,
                &mut normed,
            )?;
            gemm_batch(&self.ctx, &layer.attention.q_proj, &normed, &mut q_batch)?;
            gemm_batch(&self.ctx, &layer.attention.k_proj, &normed, &mut k_batch)?;
            gemm_batch(&self.ctx, &layer.attention.v_proj, &normed, &mut v_batch)?;
            paged_attention(
                &self.ctx,
                layer_idx,
                pool,
                &mut q_batch,
                &mut k_batch,
                &v_batch,
                &layer.attention.q_norm,
                &layer.attention.k_norm,
                &self.cos_cache,
                &self.sin_cache,
                self.config.rms_norm_eps,
                &meta,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                &mut attn_output,
            )?;
            gemm_batch(&self.ctx, &layer.attention.o_proj, &attn_output, &mut o_buf)?;
            add_batch(&self.ctx, &hidden, &o_buf, &mut hidden_out)?;
            std::mem::swap(&mut hidden, &mut hidden_out);

            rms_norm_batch(
                &self.ctx,
                &hidden,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
                &mut normed,
            )?;
            gemm_batch(&self.ctx, &layer.mlp.gate_proj, &normed, &mut gate_out)?;
            gemm_batch(&self.ctx, &layer.mlp.up_proj, &normed, &mut up_out)?;
            silu_mul(&self.ctx, &gate_out, &up_out, &mut act_out)?;
            gemm_batch(&self.ctx, &layer.mlp.down_proj, &act_out, &mut o_buf)?;
            add_batch(&self.ctx, &hidden, &o_buf, &mut hidden_out)?;
            std::mem::swap(&mut hidden, &mut hidden_out);
        }

        let mut last_hidden = DeviceVec::zeros(&self.ctx, hidden_size)?;
        copy_row_to_vec(&self.ctx, &hidden, seq_len - 1, &mut last_hidden)?;
        let mut last_normed = DeviceVec::zeros(&self.ctx, hidden_size)?;
        rms_norm_vec(
            &self.ctx,
            &last_hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut last_normed,
        )?;
        let mut logits = DeviceVec::zeros(&self.ctx, self.output_projection().rows)?;
        gemv(
            &self.ctx,
            self.output_projection(),
            &last_normed,
            &mut logits,
        )?;
        argmax(&self.ctx, &logits)
    }
}

fn validate_clean_bf16_config(config: &Qwen3Config) -> Result<()> {
    ensure!(
        config.hidden_size == config.num_attention_heads * config.head_dim,
        "Qwen3 hidden_size {} must equal num_attention_heads {} * head_dim {}",
        config.hidden_size,
        config.num_attention_heads,
        config.head_dim
    );
    ensure!(
        config
            .num_attention_heads
            .is_multiple_of(config.num_key_value_heads),
        "Qwen3 num_attention_heads must be divisible by num_key_value_heads"
    );
    Ok(())
}

#[derive(Debug)]
struct PageMeta {
    q_indptr: CudaSlice<i32>,
    kv_indptr: CudaSlice<i32>,
    kv_indices: CudaSlice<i32>,
    kv_last_page_len: CudaSlice<i32>,
    page_table_offsets: CudaSlice<i32>,
    start_positions: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    seq_len: usize,
    num_pages: usize,
}

impl PageMeta {
    fn for_slot(
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let total_len = start_pos + seq_len;
        ensure!(
            pool.seq_len(slot) == total_len,
            "PagedKVPool seq_len {} != materialized total_len {} for slot {}",
            pool.seq_len(slot),
            total_len,
            slot
        );
        let num_pages = total_len.div_ceil(pool.page_size);
        let pages = pool.page_indices(slot);
        ensure!(
            pages.len() >= num_pages,
            "slot {} has {} pages, expected at least {}",
            slot,
            pages.len(),
            num_pages
        );
        let last_page_len = total_len % pool.page_size;
        let last_page_len = if last_page_len == 0 {
            pool.page_size
        } else {
            last_page_len
        };
        let page_ids = pages[..num_pages]
            .iter()
            .map(|&page| page as i32)
            .collect::<Vec<_>>();
        Ok(Self {
            q_indptr: upload_i32(ctx, &[0, seq_len as i32])?,
            kv_indptr: upload_i32(ctx, &[0, num_pages as i32])?,
            kv_indices: upload_i32(ctx, &page_ids)?,
            kv_last_page_len: upload_i32(ctx, &[last_page_len as i32])?,
            page_table_offsets: upload_i32(ctx, &[0])?,
            start_positions: upload_i32(ctx, &[start_pos as i32])?,
            positions: upload_i32(ctx, &[(total_len - 1) as i32])?,
            seq_len,
            num_pages,
        })
    }
}

struct SafetensorLoader {
    base: PathBuf,
    shards: Vec<PathBuf>,
    weight_map: HashMap<String, usize>,
}

impl SafetensorLoader {
    fn new(base: &Path) -> Result<Self> {
        let index_path = base.join("model.safetensors.index.json");
        if index_path.exists() {
            let content = fs::read_to_string(&index_path)
                .with_context(|| format!("read {}", index_path.display()))?;
            let index: SafetensorIndex = serde_json::from_str(&content)
                .with_context(|| format!("parse {}", index_path.display()))?;
            let mut shards = Vec::<PathBuf>::new();
            let mut file_to_idx = HashMap::<String, usize>::new();
            let mut weight_map = HashMap::new();
            for (name, file) in index.weight_map {
                let idx = match file_to_idx.get(&file) {
                    Some(&idx) => idx,
                    None => {
                        let idx = shards.len();
                        shards.push(base.join(&file));
                        file_to_idx.insert(file, idx);
                        idx
                    }
                };
                weight_map.insert(name, idx);
            }
            return Ok(Self {
                base: base.to_path_buf(),
                shards,
                weight_map,
            });
        }

        let single = base.join("model.safetensors");
        if single.exists() {
            return Ok(Self {
                base: base.to_path_buf(),
                shards: vec![single],
                weight_map: HashMap::new(),
            });
        }

        let mut shards = fs::read_dir(base)
            .with_context(|| format!("scan {}", base.display()))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect::<Vec<_>>();
        shards.sort();
        ensure!(
            !shards.is_empty(),
            "no safetensors shards found under {}",
            base.display()
        );
        Ok(Self {
            base: base.to_path_buf(),
            shards,
            weight_map: HashMap::new(),
        })
    }

    fn load_vec(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceVec> {
        let tensor = self.load_tensor(name)?;
        ensure!(
            tensor.shape.len() == 1,
            "{name}: expected 1D BF16 tensor, got shape {:?}",
            tensor.shape
        );
        DeviceVec::from_safetensors(ctx, &tensor.bytes)
            .with_context(|| format!("upload tensor {name}"))
    }

    fn load_matrix(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceMatrix> {
        let tensor = self.load_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D BF16 tensor, got shape {:?}",
            tensor.shape
        );
        DeviceMatrix::from_safetensors(ctx, &tensor.bytes, tensor.shape[0], tensor.shape[1])
            .with_context(|| format!("upload tensor {name}"))
    }

    fn load_tensor(&self, name: &str) -> Result<OwnedTensor> {
        if let Some(&idx) = self.weight_map.get(name) {
            return self.load_tensor_from_shard(idx, name);
        }
        for idx in 0..self.shards.len() {
            if let Ok(tensor) = self.load_tensor_from_shard(idx, name) {
                return Ok(tensor);
            }
        }
        Err(anyhow!(
            "tensor {name} not found in safetensors under {}",
            self.base.display()
        ))
    }

    fn load_tensor_from_shard(&self, idx: usize, name: &str) -> Result<OwnedTensor> {
        let path = self
            .shards
            .get(idx)
            .ok_or_else(|| anyhow!("shard index {idx} out of range"))?;
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("deserialize {}", path.display()))?;
        let view = tensors
            .tensor(name)
            .with_context(|| format!("find tensor {name} in {}", path.display()))?;
        ensure!(
            view.dtype() == Dtype::BF16,
            "{name}: R6 clean CUDA path accepts BF16 only, got {:?}",
            view.dtype()
        );
        Ok(OwnedTensor {
            shape: view.shape().to_vec(),
            bytes: view.data().to_vec(),
        })
    }
}

#[derive(serde::Deserialize)]
struct SafetensorIndex {
    weight_map: HashMap<String, String>,
}

struct OwnedTensor {
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn precompute_rope(
    ctx: &DeviceContext,
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
    scaling: Option<&qwen3_spec::RopeScalingConfig>,
) -> Result<(DeviceVec, DeviceVec)> {
    let half_dim = head_dim / 2;
    let inv_freq = qwen3_spec::compute_scaled_inv_freq(head_dim, theta, scaling);
    let total = max_seq_len * head_dim;
    let mut cos_host = vec![bf16::ZERO; total];
    let mut sin_host = vec![bf16::ZERO; total];

    for pos in 0..max_seq_len {
        let base = pos * head_dim;
        for (i, freq) in inv_freq.iter().enumerate().take(half_dim) {
            let freq = pos as f32 * *freq;
            let cos_val = bf16::from_f32(freq.cos());
            let sin_val = bf16::from_f32(freq.sin());
            cos_host[base + i] = cos_val;
            cos_host[base + i + half_dim] = cos_val;
            sin_host[base + i] = sin_val;
            sin_host[base + i + half_dim] = sin_val;
        }
    }

    Ok((
        DeviceVec::from_host(ctx, &cos_host)?,
        DeviceVec::from_host(ctx, &sin_host)?,
    ))
}

fn upload_i32(ctx: &DeviceContext, values: &[i32]) -> Result<CudaSlice<i32>> {
    ctx.stream
        .clone_htod(values)
        .map_err(|e| anyhow!("H2D i32 upload failed: {e}"))
}

fn embedding_batch(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_ids: &CudaSlice<i32>,
    out: &mut HiddenStates,
) -> Result<()> {
    let (embed_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
    let (token_ptr, _gt) = token_ids.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::embedding_batched_cuda(
            embed_ptr as *const ffi::Half,
            token_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            embed.cols as i32,
            out.seq_len as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

fn rms_norm_batch(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_batched_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.hidden_dim as i32,
            x.seq_len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

fn rms_norm_vec(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::rms_norm_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

fn gemm_batch(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        weight.cols == x.hidden_dim,
        "gemm input dim mismatch: weight cols {}, x hidden_dim {}",
        weight.cols,
        x.hidden_dim
    );
    ensure!(
        weight.rows == out.hidden_dim && x.seq_len == out.seq_len,
        "gemm output shape mismatch: weight rows {}, out hidden_dim {}, x seq {}, out seq {}",
        weight.rows,
        out.hidden_dim,
        x.seq_len,
        out.seq_len
    );
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::gemm_cuda(
            w_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            weight.rows as i32,
            x.seq_len as i32,
            weight.cols as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

fn gemv(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &DeviceVec,
    out: &mut DeviceVec,
) -> Result<()> {
    ensure!(
        weight.cols == x.len && weight.rows == out.len,
        "gemv shape mismatch: weight [{}x{}], x {}, out {}",
        weight.rows,
        weight.cols,
        x.len,
        out.len
    );
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::gemv_cuda(
            w_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            weight.rows as i32,
            weight.cols as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

fn add_batch(
    ctx: &DeviceContext,
    a: &HiddenStates,
    b: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        a.hidden_dim == b.hidden_dim
            && a.hidden_dim == out.hidden_dim
            && a.seq_len == b.seq_len
            && a.seq_len == out.seq_len,
        "add_batch shape mismatch"
    );
    let n = a.hidden_dim * a.seq_len;
    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::add_cuda(
            a_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

fn silu_mul(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        gate.hidden_dim == up.hidden_dim
            && gate.hidden_dim == out.hidden_dim
            && gate.seq_len == up.seq_len
            && gate.seq_len == out.seq_len,
        "silu_mul shape mismatch"
    );
    let n = gate.hidden_dim * gate.seq_len;
    let (gate_ptr, _gg) = gate.data.device_ptr(&ctx.stream);
    let (up_ptr, _gu) = up.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::silu_mul_cuda(
            gate_ptr as *const ffi::Half,
            up_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn paged_attention(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    rms_eps: f32,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    if meta.seq_len == 1 {
        decode_attention(
            ctx,
            layer_idx,
            pool,
            q_batch,
            k_batch,
            v_batch,
            q_norm,
            k_norm,
            cos_cache,
            sin_cache,
            rms_eps,
            meta,
            num_q_heads,
            num_kv_heads,
            head_dim,
            out,
        )
    } else {
        prefill_attention(
            ctx,
            layer_idx,
            pool,
            q_batch,
            k_batch,
            v_batch,
            q_norm,
            k_norm,
            cos_cache,
            sin_cache,
            rms_eps,
            meta,
            num_q_heads,
            num_kv_heads,
            head_dim,
            out,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn prefill_attention(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    rms_eps: f32,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    {
        let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr_mut(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
        let (indices_ptr, _gi) = meta.kv_indices.device_ptr(&ctx.stream);
        let (offsets_ptr, _goff) = meta.page_table_offsets.device_ptr(&ctx.stream);
        let (start_ptr, _gstart) = meta.start_positions.device_ptr(&ctx.stream);
        let k_pool_ptr = pool.k_ptr(layer_idx, &ctx.stream);
        let v_pool_ptr = pool.v_ptr(layer_idx, &ctx.stream);

        unsafe {
            ffi::prefill_attention_paged_prep_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *mut ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                indices_ptr as *const i32,
                offsets_ptr as *const i32,
                pool.page_size as i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                meta.seq_len as i32,
                start_ptr as *const i32,
                rms_eps,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    run_tilelang_paged(
        ctx,
        false,
        layer_idx,
        pool,
        q_batch,
        meta,
        num_q_heads,
        num_kv_heads,
        head_dim,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_attention(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &mut HiddenStates,
    k_batch: &HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    rms_eps: f32,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    {
        let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
        let (pos_ptr, _gp) = meta.positions.device_ptr(&ctx.stream);
        let (indices_ptr, _gi) = meta.kv_indices.device_ptr(&ctx.stream);
        let (indptr_ptr, _gind) = meta.kv_indptr.device_ptr(&ctx.stream);
        let (last_ptr, _glp) = meta.kv_last_page_len.device_ptr(&ctx.stream);
        let k_pool_ptr = pool.k_ptr(layer_idx, &ctx.stream);
        let v_pool_ptr = pool.v_ptr(layer_idx, &ctx.stream);
        let stride_page = pool.kv_dim * pool.page_size;

        unsafe {
            ffi::decode_prep_paged_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *const ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                pos_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                indices_ptr as *const i32,
                indptr_ptr as *const i32,
                last_ptr as *const i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                stride_page as i32,
                1,
                rms_eps,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    run_tilelang_paged(
        ctx,
        true,
        layer_idx,
        pool,
        q_batch,
        meta,
        num_q_heads,
        num_kv_heads,
        head_dim,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_tilelang_paged(
    ctx: &DeviceContext,
    decode: bool,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &HiddenStates,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(head_dim == 128, "only HD128 TileLang kernels are wired");
    ensure!(num_kv_heads == 8, "only kv8 TileLang kernels are wired");

    let (q_ptr, _gq) = q_batch.data.device_ptr(&ctx.stream);
    let (qo_ptr, _gqo) = meta.q_indptr.device_ptr(&ctx.stream);
    let (kv_indptr_ptr, _gki) = meta.kv_indptr.device_ptr(&ctx.stream);
    let (kv_indices_ptr, _gkx) = meta.kv_indices.device_ptr(&ctx.stream);
    let (last_ptr, _glp) = meta.kv_last_page_len.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let k_pool_ptr = pool.k_ptr(layer_idx, &ctx.stream);
    let v_pool_ptr = pool.v_ptr(layer_idx, &ctx.stream);
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

    unsafe {
        match (decode, num_q_heads) {
            (false, 16) => ffi::tilelang_batch_prefill_paged_hd128_q16_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                meta.num_pages as i32,
                pool.max_total_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (false, 32) => ffi::tilelang_batch_prefill_paged_hd128_q32_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                meta.num_pages as i32,
                pool.max_total_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (false, 40) => ffi::tilelang_batch_prefill_paged_hd128_q40_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                meta.num_pages as i32,
                pool.max_total_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (false, 64) => ffi::tilelang_batch_prefill_paged_hd128_q64_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                meta.num_pages as i32,
                pool.max_total_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 16) => ffi::tilelang_batch_decode_paged_hd128_q16_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                meta.num_pages as i32,
                pool.max_total_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 32) => ffi::tilelang_batch_decode_paged_hd128_q32_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                meta.num_pages as i32,
                pool.max_total_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 40) => ffi::tilelang_batch_decode_paged_hd128_q40_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                meta.num_pages as i32,
                pool.max_total_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 64) => ffi::tilelang_batch_decode_paged_hd128_q64_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                meta.num_pages as i32,
                pool.max_total_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            _ => bail!("unsupported HD128 q/kv head config q{num_q_heads}_kv{num_kv_heads}"),
        }
    }
    Ok(())
}

fn copy_row_to_vec(
    ctx: &DeviceContext,
    batch: &HiddenStates,
    token_idx: usize,
    out: &mut DeviceVec,
) -> Result<()> {
    ensure!(
        out.len == batch.hidden_dim,
        "copy_row_to_vec output len {} != hidden_dim {}",
        out.len,
        batch.hidden_dim
    );
    let offset = token_idx * batch.hidden_dim;
    let src = batch.data.slice(offset..offset + batch.hidden_dim);
    ctx.stream
        .memcpy_dtod(&src, &mut out.data)
        .map_err(|e| anyhow!("D2D copy last hidden row failed: {e}"))
}

fn argmax(ctx: &DeviceContext, logits: &DeviceVec) -> Result<u32> {
    let mut out = ctx
        .stream
        .alloc_zeros::<i32>(1)
        .map_err(|e| anyhow!("argmax output alloc failed: {e}"))?;
    {
        let (logits_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::argmax_cuda(
                logits_ptr as *const ffi::Half,
                out_ptr as *mut i32,
                logits.len as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    ctx.sync()?;
    let token = ctx
        .stream
        .clone_dtoh(&out)
        .map_err(|e| anyhow!("D2H argmax token failed: {e}"))?;
    Ok(token[0] as u32)
}
