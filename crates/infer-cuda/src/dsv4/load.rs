//! DSv4 checkpoint loading: FP8/FP4 weight construction, DSpark draft-delta
//! import, and the config/tensor-name probes the two feed on. Split out of
//! `dsv4.rs` — load-time only, nothing here runs on a forward.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use cuda_kernels::prelude::DeviceContext;
use deepseek_spec::DeepSeekV4Config;

use cuda_kernels::tensor::WeightFormat;
use deepseek_spec::Shard;
use infer_topo::{ShardingSpec, TpConfig};
use safetensors::tensor::Dtype;

use crate::loader::{SafetensorLoader, tensor_bytes_to_f32};
use crate::moe_config::ExpertSplit;
use crate::quant_format::{QuantFormat, ScaleApply};

use super::*;

/// Load the 3-stage DSpark draft from a DSpark checkpoint. Each stage's common
/// block reuses the native MTP loaders (MLA attn + FP8 MoE + hyper-connections);
/// the position-dependent extras load conditionally: `main_proj` via the
/// fp8-block loader the attn `wq_a`/`wkv` projections use, the Markov head in
/// checkpoint BF16, and the remaining small tensors through their native loaders.
#[allow(dead_code)]
pub(crate) fn load_dspark_draft(
    loader: &SafetensorLoader,
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    split: &ExpertSplit,
    tp_cfg: &infer_topo::TpConfig,
) -> Result<Dsv4DsparkDraft> {
    ensure!(
        config.is_dspark(),
        "load_dspark_draft called on a non-DSpark config"
    );
    let compress_ratio = 0;
    let stages = (0..config.dspark_num_stages())
        .map(|stage_idx| {
            let names = config.dspark_tensor_names(stage_idx);
            let attention = loader.load_dsv4_attention(ctx, config, &names.attn, tp_cfg)?;
            let moe = loader.load_dsv4_moe_layer(
                ctx,
                &names.ffn,
                split,
                DeepSeekV4MoeRoutingKind::LearnedBias,
                false,
            )?;
            Ok(Dsv4DsparkStage {
                layer: Dsv4Layer {
                    hc_attn: loader.load_dsv4_hyper_connection(ctx, &names.hc_attn)?,
                    hc_ffn: loader.load_dsv4_hyper_connection(ctx, &names.hc_ffn)?,
                    attn_norm: loader.load_dsv4_vec(ctx, &names.attn_norm)?,
                    ffn_norm: loader.load_dsv4_vec(ctx, &names.ffn_norm)?,
                    attention,
                    moe: Some(moe),
                    mode: config.attention_mode_for_compress_ratio(compress_ratio),
                    compress_ratio,
                    dense_mlp: None,
                },
                // main_proj is fp8-block (F8_E4M3 + F8_E8M0 scale) like the attn
                // projections — load via the same fp8 path, not the plain loader.
                main_proj: names
                    .main_proj
                    .as_deref()
                    .map(|n| loader.load_dsv4_block_scaled(ctx, n))
                    .transpose()?,
                main_norm: names
                    .main_norm
                    .as_deref()
                    .map(|n| loader.load_dsv4_vec(ctx, n))
                    .transpose()?,
                hc_head: names
                    .hc_head
                    .as_ref()
                    .map(|hc| loader.load_dsv4_hyper_connection(ctx, hc))
                    .transpose()?,
                norm: names
                    .norm
                    .as_deref()
                    .map(|n| loader.load_dsv4_vec(ctx, n))
                    .transpose()?,
                markov_w1: names
                    .markov_w1
                    .as_deref()
                    .map(|n| loader.load_dsv4_bf16_matrix(ctx, n))
                    .transpose()?,
                markov_w2: names
                    .markov_w2
                    .as_deref()
                    .map(|n| loader.load_dsv4_bf16_matrix(ctx, n))
                    .transpose()?,
                confidence_proj: names
                    .confidence_proj
                    .as_deref()
                    .map(|n| loader.load_dsv4_global_matrix(ctx, n))
                    .transpose()?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Dsv4DsparkDraft { stages })
}

/// Merge DSpark spec-decode metadata from the DRAFT checkpoint into the base
/// `DeepSeekV4Config`. `--spec-type dspark` was explicitly requested, so a draft
/// config missing the required keys is a hard error, not a silent non-DSpark.
fn merge_dspark_metadata(config: &mut DeepSeekV4Config, draft_dir: &Path) -> Result<()> {
    let cfg_path = draft_dir.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow!("read DSpark draft config {}: {e}", cfg_path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| anyhow!("parse DSpark draft config {}: {e}", cfg_path.display()))?;
    let need_u64 = |key: &str| -> Result<u64> {
        v.get(key)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                anyhow!(
                    "--spec-type dspark needs `{key}` in draft config {}",
                    cfg_path.display()
                )
            })
    };

    config.dspark_block_size = need_u64("dspark_block_size")? as usize;
    config.dspark_markov_rank = need_u64("dspark_markov_rank")? as usize;
    config.dspark_noise_token_id = need_u64("dspark_noise_token_id")? as u32;
    let ids = v
        .get("dspark_target_layer_ids")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            anyhow!(
                "--spec-type dspark needs `dspark_target_layer_ids` in draft config {}",
                cfg_path.display()
            )
        })?;
    config.dspark_target_layer_ids = ids
        .iter()
        .map(|x| {
            x.as_u64().map(|n| n as usize).ok_or_else(|| {
                anyhow!(
                    "dspark_target_layer_ids in {} has a non-integer entry",
                    cfg_path.display()
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Stage count is NOT in the draft config (`num_nextn_predict_layers: 1` there
    // is the stale native-MTP field). Derive it by counting distinct `mtp.<N>`
    // stage prefixes in the draft checkpoint's weight map.
    config.dspark_num_stages = count_dspark_stages(draft_dir)?;
    ensure!(
        config.dspark_num_stages > 0,
        "DSpark draft {} yielded 0 `mtp.<N>` stages",
        draft_dir.display()
    );
    Ok(())
}

/// Count distinct `mtp.<N>.` stage prefixes in the draft checkpoint. Primary
/// source is `model.safetensors.index.json`'s `weight_map`; when absent, fall
/// back to reading the tensor names from the safetensors headers.
fn count_dspark_stages(draft_dir: &Path) -> Result<usize> {
    let index_path = draft_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path)
            .map_err(|e| anyhow!("read DSpark draft index {}: {e}", index_path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| anyhow!("parse DSpark draft index {}: {e}", index_path.display()))?;
        let map = v
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                anyhow!(
                    "DSpark draft index {} lacks a weight_map object",
                    index_path.display()
                )
            })?;
        return Ok(count_distinct_mtp_stages(map.keys().map(String::as_str)));
    }

    // No index — glob the shards and read their safetensors headers.
    let mut stages = std::collections::HashSet::new();
    for entry in std::fs::read_dir(draft_dir)
        .map_err(|e| anyhow!("read DSpark draft dir {}: {e}", draft_dir.display()))?
    {
        let path = entry
            .map_err(|e| anyhow!("read DSpark draft dir entry: {e}"))?
            .path();
        let is_shard = path.extension().is_some_and(|e| e == "safetensors")
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.contains("mtp") || n.starts_with("model-000"));
        if !is_shard {
            continue;
        }
        for name in read_safetensors_tensor_names(&path)? {
            if let Some(n) = mtp_stage_index(&name) {
                stages.insert(n);
            }
        }
    }
    Ok(stages.len())
}

/// Extract the `<N>` from a `mtp.<N>.<...>` tensor name, else `None`.
fn mtp_stage_index(name: &str) -> Option<usize> {
    name.strip_prefix("mtp.")
        .and_then(|rest| rest.split('.').next())
        .and_then(|n| n.parse::<usize>().ok())
}

fn count_distinct_mtp_stages<'a>(keys: impl Iterator<Item = &'a str>) -> usize {
    keys.filter_map(mtp_stage_index)
        .collect::<std::collections::HashSet<_>>()
        .len()
}

/// Read tensor names from a safetensors file header (8-byte LE length prefix +
/// JSON header keyed by tensor name).
fn read_safetensors_tensor_names(path: &Path) -> Result<Vec<String>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| anyhow!("open {}: {e}", path.display()))?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)
        .map_err(|e| anyhow!("read header length of {}: {e}", path.display()))?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header)
        .map_err(|e| anyhow!("read header of {}: {e}", path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&header)
        .map_err(|e| anyhow!("parse header of {}: {e}", path.display()))?;
    Ok(v.as_object()
        .map(|o| o.keys().filter(|k| *k != "__metadata__").cloned().collect())
        .unwrap_or_default())
}

/// Refuse the genuinely-unported variants up front so the loader never
/// half-loads a shape the forward can't run. CSA/HCA attention, hyper-connections
/// (`hc_mult > 1`), and hash-routed MoE layers are all wired now. MTP
/// (speculative-draft) layers are tolerated but **not loaded**: the base forward
/// loops `0..num_hidden_layers` (see [`Dsv4Model::from_fp8_safetensors`]) and the
/// MTP predictor head is a separate path with no consumer in the base decode
/// loop, so we run the production config (`num_nextn_predict_layers=1`) directly
/// rather than forcing a hand-trimmed base-only config view. Called by
/// [`crate::loader`] before any device I/O.
pub(crate) fn ensure_loadable(
    config: &DeepSeekV4Config,
    spec_decode_on: bool,
    dspark_on: bool,
) -> Result<()> {
    ensure!(
        config.num_key_value_heads == 1,
        "DSv4 MLA expects num_key_value_heads=1, got {}",
        config.num_key_value_heads
    );
    if config.num_nextn_predict_layers > 0 {
        // Report the effective decision (`spec_decode_on` = CLI MTP request or
        // fallback env), not the env alone. The MTP head load gate below uses
        // the same boolean, so the log must not claim "deferred" when
        // `--spec-type mtp` / `--mtp-draft-tokens` already made spec decode on.
        if spec_decode_on {
            if dspark_on {
                eprintln!(
                    "[dsv4] num_nextn_predict_layers={} present; spec decode on (DSpark), \
                     loading base layers plus DSpark draft (native MTP skipped).",
                    config.num_nextn_predict_layers
                );
            } else {
                eprintln!(
                    "[dsv4] num_nextn_predict_layers={} present; spec decode on, \
                     loading base layers plus mtp.0 draft head.",
                    config.num_nextn_predict_layers
                );
            }
        } else {
            eprintln!(
                "[dsv4] num_nextn_predict_layers={} present; loading the {} base layers \
                 only (MTP draft head deferred — pass --spec-type mtp / --mtp-draft-tokens).",
                config.num_nextn_predict_layers, config.num_hidden_layers
            );
        }
    }
    ensure!(
        config.hc_mult >= 1,
        "DSv4 hc_mult must be >= 1, got {}",
        config.hc_mult
    );
    Ok(())
}

impl Dsv4Model {
    /// Load a DSv4-Flash FP8 checkpoint for this TP/EP rank.
    ///
    /// EP mirrors TP (the plan's TP=8/EP=8 layout): `ep_size = world_size`,
    /// `ep_rank = rank`, so each rank owns `256 / world_size` experts. Single-GPU
    /// keeps all experts local (dev/typecheck). Weight FP8/FP4 + E8M0 scales load
    /// through the shared `cuda-kernels` DSv4 tensors; per-expert DeepGEMM caches
    /// are built at load. The forward (MLA, FP8 MoE) is Pieces 2/3.
    pub(crate) fn from_dsv4_fp8_safetensors(
        model_path: &Path,
        mtp_draft_tokens: Option<usize>,
        dspark_draft_model: Option<&Path>,
    ) -> Result<Self> {
        let tp = crate::loader::build_tp_runtime(true)?;
        Self::from_dsv4_fp8_safetensors_with_tp(
            model_path,
            tp,
            mtp_draft_tokens,
            dspark_draft_model,
        )
    }

    pub(crate) fn from_dsv4_fp8_safetensors_with_tp(
        model_path: &Path,
        #[cfg_attr(not(feature = "nccl"), allow(unused_mut))] mut tp: crate::tp::TpRuntime,
        mtp_draft_tokens: Option<usize>,
        dspark_draft_model: Option<&Path>,
    ) -> Result<Self> {
        // CP prefill (T2.b) is a qwen35 path; DSv4 shards by raw tp rank and
        // its attention reduces run on the global comm.
        ensure!(
            tp.attn_cp_size() == 1,
            "attn_cp>1 is not supported by the DSv4 executor (qwen35-only)"
        );
        // DSpark metadata (block_size / target taps / stage count) ships on the
        // DRAFT checkpoint, not the base — merged below. `dspark_on` gates the
        // shared spec-ring snapshots.
        let dspark_on = dspark_draft_model.is_some();
        // Spec decode is on when the serve config requests it — `Some(n)` from
        // `--spec-type mtp`, `dspark_on` from `--spec-type dspark`. Both routes
        // need the per-slot spec-ring snapshots, so both flip `spec_decode_on`.
        // Resolved once and stored on the model so per-slot construction reads
        // the same decision.
        let spec_decode_on = mtp_draft_tokens.is_some() || dspark_on;
        // Peek model_type: GLM-5.2 (`glm_moe_dsa`) parses through the GLM dialect
        // adapter (V32 shape, plain-o, hc_mult=1, num_nextn=0, Glm tensor names);
        // every other DSv4 checkpoint loads through the strict DSv4 parser
        // byte-unchanged.
        let config_path = model_path.join("config.json");
        let config_json = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow!("read DSv4 config {}: {e}", config_path.display()))?;
        let model_type = serde_json::from_str::<serde_json::Value>(&config_json)
            .ok()
            .and_then(|v| {
                v.get("model_type")
                    .and_then(|m| m.as_str())
                    .map(str::to_owned)
            })
            .unwrap_or_default();
        let mut config = if model_type == "glm_moe_dsa" {
            deepseek_spec::GlmMoeDsaConfig::from_json_str(&config_json)
                .and_then(|glm| glm.into_deepseek_v4())
                .map_err(|e| anyhow!("load GLM config from {}: {e}", config_path.display()))?
        } else {
            DeepSeekV4Config::from_json_str(&config_json)
                .map_err(|e| anyhow!("load DSv4 config from {}: {e}", config_path.display()))?
        };
        // DSpark metadata lives on the draft checkpoint; the base config carries
        // none (is_dspark()/T3 taps read the base). Merge it in so the base model
        // captures taps on its layers and the executor gates DSpark.
        if let Some(draft_dir) = dspark_draft_model {
            merge_dspark_metadata(&mut config, draft_dir)?;
        }
        ensure_loadable(&config, spec_decode_on, dspark_on)?;

        let moe_config = Self::moe_config_from_config(&config)?;
        let tp_cfg = *tp.config();
        let split = if tp_cfg.is_single() {
            ExpertSplit::single(config.n_routed_experts)
        } else {
            ExpertSplit::new(config.n_routed_experts, tp_cfg.world_size, tp_cfg.rank)
                .map_err(|e| anyhow!("DSv4 EP split: {e}"))?
        };
        let kv_arena = Dsv4MlaKvArena::from_config(&config)?;

        let ctx = DeviceContext::new()?;
        // One-shot small-message collectives (default-on, loud auto-degrade).
        // COLLECTIVE boot — identical construction point on every rank, BEFORE
        // the DeepEP boot so the collective sequences line up across ranks.
        #[cfg(feature = "nccl")]
        tp.init_oneshot_comm(&ctx);
        #[cfg(feature = "deepep")]
        let deepep = crate::deepep::DeepEpTransport::maybe_boot(
            &ctx,
            &tp,
            config.hidden_size,
            config.n_routed_experts,
        )?;
        let loader = SafetensorLoader::new(model_path)?;
        loader.prefetch_shards_rank0(&ctx, &tp)?;
        let names = config.tensor_names();

        let embed_tokens = loader.load_dsv4_bf16_matrix(&ctx, names.embed_tokens())?;
        let lm_head = loader.load_dsv4_global_matrix(&ctx, names.lm_head())?;

        // GLM (`glm_moe_dsa`) markers: re-encode FP8 MoE experts from `weight_scale_inv`,
        // and bypass the absent hyper-connections (`hc_mult == 1`, identity mixers).
        let glm = config.plain_o_proj;
        let hc_absent = config.is_glm();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let plan = config
                .attention_layer_plan(layer_idx)
                .ok_or_else(|| anyhow!("DSv4 layer {layer_idx} has no attention plan"))?;
            let lnames = config.layer_tensor_names(layer_idx);
            let attention = loader.load_dsv4_attention(&ctx, &config, &lnames.attn, &tp_cfg)?;
            // GLM dense layers (`per_layer_dense_mlp[i]`): a plain FFN, no experts.
            let is_dense = config
                .per_layer_dense_mlp
                .as_ref()
                .and_then(|f| f.get(layer_idx).copied())
                .unwrap_or(false);
            let (moe, dense_mlp) = if is_dense {
                let dense = loader.load_dsv4_dense_mlp(
                    &ctx,
                    lnames.ffn.dense_mlp.as_ref().ok_or_else(|| {
                        anyhow!("GLM dense layer {layer_idx} missing dense_mlp names")
                    })?,
                )?;
                (None, Some(dense))
            } else {
                let moe = loader.load_dsv4_moe_layer(
                    &ctx,
                    &lnames.ffn,
                    &split,
                    config.moe_routing_kind(layer_idx),
                    glm,
                )?;
                (Some(moe), None)
            };
            // hc-absent (GLM): identity mixers; skip the (non-existent) hc tensor loads.
            let (hc_attn, hc_ffn) = if hc_absent {
                (
                    Dsv4HyperConnection::identity_placeholder(&ctx)?,
                    Dsv4HyperConnection::identity_placeholder(&ctx)?,
                )
            } else {
                (
                    loader.load_dsv4_hyper_connection(&ctx, &lnames.hc_attn)?,
                    loader.load_dsv4_hyper_connection(&ctx, &lnames.hc_ffn)?,
                )
            };
            layers.push(Dsv4Layer {
                hc_attn,
                hc_ffn,
                attn_norm: loader.load_dsv4_vec(&ctx, &lnames.attn_norm)?,
                ffn_norm: loader.load_dsv4_vec(&ctx, &lnames.ffn_norm)?,
                attention,
                moe,
                mode: plan.mode,
                compress_ratio: plan.compress_ratio,
                dense_mlp,
            });
        }
        let norm = loader.load_dsv4_vec(&ctx, names.norm())?;
        // GLM has no head hyper-connection; identity placeholder.
        let head_hc = if hc_absent {
            Dsv4HyperConnection::identity_placeholder(&ctx)?
        } else {
            loader.load_dsv4_hyper_connection(&ctx, &names.head_hc())?
        };
        let mtp = if !dspark_on && spec_decode_on && config.num_nextn_predict_layers > 0 {
            ensure!(
                config.num_nextn_predict_layers == 1,
                "DSv4 Phase-1 MTP loader supports exactly one nextn layer, got {}",
                config.num_nextn_predict_layers
            );
            let mtp_names = config.mtp_tensor_names(0);
            let attention = loader.load_dsv4_attention(&ctx, &config, &mtp_names.attn, &tp_cfg)?;
            // The loaded MTP block uses the DSv4 MoE shape.
            let moe = loader.load_dsv4_moe_layer(
                &ctx,
                &mtp_names.ffn,
                &split,
                DeepSeekV4MoeRoutingKind::LearnedBias,
                false,
            )?;
            let compress_ratio = 0;
            Some(Dsv4MtpLayer {
                layer: Dsv4Layer {
                    hc_attn: loader.load_dsv4_hyper_connection(&ctx, &mtp_names.hc_attn)?,
                    hc_ffn: loader.load_dsv4_hyper_connection(&ctx, &mtp_names.hc_ffn)?,
                    attn_norm: loader.load_dsv4_vec(&ctx, &mtp_names.attn_norm)?,
                    ffn_norm: loader.load_dsv4_vec(&ctx, &mtp_names.ffn_norm)?,
                    attention,
                    moe: Some(moe),
                    mode: config.attention_mode_for_compress_ratio(compress_ratio),
                    compress_ratio,
                    dense_mlp: None,
                },
                head_hc: loader.load_dsv4_hyper_connection(&ctx, &mtp_names.hc_head)?,
                enorm: loader.load_dsv4_vec(&ctx, &mtp_names.enorm)?,
                hnorm: loader.load_dsv4_vec(&ctx, &mtp_names.hnorm)?,
                e_proj: loader.load_dsv4_global_matrix(&ctx, &mtp_names.e_proj)?,
                h_proj: loader.load_dsv4_global_matrix(&ctx, &mtp_names.h_proj)?,
                norm: loader.load_dsv4_vec(&ctx, &mtp_names.norm)?,
            })
        } else {
            None
        };
        ctx.sync()?;

        let probe = super::probe::Dsv4ProbeCapture::from_env(&ctx, &config, &lm_head, layers.len());
        Ok(Self {
            ctx,
            config,
            moe_config,
            split,
            kv_arena,
            embed_tokens,
            lm_head,
            layers,
            norm,
            head_hc,
            mtp,
            spec_decode_on,
            tp,
            probe: std::cell::RefCell::new(probe),
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            mega_moe: None,
            #[cfg(feature = "deepep")]
            deepep,
            graph_mode: std::sync::atomic::AtomicBool::new(false),
            graph_token_ids: std::sync::Mutex::new(None),
            graph_bufs: std::sync::Mutex::new(Vec::new()),
        })
    }
}

/// Saturating encode of `val` into FP8 E4M3FN: 1 sign + 4 exp + 3 mant, bias 7, max
/// normal
/// 0x7E = 448.0. Values are clamped to ±448; NaN input maps to zero.
fn encode_f8_e4m3fn_sat(val: f32) -> u8 {
    const E4M3_MAX: f32 = 448.0;
    const BIAS: i32 = 7;
    let val = if val.is_nan() {
        0.0
    } else {
        val.clamp(-E4M3_MAX, E4M3_MAX)
    };
    let sign: u8 = if val.is_sign_negative() { 0x80 } else { 0x00 };
    let abs = val.abs();
    if abs == 0.0 {
        return sign;
    }
    let bits = abs.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let fp8_biased_exp = f32_exp + BIAS;
    if fp8_biased_exp <= 0 {
        // Subnormal: 0.mmmm × 2^(1−7), i.e. value = mant × 2^(−9)
        let mant = ((abs * 512.0).round() as i32).clamp(0, 7) as u8;
        return sign | mant;
    }
    if fp8_biased_exp >= 15 {
        return sign | 0x7E; // saturate to ±448.0
    }
    let fp8_biased_exp = fp8_biased_exp as u8;
    let f32_mant = bits & 0x007F_FFFF;
    // Round-to-nearest: add the guard bit (bit 19) before truncating to 3 bits.
    let fp8_mant_raw = (f32_mant + (1 << 19)) >> 20;
    if fp8_mant_raw > 7 {
        let new_exp = fp8_biased_exp + 1;
        return if new_exp >= 15 {
            sign | 0x7E
        } else {
            sign | (new_exp << 3)
        };
    }
    sign | (fp8_biased_exp << 3) | (fp8_mant_raw as u8)
}

/// NVFP4→W4AFP8 per-tensor conversion result: (packed int4 weights, bf16 scales, n, k, scale_rows).
type W4Afp8Converted = (Vec<u8>, Vec<u8>, usize, usize, usize);

#[allow(dead_code)]
impl SafetensorLoader {
    /// Load a DSv4 1D norm/bias vector — BF16 or F32 in the checkpoint, normalized to
    /// BF16.
    pub(crate) fn load_dsv4_vec(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceVec> {
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 1,
            "{name}: expected 1D tensor, got shape {:?}",
            tensor.shape
        );
        DeviceVec::from_safetensors(
            ctx,
            Self::tensor_bytes_to_bf16(name, tensor.dtype, tensor.bytes())?.as_ref(),
        )
        .with_context(|| format!("upload DSv4 vec {name}"))
    }

    /// Load a DSv4 2D router gate (the only non-FP8 2D weight), normalized to BF16.
    pub(crate) fn load_dsv4_bf16_matrix(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D tensor, got shape {:?}",
            tensor.shape
        );
        DeviceMatrix::from_safetensors(
            ctx,
            Self::tensor_bytes_to_bf16(name, tensor.dtype, tensor.bytes())?.as_ref(),
            tensor.shape[0],
            tensor.shape[1],
        )
        .with_context(|| format!("upload DSv4 gate {name}"))
    }

    /// Sharded dense BF16/F32 load, for checkpoints that leave the tiny low-rank `wo_a`
    /// unquantized. Converts to bf16, then slices rows.
    /// ponytail: only Column{0}/Replicated — the only shards `wo_a` uses; bail else.
    pub(crate) fn load_dsv4_bf16_sharded(
        &self,
        ctx: &DeviceContext,
        name: &str,
        shard: Shard,
        tp: &TpConfig,
    ) -> Result<DeviceMatrix> {
        if tp.is_single() || shard == Shard::Replicated {
            return self.load_dsv4_bf16_matrix(ctx, name);
        }
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D dense tensor, got shape {:?}",
            tensor.shape
        );
        let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
        let bf16 = Self::tensor_bytes_to_bf16(name, tensor.dtype, tensor.bytes())?;
        match shard {
            Shard::Column { dim: 0 } => {
                let spec = infer_topo::column_shard(rows, tp);
                let sharded =
                    crate::shard_slice::shard_column_parallel(bf16.as_ref(), rows, cols, 2, &spec)?;
                DeviceMatrix::from_safetensors(ctx, &sharded.bytes, sharded.rows, sharded.cols)
                    .with_context(|| format!("upload DSv4 dense shard {name}"))
            }
            other => bail!(
                "{name}: DSv4 dense sharded load supports Column{{dim:0}}/Replicated, got {other:?}"
            ),
        }
    }

    /// Normalize a DSv4 block scale to E8M0 bytes. Accepts native `F8_E8M0`, or the
    /// common
    /// `F32` power-of-two serialization: a power-of-two f32's biased exponent byte IS
    /// its
    /// E8M0 code, so `(bits >> 23) & 0xff` is lossless. Errors on a non-power-of-two
    /// F32.
    fn dsv4_block_scale_e8m0(scale_name: &str, dtype: Dtype, bytes: &[u8]) -> Result<Vec<u8>> {
        match dtype {
            Dtype::F8_E8M0 => Ok(bytes.to_vec()),
            Dtype::F32 => {
                ensure!(
                    bytes.len().is_multiple_of(4),
                    "{scale_name}: F32 scale byte length {} not a multiple of 4",
                    bytes.len()
                );
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| {
                        let bits = u32::from_le_bytes(*c);
                        ensure!(
                            bits & 0x007f_ffff == 0,
                            "{scale_name}: F32 block scale {} is not a power of two; cannot map losslessly to E8M0",
                            f32::from_bits(bits)
                        );
                        Ok(((bits >> 23) & 0xff) as u8)
                    })
                    .collect()
            }
            other => {
                bail!(
                    "{scale_name}: expected F8_E8M0 or F32 power-of-two block scale, got {other:?}"
                )
            }
        }
    }

    pub(crate) fn load_dsv4_block_scaled(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D quantized tensor, got shape {:?}",
            tensor.shape
        );
        let scale_name = name
            .strip_suffix(".weight")
            .map(|prefix| format!("{prefix}.scale"))
            .ok_or_else(|| anyhow!("{name}: quantized DSv4 tensor must end with .weight"))?;
        let scale = self.borrow_raw_tensor(&scale_name)?;
        ensure!(
            scale.shape.len() == 2,
            "{scale_name}: expected 2D scale, got shape {:?}",
            scale.shape
        );
        let scale_e8m0 = Self::dsv4_block_scale_e8m0(&scale_name, scale.dtype, scale.bytes())?;
        let (scale_rows, scale_cols) = (scale.shape[0], scale.shape[1]);

        match tensor.dtype {
            Dtype::F8_E4M3 => {
                let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
                DeviceMatrix::from_dsv4_fp8_block_scaled(
                    ctx,
                    tensor.bytes(),
                    &scale_e8m0,
                    rows,
                    cols,
                    scale_rows,
                    scale_cols,
                )
                .with_context(|| format!("upload DSv4 FP8 matrix {name}"))
            }
            // FAIL-CLOSED: the FP4/MX lane NaNs from the first compressed-attention
            // layer
            // (#137); the FP4 dequant plumbing below stays for a future re-license.
            Dtype::I8 => bail!(
                "{name}: FP4/MX DSv4 checkpoints are unsupported — the compressed-attention \
                 path NaNs (#137). Use the FP8-native export."
            ),
            other => bail!("{name}: unsupported DSv4 block-scaled dtype {other:?}"),
        }
    }

    fn dsv4_scale_shard_for_value_shard(
        name: &str,
        value: &ShardingSpec,
        scale_total: usize,
        block: usize,
    ) -> Result<ShardingSpec> {
        ensure!(block > 0, "{name}: FP8 scale block must be non-zero");
        ensure!(
            value.total.div_ceil(block) == scale_total,
            "{name}: scale total {scale_total} does not match ceil({}/{block})",
            value.total
        );
        ensure!(
            value.offset.is_multiple_of(block) && value.size.is_multiple_of(block),
            "{name}: TP shard {:?} is not aligned to FP8 block size {block}",
            value.range()
        );
        Ok(ShardingSpec {
            offset: value.offset / block,
            size: value.size / block,
            total: scale_total,
        })
    }

    /// Load a DSv4 block-scaled FP8 matrix and apply a TP shard before upload. The FP8
    /// payload
    /// and E8M0 block scales must be sliced together, or the shard reads valid FP8
    /// bytes with
    /// the wrong scale blocks.
    pub(crate) fn load_dsv4_block_scaled_sharded(
        &self,
        ctx: &DeviceContext,
        name: &str,
        shard: Shard,
        tp: &TpConfig,
    ) -> Result<DeviceMatrix> {
        if tp.is_single() || shard == Shard::Replicated {
            return self.load_dsv4_block_scaled(ctx, name);
        }

        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D quantized tensor, got shape {:?}",
            tensor.shape
        );
        let scale_name = name
            .strip_suffix(".weight")
            .map(|prefix| format!("{prefix}.scale"))
            .ok_or_else(|| anyhow!("{name}: quantized DSv4 tensor must end with .weight"))?;
        let scale = self.borrow_raw_tensor(&scale_name)?;
        ensure!(
            scale.shape.len() == 2,
            "{scale_name}: expected 2D scale, got shape {:?}",
            scale.shape
        );
        let scale_e8m0 = Self::dsv4_block_scale_e8m0(&scale_name, scale.dtype, scale.bytes())?;

        match tensor.dtype {
            Dtype::F8_E4M3 => {
                let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
                let (scale_rows, scale_cols) = (scale.shape[0], scale.shape[1]);
                let (weight, scales) = match shard {
                    Shard::Column { dim: 0 } => {
                        let spec = infer_topo::column_shard(rows, tp);
                        let weight = crate::shard_slice::shard_column_parallel(
                            tensor.bytes(),
                            rows,
                            cols,
                            1,
                            &spec,
                        )?;
                        let scale_spec =
                            Self::dsv4_scale_shard_for_value_shard(name, &spec, scale_rows, 128)?;
                        let scales = crate::shard_slice::shard_column_parallel(
                            &scale_e8m0,
                            scale_rows,
                            scale_cols,
                            1,
                            &scale_spec,
                        )?;
                        (weight, scales)
                    }
                    Shard::Row { dim: 1 } => {
                        let spec = infer_topo::row_shard(cols, tp);
                        let weight = crate::shard_slice::shard_row_parallel(
                            tensor.bytes(),
                            rows,
                            cols,
                            1,
                            &spec,
                        )?;
                        let scale_spec =
                            Self::dsv4_scale_shard_for_value_shard(name, &spec, scale_cols, 128)?;
                        let scales = crate::shard_slice::shard_row_parallel(
                            &scale_e8m0,
                            scale_rows,
                            scale_cols,
                            1,
                            &scale_spec,
                        )?;
                        (weight, scales)
                    }
                    Shard::Replicated => unreachable!("replicated handled above"),
                    other => bail!("{name}: unsupported DSv4 FP8 TP shard policy {other:?}"),
                };
                DeviceMatrix::from_dsv4_fp8_block_scaled(
                    ctx,
                    &weight.bytes,
                    &scales.bytes,
                    weight.rows,
                    weight.cols,
                    scales.rows,
                    scales.cols,
                )
                .with_context(|| format!("upload sharded DSv4 FP8 matrix {name}"))
            }
            Dtype::I8 => bail!("{name}: non-replicated DSv4 FP4 TP sharding is not implemented"),
            other => bail!("{name}: unsupported DSv4 block-scaled dtype {other:?}"),
        }
    }

    /// Dense-bf16 dequant copy of one DSv4 FP8 block-scaled tensor, TP-sharded like the
    /// FP8
    /// original. Costs 2× the F8 bytes in VRAM.
    fn load_dsv4_block_scaled_bf16_copy(
        &self,
        ctx: &DeviceContext,
        name: &str,
        shard: Shard,
        tp: &TpConfig,
    ) -> Result<DeviceMatrix> {
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D quantized tensor, got shape {:?}",
            tensor.shape
        );
        let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
        let f32 = self.dequantize_dsv4_block_scaled_to_f32_host(name, rows, cols)?;
        let bytes: Vec<u8> = f32
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        if tp.is_single() || shard == Shard::Replicated {
            return DeviceMatrix::from_safetensors(ctx, &bytes, rows, cols)
                .with_context(|| format!("upload bf16 dequant copy {name}"));
        }
        match shard {
            Shard::Column { dim: 0 } => {
                let spec = infer_topo::column_shard(rows, tp);
                let sharded =
                    crate::shard_slice::shard_column_parallel(&bytes, rows, cols, 2, &spec)?;
                DeviceMatrix::from_safetensors(ctx, &sharded.bytes, sharded.rows, sharded.cols)
                    .with_context(|| format!("upload sharded bf16 dequant copy {name}"))
            }
            other => bail!("{name}: unsupported bf16-copy TP shard policy {other:?}"),
        }
    }

    fn build_dsv4_wo_a_group_tables(
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
        rows_per_group: usize,
    ) -> Result<crate::dsv4::Dsv4WoAGroupTables> {
        ensure!(
            rows_per_group > 0 && weight.rows.is_multiple_of(rows_per_group),
            "DSv4 wo_a grouped table needs rows {} divisible by rows_per_group {}",
            weight.rows,
            rows_per_group
        );
        let groups = weight.rows / rows_per_group;
        ensure!(groups > 0, "DSv4 wo_a grouped table has zero groups");
        // Dense BF16 wo_a: the grouped route-GEMV kernel is FP8/FP4-only, so these
        // tables are
        // consumed only when groups>1, which the dense path rejects. Carry the shape
        // and build
        // trivial base-pointer tables so the struct stays well-formed.
        if weight.weight_format == WeightFormat::DenseBf16 {
            let (base, _bg) = weight.data.device_ptr(&ctx.stream);
            let stride_bytes = rows_per_group
                .checked_mul(weight.cols)
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| anyhow!("DSv4 dense wo_a group stride overflow"))?;
            let ptrs: Vec<u64> = (0..groups)
                .map(|g| base + (g * stride_bytes) as u64)
                .collect();
            return Ok(crate::dsv4::Dsv4WoAGroupTables {
                weight_ptrs: ctx
                    .stream
                    .clone_htod(&ptrs)
                    .map_err(|e| anyhow!("DSv4 dense wo_a ptr table H2D failed: {e}"))?,
                scale_ptrs: ctx
                    .stream
                    .clone_htod(&ptrs)
                    .map_err(|e| anyhow!("DSv4 dense wo_a scale ptr table H2D failed: {e}"))?,
                groups,
                rows_per_group,
                cols_per_group: weight.cols,
                scale_rows_per_group: 0,
                scale_cols: 0,
            });
        }
        ensure!(
            weight.dsv4_scale_rows > 0
                && weight.dsv4_scale_cols > 0
                && weight.dsv4_scale_rows.is_multiple_of(groups),
            "DSv4 wo_a scale rows {} must be non-zero and divisible by groups {groups}",
            weight.dsv4_scale_rows
        );
        let scale_rows_per_group = weight.dsv4_scale_rows / groups;
        let qweight = weight
            .qweight
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 wo_a grouped table missing quantized weight bytes"))?;
        let scales = weight
            .dsv4_scales
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 wo_a grouped table missing E8M0 scales"))?;
        let weight_stride_bytes = match weight.weight_format {
            WeightFormat::Dsv4Fp8BlockScaled => rows_per_group.checked_mul(weight.cols),
            WeightFormat::Dsv4Fp4BlockScaled => {
                ensure!(
                    weight.cols.is_multiple_of(2),
                    "DSv4 wo_a FP4 grouped table needs even cols, got {}",
                    weight.cols
                );
                rows_per_group.checked_mul(weight.cols / 2)
            }
            other => bail!("DSv4 wo_a grouped table expected FP8/FP4 block-scaled, got {other:?}"),
        }
        .ok_or_else(|| {
            anyhow!(
                "DSv4 wo_a grouped table weight stride overflow rows_per_group={} cols={}",
                rows_per_group,
                weight.cols
            )
        })?;
        let scale_stride_bytes = scale_rows_per_group
            .checked_mul(weight.dsv4_scale_cols)
            .ok_or_else(|| {
                anyhow!(
                    "DSv4 wo_a grouped table scale stride overflow scale_rows={} scale_cols={}",
                    scale_rows_per_group,
                    weight.dsv4_scale_cols
                )
            })?;
        ensure!(
            qweight.len() == weight_stride_bytes * groups,
            "DSv4 wo_a grouped qweight len {} != groups({groups})*stride({weight_stride_bytes})",
            qweight.len()
        );
        ensure!(
            scales.len() == scale_stride_bytes * groups,
            "DSv4 wo_a grouped scale len {} != groups({groups})*stride({scale_stride_bytes})",
            scales.len()
        );

        let (weight_base, _wg) = qweight.device_ptr(&ctx.stream);
        let (scale_base, _sg) = scales.device_ptr(&ctx.stream);
        let weight_ptrs: Vec<u64> = (0..groups)
            .map(|g| weight_base + (g * weight_stride_bytes) as u64)
            .collect();
        let scale_ptrs: Vec<u64> = (0..groups)
            .map(|g| scale_base + (g * scale_stride_bytes) as u64)
            .collect();
        Ok(crate::dsv4::Dsv4WoAGroupTables {
            weight_ptrs: ctx
                .stream
                .clone_htod(&weight_ptrs)
                .map_err(|e| anyhow!("DSv4 wo_a weight ptr table H2D failed: {e}"))?,
            scale_ptrs: ctx
                .stream
                .clone_htod(&scale_ptrs)
                .map_err(|e| anyhow!("DSv4 wo_a scale ptr table H2D failed: {e}"))?,
            groups,
            rows_per_group,
            cols_per_group: weight.cols,
            scale_rows_per_group,
            scale_cols: weight.dsv4_scale_cols,
        })
    }

    /// Build the per-rank DSv4 MoE layer (FP8 DeepGEMM expert caches + router).
    /// Bias-routed layers load `gate.bias`; hash-routed layers load the host
    /// `gate.tid2eid` table instead (and skip the bias).
    pub(crate) fn load_dsv4_moe_layer(
        &self,
        ctx: &DeviceContext,
        names: &deepseek_spec::DeepSeekV4MoeTensorNames,
        split: &crate::moe_config::ExpertSplit,
        routing_kind: deepseek_spec::DeepSeekV4MoeRoutingKind,
        // GLM (`weight_scale_inv` FP8) ⇒ re-encode experts into the DSv4 FP8+E8M0
        // layout the grouped DeepGEMM cache consumes; DSv4 loads E8M0 directly.
        glm: bool,
    ) -> Result<crate::dsv4::Dsv4MoeLayer> {
        use cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache;
        use deepseek_spec::DeepSeekV4MoeRoutingKind;

        let mega_moe = matches!(
            crate::runtime_flags::dsv4_moe_transport()?,
            crate::runtime_flags::Dsv4MoeTransport::MegaMoe
        );

        // DSv4 ships FP8 E4M3 + E8M0 `<prefix>.scale`; GLM ships FP8 E4M3 + F32
        // `weight_scale_inv` (128×128 block scales), consumed losslessly by the 1D2D
        // `sfb`
        // path — no E8M0 re-encode, no dequant. Both ride the SAME
        // `build_grouped_cache`.
        let build_w13 =
            |first: &DeviceMatrix, second: &DeviceMatrix| -> Result<Dsv4Fp8DeepGemmWeightCache> {
                if glm {
                    Dsv4Fp8DeepGemmWeightCache::from_fp8_block_scaled_weight_pair_rows(
                        ctx, first, second,
                    )
                } else {
                    Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight_pair_rows(ctx, first, second)
                }
            };
        let build_w2 = |down: &DeviceMatrix| -> Result<Dsv4Fp8DeepGemmWeightCache> {
            if glm {
                Dsv4Fp8DeepGemmWeightCache::from_fp8_block_scaled_weight(ctx, down)
            } else {
                Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(ctx, down)
            }
        };
        let load_fp8 = |name: &str| -> Result<DeviceMatrix> {
            if glm {
                self.load_dsv4_glm_fp8_as_block_scaled(ctx, name)
            } else {
                self.load_dsv4_block_scaled(ctx, name)
            }
        };

        // Detect W4A16 / W4AFP8: the first routed expert's w1 carries the quant view.
        let first_expert = names.expert(split.local_expert_start);
        let first_view = self.quant_view_for_dsv4(&first_expert.w1)?;
        let is_w4a16 = first_view
            .as_ref()
            .is_some_and(|v| matches!(v.format, QuantFormat::W4A16 { .. }));
        let is_w4afp8 = first_view
            .as_ref()
            .is_some_and(|v| matches!(v.format, QuantFormat::W4Afp8));
        // NVFP4 (0731 checkpoint): E2M1 packed weight (I8/U8) + F8_E8M0 `.scale`
        // block scales. Converted to W4AFP8 on GPU at load time. FP8 E4M3 + E8M0
        // block scales is the standard DSv4 FP8 format — the scale dtype alone
        // cannot distinguish them; the weight dtype must be packed.
        let is_nvfp4 = !is_w4a16 && !is_w4afp8 && {
            let base = first_expert.w1.trim_end_matches(".weight");
            let scale_name = format!("{base}.scale");
            let headers = self.tensor_headers().ok();
            let scale_is_e8m0 = headers
                .as_ref()
                .and_then(|h| h.get(&scale_name).map(|t| t.dtype == Dtype::F8_E8M0))
                .unwrap_or(false);
            let weight_is_packed = headers
                .as_ref()
                .and_then(|h| {
                    h.get(&first_expert.w1)
                        .map(|t| matches!(t.dtype, Dtype::I8 | Dtype::U8))
                })
                .unwrap_or(false);
            scale_is_e8m0 && weight_is_packed
        };

        let (w13_grouped, w2_grouped, w13_w4a16, w2_w4a16, hidden_dim, intermediate, num_groups) =
            if is_w4a16 {
                // W4A16 path: per-expert packed INT4 + BF16 group scales.
                let mut w13 = Vec::with_capacity(split.experts_per_rank);
                let mut w2 = Vec::with_capacity(split.experts_per_rank);
                for e in split.local_expert_start..split.local_expert_end() {
                    let expert = names.expert(e);
                    let w1 = self.load_matrix_quant_aware(ctx, &expert.w1)?;
                    let w3 = self.load_matrix_quant_aware(ctx, &expert.w3)?;
                    w13.push(DeviceMatrix::fuse_rows(ctx, &w1, &w3)?);
                    w2.push(self.load_matrix_quant_aware(ctx, &expert.w2)?);
                }
                let first_w13 = w13
                    .first()
                    .ok_or_else(|| anyhow!("DSv4 MoE layer has no local experts"))?;
                let first_w2 = w2
                    .first()
                    .ok_or_else(|| anyhow!("DSv4 MoE layer has no local down experts"))?;
                let hidden_dim = first_w13.cols;
                let intermediate = first_w2.cols;
                ensure!(
                    first_w13.rows == 2 * intermediate,
                    "DSv4 W4A16 w13 rows {} != 2*intermediate {}",
                    first_w13.rows,
                    2 * intermediate
                );
                ensure!(
                    first_w2.rows == hidden_dim,
                    "DSv4 W4A16 w2 rows {} != hidden_dim {hidden_dim}",
                    first_w2.rows
                );
                let num_groups = w13.len();
                ensure!(
                    num_groups == split.experts_per_rank && w2.len() == num_groups,
                    "DSv4 W4A16 expert count mismatch: w13={} w2={} expected {}",
                    num_groups,
                    w2.len(),
                    split.experts_per_rank
                );
                (
                    None,
                    None,
                    Some(w13),
                    Some(w2),
                    hidden_dim,
                    intermediate,
                    num_groups,
                )
            } else if is_w4afp8 {
                // W4AFP8: raw bytes loaded below (SGLang CUTLASS layout); the tuple
                // only carries the dims the layer struct needs.
                let view = first_view.as_ref().expect("W4AFP8 view");
                let hidden_dim = view.logical_shape[1];
                let intermediate = view.logical_shape[0];
                (
                    None,
                    None,
                    None,
                    None,
                    hidden_dim,
                    intermediate,
                    split.experts_per_rank,
                )
            } else if is_nvfp4 {
                // NVFP4: E2M1+E8M0 converted to W4AFP8 on GPU below; the tuple
                // only carries the dims the layer struct needs.
                let w1 = self.load_raw_tensor(&first_expert.w1)?;
                let intermediate = w1.shape[0];
                let hidden_dim = w1.shape[1] * 2;
                (
                    None,
                    None,
                    None,
                    None,
                    hidden_dim,
                    intermediate,
                    split.experts_per_rank,
                )
            } else {
                // FP8 path: per-expert FP8 caches → grouped DeepGEMM caches.
                let mut w13 = Vec::with_capacity(split.experts_per_rank);
                let mut w2 = Vec::with_capacity(split.experts_per_rank);
                for e in split.local_expert_start..split.local_expert_end() {
                    let expert = names.expert(e);
                    let w1 = load_fp8(&expert.w1)?;
                    let w3 = load_fp8(&expert.w3)?;
                    w13.push(build_w13(&w1, &w3)?);
                    let down = load_fp8(&expert.w2)?;
                    w2.push(build_w2(&down)?);
                }
                let first_w13 = w13
                    .first()
                    .ok_or_else(|| anyhow!("DSv4 MoE layer has no local experts"))?;
                let first_w2 = w2
                    .first()
                    .ok_or_else(|| anyhow!("DSv4 MoE layer has no local down experts"))?;
                let hidden_dim = first_w13.cols;
                let intermediate = first_w2.cols;
                ensure!(
                    first_w13.rows == 2 * intermediate,
                    "DSv4 grouped w13 rows {} != 2*intermediate {}",
                    first_w13.rows,
                    2 * intermediate
                );
                ensure!(
                    first_w2.rows == hidden_dim,
                    "DSv4 grouped w2 rows {} != hidden_dim {hidden_dim}",
                    first_w2.rows
                );
                let w13_layout = if mega_moe {
                    crate::moe::GroupedWeightLayout::InterleavedL1
                } else {
                    crate::moe::GroupedWeightLayout::Normal
                };
                let w13_grouped = crate::moe::build_grouped_cache(
                    ctx,
                    w13.as_slice(),
                    2 * intermediate,
                    hidden_dim,
                    w13_layout,
                )?;
                let w2_grouped = crate::moe::build_grouped_cache(
                    ctx,
                    w2.as_slice(),
                    hidden_dim,
                    intermediate,
                    crate::moe::GroupedWeightLayout::Normal,
                )?;
                let num_groups = w13_grouped.groups;
                ensure!(
                    num_groups == split.experts_per_rank && w2_grouped.groups == num_groups,
                    "DSv4 grouped expert count mismatch: w13={} w2={} expected {}",
                    w13_grouped.groups,
                    w2_grouped.groups,
                    split.experts_per_rank
                );
                (
                    Some(w13_grouped),
                    Some(w2_grouped),
                    None,
                    None,
                    hidden_dim,
                    intermediate,
                    num_groups,
                )
            };

        let gate = self.load_dsv4_bf16_matrix(ctx, &names.gate_weight)?;
        let (gate_bias, hash_tid2eid_device) = match routing_kind {
            DeepSeekV4MoeRoutingKind::LearnedBias => {
                let bias_name = names
                    .gate_bias
                    .as_ref()
                    .ok_or_else(|| anyhow!("DSv4 bias-routed MoE layer missing gate.bias"))?;
                (Some(self.load_dsv4_vec(ctx, bias_name)?), None)
            }
            DeepSeekV4MoeRoutingKind::Hash => {
                let tid_name = names
                    .gate_tid2eid
                    .as_ref()
                    .ok_or_else(|| anyhow!("DSv4 hash-routed MoE layer missing gate.tid2eid"))?;
                let table = self.load_dsv4_i64_host(tid_name)?;
                let device = ctx
                    .stream
                    .clone_htod(&table)
                    .map_err(|e| anyhow!("DSv4 tid2eid H2D failed for {tid_name}: {e}"))?;
                (None, Some(device))
            }
        };

        // SHARED expert (always-on, n_shared_experts == 1); same builders as the routed
        // ones.
        let shared = names
            .shared_experts
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 expects an always-on shared expert"))?;
        let shared_w1 = load_fp8(&shared.w1)?;
        let shared_w3 = load_fp8(&shared.w3)?;
        let shared_w13 = build_w13(&shared_w1, &shared_w3)?;
        let shared_down = load_fp8(&shared.w2)?;
        let shared_w2 = build_w2(&shared_down)?;

        // W4AFP8 routed experts: load raw I8 weight + BF16 scale bytes, fuse
        // w1+w3 rows, stack experts, upload.
        let (w13_w4afp8, w2_w4afp8) = if is_w4afp8 {
            let scale_name = |w: &str| w.trim_end_matches(".weight").to_string() + ".weight_scale";
            let mut w13_w = Vec::new();
            let mut w13_s = Vec::new();
            let mut w2_w = Vec::new();
            let mut w2_s = Vec::new();
            for e in split.local_expert_start..split.local_expert_end() {
                let expert = names.expert(e);
                let w1 = self.load_raw_tensor(&expert.w1)?;
                let w3 = self.load_raw_tensor(&expert.w3)?;
                let w1s = self.load_raw_tensor(&scale_name(&expert.w1))?;
                let w3s = self.load_raw_tensor(&scale_name(&expert.w3))?;
                // w13 weight: row-major concat along axis 0 = byte concat.
                w13_w.extend_from_slice(&w1.bytes);
                w13_w.extend_from_slice(&w3.bytes);
                // w13 scales: concat along axis 1 — interleave rows.
                let row_bytes = w1s.bytes.len() / w1s.shape[0];
                let rows = w1s.shape[0];
                w13_s.reserve(rows * 2 * row_bytes);
                for r in 0..rows {
                    let start = r * row_bytes;
                    w13_s.extend_from_slice(&w1s.bytes[start..start + row_bytes]);
                    w13_s.extend_from_slice(&w3s.bytes[start..start + row_bytes]);
                }
                let w2 = self.load_raw_tensor(&expert.w2)?;
                let w2s = self.load_raw_tensor(&scale_name(&expert.w2))?;
                w2_w.extend_from_slice(&w2.bytes);
                w2_s.extend_from_slice(&w2s.bytes);
            }
            let w13_dev = ctx
                .stream
                .clone_htod(&w13_w)
                .map_err(|e| anyhow!("DSv4 W4AFP8 w13 weight upload failed: {e}"))?;
            let w13s_dev = ctx
                .stream
                .clone_htod(&w13_s)
                .map_err(|e| anyhow!("DSv4 W4AFP8 w13 scales upload failed: {e}"))?;
            let w2_dev = ctx
                .stream
                .clone_htod(&w2_w)
                .map_err(|e| anyhow!("DSv4 W4AFP8 w2 weight upload failed: {e}"))?;
            let w2s_dev = ctx
                .stream
                .clone_htod(&w2_s)
                .map_err(|e| anyhow!("DSv4 W4AFP8 w2 scales upload failed: {e}"))?;
            (
                Some(crate::moe::W4Afp8ExpertWeights {
                    weight: w13_dev,
                    scales: w13s_dev,
                    num_experts: split.experts_per_rank,
                    n: 2 * intermediate,
                    k: hidden_dim,
                }),
                Some(crate::moe::W4Afp8ExpertWeights {
                    weight: w2_dev,
                    scales: w2s_dev,
                    num_experts: split.experts_per_rank,
                    n: hidden_dim,
                    k: intermediate,
                }),
            )
        } else if is_nvfp4 {
            // NVFP4 → W4AFP8: convert E2M1+E8M0 to INT4+BF16 on GPU per expert,
            // download, fuse w1+w3 on host, upload.
            let e8m0_name = |w: &str| w.trim_end_matches(".weight").to_string() + ".scale";
            let convert = |w_name: &str| -> Result<W4Afp8Converted> {
                let weight = self.load_raw_tensor(w_name)?;
                let scale = self.load_raw_tensor(&e8m0_name(w_name))?;
                if weight.shape.len() != 2 {
                    bail!(
                        "{w_name}: expected 2D weight, got {} dims",
                        weight.shape.len()
                    );
                }
                let n = weight.shape[0];
                let k = weight.shape[1] * 2;
                let scale_rows = k / 512;
                let src_w = ctx
                    .stream
                    .clone_htod(&weight.bytes)
                    .map_err(|e| anyhow!("NVFP4 src weight upload failed: {e}"))?;
                let src_s = ctx
                    .stream
                    .clone_htod(&scale.bytes)
                    .map_err(|e| anyhow!("NVFP4 src scale upload failed: {e}"))?;
                let dst_w = ctx
                    .stream
                    .alloc_zeros::<u8>(n * (k / 2))
                    .map_err(|e| anyhow!("NVFP4 dst weight alloc failed: {e}"))?;
                let dst_s = ctx
                    .stream
                    .alloc_zeros::<u8>(scale_rows * n * 4 * 2)
                    .map_err(|e| anyhow!("NVFP4 dst scale alloc failed: {e}"))?;
                // SAFETY: all four buffers are live device allocations on `ctx.stream`,
                // sized from the tensor shape (`n*k/2` packed weights, `scale_rows*n*8` scales).
                unsafe {
                    cuda_kernels::moe::nvfp4_to_w4afp8(
                        cuda_kernels::tensor::cache_ptr(&src_w, ctx).cast::<i8>(),
                        cuda_kernels::tensor::cache_ptr(&src_s, ctx),
                        cuda_kernels::tensor::cache_ptr(&dst_w, ctx).cast::<i8>(),
                        cuda_kernels::tensor::cache_ptr(&dst_s, ctx),
                        n,
                        k,
                        ctx.stream.cu_stream(),
                    )?;
                }
                let w_host = ctx
                    .stream
                    .clone_dtoh(&dst_w)
                    .map_err(|e| anyhow!("NVFP4 dst weight download failed: {e}"))?;
                let s_host = ctx
                    .stream
                    .clone_dtoh(&dst_s)
                    .map_err(|e| anyhow!("NVFP4 dst scale download failed: {e}"))?;
                Ok((w_host, s_host, scale_rows, n, k))
            };
            let mut w13_w = Vec::new();
            let mut w13_s = Vec::new();
            let mut w2_w = Vec::new();
            let mut w2_s = Vec::new();
            // All experts must share the same w1/w3 and w2 shapes — the fused
            // w13 buffer is read with a uniform stride by the grouped GEMM.
            let mut expect_w1: Option<(usize, usize)> = None;
            let mut expect_w2: Option<(usize, usize)> = None;
            for e in split.local_expert_start..split.local_expert_end() {
                let expert = names.expert(e);
                let (w1_cw, w1_cs, w1_rows, w1_n, w1_k) = convert(&expert.w1)?;
                let (w3_cw, w3_cs, _, w3_n, w3_k) = convert(&expert.w3)?;
                if (w1_n, w1_k) != (w3_n, w3_k) {
                    bail!("expert {e}: w1 [{w1_n},{w1_k}] != w3 [{w3_n},{w3_k}]");
                }
                if let Some((en, ek)) = expect_w1 {
                    if (w1_n, w1_k) != (en, ek) {
                        bail!("expert {e}: w1 [{w1_n},{w1_k}] != expert 0 [{en},{ek}]");
                    }
                } else {
                    expect_w1 = Some((w1_n, w1_k));
                }
                w13_w.extend_from_slice(&w1_cw);
                w13_w.extend_from_slice(&w3_cw);
                let row_bytes = w1_cs.len() / w1_rows;
                w13_s.reserve(w1_rows * 2 * row_bytes);
                for r in 0..w1_rows {
                    let start = r * row_bytes;
                    w13_s.extend_from_slice(&w1_cs[start..start + row_bytes]);
                    w13_s.extend_from_slice(&w3_cs[start..start + row_bytes]);
                }
                let (w2_cw, w2_cs, _, w2_n, w2_k) = convert(&expert.w2)?;
                if let Some((en, ek)) = expect_w2 {
                    if (w2_n, w2_k) != (en, ek) {
                        bail!("expert {e}: w2 [{w2_n},{w2_k}] != expert 0 [{en},{ek}]");
                    }
                } else {
                    expect_w2 = Some((w2_n, w2_k));
                }
                w2_w.extend_from_slice(&w2_cw);
                w2_s.extend_from_slice(&w2_cs);
            }
            let w13_dev = ctx
                .stream
                .clone_htod(&w13_w)
                .map_err(|e| anyhow!("DSv4 W4AFP8 w13 weight upload failed: {e}"))?;
            let w13s_dev = ctx
                .stream
                .clone_htod(&w13_s)
                .map_err(|e| anyhow!("DSv4 W4AFP8 w13 scales upload failed: {e}"))?;
            let w2_dev = ctx
                .stream
                .clone_htod(&w2_w)
                .map_err(|e| anyhow!("DSv4 W4AFP8 w2 weight upload failed: {e}"))?;
            let w2s_dev = ctx
                .stream
                .clone_htod(&w2_s)
                .map_err(|e| anyhow!("DSv4 W4AFP8 w2 scales upload failed: {e}"))?;
            (
                Some(crate::moe::W4Afp8ExpertWeights {
                    weight: w13_dev,
                    scales: w13s_dev,
                    num_experts: split.experts_per_rank,
                    n: 2 * intermediate,
                    k: hidden_dim,
                }),
                Some(crate::moe::W4Afp8ExpertWeights {
                    weight: w2_dev,
                    scales: w2s_dev,
                    num_experts: split.experts_per_rank,
                    n: hidden_dim,
                    k: intermediate,
                }),
            )
        } else {
            (None, None)
        };

        Ok(crate::dsv4::Dsv4MoeLayer {
            w13_grouped,
            w2_grouped,
            w13_w4a16,
            w2_w4a16,
            w13_w4afp8,
            w2_w4afp8,
            num_groups,
            hidden_dim,
            intermediate,
            gate,
            gate_bias,
            hash_tid2eid_device,
            routing_kind,
            shared_w13,
            shared_w2,
            gemv_tables: std::sync::OnceLock::new(),
            w4a16_gemv_tables: std::sync::OnceLock::new(),
            w4afp8_gemv_tables: std::sync::OnceLock::new(),
        })
    }

    /// Load a GLM dense-MLP layer (`first_k_dense_replace` layers). GLM ships FP8 + F32
    /// `weight_scale_inv`; each projection is dequantized to dense bf16 because the FP8
    /// grouped caches need E8M0 scales GLM does not carry.
    pub(crate) fn load_dsv4_dense_mlp(
        &self,
        ctx: &DeviceContext,
        names: &deepseek_spec::DeepSeekV4ExpertTensorNames,
    ) -> Result<crate::dsv4::Dsv4DenseMlp> {
        // w1 = gate_proj, w3 = up_proj, w2 = down_proj.
        let gate = self.load_dsv4_block_scaled_dialect(ctx, &names.w1)?;
        let up = self.load_dsv4_block_scaled_dialect(ctx, &names.w3)?;
        let down = self.load_dsv4_block_scaled_dialect(ctx, &names.w2)?;
        let hidden_dim = gate.cols;
        let intermediate = gate.rows;
        ensure!(
            up.rows == intermediate && up.cols == hidden_dim,
            "GLM dense up_proj shape {}x{} != gate {}x{}",
            up.rows,
            up.cols,
            intermediate,
            hidden_dim
        );
        ensure!(
            down.rows == hidden_dim && down.cols == intermediate,
            "GLM dense down_proj shape {}x{} != [{hidden_dim}, {intermediate}]",
            down.rows,
            down.cols
        );
        Ok(crate::dsv4::Dsv4DenseMlp {
            gate,
            up,
            down,
            hidden_dim,
            intermediate,
        })
    }

    /// Load a DSv4 1D `i64` hash-routing table (`gate.tid2eid`) into host memory.
    pub(crate) fn load_dsv4_i64_host(&self, name: &str) -> Result<Vec<i64>> {
        use safetensors::tensor::Dtype;
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.dtype == Dtype::I64,
            "{name}: DSv4 tid2eid expected I64, got {:?}",
            tensor.dtype
        );
        ensure!(
            tensor.bytes().len() % 8 == 0,
            "{name}: I64 byte length {} is not a multiple of 8",
            tensor.bytes().len()
        );
        Ok(tensor
            .bytes()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect())
    }

    /// Load one DSv4 hyper-connection block (`base` bf16 vec, `mix_fn` matrix —
    /// bf16 or FP8/FP4 block-scaled, `scale` bf16 vec).
    pub(crate) fn load_dsv4_hyper_connection(
        &self,
        ctx: &DeviceContext,
        names: &deepseek_spec::DeepSeekV4HyperConnectionTensorNames,
    ) -> Result<crate::dsv4::Dsv4HyperConnection> {
        Ok(crate::dsv4::Dsv4HyperConnection {
            base: self.load_dsv4_vec(ctx, &names.base)?,
            mix_fn: self.load_dsv4_global_matrix(ctx, &names.mix_fn)?,
            scale: self.load_dsv4_vec(ctx, &names.scale)?,
        })
    }

    /// Load a DSv4 2D matrix dispatching on its on-disk dtype: BF16/F32 are quantized
    /// to FP8
    /// E4M3FN block-scaled (128×128 blocks, E8M0 scales) on the host so downstream
    /// linears
    /// route through `mla_linear`; F8_E4M3/I8 keep the native block-scaled path.
    pub(crate) fn load_dsv4_global_matrix(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D tensor, got shape {:?}",
            tensor.shape
        );
        let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
        match tensor.dtype {
            Dtype::BF16 | Dtype::F32 => {
                let (fp8, scales, scale_rows, scale_cols) = Self::quantize_to_dsv4_fp8_host(
                    name,
                    tensor.dtype,
                    tensor.bytes(),
                    rows,
                    cols,
                )?;
                DeviceMatrix::from_dsv4_fp8_block_scaled(
                    ctx, &fp8, &scales, rows, cols, scale_rows, scale_cols,
                )
                .with_context(|| format!("upload quantized DSv4 FP8 matrix {name}"))
            }
            Dtype::F8_E4M3 | Dtype::I8 => self.load_dsv4_block_scaled(ctx, name),
            other => bail!("{name}: unsupported DSv4 global matrix dtype {other:?}"),
        }
    }

    /// Build one DSv4 MLA attention block. The Q/KV/O LoRA matrices are FP8/FP4
    /// block-scaled;
    /// `compressor` / `indexer` matrices may be FP8/FP4 or bf16, so they route through
    /// the
    /// dtype-dispatching [`Self::load_dsv4_global_matrix`].
    pub(crate) fn load_dsv4_attention(
        &self,
        ctx: &DeviceContext,
        config: &deepseek_spec::DeepSeekV4Config,
        names: &deepseek_spec::DeepSeekV4AttentionTensorNames,
        tp: &TpConfig,
    ) -> Result<crate::dsv4::Dsv4Attention> {
        // GLM (`plain_o_proj`) ships no `attn_sink` — its MLA has no per-head sink
        // logit.
        let (attn_sink, attn_sink_f32) = if config.plain_o_proj {
            (None, None)
        } else {
            let attn_sink = self.load_dsv4_vec(ctx, &names.attn_sink)?;
            let mut dst = ctx
                .stream
                .alloc_zeros::<f32>(attn_sink.len)
                .map_err(|e| anyhow!("DSv4 attn_sink f32 mirror alloc failed: {e}"))?;
            cuda_kernels::tensor_ops::bf16_to_f32(ctx, &attn_sink.data, &mut dst, attn_sink.len)
                .map_err(|e| anyhow!("DSv4 attn_sink bf16->f32 mirror failed: {e}"))?;
            (Some(attn_sink), Some(dst))
        };
        // GLM (`plain_o_proj`) ships FP8 + F32 `weight_scale_inv`: dequant the Q/KV
        // projections
        // to dense bf16, since the FP8 DeepGEMM caches below need the E8M0
        // `dsv4_scales`
        // layout.
        let glm = config.plain_o_proj;
        let (wq_a, wkv) = if glm {
            (
                self.load_dsv4_block_scaled_dialect(ctx, &names.wq_a)?,
                self.load_dsv4_block_scaled_dialect(ctx, &names.wkv)?,
            )
        } else {
            (
                self.load_dsv4_block_scaled(ctx, &names.wq_a)?,
                self.load_dsv4_block_scaled(ctx, &names.wkv)?,
            )
        };
        let wqkv_a_deepgemm = if !glm && crate::attention::dsv4_fused_wqkv_decode_enabled()? {
            Some(
                cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight_pair_rows(
                    ctx, &wq_a, &wkv,
                )?,
            )
        } else {
            None
        };
        let wq_b = if glm {
            self.load_dsv4_block_scaled_dialect(ctx, &names.wq_b)?
        } else {
            self.load_dsv4_block_scaled_sharded(
                ctx,
                &names.wq_b,
                names
                    .shard_for(config, &names.wq_b, tp.world_size)
                    .unwrap_or(Shard::Replicated),
                tp,
            )?
        };
        let wq_b_deepgemm = self.decode_proj_cache(ctx, &wq_b)?;
        // Output projection. DSv4: low-rank wo_a→wo_b (+ per-group tables). GLM
        // (`plain_o_proj`): a single plain `o_proj`, plus the kv_b absorption split
        // (w_kc/w_vc).
        let (
            wo_a,
            wo_a_groups,
            wo_b,
            wo_a_deepgemm,
            wo_a_group_deepgemm,
            wo_b_deepgemm,
            o_proj,
            w_kc,
            w_vc,
        ) = if config.plain_o_proj {
            let o_proj = self.load_dsv4_block_scaled_dialect(
                ctx,
                names
                    .o_proj
                    .as_ref()
                    .ok_or_else(|| anyhow!("GLM plain_o_proj layer missing o_proj name"))?,
            )?;
            let (w_kc, w_vc) = self.load_dsv4_kv_b_absorb(ctx, config, names, tp)?;
            (
                None,
                None,
                None,
                None,
                None,
                None,
                Some(o_proj),
                Some(w_kc),
                Some(w_vc),
            )
        } else {
            // Some FP8 re-serializations leave the tiny low-rank `wo_a` as dense BF16.
            // Route by
            // dtype: block-scaled keeps the grouped route-GEMV + DeepGEMM levers; dense
            // BF16
            // rides `dsv4_linear` (gemm_batch). `wo_b` stays FP8.
            let wo_a_shard = names
                .shard_for(config, &names.wo_a, tp.world_size)
                .unwrap_or(Shard::Replicated);
            let wo_a = match self.borrow_raw_tensor(&names.wo_a)?.dtype {
                Dtype::F8_E4M3 | Dtype::I8 => {
                    self.load_dsv4_block_scaled_sharded(ctx, &names.wo_a, wo_a_shard, tp)?
                }
                _ => self.load_dsv4_bf16_sharded(ctx, &names.wo_a, wo_a_shard, tp)?,
            };
            let wo_a_is_block_scaled = matches!(
                wo_a.weight_format,
                WeightFormat::Dsv4Fp8BlockScaled | WeightFormat::Dsv4Fp4BlockScaled
            );
            let wo_a_groups = Self::build_dsv4_wo_a_group_tables(ctx, &wo_a, config.o_lora_rank)?;
            let wo_b = self.load_dsv4_block_scaled_sharded(
                ctx,
                &names.wo_b,
                names
                    .shard_for(config, &names.wo_b, tp.world_size)
                    .unwrap_or(Shard::Replicated),
                tp,
            )?;
            // DeepGEMM caches for the decode output projection. `wo_b` is always FP8;
            // `wo_a`
            // caches only when it is itself block-scaled (dense BF16 uses gemm_batch).
            let decode_alloc = crate::attention::dsv4_fused_wqkv_decode_enabled()?;
            let (wo_a_deepgemm, wo_a_group_deepgemm) = if decode_alloc && wo_a_is_block_scaled {
                let group_caches = if wo_a_groups.groups > 1 {
                    let mut caches = Vec::with_capacity(wo_a_groups.groups);
                    for group in 0..wo_a_groups.groups {
                        caches.push(
                            cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight_row_range(
                                ctx,
                                &wo_a,
                                group * wo_a_groups.rows_per_group,
                                wo_a_groups.rows_per_group,
                            )?,
                        );
                    }
                    Some(caches)
                } else {
                    None
                };
                let flat = (wo_a_groups.groups == 1)
                    .then(|| {
                        cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(
                            ctx, &wo_a,
                        )
                    })
                    .transpose()?;
                (flat, group_caches)
            } else {
                (None, None)
            };
            let wo_b_deepgemm = if decode_alloc {
                Some(
                    cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(ctx, &wo_b)?,
                )
            } else {
                None
            };
            (
                Some(wo_a),
                Some(wo_a_groups),
                Some(wo_b),
                wo_a_deepgemm,
                wo_a_group_deepgemm,
                wo_b_deepgemm,
                None,
                None,
                None,
            )
        };
        Ok(crate::dsv4::Dsv4Attention {
            wq_a,
            wqkv_a_deepgemm,
            q_norm: self.load_dsv4_vec(ctx, &names.q_norm)?,
            wq_b,
            wq_b_deepgemm,
            wkv,
            kv_norm: self.load_dsv4_vec(ctx, &names.kv_norm)?,
            wo_a,
            wo_a_groups,
            wo_b,
            wo_a_deepgemm,
            wo_a_group_deepgemm,
            wo_b_deepgemm,
            attn_sink,
            attn_sink_f32,
            compressor: names
                .compressor
                .as_ref()
                .map(|c| self.load_dsv4_compressor(ctx, c, glm))
                .transpose()?,
            indexer: names
                .indexer
                .as_ref()
                .map(|i| self.load_dsv4_indexer(ctx, config, i))
                .transpose()?,
            o_proj,
            w_kc,
            w_vc,
        })
    }

    /// GLM `kv_b` absorption split (SGLang `deepseek_weight_loader.py:567-590`, the
    /// non-`use_deep_gemm_bmm` path).
    ///
    /// `kv_b_proj.weight` is `[num_heads*(qk_nope_head_dim + v_head_dim),
    /// kv_lora_rank]`, FP8
    /// block-scaled. Dequantize to bf16, split per head, and emit BOTH halves in
    /// `gemm_batch`
    /// orientation `[out, in]`: `w_kc` is the per-head transpose → `[num_heads*kv_lora,
    /// qk_nope]`, `w_vc` is as-is → `[num_heads*v_head, kv_lora]`. SGLang's `bmm`
    /// orientation
    /// is transposed from this because ARLE's `gemm_batch` computes `weight·x`, not
    /// `x·weight^T`; the contraction is identical.
    pub(crate) fn load_dsv4_kv_b_absorb(
        &self,
        ctx: &DeviceContext,
        config: &deepseek_spec::DeepSeekV4Config,
        names: &deepseek_spec::DeepSeekV4AttentionTensorNames,
        tp: &TpConfig,
    ) -> Result<(DeviceMatrix, DeviceMatrix)> {
        let kv_b_name = names
            .kv_b_proj
            .as_ref()
            .ok_or_else(|| anyhow!("GLM attention missing kv_b_proj name"))?;
        let qk_nope = config.qk_nope_head_dim;
        let v_head = config.v_head_dim;
        let kv_lora = config.kv_lora_rank;
        let num_heads = config.num_attention_heads;
        ensure!(
            qk_nope > 0 && v_head > 0 && kv_lora > 0,
            "GLM kv_b split needs qk_nope_head_dim/v_head_dim/kv_lora_rank > 0, \
             got {qk_nope}/{v_head}/{kv_lora}"
        );
        let rows = num_heads * (qk_nope + v_head);
        let kv_b = self.dequantize_dsv4_block_scaled_to_f32_host(kv_b_name, rows, kv_lora)?;
        let per_head = qk_nope + v_head;
        // w_kc[h] = w_kc_src[h] transposed: [kv_lora(out), qk_nope(in)].
        let mut w_kc = vec![0.0f32; num_heads * kv_lora * qk_nope];
        // w_vc[h] = w_vc_src[h] as-is: [v_head(out), kv_lora(in)].
        let mut w_vc = vec![0.0f32; num_heads * v_head * kv_lora];
        for h in 0..num_heads {
            let head_base = h * per_head * kv_lora;
            for i in 0..qk_nope {
                for j in 0..kv_lora {
                    w_kc[h * kv_lora * qk_nope + j * qk_nope + i] =
                        kv_b[head_base + i * kv_lora + j];
                }
            }
            let vc_src_base = head_base + qk_nope * kv_lora;
            for r in 0..v_head {
                for c in 0..kv_lora {
                    w_vc[h * v_head * kv_lora + r * kv_lora + c] =
                        kv_b[vc_src_base + r * kv_lora + c];
                }
            }
        }
        let w_kc_bytes: Vec<u8> = w_kc
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        let w_vc_bytes: Vec<u8> = w_vc
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        // Replicated at the head grain; TP head-sharding of w_kc/w_vc is not
        // implemented.
        let _ = tp;
        let w_kc = DeviceMatrix::from_safetensors(ctx, &w_kc_bytes, num_heads * kv_lora, qk_nope)
            .with_context(|| format!("upload GLM w_kc from {kv_b_name}"))?;
        let w_vc = DeviceMatrix::from_safetensors(ctx, &w_vc_bytes, num_heads * v_head, kv_lora)
            .with_context(|| format!("upload GLM w_vc from {kv_b_name}"))?;
        Ok((w_kc, w_vc))
    }

    /// Load a GLM (`weight_scale_inv`) FP8 E4M3 matrix as a `Fp8BlockScaled`
    /// [`DeviceMatrix`]
    /// (raw FP8 bytes + F32 per-128×128-block scales), WITHOUT dequantizing: GLM's
    /// `weight_scale_inv` is already the `[N/128, K/128]` row-major F32 grid the
    /// `sm90_fp8_gemm_1d2d` kernel reads as `sfb` — no transpose, no lossy E8M0
    /// re-encode.
    fn load_dsv4_glm_fp8_as_block_scaled(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        const BLOCK: usize = 128;
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D FP8 tensor, got {:?}",
            tensor.shape
        );
        ensure!(
            tensor.dtype == Dtype::F8_E4M3,
            "{name}: GLM MoE FP8 weight expects F8_E4M3, got {:?}",
            tensor.dtype
        );
        let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
        let weight = tensor.bytes();
        ensure!(
            weight.len() == rows * cols,
            "{name}: FP8 byte len {} != rows*cols {}",
            weight.len(),
            rows * cols
        );
        // GLM block scale: `<prefix>.weight_scale_inv`, F32, `[N/128, K/128]`.
        let base = name
            .strip_suffix(".weight")
            .ok_or_else(|| anyhow!("{name}: quantized tensor name must end with .weight"))?;
        let scale_name = format!("{base}.weight_scale_inv");
        let scale = self.borrow_raw_tensor(&scale_name)?;
        ensure!(
            scale.dtype == Dtype::F32,
            "{scale_name}: GLM block scale expected F32, got {:?}",
            scale.dtype
        );
        ensure!(
            scale.shape.len() == 2,
            "{scale_name}: expected 2D scale, got {:?}",
            scale.shape
        );
        let (scale_rows, scale_cols) = (scale.shape[0], scale.shape[1]);
        let want_rows = rows.div_ceil(BLOCK);
        let want_cols = cols.div_ceil(BLOCK);
        ensure!(
            scale_rows == want_rows && scale_cols == want_cols,
            "{scale_name}: GLM block scale {scale_rows}x{scale_cols} != [N/128, K/128] = \
             {want_rows}x{want_cols} for weight {rows}x{cols} (block 128x128)"
        );
        let scales: Vec<f32> = scale
            .bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        DeviceMatrix::from_fp8_block_scaled(ctx, weight, &scales, rows, cols, BLOCK, BLOCK)
            .with_context(|| format!("upload GLM FP8 block-scaled MoE weight {name}"))
    }

    /// Quantize a BF16 or F32 row-major matrix to DSv4 FP8 E4M3FN block-scaled on the
    /// host:
    /// 128×128 blocks with E8M0 per-block scales, the layout natively quantized DSv4
    /// weights
    /// use. Returns `(fp8_bytes, e8m0_scale_bytes, scale_rows, scale_cols)`.
    fn quantize_to_dsv4_fp8_host(
        name: &str,
        dtype: Dtype,
        bytes: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<u8>, Vec<u8>, usize, usize)> {
        const BLOCK: usize = 128;
        const E4M3_MAX: f32 = 448.0;
        let vals: Vec<f32> = match dtype {
            Dtype::BF16 => {
                ensure!(
                    bytes.len() == rows * cols * 2,
                    "{name}: BF16 byte length {} != rows*cols*2={}",
                    bytes.len(),
                    rows * cols * 2
                );
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| half::bf16::from_le_bytes(*c).to_f32())
                    .collect()
            }
            Dtype::F32 => {
                ensure!(
                    bytes.len() == rows * cols * 4,
                    "{name}: F32 byte length {} != rows*cols*4={}",
                    bytes.len(),
                    rows * cols * 4
                );
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect()
            }
            other => bail!("{name}: quantize_to_dsv4_fp8_host: unexpected dtype {other:?}"),
        };
        let scale_rows = rows.div_ceil(BLOCK);
        let scale_cols = cols.div_ceil(BLOCK);
        let mut fp8_out = vec![0u8; rows * cols];
        let mut scale_out = vec![0u8; scale_rows * scale_cols];
        for br in 0..scale_rows {
            let row_start = br * BLOCK;
            let row_end = (row_start + BLOCK).min(rows);
            for bc in 0..scale_cols {
                let col_start = bc * BLOCK;
                let col_end = (col_start + BLOCK).min(cols);
                let mut max_abs = 0.0f32;
                for r in row_start..row_end {
                    for c in col_start..col_end {
                        let v = vals[r * cols + c].abs();
                        if v > max_abs {
                            max_abs = v;
                        }
                    }
                }
                // Smallest power-of-2 scale s.t. max_abs / scale ≤ E4M3_MAX.
                // E8M0 byte b ↔ scale = 2^(b−127).
                let scale_byte: u8 = if max_abs == 0.0 {
                    127 // 2^0 = 1; arbitrary for all-zero block
                } else {
                    let b = ((max_abs / E4M3_MAX).log2() + 127.0).ceil() as i32;
                    b.clamp(1, 254) as u8
                };
                let scale = 2.0f32.powi(scale_byte as i32 - 127);
                scale_out[br * scale_cols + bc] = scale_byte;
                for r in row_start..row_end {
                    for c in col_start..col_end {
                        fp8_out[r * cols + c] = encode_f8_e4m3fn_sat(vals[r * cols + c] / scale);
                    }
                }
            }
        }
        Ok((fp8_out, scale_out, scale_rows, scale_cols))
    }

    /// Dequantize a DSv4/GLM block-scaled FP8 E4M3 matrix to host f32. DSv4 ships
    /// `<prefix>.scale` (F8_E8M0, 1 byte/block); GLM ships `<name>.weight_scale_inv`
    /// (F32,
    /// block [128,128]). The block scale multiplies.
    fn dequantize_dsv4_block_scaled_to_f32_host(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        use crate::quant_format::decode_f8_e4m3fn;
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape == [rows, cols],
            "{name}: expected FP8 shape [{rows}, {cols}], got {:?}",
            tensor.shape
        );
        ensure!(
            tensor.dtype == Dtype::F8_E4M3,
            "{name}: GLM kv_b dequant expects F8_E4M3, got {:?}",
            tensor.dtype
        );
        let weight = tensor.bytes();
        ensure!(
            weight.len() == rows * cols,
            "{name}: FP8 byte len {} != rows*cols {}",
            weight.len(),
            rows * cols
        );
        // Prefer the GLM `weight_scale_inv` (F32) dialect, else DSv4 `<prefix>.scale`
        // (E8M0).
        let base = name
            .strip_suffix(".weight")
            .ok_or_else(|| anyhow!("{name}: quantized tensor name must end with .weight"))?;
        let glm_scale = format!("{base}.weight_scale_inv");
        let (scale_f32, scale_rows, scale_cols, block_m, block_k) =
            if let Ok(s) = self.borrow_raw_tensor(&glm_scale) {
                ensure!(
                    s.dtype == Dtype::F32,
                    "{glm_scale}: GLM block scale expected F32, got {:?}",
                    s.dtype
                );
                ensure!(s.shape.len() == 2, "{glm_scale}: expected 2D scale");
                let (sr, sc) = (s.shape[0], s.shape[1]);
                let vals: Vec<f32> = s
                    .bytes()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                // Block dims inferred from the weight/scale shapes (GLM uses
                // [128,128]).
                let block_m = rows.div_ceil(sr);
                let block_k = cols.div_ceil(sc);
                (vals, sr, sc, block_m, block_k)
            } else {
                let dsv4_scale = format!("{base}.scale");
                let s = self.borrow_raw_tensor(&dsv4_scale)?;
                ensure!(s.shape.len() == 2, "{dsv4_scale}: expected 2D scale");
                let (sr, sc) = (s.shape[0], s.shape[1]);
                // E8M0 bytes or F32 power-of-two, normalized to E8M0 → scale =
                // 2^(byte-127).
                let e8m0 = Self::dsv4_block_scale_e8m0(&dsv4_scale, s.dtype, s.bytes())?;
                let vals: Vec<f32> = e8m0.iter().map(|&b| 2.0f32.powi(b as i32 - 127)).collect();
                let block_m = rows.div_ceil(sr);
                let block_k = cols.div_ceil(sc);
                (vals, sr, sc, block_m, block_k)
            };
        ensure!(
            scale_f32.len() == scale_rows * scale_cols,
            "{name}: block-scale element count {} != {scale_rows}*{scale_cols}",
            scale_f32.len()
        );
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let sr = r / block_m;
            for c in 0..cols {
                let sc = c / block_k;
                let scale = scale_f32[sr.min(scale_rows - 1) * scale_cols + sc.min(scale_cols - 1)];
                out[r * cols + c] = decode_f8_e4m3fn(weight[r * cols + c]) * scale;
            }
        }
        Ok(out)
    }

    /// DeepGEMM repack of one decode projection weight, or `None`. Built only when the
    /// decode-DeepGEMM alloc gate is on AND the weight is raw FP8 block-scaled — the
    /// GLM
    /// dialect dequantizes to bf16, so the FP8 check alone excludes it.
    fn decode_proj_cache(
        &self,
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
    ) -> Result<Option<cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache>> {
        if crate::attention::dsv4_fused_wqkv_decode_enabled()?
            && weight.weight_format == WeightFormat::Dsv4Fp8BlockScaled
        {
            Ok(Some(
                cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(ctx, weight)?,
            ))
        } else {
            Ok(None)
        }
    }

    /// Load one compressor sub-block (`wkv`/`wgate`/`ape` matrices + `norm` vec). GLM's
    /// F32
    /// `weight_scale_inv` scales mean its compressor projections are dequantized to
    /// bf16.
    pub(crate) fn load_dsv4_compressor(
        &self,
        ctx: &DeviceContext,
        names: &deepseek_spec::DeepSeekV4CompressorTensorNames,
        glm: bool,
    ) -> Result<crate::dsv4::Dsv4Compressor> {
        let load_matrix = |name: &str| {
            if glm {
                self.load_dsv4_block_scaled_dialect(ctx, name)
            } else {
                self.load_dsv4_global_matrix(ctx, name)
            }
        };
        let wkv = load_matrix(&names.wkv)?;
        let wgate = load_matrix(&names.wgate)?;
        let fp32_probe = {
            let tensor = self.borrow_raw_tensor(&names.ape)?;
            ensure!(
                tensor.shape.len() == 2,
                "{}: expected 2D compressor APE, got {:?}",
                names.ape,
                tensor.shape
            );
            let values = match tensor.dtype {
                Dtype::F32 | Dtype::BF16 => tensor_bytes_to_f32(
                    &names.ape,
                    tensor.dtype,
                    tensor.bytes(),
                    ScaleApply::Multiply,
                )?,
                Dtype::F8_E4M3 => self.dequantize_dsv4_block_scaled_to_f32_host(
                    &names.ape,
                    tensor.shape[0],
                    tensor.shape[1],
                )?,
                other => bail!("{}: unsupported compressor APE dtype {other:?}", names.ape),
            };
            crate::dsv4::Dsv4CompressorFp32Probe {
                wkv: self.load_dsv4_bf16_matrix(ctx, &names.wkv)?,
                wgate: self.load_dsv4_bf16_matrix(ctx, &names.wgate)?,
                ape: ctx
                    .stream
                    .clone_htod(&values)
                    .map_err(|e| anyhow!("upload f32 compressor APE {}: {e}", names.ape))?,
            }
        };
        Ok(crate::dsv4::Dsv4Compressor {
            wkv_deepgemm: self.decode_proj_cache(ctx, &wkv)?,
            wgate_deepgemm: self.decode_proj_cache(ctx, &wgate)?,
            // `ape` is read RAW as bf16 by the compressor kernel, not through
            // `dsv4_linear`, so
            // it must be dense bf16 (#138).
            ape: self.load_dsv4_block_scaled_dialect(ctx, &names.ape)?,
            fp32_probe,
            norm: self.load_dsv4_vec(ctx, &names.norm)?,
            wkv,
            wgate,
        })
    }

    /// Load one indexer sub-block. DSv4 CSA: `wq_b`/`weights_proj` + a key
    /// compressor. GLM DSA (`SparseIndexed`): `wq_b`/`weights_proj` + `wk` key
    /// projection + `k_norm` (weight+bias), no compressor.
    pub(crate) fn load_dsv4_indexer(
        &self,
        ctx: &DeviceContext,
        config: &deepseek_spec::DeepSeekV4Config,
        names: &deepseek_spec::DeepSeekV4IndexerTensorNames,
    ) -> Result<crate::dsv4::Dsv4Indexer> {
        let glm = config.plain_o_proj;
        // GLM indexer wq_b is FP8 + `weight_scale_inv`; dequant to dense bf16.
        let wq_b = if glm {
            self.load_dsv4_block_scaled_dialect(ctx, &names.wq_b)?
        } else {
            self.load_dsv4_global_matrix(ctx, &names.wq_b)?
        };
        let wq_b_deepgemm = self.decode_proj_cache(ctx, &wq_b)?;
        let weights_proj = self.load_dsv4_global_matrix(ctx, &names.weights_proj)?;
        let weights_proj_deepgemm = self.decode_proj_cache(ctx, &weights_proj)?;
        let (compressor, wk, k_norm) = if glm {
            let wk = match names.wk.as_ref() {
                Some(n) => Some(self.load_dsv4_block_scaled_dialect(ctx, n)?),
                None => None,
            };
            let k_norm = match names.k_norm.as_ref() {
                Some(n) => Some(self.load_dsv4_vec(ctx, n)?),
                None => None,
            };
            (None, wk, k_norm)
        } else {
            let compressor = self.load_dsv4_compressor(
                ctx,
                names
                    .compressor
                    .as_ref()
                    .expect("DSv4 indexer always has a compressor"),
                false,
            )?;
            (Some(compressor), None, None)
        };
        Ok(crate::dsv4::Dsv4Indexer {
            wq_b,
            weights_proj,
            compressor,
            wk,
            k_norm,
            wq_b_deepgemm,
            weights_proj_deepgemm,
        })
    }

    /// Load a block-scaled FP8 matrix choosing the scale dialect by sibling: GLM
    /// `<base>.weight_scale_inv` (F32) ⇒ dequantize to a dense bf16 [`DeviceMatrix`]
    /// (the F32
    /// block scale cannot ride the E8M0 path losslessly); DSv4 `<base>.scale` (E8M0) ⇒
    /// the
    /// FP8 block-scaled path. BF16/F32 on disk falls through to the dense loader.
    pub(crate) fn load_dsv4_block_scaled_dialect(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        let tensor = self.borrow_raw_tensor(name)?;
        if tensor.dtype == Dtype::BF16 || tensor.dtype == Dtype::F32 {
            return self.load_dsv4_bf16_matrix(ctx, name);
        }
        let base = name
            .strip_suffix(".weight")
            .ok_or_else(|| anyhow!("{name}: quantized tensor name must end with .weight"))?;
        // Dequant when a block scale is present in EITHER dialect (#138).
        let has_scale = self
            .borrow_raw_tensor(&format!("{base}.weight_scale_inv"))
            .is_ok()
            || self.borrow_raw_tensor(&format!("{base}.scale")).is_ok();
        if has_scale {
            ensure!(
                tensor.shape.len() == 2,
                "{name}: expected 2D quantized tensor, got {:?}",
                tensor.shape
            );
            let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
            let f32 = self.dequantize_dsv4_block_scaled_to_f32_host(name, rows, cols)?;
            let bytes: Vec<u8> = f32
                .iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                .collect();
            DeviceMatrix::from_safetensors(ctx, &bytes, rows, cols)
                .with_context(|| format!("upload dequant bf16 {name}"))
        } else {
            self.load_dsv4_block_scaled(ctx, name)
        }
    }
}
