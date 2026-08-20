//! DSv4 checkpoint loading: FP8/FP4 weight construction, DSpark draft-delta
//! import, and the config/tensor-name probes the two feed on. Split out of
//! `dsv4.rs` — load-time only, nothing here runs on a forward.

use std::path::Path;

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::prelude::DeviceContext;
use deepseek_spec::DeepSeekV4Config;

use crate::loader::SafetensorLoader;
use crate::moe_config::ExpertSplit;

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
pub(crate) fn ensure_loadable(config: &DeepSeekV4Config, spec_decode_on: bool) -> Result<()> {
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
            eprintln!(
                "[dsv4] num_nextn_predict_layers={} present; spec decode on, \
                 loading base layers plus mtp.0 draft head.",
                config.num_nextn_predict_layers
            );
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
        ensure_loadable(&config, spec_decode_on)?;

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
        let mtp = if spec_decode_on && config.num_nextn_predict_layers > 0 {
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
        })
    }
}
