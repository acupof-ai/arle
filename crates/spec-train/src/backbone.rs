//! The DSpark draft backbone, forward.
//!
//! Ported from deepseek-ai/DeepSpec (MIT) `deepspec/modeling/dspark/qwen3/
//! modeling.py`. Tensor names come from [`qwen35_spec::dspark_tensor_names`],
//! so a checkpoint written here is one the serve loads.
//!
//! The one shape worth stating: keys and values are projected from BOTH the
//! trunk taps (context) and the draft rows, concatenated; queries are the draft
//! rows alone. [`crate::mask`] decides what each query may reach. Embeddings and
//! `lm_head` are the trunk's, frozen — the serve shares them too.

use anyhow::{Context, Result, bail, ensure};

use autograd::{Tape, Tensor, TensorId, TensorStore, ops};
use qwen35_spec::{DsparkConfig, DsparkLayerType, dspark_tensor_names};
use std::path::Path;

use crate::block::{self, Block};

pub struct Layer {
    pub q_proj: TensorId,
    pub k_proj: TensorId,
    pub v_proj: TensorId,
    pub o_proj: TensorId,
    pub q_norm: TensorId,
    pub k_norm: TensorId,
    pub input_layernorm: TensorId,
    pub post_attention_layernorm: TensorId,
    pub gate_proj: TensorId,
    pub up_proj: TensorId,
    pub down_proj: TensorId,
}

pub struct Markov {
    pub w1: TensorId,
    pub w2: TensorId,
    pub rank: usize,
}

pub struct Confidence {
    pub weight: TensorId,
    pub bias: TensorId,
    pub with_markov: bool,
}

/// The heads this run asks for. Authoritative over what a warm-start checkpoint
/// happens to carry — a DFlash checkpoint carries neither.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Heads {
    /// 0 trains no markov head.
    pub markov_rank: usize,
    /// Comes from the confidence loss weight being > 0. A head that gets no
    /// gradient still reaches the serve, which obeys it and truncates every
    /// block by noise.
    pub confidence: bool,
}

pub struct Draft {
    pub cfg: DsparkConfig,
    pub fc: TensorId,
    pub hidden_norm: TensorId,
    pub norm: TensorId,
    pub layers: Vec<Layer>,
    pub markov: Option<Markov>,
    pub confidence: Option<Confidence>,
    /// Trunk-owned and frozen.
    pub embed_tokens: TensorId,
    pub lm_head: TensorId,
}

pub struct Input<'a> {
    pub blocks: &'a [Block],
    /// `[ctx_len, taps · hidden]` — the trunk residual stream at
    /// `cfg.target_layer_ids`, concatenated on the feature axis.
    pub taps: TensorId,
    pub ctx_len: usize,
}

pub struct Output {
    /// `[rows, hidden]` after the final norm.
    pub hidden: TensorId,
    /// `[rows, vocab]`, Markov bias applied.
    pub logits: TensorId,
    /// `[rows]` pre-sigmoid accept prediction.
    pub confidence: Option<TensorId>,
}

impl Draft {
    /// Every trainable parameter, for the optimizer.
    #[must_use]
    pub fn parameters(&self) -> Vec<TensorId> {
        let mut p = vec![self.fc, self.hidden_norm, self.norm];
        for l in &self.layers {
            p.extend([
                l.q_proj,
                l.k_proj,
                l.v_proj,
                l.o_proj,
                l.q_norm,
                l.k_norm,
                l.input_layernorm,
                l.post_attention_layernorm,
                l.gate_proj,
                l.up_proj,
                l.down_proj,
            ]);
        }
        if let Some(m) = &self.markov {
            p.extend([m.w1, m.w2]);
        }
        if let Some(c) = &self.confidence {
            p.extend([c.weight, c.bias]);
        }
        p
    }

    pub fn forward(
        &self,
        input: &Input<'_>,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<Output> {
        let c = &self.cfg;
        let blocks = input.blocks;
        ensure!(!blocks.is_empty(), "no anchored blocks");
        let block_size = blocks[0].targets.len();
        let rows = blocks.len() * block_size;
        let (ctx_len, kv_len) = (input.ctx_len, input.ctx_len + rows);
        let hd = c.head_dim;
        let (nh, nkv) = (c.num_attention_heads, c.num_key_value_heads);
        let groups = nh / nkv;

        // A non-finite tap reaches every parameter through `fc` and only shows
        // up as a NaN loss much later; probe the host mirror where one exists
        // rather than scanning `[ctx_len, taps·hidden]` every forward.
        if let Some(d) = store
            .get(input.taps)
            .map(|t| t.data.as_slice())
            .filter(|d| !d.is_empty())
        {
            ensure!(
                [0, d.len() / 2, d.len() - 1]
                    .iter()
                    .all(|&i| d[i].is_finite()),
                "trunk taps are not finite"
            );
        }

        let ctx = ops::matmul_bt(input.taps, self.fc, store, tape)?;
        let ctx = ops::rmsnorm(ctx, self.hidden_norm, c.rms_norm_eps, store, tape)?;

        let noise: Vec<usize> = block::noise_token_ids(blocks, c.mask_token_id)
            .iter()
            .map(|&t| t as usize)
            .collect();
        let h = ops::embedding(self.embed_tokens, &noise, store, tape)?;
        let mut h = ops::reshape(h, &[rows, c.hidden_size], store, tape)?;

        let positions: Vec<usize> = (0..ctx_len).chain(block::draft_positions(blocks)).collect();
        let (cos, sin) = rope_tables(&positions, hd, c.rope_theta, store)?;
        let (cos_q, sin_q) = rope_tables(&positions[ctx_len..], hd, c.rope_theta, store)?;

        let window = c.sliding_window.filter(|_| !c.layer_types.is_empty());
        let mask = store.from_slice(
            &crate::mask::additive(blocks, ctx_len, window),
            &[1, rows, kv_len],
        )?;

        for l in &self.layers {
            let normed = ops::rmsnorm(h, l.input_layernorm, c.rms_norm_eps, store, tape)?;

            let q = ops::matmul_bt(normed, l.q_proj, store, tape)?;
            let q = head_norm(q, l.q_norm, rows, nh, hd, c.rms_norm_eps, store, tape)?;
            let q = ops::rope(q, cos_q, sin_q, store, tape)?;

            let k = cat_ctx_draft(ctx, normed, l.k_proj, store, tape)?;
            let k = head_norm(k, l.k_norm, kv_len, nkv, hd, c.rms_norm_eps, store, tape)?;
            let k = ops::rope(k, cos, sin, store, tape)?;
            let k = ops::repeat_kv(k, groups, store, tape)?;

            let v = cat_ctx_draft(ctx, normed, l.v_proj, store, tape)?;
            let v = ops::reshape(v, &[1, kv_len, nkv, hd], store, tape)?;
            let v = ops::transpose(v, 1, 2, store, tape)?;
            let v = ops::repeat_kv(v, groups, store, tape)?;

            let q3 = ops::reshape(q, &[nh, rows, hd], store, tape)?;
            let k3 = ops::reshape(k, &[nh, kv_len, hd], store, tape)?;
            let v3 = ops::reshape(v, &[nh, kv_len, hd], store, tape)?;
            let scores = ops::matmul(q3, ops::transpose(k3, 1, 2, store, tape)?, store, tape)?;
            let scores = ops::mul_scalar(scores, 1.0 / (hd as f32).sqrt(), store, tape)?;
            let scores = ops::add_broadcast(scores, mask, store, tape)?;
            let attn = ops::matmul(ops::softmax(scores, store, tape)?, v3, store, tape)?;
            let attn = ops::transpose(attn, 0, 1, store, tape)?;
            let attn = ops::reshape(attn, &[rows, nh * hd], store, tape)?;
            h = ops::add(h, ops::matmul_bt(attn, l.o_proj, store, tape)?, store, tape)?;

            let normed = ops::rmsnorm(h, l.post_attention_layernorm, c.rms_norm_eps, store, tape)?;
            let gate = ops::silu(
                ops::matmul_bt(normed, l.gate_proj, store, tape)?,
                store,
                tape,
            )?;
            let up = ops::matmul_bt(normed, l.up_proj, store, tape)?;
            let mlp = ops::mul(gate, up, store, tape)?;
            h = ops::add(
                h,
                ops::matmul_bt(mlp, l.down_proj, store, tape)?,
                store,
                tape,
            )?;
        }

        let hidden_out = ops::rmsnorm(h, self.norm, c.rms_norm_eps, store, tape)?;
        let mut logits = ops::matmul_bt(hidden_out, self.lm_head, store, tape)?;

        let prev: Vec<usize> = blocks
            .iter()
            .flat_map(|b| b.prev.iter().map(|&t| t as usize))
            .collect();
        let prev_embed = match &self.markov {
            Some(m) => {
                let e = ops::embedding(m.w1, &prev, store, tape)?;
                let e = ops::reshape(e, &[rows, m.rank], store, tape)?;
                let bias = ops::matmul_bt(e, m.w2, store, tape)?;
                logits = ops::add(logits, bias, store, tape)?;
                Some(e)
            }
            None => None,
        };

        let confidence = match &self.confidence {
            Some(cf) => {
                let feat = if cf.with_markov {
                    let e = prev_embed.context("confidence head wants a markov head")?;
                    ops::cat(&[hidden_out, e], 1, store, tape)?
                } else {
                    hidden_out
                };
                let z = ops::matmul_bt(feat, cf.weight, store, tape)?;
                let z = ops::add_broadcast(z, cf.bias, store, tape)?;
                Some(ops::reshape(z, &[rows], store, tape)?)
            }
            None => None,
        };

        Ok(Output {
            hidden: hidden_out,
            logits,
            confidence,
        })
    }
}

/// `[n, heads·dim] -> rmsnorm over dim -> [1, heads, n, dim]`, the layout rope
/// and the score matmul both want.
#[allow(clippy::too_many_arguments)]
fn head_norm(
    x: TensorId,
    weight: TensorId,
    n: usize,
    heads: usize,
    dim: usize,
    eps: f32,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let x = ops::reshape(x, &[n * heads, dim], store, tape)?;
    let x = ops::rmsnorm(x, weight, eps, store, tape)?;
    let x = ops::reshape(x, &[1, n, heads, dim], store, tape)?;
    Ok(ops::transpose(x, 1, 2, store, tape)?)
}

/// One projection applied to the context and the draft rows, concatenated into
/// the key/value sequence.
fn cat_ctx_draft(
    ctx: TensorId,
    draft: TensorId,
    proj: TensorId,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<TensorId> {
    let a = ops::matmul_bt(ctx, proj, store, tape)?;
    let b = ops::matmul_bt(draft, proj, store, tape)?;
    Ok(ops::cat(&[a, b], 0, store, tape)?)
}

/// `cos`/`sin` rows for arbitrary absolute positions — draft rows sit at their
/// anchors, not at `0..n`.
fn rope_tables(
    positions: &[usize],
    head_dim: usize,
    theta: f32,
    store: &mut TensorStore,
) -> Result<(TensorId, TensorId)> {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);
    for &p in positions {
        for &f in &inv {
            let a = p as f32 * f;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    let shape = [positions.len(), half];
    Ok((
        store.from_slice(&cos, &shape)?,
        store.from_slice(&sin, &shape)?,
    ))
}

/// Fresh weights: normal(0, 0.02) for projections, ones for norms — HF's
/// `_init_weights` for Qwen3, which is what the reference trains from. Vocab is
/// the trunk `lm_head`'s row count, whose distribution the draft predicts.
pub fn init(
    cfg: DsparkConfig,
    heads: Heads,
    seed: u64,
    embed_tokens: TensorId,
    lm_head: TensorId,
    store: &mut TensorStore,
) -> Result<Draft> {
    let mut init = Init(seed);
    let (h, hd) = (cfg.hidden_size, cfg.head_dim);
    let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
    let vocab = shape_of(store, lm_head)?[0];

    let layers = (0..cfg.num_hidden_layers)
        .map(|_| {
            Ok(Layer {
                q_proj: init.normal(nh * hd, h, store)?,
                k_proj: init.normal(nkv * hd, h, store)?,
                v_proj: init.normal(nkv * hd, h, store)?,
                o_proj: init.normal(h, nh * hd, store)?,
                q_norm: ones(hd, store)?,
                k_norm: ones(hd, store)?,
                input_layernorm: ones(h, store)?,
                post_attention_layernorm: ones(h, store)?,
                gate_proj: init.normal(cfg.intermediate_size, h, store)?,
                up_proj: init.normal(cfg.intermediate_size, h, store)?,
                down_proj: init.normal(h, cfg.intermediate_size, store)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let markov = match heads.markov_rank {
        0 => None,
        rank => Some(init.markov(rank, vocab, store)?),
    };
    let confidence = heads
        .confidence
        .then(|| init.confidence(h, markov.as_ref(), store))
        .transpose()?;

    Ok(Draft {
        fc: init.normal(h, cfg.target_layer_ids.len() * h, store)?,
        hidden_norm: ones(h, store)?,
        norm: ones(h, store)?,
        cfg,
        layers,
        markov,
        confidence,
        embed_tokens,
        lm_head,
    })
}

fn ones(n: usize, store: &mut TensorStore) -> Result<TensorId> {
    Ok(store.alloc(Tensor::new(vec![1.0; n], vec![n], true)?))
}

/// One splitmix64 stream, so a run reproduces from `--seed` alone. Shared by
/// [`init`] and the heads [`load`] has to create for a warm start.
struct Init(u64);

impl Init {
    fn normal(&mut self, rows: usize, cols: usize, store: &mut TensorStore) -> Result<TensorId> {
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| 0.02 * gauss(&mut self.0))
            .collect();
        Ok(store.alloc(Tensor::new(data, vec![rows, cols], true)?))
    }

    fn markov(&mut self, rank: usize, vocab: usize, store: &mut TensorStore) -> Result<Markov> {
        Ok(Markov {
            w1: self.normal(vocab, rank, store)?,
            w2: self.normal(vocab, rank, store)?,
            rank,
        })
    }

    fn confidence(
        &mut self,
        hidden: usize,
        markov: Option<&Markov>,
        store: &mut TensorStore,
    ) -> Result<Confidence> {
        Ok(Confidence {
            weight: self.normal(1, hidden + markov.map_or(0, |m| m.rank), store)?,
            bias: store.alloc(Tensor::new(vec![0.0], vec![1], true)?),
            with_markov: markov.is_some(),
        })
    }
}

/// Box-Muller over splitmix64 — deterministic so a re-init reproduces a run.
fn gauss(seed: &mut u64) -> f32 {
    let mut next = || {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = *seed;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((x ^ (x >> 31)) >> 11) as f32 / (1u64 << 53) as f32
    };
    let u1 = next().max(f32::MIN_POSITIVE);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * next()).cos()
}

fn shape_of(store: &TensorStore, id: TensorId) -> Result<Vec<usize>> {
    Ok(store
        .get(id)
        .ok_or_else(|| anyhow::anyhow!("tensor {id:?} was freed"))?
        .shape
        .clone())
}

/// Host values of one tensor. `TensorStore::to_host` would want the store
/// mutably, and a checkpoint write has no business borrowing it that way, so a
/// device-resident parameter is read back through the backend directly.
fn host_of(store: &TensorStore, id: TensorId) -> Result<Vec<f32>> {
    let t = store
        .get(id)
        .ok_or_else(|| anyhow::anyhow!("tensor {id:?} was freed"))?;
    if t.dirty != autograd::tensor::Dirty::Device {
        return Ok(t.data.clone());
    }
    let handle = t
        .device_handle
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("tensor {id:?} is on device with no handle"))?;
    store.backend().eval(&[handle])?;
    Ok(store.backend().readback(handle)?)
}

/// Read a draft checkpoint. `embed_tokens` / `lm_head` are the trunk's.
///
/// `heads` decides which heads the returned draft has, not the shard: a warm
/// start from a DFlash checkpoint carries the backbone alone, and deriving the
/// heads from tensor presence there yields a draft the serve cannot use.
pub fn load(
    dir: &Path,
    heads: Heads,
    seed: u64,
    embed_tokens: TensorId,
    lm_head: TensorId,
    store: &mut TensorStore,
) -> Result<Draft> {
    let cfg = DsparkConfig::from_dir(dir)?;
    let names = dspark_tensor_names(cfg.num_hidden_layers);
    let shards = Shards::open(dir)?;

    let layers = names
        .layers
        .iter()
        .map(|n| {
            Ok(Layer {
                q_proj: shards.param(&n.q_proj, store)?,
                k_proj: shards.param(&n.k_proj, store)?,
                v_proj: shards.param(&n.v_proj, store)?,
                o_proj: shards.param(&n.o_proj, store)?,
                q_norm: shards.param(&n.q_norm, store)?,
                k_norm: shards.param(&n.k_norm, store)?,
                input_layernorm: shards.param(&n.input_layernorm, store)?,
                post_attention_layernorm: shards.param(&n.post_attention_layernorm, store)?,
                gate_proj: shards.param(&n.gate_proj, store)?,
                up_proj: shards.param(&n.up_proj, store)?,
                down_proj: shards.param(&n.down_proj, store)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    for gated in [&names.markov_gate_proj, &names.markov_joint_proj] {
        ensure!(
            !shards.has(gated),
            "{gated}: only the vanilla w1/w2 markov head is wired"
        );
    }
    let mut init = Init(seed);
    let markov = match shards.has(&names.markov_w1) {
        true => {
            let w1 = shards.param(&names.markov_w1, store)?;
            let rank = shape_of(store, w1)?[1];
            ensure!(
                heads.markov_rank == rank,
                "checkpoint markov rank {rank}, run asks for {}",
                heads.markov_rank
            );
            Some(Markov {
                w1,
                w2: shards.param(&names.markov_w2, store)?,
                rank,
            })
        }
        false => match heads.markov_rank {
            0 => None,
            rank => Some(init.markov(rank, shape_of(store, lm_head)?[0], store)?),
        },
    };
    let confidence = match shards.has(&names.confidence_weight) {
        true => {
            ensure!(
                heads.confidence,
                "checkpoint carries a confidence head, the run's confidence loss weight is 0"
            );
            let weight = shards.param(&names.confidence_weight, store)?;
            let width = shape_of(store, weight)?[1];
            let with_markov = width > cfg.hidden_size;
            let rank = markov.as_ref().map_or(0, |m| m.rank);
            ensure!(
                width == cfg.hidden_size + usize::from(with_markov) * rank,
                "confidence head takes {width} features, this draft supplies {}",
                cfg.hidden_size + rank
            );
            Some(Confidence {
                weight,
                bias: shards.param(&names.confidence_bias, store)?,
                with_markov,
            })
        }
        false => heads
            .confidence
            .then(|| init.confidence(cfg.hidden_size, markov.as_ref(), store))
            .transpose()?,
    };

    Ok(Draft {
        fc: shards.param(&names.fc, store)?,
        hidden_norm: shards.param(&names.hidden_norm, store)?,
        norm: shards.param(&names.norm, store)?,
        cfg,
        layers,
        markov,
        confidence,
        embed_tokens,
        lm_head,
    })
}

/// One frozen table (`embed_tokens` / `lm_head`) out of a model directory.
/// The draft shares the trunk's, so they are read here and never stepped.
pub fn load_frozen(dir: &Path, name: &str, store: &mut TensorStore) -> Result<TensorId> {
    Shards::open(dir)?.tensor(name, false, store)
}

/// Whether a directory carries any of the draft's own weights. Only "no
/// safetensors here" means train from scratch; a corrupt or unreadable shard
/// is an error, because silently restarting would overwrite the checkpoint the
/// caller meant to resume.
pub fn has_weights(dir: &Path) -> Result<bool> {
    let any = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .any(|p| p.extension().is_some_and(|e| e == "safetensors"));
    if !any {
        return Ok(false);
    }
    Ok(Shards::open(dir)?.has(&dspark_tensor_names(1).fc))
}

/// The `config.json` both readers need beside the shard: [`load`] here and the
/// serve's `load_dspark_head` go through `DsparkConfig::from_dir`.
///
/// `target_layer_ids` is part of the weights, not metadata — `fc` is trained
/// against that tap order and reading it back in another order is silent
/// garbage. `speculative_tokens` carries the next-token/same-position bit;
/// `architectures` is identical for both flavors and never decides it. The head
/// fields are the reference's `build_draft_config` names.
fn config_json(draft: &Draft) -> serde_json::Value {
    let cfg = &draft.cfg;
    let layer_types: Vec<&str> = cfg
        .layer_types
        .iter()
        .map(|t| match t {
            DsparkLayerType::Full => "full_attention",
            DsparkLayerType::Sliding => "sliding_attention",
        })
        .collect();
    serde_json::json!({
        "architectures": ["DSparkDraftModel"],
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "sliding_window": cfg.sliding_window,
        "layer_types": layer_types,
        "block_size": cfg.block_size,
        "mask_token_id": cfg.mask_token_id,
        "target_layer_ids": cfg.target_layer_ids,
        "speculative_tokens": cfg.max_draft_tokens(),
        "markov_rank": draft.markov.as_ref().map_or(0, |m| m.rank),
        "markov_head_type": "vanilla",
        "enable_confidence_head": draft.confidence.is_some(),
        "confidence_head_with_markov":
            draft.confidence.as_ref().is_some_and(|c| c.with_markov),
    })
}

/// Write a checkpoint DIRECTORY the serve can load: `config.json` +
/// `model.safetensors`. Embeddings and `lm_head` stay out: the serve reads the
/// trunk's and warns when a draft checkpoint carries its own.
///
/// One shard named `model.safetensors` and no index — the serve's
/// `SafetensorLoader::new` then takes its single-file branch and never scans
/// for siblings, so a stray file in `dir` cannot change what loads.
pub fn save(draft: &Draft, dir: &Path, store: &TensorStore) -> Result<()> {
    let names = dspark_tensor_names(draft.cfg.num_hidden_layers);
    let mut named: Vec<(String, TensorId)> = vec![
        (names.fc, draft.fc),
        (names.hidden_norm, draft.hidden_norm),
        (names.norm, draft.norm),
    ];
    for (n, l) in names.layers.iter().zip(&draft.layers) {
        named.extend([
            (n.q_proj.clone(), l.q_proj),
            (n.k_proj.clone(), l.k_proj),
            (n.v_proj.clone(), l.v_proj),
            (n.o_proj.clone(), l.o_proj),
            (n.q_norm.clone(), l.q_norm),
            (n.k_norm.clone(), l.k_norm),
            (n.input_layernorm.clone(), l.input_layernorm),
            (
                n.post_attention_layernorm.clone(),
                l.post_attention_layernorm,
            ),
            (n.gate_proj.clone(), l.gate_proj),
            (n.up_proj.clone(), l.up_proj),
            (n.down_proj.clone(), l.down_proj),
        ]);
    }
    if let Some(m) = &draft.markov {
        named.extend([(names.markov_w1, m.w1), (names.markov_w2, m.w2)]);
    }
    if let Some(c) = &draft.confidence {
        named.extend([
            (names.confidence_weight, c.weight),
            (names.confidence_bias, c.bias),
        ]);
    }

    let mut buffers = Vec::with_capacity(named.len());
    for (name, id) in named {
        let shape = shape_of(store, id)?;
        let bytes: Vec<u8> = host_of(store, id)?
            .iter()
            .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
            .collect();
        buffers.push((name, shape, bytes));
    }
    let views = buffers
        .iter()
        .map(|(name, shape, bytes)| {
            let v = safetensors::tensor::TensorView::new(
                safetensors::Dtype::BF16,
                shape.clone(),
                bytes,
            )
            .expect("bf16 view over own buffer");
            (name.clone(), v)
        })
        .collect::<Vec<_>>();
    std::fs::create_dir_all(dir)?;
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config_json(draft))?,
    )?;
    let tmp = dir.join("model.safetensors.tmp");
    safetensors::serialize_to_file(views, None, &tmp)
        .map_err(|e| anyhow::anyhow!("draft save to {} failed: {e}", tmp.display()))?;
    std::fs::rename(&tmp, dir.join("model.safetensors"))?;
    Ok(())
}

struct Shards {
    files: Vec<(memmap2::Mmap, Vec<String>)>,
}

impl Shards {
    fn open(dir: &Path) -> Result<Self> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        ensure!(!paths.is_empty(), "no safetensors in {}", dir.display());
        paths.sort();
        let files = paths
            .into_iter()
            .map(|p| {
                let f = std::fs::File::open(&p)?;
                // SAFETY: checkpoint files are not mutated while training runs.
                let mmap = unsafe { memmap2::Mmap::map(&f)? };
                let names = safetensors::SafeTensors::deserialize(&mmap)
                    .map_err(|e| anyhow::anyhow!("parse {}: {e}", p.display()))?
                    .names()
                    .into_iter()
                    .map(String::from)
                    .collect();
                Ok((mmap, names))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { files })
    }

    fn has(&self, name: &str) -> bool {
        self.files.iter().any(|(_, n)| n.iter().any(|x| x == name))
    }

    fn param(&self, name: &str, store: &mut TensorStore) -> Result<TensorId> {
        self.tensor(name, true, store)
    }

    fn tensor(&self, name: &str, grad: bool, store: &mut TensorStore) -> Result<TensorId> {
        for (mmap, names) in &self.files {
            if !names.iter().any(|x| x == name) {
                continue;
            }
            let st = safetensors::SafeTensors::deserialize(mmap)
                .map_err(|e| anyhow::anyhow!("parse shard for {name}: {e}"))?;
            let t = st
                .tensor(name)
                .map_err(|e| anyhow::anyhow!("read {name}: {e}"))?;
            let data: Vec<f32> = match t.dtype() {
                safetensors::Dtype::BF16 => autograd::safetensors_io::bf16_bytes_to_f32(t.data()),
                safetensors::Dtype::F16 => t
                    .data()
                    .chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F32 => t
                    .data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
                d => bail!("{name}: unsupported dtype {d:?}"),
            };
            return Ok(store.alloc(Tensor::new(data, t.shape().to_vec(), grad)?));
        }
        bail!("{name} missing from the draft checkpoint")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::build_block;
    use autograd::{Backend, CpuBackend};
    use qwen35_spec::DsparkLayerType;
    use std::sync::Arc;

    const VOCAB: usize = 32;
    const HIDDEN: usize = 8;
    const CTX: usize = 8;

    fn tiny_cfg() -> DsparkConfig {
        DsparkConfig {
            hidden_size: HIDDEN,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            sliding_window: Some(4096),
            layer_types: vec![DsparkLayerType::Full; 2],
            block_size: 3,
            mask_token_id: 31,
            target_layer_ids: vec![-1, 0],
            next_token_heads: true,
        }
    }

    const HEADS: Heads = Heads {
        markov_rank: 4,
        confidence: true,
    };
    /// The draw `taped_gradients_match_finite_differences` is calibrated
    /// against; other seeds land 0.08-0.17 against its 5e-2 bar.
    const SEED: u64 = 0x5DEE_C0DE;

    fn setup() -> (TensorStore, Tape, Draft) {
        let (mut store, embed, lm_head) = tables();
        let draft = init(tiny_cfg(), HEADS, SEED, embed, lm_head, &mut store).unwrap();
        (store, Tape::new(), draft)
    }

    fn tables() -> (TensorStore, TensorId, TensorId) {
        let mut store = TensorStore::with_backend(Arc::new(CpuBackend) as Arc<dyn Backend>);
        let embed = init_table(VOCAB, HIDDEN, 1, &mut store);
        let lm_head = init_table(VOCAB, HIDDEN, 2, &mut store);
        (store, embed, lm_head)
    }

    fn init_table(rows: usize, cols: usize, salt: u64, store: &mut TensorStore) -> TensorId {
        let mut seed = salt;
        let data: Vec<f32> = (0..rows * cols).map(|_| 0.05 * gauss(&mut seed)).collect();
        store.alloc(Tensor::new(data, vec![rows, cols], false).unwrap())
    }

    fn run(taps: &[f32], ctx_len: usize, anchors: &[usize]) -> Vec<f32> {
        let (mut store, mut tape, draft) = setup();
        let ids: Vec<u32> = (0..ctx_len as u32).map(|i| i % 20).collect();
        let blocks: Vec<Block> = anchors
            .iter()
            .map(|&a| build_block(&ids, &vec![true; ctx_len], a, 3).unwrap())
            .collect();
        let taps = store.from_slice(taps, &[ctx_len, 2 * HIDDEN]).unwrap();
        let out = draft
            .forward(
                &Input {
                    blocks: &blocks,
                    taps,
                    ctx_len,
                },
                &mut store,
                &mut tape,
            )
            .unwrap();
        store.to_host(out.logits).unwrap()
    }

    fn noise(n: usize, salt: u64, scale: f32) -> Vec<f32> {
        let mut seed = salt;
        (0..n).map(|_| scale * gauss(&mut seed)).collect()
    }

    fn one_block() -> Vec<Block> {
        let ids: Vec<u32> = (0..CTX as u32).collect();
        vec![build_block(&ids, &[true; CTX], 3, 3).unwrap()]
    }

    /// One chunk's forward + loss, composed as `trainer::chunk_backward` does.
    /// `target_host` is a tensor of its own: reusing the draft's own logits
    /// makes `diff` identically zero, which cancels the 0.9-weight TV gradient
    /// and leaves only the 0.1-weight CE term under test.
    fn forward_loss(
        draft: &Draft,
        blocks: &[Block],
        taps_host: &[f32],
        target_host: &[f32],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> TensorId {
        let taps = store.from_slice(taps_host, &[CTX, 2 * HIDDEN]).unwrap();
        let out = draft
            .forward(
                &Input {
                    blocks,
                    taps,
                    ctx_len: CTX,
                },
                store,
                tape,
            )
            .unwrap();
        let rows = blocks.len() * blocks[0].targets.len();
        let target_logits = store.from_slice(target_host, &[rows, VOCAB]).unwrap();
        let targets: Vec<usize> = blocks
            .iter()
            .flat_map(|b| b.targets.iter().map(|&t| t as usize))
            .collect();
        let weights: Vec<f32> = blocks
            .iter()
            .flat_map(|b| crate::block::row_weights(b, Some(4.0)))
            .collect();
        let (loss, _) = crate::loss::dspark_loss(
            &crate::loss::Batch {
                draft_logits: out.logits,
                target_logits,
                targets: &targets,
                weights: &weights,
                denom: weights.iter().sum(),
                conf_logits: out.confidence,
                block_size: blocks[0].targets.len(),
            },
            crate::loss::Weights::default(),
            store,
            tape,
        )
        .unwrap();
        loss
    }

    /// The serve's ctx ring stops strictly below the anchor: `last_token` has not
    /// been forwarded when the draft runs, so its tap does not exist yet
    /// (`executor/qwen35.rs` appends it after the verify). Reaching `taps[anchor]`
    /// is invisible to every shape and index assertion — only one key's content
    /// differs — and it is the residual row 0's own target is projected from.
    #[test]
    fn a_block_is_blind_from_its_anchor_onward() {
        let (ctx_len, row) = (12, 2 * HIDDEN);
        let taps = noise(ctx_len * row, 7, 1.0);
        let base = run(&taps, ctx_len, &[4, 9]);
        let block = 3 * VOCAB;

        let bump = |lo: usize, hi: usize| {
            let mut t = taps.clone();
            for x in &mut t[lo * row..hi * row] {
                *x += 10.0;
            }
            run(&t, ctx_len, &[4, 9])
        };
        let moved = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max)
        };

        let before_9 = bump(8, 9);
        assert!(
            moved(&base[block..], &before_9[block..]) > 1e-4,
            "positive control: anchor-9 block must reach taps[8]"
        );
        let from_9 = bump(9, ctx_len);
        assert!(
            moved(&base[block..], &from_9[block..]) < 1e-5,
            "anchor-9 block reaches taps[9] or beyond"
        );
        let from_4 = bump(4, ctx_len);
        assert!(
            moved(&base[..block], &from_4[..block]) < 1e-5,
            "anchor-4 block reaches taps[4] or beyond"
        );
    }

    #[test]
    fn backward_reaches_every_parameter() {
        let (mut store, mut tape, draft) = setup();
        let blocks = one_block();
        let loss = forward_loss(
            &draft,
            &blocks,
            &noise(CTX * 2 * HIDDEN, 3, 1.0),
            &noise(3 * VOCAB, 5, 0.5),
            &mut store,
            &mut tape,
        );
        let grads = tape.backward(loss, &mut store).unwrap();

        for p in draft.parameters() {
            let g = *grads.get(&p).expect("no gradient");
            let norm: f32 = store.to_host(g).unwrap().iter().map(|x| x * x).sum();
            assert!(norm > 0.0, "parameter {p:?} got a zero gradient");
        }
    }

    /// Central differences along one fixed direction per parameter tensor, at
    /// the default weights so all three loss terms are live. `backward_reaches_
    /// every_parameter` only proves a gradient is non-zero; a transposed or
    /// mis-scaled backward survives it and still lets the loss fall.
    #[test]
    fn taped_gradients_match_finite_differences() {
        // Step per ELEMENT, relative to the tensor's own scale: norms sit at
        // 1.0 and projections at 0.02, and one absolute step cannot serve both.
        const REL: f32 = 0.04;

        let (mut store, mut tape, draft) = setup();
        let blocks = one_block();
        let taps = noise(CTX * 2 * HIDDEN, 3, 1.0);
        let target = noise(3 * VOCAB, 5, 0.5);
        let value = |store: &mut TensorStore, tape: &mut Tape| {
            tape.entries.clear();
            tape.set_enabled(true);
            let loss = forward_loss(&draft, &blocks, &taps, &target, store, tape);
            (store.to_host(loss).unwrap()[0], loss)
        };

        let (base, loss) = value(&mut store, &mut tape);
        let grads = tape.backward(loss, &mut store).unwrap();

        let params = draft.parameters();
        let dirs: Vec<Vec<f32>> = params
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let mut d = noise(store.get(p).unwrap().size, 0x51ED + i as u64, 1.0);
                let scale = d.iter().map(|x| x * x).sum::<f32>().sqrt();
                d.iter_mut().for_each(|x| *x /= scale);
                d
            })
            .collect();
        let analytic: Vec<f32> = params
            .iter()
            .zip(&dirs)
            .map(|(&p, d)| {
                let g = store.to_host(grads[&p]).unwrap();
                g.iter().zip(d).map(|(a, b)| a * b).sum()
            })
            .collect();

        let (mut worst, mut skipped) = (0.0f32, 0usize);
        for (i, (&p, d)) in params.iter().zip(&dirs).enumerate() {
            let w = store.to_host(p).unwrap();
            let rms = (w.iter().map(|x| x * x).sum::<f32>() / w.len() as f32).sqrt();
            let h = REL * rms.max(1e-3) * (w.len() as f32).sqrt();
            let shift = |store: &mut TensorStore, s: f32| {
                let t = store.get_mut(p).unwrap();
                t.data.iter_mut().zip(d).for_each(|(x, dx)| *x += s * dx);
            };
            shift(&mut store, h);
            let plus = value(&mut store, &mut tape).0;
            shift(&mut store, -2.0 * h);
            let minus = value(&mut store, &mut tape).0;
            shift(&mut store, h);

            let numeric = (plus - minus) / (2.0 * h);
            // The f32 loss resolves to `|L|·EPSILON`, so `|L|·EPSILON/2h` is
            // the noise in the difference; below 30× that there is no signal.
            let floor = 30.0 * base.abs() * f32::EPSILON / (2.0 * h);
            let scale = analytic[i].abs().max(numeric.abs());
            if scale < floor {
                skipped += 1;
                continue;
            }
            worst = worst.max((numeric - analytic[i]).abs() / scale);
        }
        assert!(worst < 5e-2, "worst relative gradient error {worst}");
        assert!(
            skipped * 2 < params.len(),
            "{skipped} of {} directional derivatives sat under the f32 noise \
             floor — the check is passing on too little",
            params.len()
        );
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "arle-spec-train-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// `save` -> `load` is the only path by which a trained draft reaches the
    /// serve, and the config it writes is what binds `fc` to its tap order.
    #[test]
    fn a_saved_draft_reloads_as_the_same_model() {
        let (mut store, _tape, draft) = setup();
        let dir = scratch("roundtrip");
        save(&draft, &dir, &store).unwrap();

        let back = load(
            &dir,
            HEADS,
            SEED,
            draft.embed_tokens,
            draft.lm_head,
            &mut store,
        )
        .unwrap();
        assert_eq!(back.cfg, draft.cfg);
        assert!(has_weights(&dir).unwrap());
        // `zip` below would pass on a draft that came back with fewer heads.
        assert_eq!(back.parameters().len(), draft.parameters().len());
        assert_eq!(
            back.markov.as_ref().map(|m| m.rank),
            Some(HEADS.markov_rank)
        );
        assert!(back.confidence.as_ref().is_some_and(|c| c.with_markov));

        for (a, b) in draft.parameters().into_iter().zip(back.parameters()) {
            assert_eq!(shape_of(&store, a).unwrap(), shape_of(&store, b).unwrap());
            let (x, y) = (store.to_host(a).unwrap(), store.to_host(b).unwrap());
            // bf16 storage: half an ulp, 2^-8 relative.
            let off = x
                .iter()
                .zip(&y)
                .map(|(p, q)| (p - q).abs() / p.abs().max(f32::MIN_POSITIVE))
                .fold(0.0, f32::max);
            assert!(off <= 0.0039, "parameter {a:?} came back {off} off");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A confidence head the loss never touched still reaches the serve, which
    /// obeys it; and a DFlash warm start carries no heads at all, so neither
    /// end may be read off the shard.
    #[test]
    fn heads_follow_the_request_not_the_checkpoint() {
        let (mut store, embed, lm_head) = tables();
        let bare = init(
            tiny_cfg(),
            Heads::default(),
            SEED,
            embed,
            lm_head,
            &mut store,
        )
        .unwrap();
        assert!(bare.markov.is_none() && bare.confidence.is_none());

        let dir = scratch("heads");
        save(&bare, &dir, &store).unwrap();
        let grown = load(&dir, HEADS, SEED, embed, lm_head, &mut store).unwrap();
        assert_eq!(grown.markov.as_ref().map(|m| m.rank), Some(4));
        assert_eq!(
            shape_of(&store, grown.confidence.as_ref().unwrap().weight).unwrap(),
            vec![1, HIDDEN + 4]
        );

        save(&grown, &dir, &store).unwrap();
        let clash = Heads {
            markov_rank: 8,
            ..HEADS
        };
        assert!(load(&dir, clash, SEED, embed, lm_head, &mut store).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Two runs at one `--seed` must start from the same weights; the point of
    /// the flag is that a bench A/B differs only by the arm.
    #[test]
    fn init_starts_from_the_seed() {
        let weights = |seed| {
            let (mut store, embed, lm_head) = tables();
            let draft = init(tiny_cfg(), HEADS, seed, embed, lm_head, &mut store).unwrap();
            draft
                .parameters()
                .into_iter()
                .flat_map(|p| store.to_host(p).unwrap())
                .collect::<Vec<f32>>()
        };
        assert_eq!(weights(42), weights(42));
        assert_ne!(weights(42), weights(43));
    }
}
