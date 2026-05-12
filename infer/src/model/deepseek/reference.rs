//! Slow Rust reference executor for the local DeepSeek V4 1B checkpoint.
//!
//! This is a CPU smoke path for `cpu_serve` and local correctness work. It is
//! intentionally not a performance path: CUDA/Metal kernels remain the serving
//! target. The important property is that it consumes the same V4 config and
//! safetensors weights as the runtime model, with no Python/PyTorch dependency.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};
use deepseek_spec::{DeepSeekV4CompressorTensorNames, DeepSeekV4Config, DeepSeekV4MoeTensorNames};
use half::bf16;
use memmap2::Mmap;
use serde::Deserialize;

use crate::deepseek_v4_manifest::validate_deepseek_v4_checkpoint_manifest;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorDtype {
    Bf16,
    F32,
    I64,
    Unsupported,
}

#[derive(Debug, Clone)]
struct TensorMeta {
    shard: usize,
    dtype: TensorDtype,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct SafeTensorStore {
    mmaps: Vec<Mmap>,
    tensors: HashMap<String, TensorMeta>,
}

#[derive(Debug, Deserialize)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub(crate) struct DeepseekV4ReferenceModel {
    config: DeepSeekV4Config,
    tensors: SafeTensorStore,
}

impl DeepseekV4ReferenceModel {
    pub(crate) fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = DeepSeekV4Config::from_json_file(model_dir.join("config.json"))
            .with_context(|| format!("loading DeepSeek V4 config from {}", model_dir.display()))?;
        let tensors = SafeTensorStore::open_model_dir(model_dir)?;
        validate_deepseek_v4_checkpoint_manifest(model_dir, &config)?;
        Ok(Self { config, tensors })
    }

    pub(crate) fn generate_greedy(
        &self,
        prompt: &str,
        tokenizer: &Tokenizer,
        max_new_tokens: usize,
        stop_token_ids: &[u32],
        ignore_eos: bool,
    ) -> Result<(String, usize)> {
        let mut tokens = tokenizer.encode(prompt)?;
        ensure!(
            !tokens.is_empty(),
            "DeepSeek V4 reference path requires a non-empty prompt"
        );
        let prompt_len = tokens.len();
        let mut generated = Vec::new();
        for _ in 0..max_new_tokens {
            let logits = self.forward_last_logits(&tokens)?;
            let next = argmax(&logits) as u32;
            tokens.push(next);
            if !ignore_eos
                && (self.config.eos_token_id == Some(next) || stop_token_ids.contains(&next))
            {
                break;
            }
            generated.push(next);
        }
        Ok((tokenizer.decode(&generated)?, tokens.len() - prompt_len))
    }

    pub(crate) fn forward_last_logits(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty(), "DeepSeek V4 forward needs tokens");
        ensure!(
            tokens.len() <= self.config.sliding_window,
            "DeepSeek V4 CPU reference currently caps prompt length at sliding_window={} tokens",
            self.config.sliding_window
        );

        let seq = tokens.len();
        let d = self.config.hidden_size;
        let n_hc = self.config.hc_mult;
        let mut stream = vec![0.0_f32; seq * n_hc * d];
        for (pos, &token) in tokens.iter().enumerate() {
            let embed = self.tensors.row_f32("embed.weight", token as usize)?;
            ensure!(embed.len() == d, "embed row width mismatch");
            for hc in 0..n_hc {
                stream_slice_mut(&mut stream, pos, hc, n_hc, d).copy_from_slice(&embed);
            }
        }

        let (rope_cos, rope_sin) =
            build_rope_cache(seq, self.config.qk_rope_head_dim, self.config.rope_theta);
        let (rope_cos_c, rope_sin_c) = build_rope_cache(
            seq.max(1),
            self.config.qk_rope_head_dim,
            self.config.compress_rope_theta,
        );

        for layer_idx in 0..self.config.num_hidden_layers {
            stream = self.layer_forward(
                layer_idx,
                &stream,
                tokens,
                &rope_cos,
                &rope_sin,
                &rope_cos_c,
                &rope_sin_c,
            )?;
        }

        let last = seq - 1;
        let head_pre =
            self.gen_head_pre(last, &stream, "hc_head_fn", "hc_head_base", "hc_head_scale")?;
        let mut hidden = vec![0.0_f32; d];
        for hc in 0..n_hc {
            let src = stream_slice(&stream, last, hc, n_hc, d);
            let weight = head_pre[hc];
            for col in 0..d {
                hidden[col] += weight * src[col];
            }
        }
        let hidden = self.rms_norm(&hidden, "norm.weight")?;
        self.tensors.matvec("head.weight", &hidden)
    }

    fn layer_forward(
        &self,
        layer_idx: usize,
        stream: &[f32],
        tokens: &[u32],
        rope_cos: &[f32],
        rope_sin: &[f32],
        rope_cos_c: &[f32],
        rope_sin_c: &[f32],
    ) -> Result<Vec<f32>> {
        let names = self.config.layer_tensor_names(layer_idx);
        let seq = tokens.len();
        let d = self.config.hidden_size;
        let n_hc = self.config.hc_mult;

        let residual = stream.to_vec();
        let mhc = self.gen_mhc_params(
            stream,
            seq,
            &names.hc_attn.mix_fn,
            &names.hc_attn.base,
            &names.hc_attn.scale,
        )?;
        let mut sub_in = hc_pre(stream, &mhc.pre, seq, n_hc, d);
        self.rms_norm_in_place(&mut sub_in, &names.attn_norm)?;
        let attn_out = self.attention_forward(
            layer_idx,
            &sub_in,
            &names.attn.prefix,
            rope_cos,
            rope_sin,
            rope_cos_c,
            rope_sin_c,
        )?;
        let stream = hc_post(&attn_out, &residual, &mhc.post, &mhc.comb, seq, n_hc, d);

        let residual = stream.clone();
        let mhc = self.gen_mhc_params(
            &stream,
            seq,
            &names.hc_ffn.mix_fn,
            &names.hc_ffn.base,
            &names.hc_ffn.scale,
        )?;
        let mut sub_in = hc_pre(&stream, &mhc.pre, seq, n_hc, d);
        self.rms_norm_in_place(&mut sub_in, &names.ffn_norm)?;
        let ffn_out = self.moe_forward(layer_idx, &sub_in, tokens, &names.ffn)?;
        Ok(hc_post(
            &ffn_out, &residual, &mhc.post, &mhc.comb, seq, n_hc, d,
        ))
    }

    fn attention_forward(
        &self,
        layer_idx: usize,
        x: &[f32],
        prefix: &str,
        rope_cos: &[f32],
        rope_sin: &[f32],
        rope_cos_c: &[f32],
        rope_sin_c: &[f32],
    ) -> Result<Vec<f32>> {
        let seq = x.len() / self.config.hidden_size;
        let d = self.config.hidden_size;
        let heads = self.config.num_attention_heads;
        let c = self.config.head_dim;
        let rope_dim = self.config.qk_rope_head_dim;
        let compress_ratio = self.config.compress_ratios[layer_idx];
        let mode = self
            .config
            .attention_mode_for_compress_ratio(compress_ratio);

        let mut c_q = vec![0.0_f32; seq * self.config.q_lora_rank];
        let mut q = vec![0.0_f32; seq * heads * c];
        let mut kv_sw = vec![0.0_f32; seq * c];
        for t in 0..seq {
            let xt = &x[t * d..(t + 1) * d];
            let mut cq = self.tensors.matvec(&format!("{prefix}.wq_a.weight"), xt)?;
            self.rms_norm_vec_in_place(&mut cq, &format!("{prefix}.q_norm.weight"))?;
            c_q[t * self.config.q_lora_rank..(t + 1) * self.config.q_lora_rank]
                .copy_from_slice(&cq);

            let q_raw = self.tensors.matvec(&format!("{prefix}.wq_b.weight"), &cq)?;
            ensure!(q_raw.len() == heads * c, "attention q width mismatch");
            for h in 0..heads {
                let dst = &mut q[(t * heads + h) * c..(t * heads + h + 1) * c];
                dst.copy_from_slice(&q_raw[h * c..(h + 1) * c]);
                fixed_rms_norm_in_place(dst, self.config.rms_norm_eps);
                apply_partial_rope(
                    dst,
                    &rope_cos[t * rope_dim..(t + 1) * rope_dim],
                    &rope_sin[t * rope_dim..(t + 1) * rope_dim],
                    rope_dim,
                    1.0,
                );
            }

            let mut kv = self.tensors.matvec(&format!("{prefix}.wkv.weight"), xt)?;
            self.rms_norm_vec_in_place(&mut kv, &format!("{prefix}.kv_norm.weight"))?;
            apply_partial_rope(
                &mut kv,
                &rope_cos[t * rope_dim..(t + 1) * rope_dim],
                &rope_sin[t * rope_dim..(t + 1) * rope_dim],
                rope_dim,
                1.0,
            );
            kv_sw[t * c..(t + 1) * c].copy_from_slice(&kv);
        }

        let (kv_comp, csa_selected) = if compress_ratio > 0 {
            let attn_names = self.config.layer_tensor_names(layer_idx).attn;
            let comp = self.compressor_forward(
                attn_names
                    .compressor
                    .as_ref()
                    .expect("compress_ratio>0 has compressor"),
                x,
                c,
                compress_ratio,
                compress_ratio < 16,
            )?;
            let nb = comp.len() / c;
            let mut comp_rope = comp;
            for block in 0..nb {
                let pos = (block * compress_ratio + (compress_ratio - 1)).min(seq - 1);
                apply_partial_rope(
                    &mut comp_rope[block * c..(block + 1) * c],
                    &rope_cos_c[pos * rope_dim..(pos + 1) * rope_dim],
                    &rope_sin_c[pos * rope_dim..(pos + 1) * rope_dim],
                    rope_dim,
                    1.0,
                );
            }
            let selected = if compress_ratio < 16 {
                Some(self.csa_selected_blocks(
                    layer_idx,
                    x,
                    &c_q,
                    attn_names.indexer.as_ref().expect("CSA has indexer"),
                    compress_ratio,
                )?)
            } else {
                None
            };
            (Some(comp_rope), selected)
        } else {
            (None, None)
        };

        let mut attn_out = vec![0.0_f32; seq * heads * c];
        let sink = self.tensors.vec_f32(&format!("{prefix}.attn_sink"))?;
        let scale = 1.0 / (c as f32).sqrt();
        for t in 0..seq {
            for h in 0..heads {
                let qh = &q[(t * heads + h) * c..(t * heads + h + 1) * c];
                let mut logits = Vec::new();
                let mut values = Vec::new();

                if let Some(kv_comp) = &kv_comp {
                    let nb = kv_comp.len() / c;
                    match mode {
                        deepseek_spec::DeepSeekV4AttentionMode::HybridCompressed => {
                            for block in 0..nb {
                                let block_end = block * compress_ratio + (compress_ratio - 1);
                                if block_end < t {
                                    let value = &kv_comp[block * c..(block + 1) * c];
                                    logits.push(dot(qh, value) * scale);
                                    values.push(value);
                                }
                            }
                        }
                        deepseek_spec::DeepSeekV4AttentionMode::CompressedSparse => {
                            if let Some(selected) = &csa_selected {
                                for &block in &selected[t] {
                                    let value = &kv_comp[block * c..(block + 1) * c];
                                    logits.push(dot(qh, value) * scale);
                                    values.push(value);
                                }
                            }
                        }
                        deepseek_spec::DeepSeekV4AttentionMode::SlidingWindow => {}
                    }
                }

                let sw_start = (t + 1).saturating_sub(self.config.sliding_window);
                for key in sw_start..=t {
                    let value = &kv_sw[key * c..(key + 1) * c];
                    logits.push(dot(qh, value) * scale);
                    values.push(value);
                }

                let probs = sink_softmax(&logits, sink[h]);
                let dst = &mut attn_out[(t * heads + h) * c..(t * heads + h + 1) * c];
                for (prob, value) in probs.iter().zip(values) {
                    for col in 0..c {
                        dst[col] += prob * value[col];
                    }
                }
                apply_partial_rope(
                    dst,
                    &rope_cos[t * rope_dim..(t + 1) * rope_dim],
                    &rope_sin[t * rope_dim..(t + 1) * rope_dim],
                    rope_dim,
                    -1.0,
                );
            }
        }

        self.output_projection(prefix, &attn_out, seq)
    }

    fn output_projection(&self, prefix: &str, attn_out: &[f32], seq: usize) -> Result<Vec<f32>> {
        let heads_per_group = self.config.num_attention_heads / self.config.o_groups;
        let group_width = heads_per_group * self.config.head_dim;
        let rank = self.config.o_lora_rank;
        let latent_width = self.config.o_groups * rank;
        let d = self.config.hidden_size;
        let wo_a_name = format!("{prefix}.wo_a.weight");
        let wo_b_name = format!("{prefix}.wo_b.weight");
        let mut out = vec![0.0_f32; seq * d];
        for t in 0..seq {
            let mut latent = vec![0.0_f32; latent_width];
            for group in 0..self.config.o_groups {
                let mut group_in = vec![0.0_f32; group_width];
                for local_head in 0..heads_per_group {
                    let head = group * heads_per_group + local_head;
                    let src = &attn_out[(t * self.config.num_attention_heads + head)
                        * self.config.head_dim
                        ..(t * self.config.num_attention_heads + head + 1) * self.config.head_dim];
                    group_in[local_head * self.config.head_dim
                        ..(local_head + 1) * self.config.head_dim]
                        .copy_from_slice(src);
                }
                for r in 0..rank {
                    let row = group * rank + r;
                    latent[row] = self.tensors.row_dot(&wo_a_name, row, &group_in)?;
                }
            }
            let projected = self.tensors.matvec(&wo_b_name, &latent)?;
            out[t * d..(t + 1) * d].copy_from_slice(&projected);
        }
        Ok(out)
    }

    fn compressor_forward(
        &self,
        names: &DeepSeekV4CompressorTensorNames,
        x: &[f32],
        head_dim: usize,
        ratio: usize,
        overlap: bool,
    ) -> Result<Vec<f32>> {
        let seq = x.len() / self.config.hidden_size;
        let d = self.config.hidden_size;
        let coeff = if overlap { 2 } else { 1 };
        let width = coeff * head_dim;
        let padded = seq.next_multiple_of(ratio);
        let nb = padded / ratio;
        let ape = self.tensors.vec_f32(&names.ape)?;
        let mut kv = vec![0.0_f32; padded * width];
        let mut score = vec![0.0_f32; padded * width];
        for t in 0..seq {
            let xt = &x[t * d..(t + 1) * d];
            let k = self.tensors.matvec(&names.wkv, xt)?;
            let s = self.tensors.matvec(&names.wgate, xt)?;
            kv[t * width..(t + 1) * width].copy_from_slice(&k);
            score[t * width..(t + 1) * width].copy_from_slice(&s);
        }
        for t in 0..padded {
            let pos = t % ratio;
            for col in 0..width {
                score[t * width + col] += ape[pos * width + col];
            }
        }

        let mut out = vec![0.0_f32; nb * head_dim];
        for block in 0..nb {
            for col in 0..head_dim {
                let mut logits = Vec::with_capacity(if overlap { 2 * ratio } else { ratio });
                let mut values = Vec::with_capacity(logits.capacity());
                if overlap {
                    for pos in 0..ratio {
                        if block == 0 {
                            logits.push(f32::NEG_INFINITY);
                            values.push(0.0);
                        } else {
                            let t = (block - 1) * ratio + pos;
                            logits.push(score[t * width + col]);
                            values.push(kv[t * width + col]);
                        }
                    }
                    for pos in 0..ratio {
                        let t = block * ratio + pos;
                        logits.push(score[t * width + head_dim + col]);
                        values.push(kv[t * width + head_dim + col]);
                    }
                } else {
                    for pos in 0..ratio {
                        let t = block * ratio + pos;
                        logits.push(score[t * width + col]);
                        values.push(kv[t * width + col]);
                    }
                }
                let probs = softmax(&logits);
                out[block * head_dim + col] =
                    probs.iter().zip(values).map(|(p, v)| p * v).sum::<f32>();
            }
        }
        self.rms_norm_matrix_rows_in_place(&mut out, head_dim, &names.norm)?;
        Ok(out)
    }

    fn csa_selected_blocks(
        &self,
        layer_idx: usize,
        x: &[f32],
        c_q: &[f32],
        names: &deepseek_spec::DeepSeekV4IndexerTensorNames,
        ratio: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let seq = x.len() / self.config.hidden_size;
        let d = self.config.hidden_size;
        let keys = self.compressor_forward(
            &names.compressor,
            x,
            self.config.index_head_dim,
            ratio,
            true,
        )?;
        let nb = keys.len() / self.config.index_head_dim;
        let score_scale = (self.config.index_head_dim as f32).powf(-0.5)
            * (self.config.index_n_heads as f32).powf(-0.5);
        let mut out = vec![Vec::new(); seq];
        for t in 0..seq {
            let cq = &c_q[t * self.config.q_lora_rank..(t + 1) * self.config.q_lora_rank];
            let q_i = self.tensors.matvec(&names.wq_b, cq)?;
            let mut w_i = self
                .tensors
                .matvec(&names.weights_proj, &x[t * d..(t + 1) * d])?;
            for value in &mut w_i {
                *value *= score_scale;
            }
            let mut scored = Vec::new();
            for block in 0..nb {
                if block >= t / ratio {
                    continue;
                }
                let key = &keys
                    [block * self.config.index_head_dim..(block + 1) * self.config.index_head_dim];
                let mut score = 0.0_f32;
                for head in 0..self.config.index_n_heads {
                    let qh = &q_i[head * self.config.index_head_dim
                        ..(head + 1) * self.config.index_head_dim];
                    score += w_i[head] * dot(qh, key).max(0.0);
                }
                if score.is_finite() {
                    scored.push((score, block));
                }
            }
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            scored.truncate(self.config.index_topk.min(scored.len()));
            out[t] = scored.into_iter().map(|(_, block)| block).collect();
        }
        let _ = layer_idx;
        Ok(out)
    }

    fn moe_forward(
        &self,
        layer_idx: usize,
        x: &[f32],
        tokens: &[u32],
        names: &DeepSeekV4MoeTensorNames,
    ) -> Result<Vec<f32>> {
        let seq = tokens.len();
        let d = self.config.hidden_size;
        let mut out = vec![0.0_f32; seq * d];
        let gate_bias = names
            .gate_bias
            .as_ref()
            .map(|name| self.tensors.vec_f32(name))
            .transpose()?;
        let hash_routing = layer_idx < self.config.num_hash_layers;

        for t in 0..seq {
            let xt = &x[t * d..(t + 1) * d];
            let logits = self.tensors.matvec(&names.gate_weight, xt)?;
            let scores = self.config.router_scores_from_logits(&logits)?;
            let routes = if hash_routing {
                let table = names
                    .gate_tid2eid
                    .as_ref()
                    .context("hash-routed V4 layer missing tid2eid")?;
                let experts = self.tensors.row_usize(table, tokens[t] as usize)?;
                self.config
                    .moe_routes_from_scores(layer_idx, t, &scores, None, Some(&experts))?
            } else {
                self.config.moe_routes_from_scores(
                    layer_idx,
                    t,
                    &scores,
                    gate_bias.as_deref(),
                    None,
                )?
            };

            for route in routes {
                let expert = names.expert(route.expert_idx);
                let y = self.expert_forward(&expert.prefix, xt)?;
                for col in 0..d {
                    out[t * d + col] += route.weight * y[col];
                }
            }
            if let Some(shared) = &names.shared_experts {
                let y = self.expert_forward(&shared.prefix, xt)?;
                for col in 0..d {
                    out[t * d + col] += y[col];
                }
            }
        }
        Ok(out)
    }

    fn expert_forward(&self, prefix: &str, x: &[f32]) -> Result<Vec<f32>> {
        let mut gate = self.tensors.matvec(&format!("{prefix}.w1.weight"), x)?;
        let mut up = self.tensors.matvec(&format!("{prefix}.w3.weight"), x)?;
        for value in &mut up {
            *value = value.clamp(-self.config.swiglu_limit, self.config.swiglu_limit);
        }
        for value in &mut gate {
            *value = value.min(self.config.swiglu_limit);
        }
        let hidden = gate
            .into_iter()
            .zip(up)
            .map(|(g, u)| silu(g) * u)
            .collect::<Vec<_>>();
        self.tensors.matvec(&format!("{prefix}.w2.weight"), &hidden)
    }

    fn gen_mhc_params(
        &self,
        stream: &[f32],
        seq: usize,
        fn_name: &str,
        base_name: &str,
        scale_name: &str,
    ) -> Result<MhcParams> {
        let d = self.config.hidden_size;
        let n = self.config.hc_mult;
        let flat = n * d;
        let mix = (2 + n) * n;
        let base = self.tensors.vec_f32(base_name)?;
        let scale = self.tensors.vec_f32(scale_name)?;
        let mut pre = vec![0.0_f32; seq * n];
        let mut post = vec![0.0_f32; seq * n];
        let mut comb = vec![0.0_f32; seq * n * n];
        for t in 0..seq {
            let row = &stream[t * flat..(t + 1) * flat];
            let rsqrt = rms_rsqrt(row, self.config.hc_eps);
            let mut mixes = vec![0.0_f32; mix];
            for m in 0..mix {
                mixes[m] = self.tensors.row_dot(fn_name, m, row)? * rsqrt;
            }
            for i in 0..n {
                pre[t * n + i] = sigmoid(scale[0] * mixes[i] + base[i]) + self.config.hc_eps;
                post[t * n + i] = 2.0 * sigmoid(scale[1] * mixes[n + i] + base[n + i]);
            }
            let mut raw = vec![0.0_f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    let idx = i * n + j;
                    raw[idx] = scale[2] * mixes[2 * n + idx] + base[2 * n + idx];
                }
            }
            row_softmax_plus_eps(&mut raw, n, self.config.hc_eps);
            column_normalize(&mut raw, n, self.config.hc_eps);
            for _ in 1..self.config.hc_sinkhorn_iters {
                row_normalize(&mut raw, n, self.config.hc_eps);
                column_normalize(&mut raw, n, self.config.hc_eps);
            }
            comb[t * n * n..(t + 1) * n * n].copy_from_slice(&raw);
        }
        Ok(MhcParams { pre, post, comb })
    }

    fn gen_head_pre(
        &self,
        token_idx: usize,
        stream: &[f32],
        fn_name: &str,
        base_name: &str,
        scale_name: &str,
    ) -> Result<Vec<f32>> {
        let d = self.config.hidden_size;
        let n = self.config.hc_mult;
        let flat = n * d;
        let row = &stream[token_idx * flat..(token_idx + 1) * flat];
        let rsqrt = rms_rsqrt(row, self.config.hc_eps);
        let base = self.tensors.vec_f32(base_name)?;
        let scale = self.tensors.vec_f32(scale_name)?;
        let s = scale.first().copied().unwrap_or(1.0);
        let mut pre = vec![0.0_f32; n];
        for i in 0..n {
            let mix = self.tensors.row_dot(fn_name, i, row)? * rsqrt;
            pre[i] = sigmoid(s * mix + base[i]) + self.config.hc_eps;
        }
        Ok(pre)
    }

    fn rms_norm(&self, x: &[f32], weight_name: &str) -> Result<Vec<f32>> {
        let mut out = x.to_vec();
        self.rms_norm_vec_in_place(&mut out, weight_name)?;
        Ok(out)
    }

    fn rms_norm_in_place(&self, x: &mut [f32], weight_name: &str) -> Result<()> {
        let d = self.config.hidden_size;
        for row in x.chunks_exact_mut(d) {
            self.rms_norm_vec_in_place(row, weight_name)?;
        }
        Ok(())
    }

    fn rms_norm_matrix_rows_in_place(
        &self,
        x: &mut [f32],
        row_width: usize,
        weight_name: &str,
    ) -> Result<()> {
        for row in x.chunks_exact_mut(row_width) {
            self.rms_norm_vec_in_place(row, weight_name)?;
        }
        Ok(())
    }

    fn rms_norm_vec_in_place(&self, x: &mut [f32], weight_name: &str) -> Result<()> {
        let weight = self.tensors.vec_f32(weight_name)?;
        ensure!(
            weight.len() == x.len(),
            "RMSNorm weight {} len {} does not match row len {}",
            weight_name,
            weight.len(),
            x.len()
        );
        let scale = rms_rsqrt(x, self.config.rms_norm_eps);
        for (value, weight) in x.iter_mut().zip(weight) {
            *value *= scale * weight;
        }
        Ok(())
    }
}

impl SafeTensorStore {
    fn open_model_dir(model_dir: &Path) -> Result<Self> {
        let shard_paths = resolve_safetensor_shards(model_dir)?;
        Self::open_shards(&shard_paths)
    }

    fn open_shards(shard_paths: &[String]) -> Result<Self> {
        ensure!(
            !shard_paths.is_empty(),
            "DeepSeek V4 reference needs at least one safetensors shard"
        );
        let mut mmaps = Vec::with_capacity(shard_paths.len());
        let mut tensors = HashMap::new();
        for (shard, path) in shard_paths.iter().enumerate() {
            let mmap = Self::mmap_one(Path::new(path))?;
            Self::parse_header(shard, path, &mmap, &mut tensors)?;
            mmaps.push(mmap);
        }
        Ok(Self { mmaps, tensors })
    }

    fn mmap_one(path: &Path) -> Result<Mmap> {
        let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        unsafe { Mmap::map(&file) }.with_context(|| format!("mmap safetensors {}", path.display()))
    }

    fn parse_header(
        shard: usize,
        path: &str,
        mmap: &Mmap,
        tensors: &mut HashMap<String, TensorMeta>,
    ) -> Result<()> {
        ensure!(mmap.len() >= 8, "{path} is too small");
        let mut len_bytes = [0_u8; 8];
        len_bytes.copy_from_slice(&mmap[..8]);
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        let header_start = 8;
        let header_end = header_start + header_len;
        ensure!(
            header_end <= mmap.len(),
            "{} safetensors header exceeds file size",
            path
        );
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&mmap[header_start..header_end])
                .with_context(|| format!("parsing safetensors header {path}"))?;
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let tensor: HeaderTensor = serde_json::from_value(value)
                .with_context(|| format!("parsing tensor metadata {name}"))?;
            let dtype = match tensor.dtype.as_str() {
                "BF16" => TensorDtype::Bf16,
                "F32" => TensorDtype::F32,
                "I64" => TensorDtype::I64,
                _ => TensorDtype::Unsupported,
            };
            let start = header_end + tensor.data_offsets[0];
            let end = header_end + tensor.data_offsets[1];
            ensure!(
                start <= end && end <= mmap.len(),
                "tensor {name} offsets out of range"
            );
            ensure!(
                tensors
                    .insert(
                        name.clone(),
                        TensorMeta {
                            shard,
                            dtype,
                            shape: tensor.shape,
                            start,
                            end,
                        },
                    )
                    .is_none(),
                "duplicate tensor {name} across DeepSeek V4 shards"
            );
        }
        Ok(())
    }

    fn meta(&self, name: &str) -> Result<&TensorMeta> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))
    }

    fn data(&self, meta: &TensorMeta) -> &[u8] {
        &self.mmaps[meta.shard][meta.start..meta.end]
    }

    fn vec_f32(&self, name: &str) -> Result<Vec<f32>> {
        let meta = self.meta(name)?;
        let len = meta.shape.iter().product();
        (0..len).map(|idx| self.f32_at(meta, idx)).collect()
    }

    fn row_f32(&self, name: &str, row: usize) -> Result<Vec<f32>> {
        let meta = self.meta(name)?;
        ensure!(meta.shape.len() == 2, "tensor {name} must be 2D");
        let rows = meta.shape[0];
        let cols = meta.shape[1];
        ensure!(
            row < rows,
            "row {row} out of range for {name}[{rows},{cols}]"
        );
        (0..cols)
            .map(|col| self.f32_at(meta, row * cols + col))
            .collect()
    }

    fn row_usize(&self, name: &str, row: usize) -> Result<Vec<usize>> {
        let meta = self.meta(name)?;
        ensure!(meta.dtype == TensorDtype::I64, "tensor {name} must be I64");
        ensure!(meta.shape.len() == 2, "tensor {name} must be 2D");
        let rows = meta.shape[0];
        let cols = meta.shape[1];
        ensure!(row < rows, "row {row} out of range for {name}");
        (0..cols)
            .map(|col| {
                let offset = (row * cols + col) * 8;
                let mut bytes = [0_u8; 8];
                bytes.copy_from_slice(&self.data(meta)[offset..offset + 8]);
                let value = i64::from_le_bytes(bytes);
                ensure!(value >= 0, "tensor {name} has negative expert id {value}");
                Ok(value as usize)
            })
            .collect()
    }

    fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        let meta = self.meta(name)?;
        ensure!(meta.shape.len() == 2, "tensor {name} must be 2D");
        let rows = meta.shape[0];
        let cols = meta.shape[1];
        ensure!(
            x.len() == cols,
            "matvec {name} expects input len {cols}, got {}",
            x.len()
        );
        let mut out = vec![0.0_f32; rows];
        for row in 0..rows {
            out[row] = self.row_dot(name, row, x)?;
        }
        Ok(out)
    }

    fn row_dot(&self, name: &str, row: usize, x: &[f32]) -> Result<f32> {
        let meta = self.meta(name)?;
        ensure!(meta.shape.len() == 2, "tensor {name} must be 2D");
        let rows = meta.shape[0];
        let cols = meta.shape[1];
        ensure!(row < rows, "row {row} out of range for {name}");
        ensure!(
            x.len() == cols,
            "row_dot {name} expects input len {cols}, got {}",
            x.len()
        );
        let base = row * cols;
        let mut acc = 0.0_f32;
        for col in 0..cols {
            acc += self.f32_at(meta, base + col)? * x[col];
        }
        Ok(acc)
    }

    fn f32_at(&self, meta: &TensorMeta, idx: usize) -> Result<f32> {
        match meta.dtype {
            TensorDtype::Bf16 => {
                let offset = idx * 2;
                ensure!(
                    offset + 2 <= meta.end - meta.start,
                    "BF16 tensor read out of range"
                );
                let data = self.data(meta);
                Ok(bf16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]])).to_f32())
            }
            TensorDtype::F32 => {
                let offset = idx * 4;
                ensure!(
                    offset + 4 <= meta.end - meta.start,
                    "F32 tensor read out of range"
                );
                let mut bytes = [0_u8; 4];
                bytes.copy_from_slice(&self.data(meta)[offset..offset + 4]);
                Ok(f32::from_le_bytes(bytes))
            }
            TensorDtype::I64 => bail!("cannot read I64 tensor as f32"),
            TensorDtype::Unsupported => bail!("cannot read unsupported tensor dtype as f32"),
        }
    }
}

struct MhcParams {
    pre: Vec<f32>,
    post: Vec<f32>,
    comb: Vec<f32>,
}

fn resolve_safetensor_shards(model_dir: &Path) -> Result<Vec<String>> {
    let single = model_dir.join("model.safetensors");
    let index = model_dir.join("model.safetensors.index.json");
    if single.exists() && !index.exists() {
        return Ok(vec![single.to_string_lossy().into_owned()]);
    }

    let content = fs::read_to_string(&index)
        .with_context(|| format!("reading safetensors index {}", index.display()))?;
    let index_json: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("parsing safetensors index {}", index.display()))?;
    let weight_map = index_json["weight_map"]
        .as_object()
        .with_context(|| format!("{} missing weight_map", index.display()))?;

    let mut shard_names = weight_map
        .values()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .with_context(|| format!("{} contains a non-string shard path", index.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    shard_names.sort();
    shard_names.dedup();
    ensure!(
        !shard_names.is_empty(),
        "{} does not reference any safetensors shards",
        index.display()
    );
    Ok(shard_names
        .into_iter()
        .map(|name| model_dir.join(name).to_string_lossy().into_owned())
        .collect())
}

fn stream_slice(stream: &[f32], token: usize, hc: usize, n_hc: usize, d: usize) -> &[f32] {
    let start = (token * n_hc + hc) * d;
    &stream[start..start + d]
}

fn stream_slice_mut(
    stream: &mut [f32],
    token: usize,
    hc: usize,
    n_hc: usize,
    d: usize,
) -> &mut [f32] {
    let start = (token * n_hc + hc) * d;
    &mut stream[start..start + d]
}

fn hc_pre(stream: &[f32], pre: &[f32], seq: usize, n_hc: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; seq * d];
    for t in 0..seq {
        for hc in 0..n_hc {
            let weight = pre[t * n_hc + hc];
            let src = stream_slice(stream, t, hc, n_hc, d);
            for col in 0..d {
                out[t * d + col] += weight * src[col];
            }
        }
    }
    out
}

fn hc_post(
    new_x: &[f32],
    residual: &[f32],
    post: &[f32],
    comb: &[f32],
    seq: usize,
    n_hc: usize,
    d: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; seq * n_hc * d];
    for t in 0..seq {
        for i in 0..n_hc {
            let dst = stream_slice_mut(&mut out, t, i, n_hc, d);
            let post_weight = post[t * n_hc + i];
            for col in 0..d {
                dst[col] = post_weight * new_x[t * d + col];
            }
            for j in 0..n_hc {
                let weight = comb[(t * n_hc + i) * n_hc + j];
                let src = stream_slice(residual, t, j, n_hc, d);
                for col in 0..d {
                    dst[col] += weight * src[col];
                }
            }
        }
    }
    out
}

fn build_rope_cache(seq: usize, dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    if dim == 0 {
        return (Vec::new(), Vec::new());
    }
    let half = dim / 2;
    let inv_freq = (0..half)
        .map(|i| 1.0_f32 / base.powf((2 * i) as f32 / dim as f32))
        .collect::<Vec<_>>();
    let mut cos = vec![0.0_f32; seq * dim];
    let mut sin = vec![0.0_f32; seq * dim];
    for pos in 0..seq {
        for i in 0..half {
            let value = pos as f32 * inv_freq[i];
            let c = value.cos();
            let s = value.sin();
            cos[pos * dim + i] = c;
            cos[pos * dim + half + i] = c;
            sin[pos * dim + i] = s;
            sin[pos * dim + half + i] = s;
        }
    }
    (cos, sin)
}

fn apply_partial_rope(x: &mut [f32], cos: &[f32], sin: &[f32], rope_dim: usize, sign: f32) {
    if rope_dim == 0 {
        return;
    }
    let start = x.len() - rope_dim;
    let half = rope_dim / 2;
    for idx in 0..half {
        let a = x[start + idx];
        let b = x[start + half + idx];
        let s = sign * sin[idx];
        x[start + idx] = a * cos[idx] - b * s;
        x[start + half + idx] = b * cos[idx] + a * s;
    }
}

fn sink_softmax(logits: &[f32], sink: f32) -> Vec<f32> {
    let max = logits.iter().copied().fold(sink, f32::max);
    let denom = logits.iter().map(|value| (*value - max).exp()).sum::<f32>() + (sink - max).exp();
    logits
        .iter()
        .map(|value| (*value - max).exp() / denom)
        .collect()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![0.0; logits.len()];
    }
    let exp = logits
        .iter()
        .map(|value| (*value - max).exp())
        .collect::<Vec<_>>();
    let denom = exp.iter().sum::<f32>();
    exp.into_iter().map(|value| value / denom).collect()
}

fn row_softmax_plus_eps(raw: &mut [f32], n: usize, eps: f32) {
    for row in 0..n {
        let start = row * n;
        let probs = softmax(&raw[start..start + n]);
        for col in 0..n {
            raw[start + col] = probs[col] + eps;
        }
    }
}

fn row_normalize(raw: &mut [f32], n: usize, eps: f32) {
    for row in 0..n {
        let start = row * n;
        let sum = raw[start..start + n].iter().sum::<f32>() + eps;
        for col in 0..n {
            raw[start + col] /= sum;
        }
    }
}

fn column_normalize(raw: &mut [f32], n: usize, eps: f32) {
    for col in 0..n {
        let mut sum = eps;
        for row in 0..n {
            sum += raw[row * n + col];
        }
        for row in 0..n {
            raw[row * n + col] /= sum;
        }
    }
}

fn fixed_rms_norm_in_place(x: &mut [f32], eps: f32) {
    let scale = rms_rsqrt(x, eps);
    for value in x {
        *value *= scale;
    }
}

fn rms_rsqrt(x: &[f32], eps: f32) -> f32 {
    let mean = x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
    1.0 / (mean + eps).sqrt()
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models/dsv4-mini-1B-init")
    }

    #[test]
    fn dsv4_reference_resolves_sharded_safetensors_index() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{
              "metadata": {},
              "weight_map": {
                "embed.weight": "model-00002-of-00002.safetensors",
                "head.weight": "model-00001-of-00002.safetensors",
                "norm.weight": "model-00002-of-00002.safetensors"
              }
            }"#,
        )
        .unwrap();

        let shards = resolve_safetensor_shards(dir.path()).unwrap();
        assert_eq!(
            shards,
            vec![
                dir.path()
                    .join("model-00001-of-00002.safetensors")
                    .to_string_lossy()
                    .into_owned(),
                dir.path()
                    .join("model-00002-of-00002.safetensors")
                    .to_string_lossy()
                    .into_owned(),
            ]
        );
    }

    #[test]
    fn dsv4_reference_loads_1b_manifest() {
        let model = DeepseekV4ReferenceModel::load(model_path()).unwrap();
        assert_eq!(model.config.model_type, "deepseek_v4");
        assert_eq!(model.config.hidden_size, 1024);
        assert_eq!(model.config.num_hidden_layers, 24);
        assert_eq!(model.config.vocab_size, 129_280);
    }

    #[test]
    #[ignore = "slow CPU reference forward over the full 1B checkpoint"]
    fn dsv4_reference_one_token_forward_logits_shape() {
        let model = DeepseekV4ReferenceModel::load(model_path()).unwrap();
        let logits = model.forward_last_logits(&[0]).unwrap();
        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(logits.iter().all(|value| value.is_finite()));
    }
}
