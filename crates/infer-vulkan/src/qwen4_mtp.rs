//! The `qwen4_exp` MTP head — the model's own speculative-decode DRAFTER.
//!
//! `mtp.*` is one extra decoder layer (full attention + top-10-of-512 MoE +
//! hyper-connections) with four fusion tensors and its OWN
//! `hyper_connection_mixer`. No reference forward exists — transformers
//! ignores the prefix on load (`_keys_to_ignore_on_load_unexpected =
//! [r"^mtp.*"]`, modeling_qwen4_exp.py:1256/1535) — so the semantics here are
//! pinned two ways: every `mtp.layers.0.*` submodule reuses the text decoder's
//! exact suffix vocabulary (verified class by class in `qwen4_names`), and the
//! four `mtp.*`-only tensors are the DeepSeek-V3 MTP fusion adapted to
//! hyper-connections, pinned by SHAPE:
//!
//! - `pre_fc_norm_hidden` is `[10240]` — it norms the target's PRE-mixer
//!   4-stream residual (a `[2560]` norm would mean the post-mixer state);
//!   `pre_fc_norm_embedding` `[2560]` norms the next token's embedding row.
//! - `fc_hidden`/`fc_embedding` (both `[2560, 2560]`) fuse the two into the
//!   10240-wide input of `mtp.layers.0`: `fc_hidden` applies PER STREAM,
//!   `fc_embedding` broadcasts —
//!   `h_in[s] = fc_hidden @ norm(h)[s] + fc_embedding @ norm(e)`.
//!
//! Recurrent drafting (DeepSeek-V3 / vLLM MTP style): draft step 1 fuses the
//! target's pre-mixer `h` at the last verified position with the embedding of
//! the last EMITTED token; step `j > 1` fuses the MTP block's own 10240-wide
//! output from step `j - 1` with the embedding of draft `j - 1`. Logits come
//! from the MTP's own mixer through the SHARED `lm_head`. RoPE positions are
//! token-side: the entry that embeds the token at absolute position `q + 1`
//! ropes at `q + 1`.
//!
//! Drafts are PROPOSALS: any numeric or semantic error here costs acceptance
//! rate, never correctness — the greedy equivalence gate in
//! `tests/qwen4_speculative.rs` holds no matter what this module outputs.
//!
//! # KV canon
//!
//! The MTP layer attends over its own KV. A canonical entry at position `q`
//! was built from the TARGET's `h[q]` and the real token at `q + 1`; draft
//! steps past the first build entries from the MTP's own recurrent `h` and
//! unverified tokens, which are speculative and get truncated before the next
//! draft call. Accepted positions re-enter as catch-up entries (KV-only
//! forwards — fuse + attention-site gated residual + k/v projection, no
//! SDPA/MoE/logits) using the verify pass's `h` rows, so the canon never
//! drifts from what training saw. After a prompt prefill the canon starts at
//! the last prefill chunk (earlier chunks' `h` rows are gone) — a shorter
//! attention prefix costs draft quality only.
//!
//! # Host oracle vs device route
//!
//! [`MtpHead::forward`] is ONE transcription with a routing seam, the same
//! split as the main model's host lane: `route = None` is the pure host
//! oracle; `Some((dev, weights))` moves every wide matvec onto the device
//! through [`DenseGemv`] and the hyper-connection sites through the
//! `Qwen4HcMix` kernels ([`MTP_HC_LAYER`]). The recurrence-free math — norms,
//! SiLU/sigmoid, SDPA over the MTP's own KV — stays host-side in both.
//! `tests/qwen4_speculative.rs` diffs the two routes on real weights.

use anyhow::{Result, anyhow, ensure};
use infer_gguf::safetensors::SafeTensorsDir;

use crate::model_qwen4_exp::{
    DenseGemv, HostDense, HostFullAttn, HostKv, HostMoe, Qwen4Dev, head_rms_norm_bias,
    host_router_topk, host_shared_expert, load_hc, rope_partial, round_to_f16, sigmoid32, silu32,
};
use crate::qwen4_config::Qwen4ExpConfig;
use crate::qwen4_hc::{
    GatedResidualWeights, HyperConnectionConfig, gated_residual, grouped_rmsnorm,
    inject_block_output,
};
use crate::qwen4_names::{ExpertProj, HcSite};
use crate::qwen4_upload::{MTP_HC_LAYER, MTP_PREFIX, Qwen4Weights, mtp_expert_slice_name};

/// The device half of the routing seam: the runner plus the resident weights.
pub type MtpRoute<'a, 'ctx, 'st> = (&'a mut Qwen4Dev<'ctx>, &'a Qwen4Weights<'ctx, 'st>);

/// Host weights of the MTP head (borrowed from the checkpoint mmaps).
pub struct MtpHead<'st> {
    /// `fc_embedding` `[2560, 2560]` — broadcast half of the fusion.
    fc_embedding: HostDense<'st>,
    /// `fc_hidden` `[2560, 2560]` — per-stream half.
    fc_hidden: HostDense<'st>,
    /// `pre_fc_norm_embedding` `[2560]`, RAW (host applies `1 + w`).
    pre_fc_norm_embedding: Vec<f32>,
    /// `pre_fc_norm_hidden` `[10240]`, RAW, grouped at `hidden_size`.
    pre_fc_norm_hidden: Vec<f32>,
    attn_hc: GatedResidualWeights,
    mlp_hc: GatedResidualWeights,
    /// The MTP's own mixer (`use_combine = False`).
    mixer: GatedResidualWeights,
    full: HostFullAttn<'st>,
    moe: HostMoe<'st>,
    /// Stacked `experts.gate_up_proj` bytes, `[n_experts][2*inter][hidden]`.
    gate_up: &'st [u8],
    /// Stacked `experts.down_proj` bytes, `[n_experts][hidden][inter]`.
    down: &'st [u8],
    n_experts: usize,
    inter: usize,
    hidden: usize,
}

/// One MTP forward's outputs.
pub struct MtpForwardOut {
    /// The block's 10240-wide output (pre its own mixer) — the next draft
    /// step's recurrent conditioning.
    pub h_out: Vec<f32>,
    /// `lm_head` logits, when asked for (`None` on KV-only catch-up).
    pub logits: Option<Vec<f32>>,
}

impl<'st> MtpHead<'st> {
    /// Load the head's host weights off the checkpoint.
    pub fn load(st: &'st SafeTensorsDir, cfg: &Qwen4ExpConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        let hc = crate::model_qwen4_exp::hc_config(cfg);
        let hd = cfg.head_dim;
        let name = |sfx: &str| format!("{MTP_PREFIX}layers.0.{sfx}");
        let full = HostFullAttn {
            q: HostDense::load(
                st,
                &name("self_attn.q_proj.weight"),
                h,
                cfg.num_attention_heads * hd * 2,
            )?,
            k: HostDense::load(
                st,
                &name("self_attn.k_proj.weight"),
                h,
                cfg.num_key_value_heads * hd,
            )?,
            v: HostDense::load(
                st,
                &name("self_attn.v_proj.weight"),
                h,
                cfg.num_key_value_heads * hd,
            )?,
            o: HostDense::load(
                st,
                &name("self_attn.o_proj.weight"),
                cfg.num_attention_heads * hd,
                h,
            )?,
            q_norm: crate::model_qwen4_exp::f32_tensor(st, &name("self_attn.q_norm.weight"), hd)?,
            k_norm: crate::model_qwen4_exp::f32_tensor(st, &name("self_attn.k_norm.weight"), hd)?,
        };
        let moe = HostMoe {
            router: HostDense::load(st, &name("mlp.gate.weight"), h, cfg.num_experts)?,
            shexp_gate: HostDense::load(st, &name("mlp.shared_expert_gate.weight"), h, 1)?,
            sh_gate: HostDense::load(
                st,
                &name("mlp.shared_expert.gate_proj.weight"),
                h,
                cfg.shared_expert_intermediate_size,
            )?,
            sh_up: HostDense::load(
                st,
                &name("mlp.shared_expert.up_proj.weight"),
                h,
                cfg.shared_expert_intermediate_size,
            )?,
            sh_down: HostDense::load(
                st,
                &name("mlp.shared_expert.down_proj.weight"),
                cfg.shared_expert_intermediate_size,
                h,
            )?,
        };
        let stack = |sfx: &str, want_dims: [usize; 3]| -> Result<&'st [u8]> {
            let full_name = name(sfx);
            let info = st
                .tensor(&full_name)
                .ok_or_else(|| anyhow!("missing `{full_name}`"))?;
            ensure!(info.dtype == "BF16", "`{full_name}` is {}", info.dtype);
            // `dims` is GGUF ne order (innermost first).
            let got: Vec<u64> = info.dims.to_vec();
            let want: Vec<u64> = want_dims.iter().rev().map(|&d| d as u64).collect();
            ensure!(
                got == want,
                "`{full_name}` dims {got:?} != {want:?} (ne order)"
            );
            st.tensor_data(&full_name)
        };
        let inter = cfg.moe_intermediate_size;
        let gate_up = stack("mlp.experts.gate_up_proj", [cfg.num_experts, 2 * inter, h])?;
        let down = stack("mlp.experts.down_proj", [cfg.num_experts, h, inter])?;
        Ok(Self {
            fc_embedding: HostDense::load(st, &format!("{MTP_PREFIX}fc_embedding.weight"), h, h)?,
            fc_hidden: HostDense::load(st, &format!("{MTP_PREFIX}fc_hidden.weight"), h, h)?,
            pre_fc_norm_embedding: crate::model_qwen4_exp::f32_tensor(
                st,
                &format!("{MTP_PREFIX}pre_fc_norm_embedding.weight"),
                h,
            )?,
            pre_fc_norm_hidden: crate::model_qwen4_exp::f32_tensor(
                st,
                &format!("{MTP_PREFIX}pre_fc_norm_hidden.weight"),
                hc.hc_count * h,
            )?,
            attn_hc: load_hc(
                st,
                &format!("{MTP_PREFIX}layers.0.attn_hyper_connection"),
                &hc,
                false,
            )?,
            mlp_hc: load_hc(
                st,
                &format!("{MTP_PREFIX}layers.0.mlp_hyper_connection"),
                &hc,
                false,
            )?,
            mixer: load_hc(
                st,
                &format!("{MTP_PREFIX}hyper_connection_mixer"),
                &hc,
                true,
            )?,
            full,
            moe,
            gate_up,
            down,
            n_experts: cfg.num_experts,
            inter,
            hidden: h,
        })
    }

    /// One routed expert's projection as a [`HostDense`] view into the stack.
    /// The name is the slice's device-twin key, so [`DenseGemv`] routes it to
    /// the uploaded per-expert suballocation when resident. Public because the
    /// parity harness diffs individual slices host-vs-device — the slice
    /// addressing is the one piece of the MTP route no text-stream test
    /// exercises.
    pub fn expert_dense(&self, expert: usize, proj: ExpertProj) -> Result<HostDense<'st>> {
        ensure!(expert < self.n_experts, "expert {expert} out of range");
        let (bytes, rows_per_expert, row_at, nrows, ncols) = match proj {
            // Fused [gate; up]: gate rows first (`chunk(2, dim=-1)` of the
            // linear output in Qwen4ExpTextExperts.forward).
            ExpertProj::Gate => (self.gate_up, 2 * self.inter, 0, self.inter, self.hidden),
            ExpertProj::Up => (
                self.gate_up,
                2 * self.inter,
                self.inter,
                self.inter,
                self.hidden,
            ),
            ExpertProj::Down => (self.down, self.hidden, 0, self.hidden, self.inter),
        };
        let start = (expert * rows_per_expert + row_at) * ncols * 2;
        let len = nrows * ncols * 2;
        HostDense::from_bf16_rows(
            mtp_expert_slice_name(u32::try_from(expert)?, proj),
            &bytes[start..start + len],
            ncols,
            nrows,
        )
    }

    /// `Qwen4ExpTextSparseMoeBlock` over the stacked BF16 experts: router
    /// softmax top-k (renormalised), routed experts, sigmoid-gated shared
    /// expert. Matvecs route through `gemv` when present.
    fn moe(
        &self,
        cfg: &Qwen4ExpConfig,
        x: &[f32],
        mut route: Option<&mut MtpRoute<'_, '_, '_>>,
    ) -> Result<Vec<f32>> {
        let logits = match route.as_mut() {
            Some((d, w)) => DenseGemv::new(d, w).matvec(&self.moe.router, x)?,
            None => self.moe.router.matvec(x),
        };
        let (ids, weights) = host_router_topk(&logits, cfg.num_experts_per_tok, cfg.norm_topk_prob);
        let mut y = vec![0.0f32; self.hidden];
        for (&e, &wt) in ids.iter().zip(&weights) {
            let e = usize::try_from(e).map_err(|_| anyhow!("negative expert id"))?;
            let gate_m = self.expert_dense(e, ExpertProj::Gate)?;
            let up_m = self.expert_dense(e, ExpertProj::Up)?;
            let mut pair = match route.as_mut() {
                Some((d, w)) => DenseGemv::new(d, w).matvec_many(&[&gate_m, &up_m], x)?,
                None => vec![gate_m.matvec(x), up_m.matvec(x)],
            }
            .into_iter();
            let g = pair.next().expect("gate");
            let u = pair.next().expect("up");
            let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu32(g) * u).collect();
            let down_m = self.expert_dense(e, ExpertProj::Down)?;
            let d_out = match route.as_mut() {
                Some((d, w)) => DenseGemv::new(d, w).matvec(&down_m, &act)?,
                None => down_m.matvec(&act),
            };
            for (yv, &dv) in y.iter_mut().zip(&d_out) {
                *yv += wt * dv;
            }
        }
        let shared = match route.as_mut() {
            Some((d, w)) => {
                let mut gemv = DenseGemv::new(d, w);
                host_shared_expert(&self.moe, x, Some(&mut gemv))?
            }
            None => host_shared_expert(&self.moe, x, None)?,
        };
        for (yv, &sv) in y.iter_mut().zip(&shared) {
            *yv += sv;
        }
        Ok(y)
    }

    /// The MTP layer's full attention over its OWN KV, decoupled from the
    /// text oracle's `pos == kv entries` invariant: the MTP canon starts
    /// mid-sequence, so `rope_pos` (token-side absolute position) and the KV
    /// length are independent. `kv_only` computes and appends this entry's
    /// K/V and returns `None` — the catch-up path, which needs no attention
    /// output. Otherwise the block output `[hidden]` comes back.
    fn attention(
        &self,
        cfg: &Qwen4ExpConfig,
        x: &[f32],
        rope_pos: usize,
        kv: &mut HostKv,
        kv_only: bool,
        mut route: Option<&mut MtpRoute<'_, '_, '_>>,
    ) -> Result<Option<Vec<f32>>> {
        let hd = cfg.head_dim;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let group = nq / nkv;
        let kv_dim = nkv * hd;

        let (q_full, mut k_new, v_new) = if kv_only {
            let mut kv_proj = match route.as_mut() {
                Some((d, w)) => {
                    DenseGemv::new(d, w).matvec_many(&[&self.full.k, &self.full.v], x)?
                }
                None => vec![self.full.k.matvec(x), self.full.v.matvec(x)],
            }
            .into_iter();
            (
                Vec::new(),
                kv_proj.next().expect("k"),
                kv_proj.next().expect("v"),
            )
        } else {
            let mut proj = match route.as_mut() {
                Some((d, w)) => DenseGemv::new(d, w)
                    .matvec_many(&[&self.full.q, &self.full.k, &self.full.v], x)?,
                None => vec![
                    self.full.q.matvec(x),
                    self.full.k.matvec(x),
                    self.full.v.matvec(x),
                ],
            }
            .into_iter();
            (
                proj.next().expect("q"),
                proj.next().expect("k"),
                proj.next().expect("v"),
            )
        };

        for h in 0..nkv {
            let head = &mut k_new[h * hd..(h + 1) * hd];
            head_rms_norm_bias(head, &self.full.k_norm, cfg.rms_norm_eps);
            rope_partial(head, cfg.rotary_dim, rope_pos, cfg.rope_theta);
        }
        // Same f16 write contract as the text KV caches.
        kv.k.extend(k_new.iter().map(|&v| round_to_f16(v)));
        kv.v.extend(v_new.iter().map(|&v| round_to_f16(v)));
        if kv_only {
            return Ok(None);
        }

        let mut q_roped = vec![0.0f32; nq * hd];
        for h in 0..nq {
            q_roped[h * hd..(h + 1) * hd].copy_from_slice(&q_full[h * 2 * hd..h * 2 * hd + hd]);
            let head = &mut q_roped[h * hd..(h + 1) * hd];
            head_rms_norm_bias(head, &self.full.q_norm, cfg.rms_norm_eps);
            rope_partial(head, cfg.rotary_dim, rope_pos, cfg.rope_theta);
        }

        // Causal SDPA over every entry (the canon + this one), f64 softmax.
        let entries = kv.k.len() / kv_dim;
        let scale = 1.0 / (hd as f64).sqrt();
        let mut attn = vec![0.0f32; nq * hd];
        for h in 0..nq {
            let kvh = h / group;
            let q = &q_roped[h * hd..(h + 1) * hd];
            let mut scores = Vec::with_capacity(entries);
            let mut max = f64::NEG_INFINITY;
            for t in 0..entries {
                let krow = &kv.k[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                let dot: f64 = q
                    .iter()
                    .zip(krow)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let s = dot * scale;
                max = max.max(s);
                scores.push(s);
            }
            let mut denom = 0.0f64;
            for s in &mut scores {
                *s = (*s - max).exp();
                denom += *s;
            }
            let mut acc = vec![0.0f64; hd];
            for (t, &p) in scores.iter().enumerate() {
                let vrow = &kv.v[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                for (a, &v) in acc.iter_mut().zip(vrow) {
                    *a += p * f64::from(v);
                }
            }
            for (o, a) in attn[h * hd..(h + 1) * hd].iter_mut().zip(acc) {
                *o = (a / denom) as f32;
            }
        }

        // Per-element sigmoid gate from the interleaved q projection.
        let mut gated = vec![0.0f32; nq * hd];
        for h in 0..nq {
            for d in 0..hd {
                let gate = q_full[h * 2 * hd + hd + d];
                gated[h * hd + d] = attn[h * hd + d] * sigmoid32(gate);
            }
        }
        let y = match route.as_mut() {
            Some((d, w)) => DenseGemv::new(d, w).matvec(&self.full.o, &gated)?,
            None => self.full.o.matvec(&gated),
        };
        Ok(Some(y))
    }

    /// Fuse the conditioning pair into the 10240-wide layer input:
    /// `h_in[s] = fc_hidden @ norm(h)[s] + fc_embedding @ norm(e)`.
    fn fuse(
        &self,
        hc: &HyperConnectionConfig,
        h_premixer: &[f32],
        token_embed: &[f32],
        mut route: Option<&mut MtpRoute<'_, '_, '_>>,
    ) -> Result<Vec<f32>> {
        let h = hc.hidden_size;
        ensure!(h_premixer.len() == hc.hc_hidden(), "fuse: hidden width");
        ensure!(token_embed.len() == h, "fuse: embedding width");
        let hn = grouped_rmsnorm(h_premixer, &self.pre_fc_norm_hidden, h, hc.rms_norm_eps)?;
        // group == the whole vector: a plain (1 + w) RMS norm.
        let en = grouped_rmsnorm(token_embed, &self.pre_fc_norm_embedding, h, hc.rms_norm_eps)?;
        let fe = match route.as_mut() {
            Some((d, w)) => DenseGemv::new(d, w).matvec(&self.fc_embedding, &en)?,
            None => self.fc_embedding.matvec(&en),
        };
        let mut h_in = vec![0.0f32; hc.hc_hidden()];
        for s in 0..hc.hc_count {
            let fh = match route.as_mut() {
                Some((d, w)) => {
                    DenseGemv::new(d, w).matvec(&self.fc_hidden, &hn[s * h..(s + 1) * h])?
                }
                None => self.fc_hidden.matvec(&hn[s * h..(s + 1) * h]),
            };
            for (dst, (&a, &b)) in h_in[s * h..(s + 1) * h].iter_mut().zip(fh.iter().zip(&fe)) {
                *dst = a + b;
            }
        }
        Ok(h_in)
    }

    /// One MTP step. `kv_only` appends this entry's K/V and stops (catch-up);
    /// otherwise the full block runs and, when `lm_head` is given, the mixer
    /// collapses `h_out` and the shared `lm_head` produces draft logits.
    ///
    /// `route = None` is the pure host oracle; `Some` moves the wide matvecs
    /// and the hyper-connection sites onto the device (the hc sites fall back
    /// to the host transcription when the MTP hc weights are not resident, so
    /// a partial residency degrades to slower, never to wrong).
    #[expect(
        clippy::too_many_arguments,
        reason = "the conditioning pair is this wide"
    )]
    pub fn forward(
        &self,
        cfg: &Qwen4ExpConfig,
        hc: &HyperConnectionConfig,
        h_premixer: &[f32],
        token_embed: &[f32],
        rope_pos: usize,
        kv: &mut HostKv,
        kv_only: bool,
        mut route: Option<MtpRoute<'_, '_, '_>>,
        lm_head: Option<&HostDense<'_>>,
    ) -> Result<MtpForwardOut> {
        let hc_dev = route
            .as_ref()
            .is_some_and(|(_, w)| w.hyper_connection(Some(MTP_HC_LAYER), HcSite::Attn).is_ok());
        let mut h_state = self.fuse(hc, h_premixer, token_embed, route.as_mut())?;

        // ── attention sub-block ─────────────────────────────────────────
        let (x, host_gr) = if hc_dev {
            let (d, w) = route.as_mut().expect("hc_dev checked");
            (
                d.hc_pre(w, hc, Some(MTP_HC_LAYER), HcSite::Attn, &h_state)?,
                None,
            )
        } else {
            let gr = gated_residual(hc, &self.attn_hc, &h_state)?;
            (gr.block_input.clone(), Some(gr))
        };
        let attn_out = self.attention(cfg, &x, rope_pos, kv, kv_only, route.as_mut())?;
        let Some(y) = attn_out else {
            // Catch-up: the KV row is written; nothing downstream is needed.
            return Ok(MtpForwardOut {
                h_out: Vec::new(),
                logits: None,
            });
        };
        if hc_dev {
            let (d, w) = route.as_mut().expect("hc_dev checked");
            h_state = d.hc_combine(w, hc, Some(MTP_HC_LAYER), HcSite::Attn, &y)?;
        } else {
            let gr = host_gr.expect("host gated residual");
            let inj = gr
                .injection_weights
                .as_ref()
                .expect("layer site has injection");
            inject_block_output(hc, &mut h_state, inj, &y)?;
        }

        // ── MoE sub-block ───────────────────────────────────────────────
        let (x, host_gr) = if hc_dev {
            let (d, w) = route.as_mut().expect("hc_dev checked");
            (
                d.hc_pre(w, hc, Some(MTP_HC_LAYER), HcSite::Mlp, &h_state)?,
                None,
            )
        } else {
            let gr = gated_residual(hc, &self.mlp_hc, &h_state)?;
            (gr.block_input.clone(), Some(gr))
        };
        let y = self.moe(cfg, &x, route.as_mut())?;
        if hc_dev {
            let (d, w) = route.as_mut().expect("hc_dev checked");
            h_state = d.hc_combine(w, hc, Some(MTP_HC_LAYER), HcSite::Mlp, &y)?;
        } else {
            let gr = host_gr.expect("host gated residual");
            let inj = gr
                .injection_weights
                .as_ref()
                .expect("layer site has injection");
            inject_block_output(hc, &mut h_state, inj, &y)?;
        }

        // ── the MTP's own mixer + the shared lm_head ────────────────────
        let logits = match lm_head {
            None => None,
            Some(lm) => {
                let xm = if hc_dev {
                    let (d, w) = route.as_mut().expect("hc_dev checked");
                    d.hc_pre(w, hc, Some(MTP_HC_LAYER), HcSite::Mixer, &h_state)?
                } else {
                    gated_residual(hc, &self.mixer, &h_state)?.block_input
                };
                Some(match route.as_mut() {
                    Some((d, w)) => DenseGemv::new(d, w).matvec(lm, &xm)?,
                    None => lm.matvec(&xm),
                })
            }
        };
        Ok(MtpForwardOut {
            h_out: h_state,
            logits,
        })
    }
}

/// The drafter: the head plus its KV canon and catch-up queue.
pub struct MtpDrafter<'st> {
    pub head: MtpHead<'st>,
    /// The MTP layer's own KV (host-side; the draft SDPA is host math).
    kv: HostKv,
    /// Absolute position of KV entry 0; `None` until the first entry.
    base_pos: Option<usize>,
    /// Entries 0..canonical were built from true target `h` + real tokens;
    /// anything past them is speculative and truncated per draft call.
    canonical: usize,
    /// `(position, target h, next token)` waiting to become canonical
    /// entries — pushed by the driver from each verify pass's accepted rows.
    catchup: Vec<(usize, Vec<f32>, u32)>,
}

impl<'st> MtpDrafter<'st> {
    #[must_use]
    pub fn new(head: MtpHead<'st>) -> Self {
        Self {
            head,
            kv: HostKv::default(),
            base_pos: None,
            canonical: 0,
            catchup: Vec::new(),
        }
    }

    /// Absolute position the NEXT canonical entry must sit at (`None` before
    /// the first entry, when any position is acceptable).
    #[must_use]
    pub fn next_entry_pos(&self) -> Option<usize> {
        self.base_pos.map(|b| b + self.canonical)
    }

    /// Queue one accepted position for KV catch-up: the target's pre-mixer
    /// `h` at `pos` and the (verified) token at `pos + 1`.
    pub fn push_catchup(&mut self, pos: usize, h: Vec<f32>, next_token: u32) {
        self.catchup.push((pos, h, next_token));
    }

    /// Canonical entry count, for the harness.
    #[must_use]
    pub fn canonical_entries(&self) -> usize {
        self.canonical
    }

    /// Split borrows for the driver: the head, the KV and the bookkeeping.
    #[expect(clippy::type_complexity, reason = "a split-borrow accessor")]
    pub(crate) fn parts(
        &mut self,
    ) -> (
        &MtpHead<'st>,
        &mut HostKv,
        &mut Option<usize>,
        &mut usize,
        &mut Vec<(usize, Vec<f32>, u32)>,
    ) {
        (
            &self.head,
            &mut self.kv,
            &mut self.base_pos,
            &mut self.canonical,
            &mut self.catchup,
        )
    }
}

/// How many tail-of-prompt positions seed the MTP KV canon after a prefill.
/// Bounded because each is one KV-only MTP forward (~1-2 ms on the device
/// route); `ARLE_QWEN4_MTP_WARMUP` overrides for the acceptance sweep.
fn mtp_warmup_cap() -> usize {
    std::env::var("ARLE_QWEN4_MTP_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
}

impl<'ctx, 'st> crate::model_qwen4_exp::Qwen4DraftSource<'ctx, 'st> for MtpDrafter<'st> {
    fn warmup(
        &mut self,
        model: &crate::model_qwen4_exp::VulkanQwen4ExpModel<'ctx, 'st>,
        prompt: &[u32],
        chunk_start: usize,
    ) -> Result<()> {
        // Catch-up entries want the token AFTER each position, so the last
        // usable position is prompt.len() - 2; the queue is absorbed (and the
        // forwards actually run) on the first `draft` call.
        let from = chunk_start.max(prompt.len().saturating_sub(1 + mtp_warmup_cap()));
        for q in from..prompt.len().saturating_sub(1) {
            self.push_catchup(q, model.prefill_h_row(q - chunk_start)?, prompt[q + 1]);
        }
        Ok(())
    }

    fn note_accepted(&mut self, pos: usize, h: &[f32], next_token: u32) {
        self.push_catchup(pos, h.to_vec(), next_token);
    }

    fn draft(
        &mut self,
        model: &mut crate::model_qwen4_exp::VulkanQwen4ExpModel<'ctx, 'st>,
        h_last: &[f32],
        h_last_pos: usize,
        last_token: u32,
        k: usize,
    ) -> Result<Vec<u32>> {
        model.mtp_draft(self, h_last, h_last_pos, last_token, k)
    }
}
