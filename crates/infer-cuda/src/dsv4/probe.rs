//! Non-invasive logit-lens probe: per-layer stream capture during forward,
//! post-forward logit lens (norm + lm_head + log_softmax) on GPU.
//!
//! Gated by ARLE_PROBE_JSONL; zero cost when off.
//! ARLE_PROBE_LENS_LAYERS: comma-separated layer indices (default: all).

use std::fs::File;
use std::io::{BufWriter, Write};

use super::*;

const MAX_PROBE_ROWS: usize = 64;

pub(crate) struct Dsv4ProbeCapture {
    layers: Vec<HiddenStates>,
    lens_layers: Vec<usize>,
    head_normed: HiddenStates,
    logits: HiddenStates,
    host_logits: Vec<f32>,
    writer: BufWriter<File>,
    captured_rows: usize,
    positions: Vec<usize>,
}

impl Dsv4ProbeCapture {
    pub(super) fn from_env(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        lm_head: &DeviceMatrix,
        n_layers: usize,
    ) -> Option<Self> {
        let path = std::env::var("ARLE_PROBE_JSONL").ok()?;
        let stream_dim = config.hidden_size * config.hc_mult;
        let vocab = lm_head.rows;

        let lens_layers: Vec<usize> = std::env::var("ARLE_PROBE_LENS_LAYERS")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.split(',')
                    .filter_map(|x| x.trim().parse().ok())
                    .filter(|&l| l < n_layers)
                    .collect()
            })
            .unwrap_or_else(|| (0..n_layers).collect());

        // SAFETY: uninit device scratch; fully written by the capture D2D before read.
        let layers = (0..n_layers)
            .map(|_| unsafe { HiddenStates::uninit(ctx, stream_dim, MAX_PROBE_ROWS) })
            .collect::<Result<Vec<_>>>()
            .ok()?;
        // SAFETY: uninit device scratch; fully written by head_normed_rows before read.
        let head_normed =
            unsafe { HiddenStates::uninit(ctx, config.hidden_size, MAX_PROBE_ROWS) }.ok()?;
        // SAFETY: uninit device scratch; fully written by lm_head_project_batch before read.
        let logits = unsafe { HiddenStates::uninit(ctx, vocab, MAX_PROBE_ROWS) }.ok()?;
        let host_logits = vec![0.0f32; vocab * MAX_PROBE_ROWS];

        let file = File::create(&path).ok()?;
        eprintln!(
            "[probe] lens → {path} ({} layers, ≤{MAX_PROBE_ROWS} rows)",
            lens_layers.len()
        );

        Some(Self {
            layers,
            lens_layers,
            head_normed,
            logits,
            host_logits,
            writer: BufWriter::new(file),
            captured_rows: 0,
            positions: Vec::new(),
        })
    }

    fn capture(
        &mut self,
        ctx: &DeviceContext,
        stream: &HiddenStates,
        layer_idx: usize,
        positions: &[usize],
    ) -> Result<()> {
        if !self.lens_layers.contains(&layer_idx) {
            return Ok(());
        }
        let stream_dim = stream.hidden_dim;
        let rows = stream.seq_len.min(MAX_PROBE_ROWS);
        let offset = (stream.seq_len - rows) * stream_dim;
        let elems = rows * stream_dim;

        let src = stream.data.slice(offset..offset + elems);
        let mut dst = self.layers[layer_idx].data.slice_mut(0..elems);
        ctx.stream
            .memcpy_dtod(&src, &mut dst)
            .map_err(|e| anyhow!("probe capture D2D failed: {e}"))?;

        if self.captured_rows == 0 {
            self.captured_rows = rows;
            self.positions = positions[positions.len() - rows..].to_vec();
        }
        Ok(())
    }

    fn flush(&mut self, model: &Dsv4Model) -> Result<()> {
        let rows = self.captured_rows;
        if rows == 0 {
            return Ok(());
        }
        let vocab = model.lm_head.rows;
        let last_layer = *self.lens_layers.last().unwrap();

        // Reference tokens from the last captured layer.
        self.compute_layer_logits(model, last_layer, rows)?;
        self.d2h(model, rows)?;
        let reference: Vec<u32> = (0..rows)
            .map(|r| argmax(&self.host_logits[r * vocab..(r + 1) * vocab]))
            .collect();

        let lens_layers = self.lens_layers.clone();
        for &layer_idx in &lens_layers {
            if layer_idx != last_layer {
                self.compute_layer_logits(model, layer_idx, rows)?;
                self.d2h(model, rows)?;
            }
            for (r, &pos) in self.positions.iter().take(rows).enumerate() {
                let base = r * vocab;
                let slice = &self.host_logits[base..base + vocab];
                let (top1, top1_logprob, nll, agree) = compute_stats(slice, reference[r]);
                writeln!(
                    self.writer,
                    r#"{{"phase":"lens","pos":{},"layer":{},"top1":{},"top1_logprob":{:.6},"nll":{:.6},"agree":{}}}"#,
                    pos, layer_idx, top1, top1_logprob, nll, agree
                )?;
            }
        }

        self.writer.flush()?;
        self.captured_rows = 0;
        self.positions.clear();
        Ok(())
    }

    fn compute_layer_logits(
        &mut self,
        model: &Dsv4Model,
        layer_idx: usize,
        rows: usize,
    ) -> Result<()> {
        self.head_normed.seq_len = rows;
        self.logits.seq_len = rows;
        model.head_normed_rows(&self.layers[layer_idx], 0..rows, &mut self.head_normed)?;
        model.lm_head_project_batch(&self.head_normed, &mut self.logits)?;
        Ok(())
    }

    fn d2h(&mut self, model: &Dsv4Model, rows: usize) -> Result<()> {
        let vocab = model.lm_head.rows;
        let elems = rows * vocab;
        let mut host_bf16 = vec![half::bf16::default(); elems];
        let src = self.logits.data.slice(0..elems);
        model
            .ctx
            .stream
            .memcpy_dtoh(&src, &mut host_bf16[..])
            .map_err(|e| anyhow!("probe D2H failed: {e}"))?;
        model
            .ctx
            .stream
            .synchronize()
            .map_err(|e| anyhow!("probe sync failed: {e}"))?;
        for (i, &v) in host_bf16.iter().enumerate() {
            self.host_logits[i] = v.to_f32();
        }
        Ok(())
    }
}

impl Dsv4Model {
    pub(super) fn probe_capture(
        &self,
        stream: &HiddenStates,
        layer_idx: usize,
        positions: &[usize],
    ) -> Result<()> {
        if let Some(probe) = self.probe.borrow_mut().as_mut() {
            probe.capture(&self.ctx, stream, layer_idx, positions)?;
        }
        Ok(())
    }

    pub(super) fn probe_flush(&self) -> Result<()> {
        let probe = self.probe.borrow_mut().take();
        if let Some(mut probe) = probe {
            let result = probe.flush(self);
            *self.probe.borrow_mut() = Some(probe);
            result?;
        }
        Ok(())
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i as u32, v) } else { (bi, bv) }
        })
        .0
}

fn compute_stats(logits: &[f32], reference: u32) -> (u32, f32, f32, bool) {
    let mut max_logit = f32::NEG_INFINITY;
    let mut top1 = 0u32;
    for (i, &logit) in logits.iter().enumerate() {
        if logit > max_logit {
            max_logit = logit;
            top1 = i as u32;
        }
    }
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let log_sum = sum_exp.ln();
    let top1_logprob = logits[top1 as usize] - max_logit - log_sum;
    let nll = -(logits[reference as usize] - max_logit - log_sum);
    let agree = top1 == reference;
    (top1, top1_logprob, nll, agree)
}
