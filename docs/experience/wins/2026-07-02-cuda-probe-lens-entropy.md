# CUDA inference probe: decode logit lens + per-token entropy JSONL

> Status: **pending-remote** — H20 pod verification queued (smoke self-check:
> final-layer lens NLL ≈ decode NLL; `dsv4/stage/lm_head_project` stage_profile
> for the measured per-layer lens cost; probe-off A/B vs parent commit, Δ≈0
> required). This stub becomes the full bench entry when the pod run lands.

## Context

Analysis probe for two questions: (1) at which layer does the decode
distribution's perplexity plateau (early-exit depth — logit lens over the last
N layers, default 10); (2) per-token entropy/NLL for every prefill position and
every sampled decode token. `arle serve --probe-out <path>
[--probe-lens-layers N] [--probe-token-entropy BOOL]` → rank-0 JSONL; env
transport `ARLE_PROBE_*` (docs/environment.md).

## What Worked

- Zero new device code: lens = existing head recipe (`head_normed_rows` fold +
  `rms_norm_vec` + `lm_head_project_batch`) applied to intermediate streams;
  entropy hooks ride the single-row sampler convergence points
  (`sample_cuda_token`/`_scratched`), which already run outside CUDA-graph
  capture and cover single/batched/graph decode + the prefill last token.
- No mid-loop sync: the layer loop stashes DEVICE logits; D2H + math + emit
  happen only at the existing post-sampling sync point (`lens_flush`).
- Probe off = one `OnceLock` load per hook / one `Option` compare per layer;
  default path byte-identical (decode-graph dispatch only gains a
  `lens_layers()==0` conjunct).

## Rule

Instrument at existing convergence/sync points (sampler entry, post-sampling
sync) instead of inside per-layer hot loops; stash device buffers and defer
D2H to a sync point the path already pays for.
