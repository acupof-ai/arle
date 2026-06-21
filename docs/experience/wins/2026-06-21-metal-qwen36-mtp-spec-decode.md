# Metal Qwen3.6 spec decode via the NextN/MTP head — 12.3 → 17.75 tok/s (+44%), breaks the bandwidth floor

## Context
OptiQ on arle Metal c=1 was **bandwidth-bound at ~12.3 tok/s** — measured at 95% of mlx_lm (13.05)
and within ~1.2× of the M4 Pro physical floor (18 GB weights / 273 GB/s ≈ 15.2 tok/s). The MLP
compile was the last autoregressive lever. **The only way past the floor is the algorithm:
speculative decoding amortizes the weight read over N tokens per forward.**

## What Worked
`mlx-community/Qwen3.6-27B-MTP-4bit` is the trained NextN/MTP head (31 tensors: `fc` +
`pre_fc_norm_embedding/hidden` + one transformer layer + `norm`; head_dim 256 matches the base).
Converged it into arle's existing DFlash/EAGLE machinery as a second config of ONE head —
`DraftKind { DFlashEagle | Qwen35Mtp }` threaded config→loader→C++ forward, auto-detected from the
head; the z-lab DFlashEagle path is byte-for-byte unchanged (fully additive). The hard spec parts
(verify, accept, KV+GDR rollback, the `qwen35_speculative_block` loop) were **reused, not rebuilt**.

Reference-driven, not guessed (the load-bearing point): the head's `layers.0` is **not** a vanilla
layer — it's a **gated full-attention layer** (q_proj = 2×heads·head_dim = 12288 with an output
gate `sigmoid(gate)·attn`, q/k norm at head_dim 256, **partial rotary** rotary_dim = 256×0.25 = 64,
θ=1e7), and the head is **autoregressive** (1 token/forward, recursive: a depth-3 block = sequential
head forwards chaining the head's own hidden). Both confirmed against **SGLang `qwen3_5_mtp.py` +
`eagle_worker.py`** — which corrected the initial spec. The head consumes the base **residual after
the last layer** (`target_layer_ids = [num_layers-1]`, raw — its own `pre_fc_norm_hidden` normalizes).

## Results (OptiQ base + MTP head draft, depth 3, M4 Pro, temp 0, same-session A/B)
| Config | tok/s | |
|--------|-------|--|
| Baseline (no draft) | 12.30 | |
| **Spec (MTP head, d3)** | **17.75** (×3: 17.75/17.75/17.74) | **+44%** |

- **Beats the 15.2 bandwidth floor** — the algorithm lever does what no framework/kernel could.
- Acceptance (subagent's 200-step trace): **68.8% draft-token / 2.375 of 3 per block** — clears the
  >50% bar; **no OptiQ-VL-vs-text-head mismatch** (the worried-about risk was a non-issue).
- Output coherent + matches the no-spec baseline (verify guarantees target-correct tokens).
- OptiQ+spec (17.75) now beats even plain 4bit (14.6) AND keeps the better quality (PPL 7.82).
- Build/clippy clean; 30/30 infer-metal tests green; one bug fixed (MTP forward can't `mx::compile` —
  shapeless split — so `finalize()` runs it eagerly, 1 layer, cheap).

## Rule
- **The bandwidth floor is the AUTOREGRESSIVE ceiling, not the ultimate.** When a memory-bound
  decode is pinned at ~1.2× the HBM floor and the reference framework (mlx_lm) is too, stop tuning
  the framework — switch the algorithm. Spec decode reads the weights once and emits N tokens.
- **Acceptance rate is a consequence of implementation correctness, not a base-match gamble**
  (ckl's call, 2026-06-21). The fix for a low acceptance is to find the reference and match it
  EXACTLY (the gated full-attn layer + partial rotary + recursive draft + raw-hidden source), not to
  swap bases and pray. Implement against SGLang/the modeling code; the acceptance number is the
  echo of correctness, and here it read 68.8% on the first correct implementation.
- **Converge into existing machinery, don't fork.** A second `DraftKind` config reusing the whole
  verify/rollback path beat a parallel draft model; the existing path stayed byte-identical.
