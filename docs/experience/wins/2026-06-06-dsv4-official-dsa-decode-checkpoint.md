# DSv4 Official DSA Decode Checkpoint

## Context

DSv4 decode at 4096 context is dominated by CSA selection. The official SGLang
DSA indexer stack is the target path, but the first landed tranche is a
default-off decode-only checkpoint.

## What Worked

- Mirrored the SGLang DSA posture for paged decode:
  - Hadamard rotation before FP8 quant, scaled by `hidden_size^-0.5`.
  - FP8 E4M3 activation quant with per-128 scale `max(abs(x), 1e-4) / 448`.
  - Fused index-K cache layout `[page][64][128 fp8 values | 64 fp32 scales]`.
  - DeepGEMM paged-MQA logits with `weights_proj * n_heads^-0.5 * q_scale * softmax_scale`.
  - Official `deepseek_v4_topk_transform_512` transformed output.
- Kept legacy `dsv4_csa_select` as the default path.
- Added `ARLE_DSV4_DSA_INDEXER=1` as an opt-in for the official decode path.

## Evidence

- Local: `cargo fmt`.
- Local: `CUDARC_CUDA_VERSION=12080 cargo check -p infer-cuda --features cuda,no-cuda`.
- Pod: `cargo build --release -p infer-cuda --features cuda,nccl,deepep --example dsv4_parity`.
- Pod short smoke with official DSA decode path reached `clean_tokens=[344, 34837]`.

## Limits

- Prefill/extend still fall back to legacy CSA selection.
- Long-prompt needle validation was run with `ARLE_DSV4_FLASHMLA_DECODE=0`
  because the rebuilt FlashMLA decode metadata path returns
  `CUDA_ERROR_NOT_SUPPORTED`.
- 4096-token greedy output has a broad same-config nondeterminism floor, so this
  checkpoint is not licensed as a default-on production path yet.

## Rule

Official-kernel adoption should land in narrow, reversible tranches. A
decode-only path stays opt-in until it validates under the real FlashMLA decode
configuration and the ragged prefill/extend path is explicitly handled.
