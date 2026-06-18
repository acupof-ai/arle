# Qwen3.6 F2 coherence gate — BF16 and FP8

## Context

`errors/2026-06-11-qwen35-cuda-rewrite-35b-degenerate-output.md` recorded the
real Qwen3.6-35B-A3B CUDA rewrite failure: deterministic garbage on both TP=1
and TP=2. The original narrowed suspect was layer-0 gated-delta
`linear_attention`, but the live activation dump killed that hypothesis: layer-0
`in_proj_qkv`, `conv1d_silu_qkv`, `gdr_out`, `gated_norm_out`, and `out_proj`
were finite with sane magnitudes.

The actual blocking failure was the CUDA 12.9 cuBLAS/Lt small-N BF16 GEMM
divide-error fixed in `gemv.cu`: small continuation/chat chunks now route
`N <= 16` through the handwritten BF16 GEMV loop instead of any cuBLAS API.

## Evidence

Remote `.62` / `iv-ye8is8fbi8s6iplibbg7`, GPU3 avoided:

```text
source=/data01/arle-build
binary=/data01/arle-build/target/release/arle
build=CUDARC_CUDA_VERSION=12090 CUDA_HOME=/usr/local/cuda \
  ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python \
  cargo build --release --features cuda -p agent-infer --bin arle
PASS
```

Local gates:

```text
cargo fmt --check
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
PASS
```

`cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda --lib -- -D warnings`
was attempted and failed on pre-existing `crates/infer-cuda/src/dsv4.rs`
`needless_option_as_deref` warnings, not on this change.

## BF16 gate

Serve:

```text
model=/data01/models/Qwen3.6-35B-A3B
CUDA_VISIBLE_DEVICES=2
port=18180
ARLE_CUDA_DISABLE_FLASHMLA=1
ARLE_QWEN35_DEEPGEMM=0
```

Decoded coherence probes:

```text
PROMPT0 usage={'prompt_tokens': 26, 'completion_tokens': 96, 'total_tokens': 122}
out='Thinking Process:
...
3.  **Fact-check:** What is the capital of France? It is Paris.
...
</think>'

PROMPT1 usage={'prompt_tokens': 28, 'completion_tokens': 96, 'total_tokens': 124}
out='Thinking Process:
...
3.  **Perform the calculation:** $17 + 25 = 42$.
...
5.  **Final Output Generation:** 42.✅
</think>

'
```

Needle gate, using the script's Qwen3 non-thinking raw template to avoid the
chat endpoint spending all 16 generated tokens on the thinking header:

```text
PORT=18180 MODEL=Qwen3.6-35B-A3B RAW=1 TEMPLATE=qwen3_nonthink \
python3 scripts/needle_gate.py 115,300 2 0.0

len=115 depth=0.00 run=0 pt=149 cls=exact wall=1.1s out='The secret access code is 738291.'
len=115 depth=0.00 run=1 pt=149 cls=exact wall=1.1s out='The secret access code is 738291.'
SUMMARY len=115 depth=0.00 exact=2 partial=0 miss=0 DET
len=300 depth=0.00 run=0 pt=314 cls=exact wall=1.9s out='738291'
len=300 depth=0.00 run=1 pt=314 cls=exact wall=2.0s out='738291'
SUMMARY len=300 depth=0.00 exact=2 partial=0 miss=0 DET
```

## FP8 gate

Serve:

```text
model=/data01/models/Qwen3.6-35B-A3B-FP8
CUDA_VISIBLE_DEVICES=4
port=18181
ARLE_CUDA_DISABLE_FLASHMLA=1
ARLE_QWEN35_DEEPGEMM=0
```

Decoded coherence probes:

```text
PROMPT0 usage={'prompt_tokens': 26, 'completion_tokens': 96, 'total_tokens': 122}
out='Thinking Process:
...
3.  **Identify the capital:** The capital of France is Paris.
...
5.  **Construct the final answer:** "Paris".
...
'

PROMPT1 usage={'prompt_tokens': 28, 'completion_tokens': 96, 'total_tokens': 124}
out='Thinking Process:
...
2.  **Perform the Calculation:**
    *   $17 + 25$
    *   $7 + 5 = 12$ (write down 2, carry over 1)
...
'
```

Needle gate:

```text
PORT=18181 MODEL=Qwen3.6-35B-A3B-FP8 RAW=1 TEMPLATE=qwen3_nonthink \
python3 scripts/needle_gate.py 115,300 2 0.0

len=115 depth=0.00 run=0 pt=149 cls=exact wall=7.7s out='The secret access code is 738291.'
len=115 depth=0.00 run=1 pt=149 cls=exact wall=7.7s out='The secret access code is 738291.'
SUMMARY len=115 depth=0.00 exact=2 partial=0 miss=0 DET
len=300 depth=0.00 run=0 pt=314 cls=exact wall=15.8s out='738291'
len=300 depth=0.00 run=1 pt=314 cls=exact wall=15.8s out='738291'
SUMMARY len=300 depth=0.00 exact=2 partial=0 miss=0 DET
```

## FP8 training finite-diff tail gate

The training-side tail was rerun from a clean tracked-only archive of
`c1f5b519` on `.62` GPU2, not from the dirty `/data01/arle-build` tree:

```text
source=/data01/arle-codex-c1f5b519
target=/data01/arle-target-codex-c1f5b519
model=/data01/models/Qwen3.6-35B-A3B-FP8
CUDA_VISIBLE_DEVICES=2
CUDA_HOME=/usr/local/cuda
CUDARC_CUDA_VERSION=12090
ARLE_CUDA_DISABLE_FLASHMLA=1
INFER_TILELANG_PYTHON=/root/tl-venv/bin/python
```

A0 MoE FD gate, CUDA backend, `eps=1e-3` and relative tolerance:

```text
a0_moe_finite_diff backend=cuda eps=1.0e-3 checked_values=4864
relative_values=1036 tiny_values=3828 max_abs_at_worst_rel=5.588241e-7
max_rel=2.673310e-3 worst=a0_moe.router.weight[13]
analytic=-2.090383e-4 numeric=-2.084794e-4
max_tiny_abs=9.490354e-7 tiny_abs_failures=0
test result: ok. 3 passed
```

Real-checkpoint Qwen3.6 FP8 LoRA MLP-layer FD gate:

```text
qwen36_fp8_lora_fd_gate ... --target-set all-linear \
  --target-adapter auto:routed-up --mode mlp-layer --layer 0 \
  --eps 1e-3 --profile-backward

qwen36_fp8_lora_fd_backward_profile total_seconds=0.095036
op_seconds=0.041475 merge_grad_seconds=0.053254
qwen36_fp8_lora_fd_gate_result load_seconds=13.909568
analytic_seconds=0.103256 plus_seconds=0.006681 minus_seconds=0.006814
target=model.language_model.layers.0.mlp.experts.210.up_proj.weight.lora_b
index=186 eps=1.0e-3 analytic=-2.581576268e-7
numeric=-2.587512427e-7 rel_err=2.294e-3
qwen36_fp8_lora_fd_gate PASS
```

The unfrozen and route-frozen full-model scalar FD diagnostics remain killed as
oracles by the existing route/noise entries; the licensed gradient gate is the
real-checkpoint MLP-layer FD above plus the A0 relative FD test. Do not relabel
the full-model scalar diagnostic as a pass.

## Verdict

F2 is closed for the exercised CUDA 35B-A3B hand-kernel teacher paths
(`ARLE_QWEN35_DEEPGEMM=0`): BF16 and FP8 both
serve, generate coherent decoded text, and retrieve the needle exactly under
the Qwen3 non-thinking template. The prior deterministic garbage signature is
not present.

This is not yet a default-configuration verdict for Qwen DeepGEMM-enabled serve;
that needs a same-template BF16+FP8 coherence rerun with the default DeepGEMM
setting before broadening the claim.

The chat endpoint still emits visible thinking text for this checkpoint even
when the user asks for final-only. That is a prompt/template behavior issue, not
the F2 forward-garbage failure; raw Qwen3 non-thinking template is the clean
needle gate for this model.

## Rule

When a model previously emitted deterministic garbage, close the correctness
track only with decoded tokens plus a needle gate on the actual serve path.
Layer-local finite values and a non-crashing forward are necessary but not
sufficient.
