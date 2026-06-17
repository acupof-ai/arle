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

## Verdict

F2 is closed for the exercised CUDA 35B-A3B teacher paths: BF16 and FP8 both
serve, generate coherent decoded text, and retrieve the needle exactly under
the Qwen3 non-thinking template. The prior deterministic garbage signature is
not present.

The chat endpoint still emits visible thinking text for this checkpoint even
when the user asks for final-only. That is a prompt/template behavior issue, not
the F2 forward-garbage failure; raw Qwen3 non-thinking template is the clean
needle gate for this model.

## Rule

When a model previously emitted deterministic garbage, close the correctness
track only with decoded tokens plus a needle gate on the actual serve path.
Layer-local finite values and a non-crashing forward are necessary but not
sufficient.
