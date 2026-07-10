# Qwen3.5 9B ModelScope GGUF smoke quality gap

## Context

User redirected local work from DSv4 operator optimization to Qwen 3.6 /
Qwen3.5-family bring-up on the current single RTX 4070 Ti SUPER host, starting
with a 9B quantized checkpoint from ModelScope.

The official ModelScope `Qwen/Qwen3.5-9B` checkpoint is Apache-2.0 and exposes
`Qwen3_5ForConditionalGeneration`, but its BF16 safetensors shards total about
19 GB. For this 16 GB local GPU, the test used the quantized ModelScope GGUF
repo `unsloth/Qwen3.5-9B-GGUF`, file `Qwen3.5-9B-Q4_K_M.gguf`.

Local files are under gitignored `infer/models/Qwen3.5-9B-GGUF/`:

```text
Qwen3.5-9B-Q4_K_M.gguf 5680522464 bytes
config.json             from Qwen/Qwen3.5-9B
tokenizer.json          from Qwen/Qwen3.5-9B
tokenizer_config.json   from Qwen/Qwen3.5-9B
chat_template.jinja     from Qwen/Qwen3.5-9B
```

ModelScope file evidence for the GGUF:

```text
repo: unsloth/Qwen3.5-9B-GGUF
file: Qwen3.5-9B-Q4_K_M.gguf
size: 5680522464
sha256: 03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8
revision: ae90f0d1c1be2b9250b0ef68265615f6fe3c777b
```

## Root Cause

The first smoke run loaded the model but failed on the first decode step:

```text
Batched decode failed: PostMlpAllReduce buffer len 16384 does not match logical len 4096
```

This was a single-rank LayerCommunicator guard-order bug. Qwen3.5 batched
decode preallocates hidden buffers for max slots, so the backing CUDA slice
capacity can be `max_slots * hidden_dim` while the current logical batch is
`batch * hidden_dim`. In TP=1 the post-MLP all-reduce is a strict no-op and
must not reject an overallocated scratch buffer.

Fix landed in `infer/src/model/layer_communicator.rs`: compute world size and
return `NoopSingleRank` before enforcing `buffer.len() == logical_len`. The
multi-rank guard is unchanged.

## Evidence

Build/smoke command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
INFER_Q35_PATH=infer/models/Qwen3.5-9B-GGUF \
cargo test --release -p infer --test smoke_qwen35_gguf --features cuda \
  qwen35_gguf_generate -- --ignored --nocapture
```

After the fix:

```text
Qwen3.5 GGUF loaded in 7697ms (32 layers)
GPU memory @ post_model_load: free=8.81 GB / total=16.72 GB
TokenKVPool: 102624 max tokens (6414 pages @ page_size=16), 3.4 GB
test qwen35_gguf_generate ... ok
```

Generated text remains incoherent under greedy sampling:

```text
prompt="The capital of France is"
text="锻现金流量现金流现金流 careers哪儿市民市民 Μ沿quip expertise expedona piston piston"

prompt="1 + 1 = "
text=" تم チ法规和 fel hanging主 गु simply优化 Inspir围绕着金融机构金融机构金融机构agnost和政府"
```

This means the ModelScope 9B Q4_K_M path is now loader/scheduler-executable,
but model-quality correctness is still not established. Treat output quality as
FAIL, not a win. This aligns with the older unresolved CUDA GGUF quality entries
for Qwen3/Qwen3.5 Q4_K paths.

## Fix

For this tranche:

- Downloaded the ModelScope 9B Q4_K_M GGUF and official sidecar tokenizer/config
  files into gitignored `infer/models/Qwen3.5-9B-GGUF/`.
- Fixed single-rank `LayerCommunicator` no-op handling for overallocated decode
  scratch buffers.
- Verified the existing ignored CUDA smoke reaches load, prefill, decode, and
  completion on this GPU.

Deferred:

- Coherence/parity investigation for Qwen3.5 9B GGUF Q4_K_M.
- Any Qwen3.6 MoE CUDA bring-up.
- Any DSv4 local operator work.

## Rule

Single-rank collectives must be true pass-throughs. Capacity-vs-logical-length
checks belong only on real multi-rank collectives, where reducing extra scratch
capacity would be incorrect.

Also: "smoke passes" is not a quality claim. For GGUF/Q4_K paths, require a
separate coherence or parity gate before recording a wins entry.
