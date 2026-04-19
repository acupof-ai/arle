# Plan — pretrained-weight bridge (train ↔ infer)

**Directive (2026-04-19):** User approved the 5-step critical path in
`docs/experience/wins/2026-04-19-m5.3a-m5.3b-add-device-resident-tensor.md`
follow-up chat: wire up a real pretrained weight → autograd training → serve
loop so "能训出有用模型" is reachable, not blocked on M5.3 kernel phases.

This plan is **orthogonal to M5.3b CUDA kernels** (`m5.3b-cuda-autograd-kernels.md`).
M5.3b makes training faster on GPU; this plan makes training **useful at all**
by connecting it to real tokenizers, real weights, and real eval.

## Goal

End-to-end demonstrable loop on a laptop:

```
Qwen3-0.5B safetensors
  → load into autograd TensorStore          (step 2)
  → run autograd-side Qwen3 forward         (step 3)
  → SFT one step on a small JSONL           (step 4)
  → export to safetensors                   (step 5)
  → serve via `cli` HTTP              (already built)
  → chat in OpenAI v1 client and get coherent output  (acceptance gate)
```

Model size: **Qwen3-0.5B** for the first land. Fits in f32 on Mac (2 GB
weight). Qwen3-4B is a stretch goal once Metal/CUDA paths compose.

## Steps (dependency-ordered)

### Step 1 — Tokenizer bridge

**Files:**
- `crates/train/Cargo.toml` — add `tokenizers = "0.22"` (already workspace-pinned in `infer/Cargo.toml`; lift to root `Cargo.toml` `[workspace.dependencies]` for sharing).
- `crates/train/src/tokenizer.rs` *(new)* — thin wrapper:
  - `TrainTokenizer::from_file(path: &Path) -> Result<Self>` — loads `tokenizer.json`.
  - `encode(text: &str, add_special: bool) -> Vec<u32>`
  - `decode(ids: &[u32], skip_special: bool) -> String`
  - `vocab_size() -> usize`
  - Optional: `chat_template(messages: &[ChatMsg]) -> String` — Qwen-style `<|im_start|>role\ncontent<|im_end|>\n` (hardcoded; Qwen3/Qwen3.5 share it).
- `crates/train/src/lib.rs` — `pub mod tokenizer;`
- `crates/train/tests/test_tokenizer.rs` *(new)* — roundtrip on a Qwen3 tokenizer.json (gated on `INFER_TEST_MODEL_PATH`).

**Acceptance:** `encode("hello world")` matches the corresponding infer-side encode byte-for-byte (reference: decode the bytes that `infer/` produces for the same string — no infer-side encoder exists today, so acceptance degrades to "encode→decode roundtrip lossless, vocab_size matches config").

**Estimate:** 0.5 d.

---

### Step 2 — Safetensors ↔ `TensorStore` load path

**Design:** `SafetensorsRegistry` owns a name→TensorId map. Caller:
1. Builds the autograd module (`Module` trait in `autograd/src/module.rs`).
2. Registers each parameter TensorId with its HF-convention name via a builder method on the new `Qwen3Autograd` module.
3. Passes the registry + a safetensors path to `registry.load_into(&mut store, path)?`.

**Files:**
- `Cargo.toml` — add `safetensors`, `tokenizers`, `memmap2` to `[workspace.dependencies]`.
- `crates/autograd/Cargo.toml` — optional `safetensors` + `memmap2` deps behind a `safetensors` feature (off by default; `train` turns it on).
- `crates/autograd/src/safetensors_io.rs` *(new)*, feature-gated:
  - `pub struct SafetensorsRegistry { map: HashMap<String, TensorId> }`
  - `pub fn register(&mut self, name: &str, id: TensorId)`
  - `pub fn load_into(&self, store: &mut TensorStore, path: &Path) -> Result<()>`:
    - mmaps safetensors via `safetensors::SafeTensors::deserialize`.
    - For each registered name: fetch `TensorView`, validate shape == `store.tensor(id).shape`, dtype convert (bf16/f16 → f32), overwrite `store.tensor_mut(id).data`. Mark `Dirty::Host`.
    - Errors out on missing keys (strict by default; `allow_missing: bool` ctor flag for partial loads).
  - `pub fn save_from(&self, store: &TensorStore, path: &Path) -> Result<()>` — symmetric, writes f32.
  - dtype conversion helpers: `bf16_to_f32_slice`, `f16_to_f32_slice`.
- `crates/autograd/src/lib.rs` — `#[cfg(feature = "safetensors")] pub mod safetensors_io;`
- `crates/autograd/tests/test_safetensors.rs` *(new)* — roundtrip f32 tensor, bf16→f32 conversion, shape mismatch error.

**Acceptance:** Load a hand-rolled 2-tensor safetensors file; values match bit-exact f32; bf16 converts with ≤1e-7 abs.

**Estimate:** 1 d.

---

### Step 3 — Qwen3 architecture in autograd

**Model target:** Qwen3-0.5B (24 layers, 896 d_model, 14 heads, 2 kv_heads, GQA). Config read from `config.json`.

**New autograd ops required:**
- `rope(x, sin_cos_cache, position_ids, traditional=false)` — RoPE rotation. CPU impl + Metal via `mlx_fast_rope` + CUDA via new `rope.cu` kernel (follow-on to M5.3b).
- `silu(x)` — `x * sigmoid(x)`. CPU impl + Metal via `mlx_sigmoid`+`mlx_multiply` + CUDA via new kernel (copyable from mni-ml pattern).
- `causal_sdpa(q, k, v, mask)` — composable as `softmax(matmul(q, k.T) / sqrt(d) + mask) @ v`. MVP composes existing ops; no flash. Later: Metal `mlx_fast_sdpa` + CUDA FlashInfer port (out of scope here).

**Files:**
- `crates/autograd/src/ops/rope.rs` *(new)* — `rope` forward + backward (angle derivative).
- `crates/autograd/src/ops/activation.rs` — extend with `silu` + backward.
- `crates/autograd/src/ops/attention.rs` *(new)* — `causal_sdpa` composed from matmul/softmax/add/mask.
- `crates/autograd/src/ops.rs` — re-exports.
- `crates/train/src/qwen3_autograd.rs` *(new)* — the model itself:
  - `pub struct Qwen3Config { vocab_size, hidden_size, num_hidden_layers, num_attention_heads, num_kv_heads, head_dim, intermediate_size, max_position_embeddings, rms_norm_eps, rope_theta }`
  - `Qwen3Config::from_json(path: &Path)`
  - `pub struct Qwen3Model { layers: Vec<Qwen3Layer>, embed, final_norm, lm_head, ... }`
  - `Qwen3Model::new(cfg, store) -> Self` — allocates weights, returns `SafetensorsRegistry` via side method so caller can populate it.
  - `Qwen3Model::forward(store, tape, input_ids: TensorId, position_ids: TensorId) -> logits: TensorId` — matches HF layer-by-layer order.
  - GQA: expand K/V heads by `num_heads / num_kv_heads` repeat-interleave before SDPA; autograd op `repeat_kv(x, n_rep)` helper (compose via reshape+broadcast — no new kernel).
- `crates/train/src/lib.rs` — `pub mod qwen3_autograd;`
- `crates/train/tests/test_qwen3_autograd_forward.rs` *(new)* — forward with loaded Qwen3-0.5B weights produces logits whose top-1 on "The capital of France is" is ` Paris` or similar (fuzzy gate).

**Acceptance:**
1. Load Qwen3-0.5B weights into `Qwen3Model`.
2. Forward "The capital of France is" (tokenized via Step 1).
3. Top-1 over final token is a sane continuation (`Paris`/` Paris`/` the` — whatever infer-side produces on the same input, we match within top-5).

**Estimate:** 3 d (RoPE + SiLU are 0.5 d each; SDPA compose + Qwen3Model wiring is 2 d; test + debug is 1 d).

---

### Step 4 — SFT training loop binary

**Files:**
- `crates/train/src/bin/train_sft.rs` *(new)* — CLI:
  - `--model <path>` (HF dir with config.json + model.safetensors + tokenizer.json)
  - `--data <jsonl>` (lines: `{"messages": [{"role":"user","content":...}, {"role":"assistant","content":...}]}`)
  - `--out <checkpoint-dir>` (writes .safetensors every N steps)
  - `--steps N --batch B --lr 2e-5 --seq-len 1024`
  - `--backend cpu|metal|cuda`
- Loop: tokenize batch → pad → forward → cross-entropy on assistant-only tokens (mask user tokens out of loss) → `tape.backward()` → `AdamW.step()` → log loss.
- `crates/train/src/sft_data.rs` *(new)* — JSONL reader + chat-template tokenization + assistant-mask generation.
- `crates/train/Cargo.toml` — add `serde_json` (workspace) + bin entry.

**Acceptance:** Run 100 SFT steps on a hand-crafted 10-line JSONL; loss monotonically decreases (noise-adjusted); output checkpoint loads back in.

**Estimate:** 1 d.

---

### Step 5 — Safetensors export from `TensorStore`

**Files:**
- `crates/autograd/src/safetensors_io.rs` — already gained `save_from` in Step 2; now exercise it.
- `crates/train/src/bin/train_sft.rs` — call `registry.save_from(&store, out_dir.join("model.safetensors"))` at checkpoint intervals; copy tokenizer.json + write config.json alongside.
- `crates/train/src/bin/train_sft.rs` — final export on exit.

**Acceptance:**
```bash
cargo run -p train --bin train_sft --release -- --model Qwen3-0.5B --data tiny_sft.jsonl --steps 100 --out /tmp/trained
cargo run -p cli --release -- chat --model /tmp/trained "hello"
```
Chat response is coherent (not garbage). Bit-identical reload would be ideal but f32→bf16 round-trip will drift; accept ≤1e-3 logit delta on the tokenized test prompt.

**Estimate:** 0.5 d (mostly plumbing once Step 2 is green).

---

### Step 6 (optional, deferred) — infer-engine-based rollout for RL

Out of scope for this plan. Tracked for follow-up.

## Dependencies and parallel-izability

- **Step 1 ⊥ Codex C1** — touches only `train/` Cargo + new file. Start now.
- **Step 2 ⊥ Codex C1** — touches `autograd/src/safetensors_io.rs` (new file) + `autograd/Cargo.toml`. Codex is editing `autograd/src/backend.rs` + `Cargo.toml`. Cargo.toml is a conflict risk — **wait for C1 to commit before Step 2 edits `autograd/Cargo.toml`**.
- **Step 3 depends on Step 2** (needs the registry to load weights into the new model).
- **Step 4 depends on Step 3** (needs a working model).
- **Step 5 depends on Step 2** (export helper).

## Risks / open questions

1. **Qwen3-0.5B total f32 params = ~500M × 4 bytes = ~2 GB.** `TensorStore.data: Vec<f32>` holds host-side copies even for device-resident tensors today (dirty-tracked but still allocated). Memory budget OK on 18GB+ Mac; tight on 8GB. **Fix if needed:** lazy host-side allocation driven by `Dirty` — only keep host `Vec<f32>` when `Dirty::Host` or `Dirty::Both`.
2. **Backward memory.** SFT on seq_len=1024 with 24 layers needs ~2 GB activations per batch-1 forward. Acceptable; gradient checkpointing is the next lever.
3. **Tokenizer: add_bos / chat template correctness.** Qwen3 tokenizer has its own special-token handling. Test against infer-side tokenization of the same text.
4. **bf16 on disk.** Most HF Qwen3 safetensors ship bf16. Step 2's `bf16_to_f32_slice` must be correct (bit-exact roundtrip → f32 up-cast has no loss). CUDA/Metal backends stay f32 for MVP.
5. **AdamW state size.** 2× weight params = ~4 GB for Qwen3-0.5B. Fits.
6. **GQA `repeat_kv` compose-via-reshape** — need a `repeat_interleave` autograd op or a reshape+broadcast trick. Reshape+broadcast is zero-extra-ops; try that first.

## Verification plan

Per step, not just at the end:

| Step | Verify |
|------|--------|
| 1 | `cargo test -p train --release tokenizer` — roundtrip, vocab size. |
| 2 | `cargo test -p autograd --release --features safetensors --test test_safetensors`. |
| 3 | `cargo test -p train --release --test test_qwen3_autograd_forward` (hand-checked top-1 "Paris"). |
| 4 | Run 100-step SFT on `tiny_sft.jsonl`; assert loss drop ≥30%. |
| 5 | Run Step-5 acceptance shell snippet above; coherent chat output. |

No new bench entries required per step — the **headline bench** is "can it chat coherently?", not tok/s. If (and when) Step 4/5 land, a followup wins entry compares loss curves vs HF/PyTorch on the same data.

## Out of scope

- Distributed training (single-node only).
- bf16/fp16 **training** (load-time bf16→f32 is in scope; training remains f32).
- KV cache during training (recompute every step; acceptable at seq_len=1024).
- Pretraining from scratch on web-scale data (infrastructure cost prohibitive on a laptop).
- RL using infer-engine rollouts (step 6 — separate plan).
- Continuous training / online learning / RLAIF (separate plans).
