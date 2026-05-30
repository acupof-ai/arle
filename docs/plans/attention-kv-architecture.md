# Attention / KV Architecture — grounded trait boundaries

> Root-cause treatment of "架构不清晰". Output of a 6-round ground→design→polish
> pass (workflow `attn-kv-trait-design`, 2026-05-30), every claim source-verified.
> Companion: [`gpu-dispatch-governance.md`](gpu-dispatch-governance.md),
> [`backend-operator-library.md`](backend-operator-library.md).

## TL;DR — what the polish established (求真务实)

The proposed clean sketch —

```rust
trait AttentionBackend { fn forward(&self, q: &Tensor, kv: &mut KvCache, meta: &SeqMeta) -> Tensor; }
trait KvLayout { fn alloc(&self,..) -> KvCache; fn append(&self, &mut KvCache, ..); }
```

— does **not** survive contact with the source. Five rounds of adversarial polish,
grounded in the real code, killed two assumptions before they became 脏乱差:

1. **`AttentionBackend::forward` is not groundable as one trait.** Attention here is
   not a uniform "q, kv, meta → out":
   - **Qwen3.5 decode is a per-layer hybrid.** `decode_batch_grouped`
     (`qwen35/batch_decode.rs:984`) interleaves **recurrent LinearAttention** layers
     (`decode_batch_linear_attn_layer_graphable:1051`, mutating `&mut state_ptrs` in a
     **shared CUDA-graph capture group**) with full-attention layers **3:1**
     (`num_linear_layers.div_ceil(3)`). A stateless `forward()` cannot express the
     recurrent sibling layer that shares the capture.
   - **DeepSeek-V4 is MLA latent** (a compressor + indexer over a compressed cache),
     not paged-softmax over `(q, kv)`.
   - **Qwen3-mixed smears the format decision across 9+ sites**
     (`batch_decode.rs:390/547/984/1014/1121/1145/1161/1178-1181 + :1513/:1585 +
     :2061/:2424/:2529`) — write+read fused, not a single dispatch point.

   A single `forward()` trait spanning recurrent + MLA + hybrid + 9-site-smear is a
   **speculative abstraction** (`feedback_no_speculative_interface_shaping`). It would
   make the architecture *less* clear, not more.

2. **`KvLayout::{append, read}` are not pool methods.** `rg 'fn .*attention|fn .*read|
   fn .*gather' paged_kv.rs` is empty. The real *append* is `ops::decode_prep_paged` +
   `quantize_paged_kv_*` (model-side); the real *read* is the per-format
   `decode_attention_*` / `turboquant_fused_decode_attention` arms (model-side). The
   pool only owns **page/byte lifecycle**; the model launches the kernels.

**So the grounded boundaries are:**

| Sketch | Grounded form | Why |
|---|---|---|
| `AttentionBackend::forward` (trait) | **`attention_plan(&AttnShape) -> Result<AttnKernel, String>`** — a backend-neutral SELECTION **value** (free fn, the proven `oplib::linear::plan` pattern), **not** a trait | attention is too heterogeneous to launch behind one method; but the *kernel selection* (the `match kv_pool.format`) **is** a clean fold |
| `KvLayout::{alloc, append, read}` | **`KvLayout`** = a CUDA-bound **page-lifecycle trait** (`alloc_tokens` / `attach_pages` / `alloc_detached_pages` / `free_slot` / `budget_bytes`); **no** append/read | append/read are model-side kernel launches, not pool surface |

This **is** the root fix for 架构不清晰: the scattered `match kv_pool.format`
becomes one inspectable, CPU-testable `attention_plan`, and the pool's lifecycle
becomes one trait — *without* inventing a forward() boundary that the recurrent /
MLA / mixed paths would immediately violate.

---

## The 脉络 — how the whole attention/KV path connects

```
server_engine::InferenceEngine                       front door (HTTP / agent CLI, one contract)
  └ InferenceBackend  (CUDA | Metal)                 backend isolation (server_engine enum)
      └ ModelForward  (qwen3 | qwen35 | deepseek)    a model = a sequence of layers
          │
          ├ per layer, the model picks a KERNEL via a pure SELECTOR (the oplib plane):
          │     oplib::linear::plan(inputs, policy)      -> LinearKernel     (GEMMs)
          │     oplib::attention::attention_plan(shape)  -> AttnKernel       (paged-decode read)   ← NEW
          │     oplib::attention::head_config{,_hd128}   -> HeadConfig       (AOT head specialization)
          │     oplib::kv_dispatch / deepseek            -> {scheme, grouped} (quant / MoE)        ← planned
          │         · all pure, backend-neutral, CPU-testable (no cudarc/mlx types)
          │
          ├ KV lifecycle behind one trait:
          │     KvLayout (TokenKVPool): alloc_tokens / attach_pages / free_slot / budget_bytes
          │         · CUDA-bound page/byte arena; owned by the scheduler
          │
          └ then launches the chosen kernel (bespoke per arm — genuinely incompatible sigs):
                crates/cuda-kernels: TileLang (HD128/HD256) · Marlin · FlashMLA · kv_quant · …

   what stays MODEL-SPECIFIC by evidence (NOT forced into a trait — 不是漏掉,是它们真不同):
     · qwen35 recurrent LinearAttention (gated-delta-rule, shared capture, &mut state)
     · DeepSeek-V4 MLA latent attention (compressor/indexer)
     · the 5 per-arm decode-attention launch bodies (5 incompatible arg lists)

   横切 governance (the 4 gates ride ON these boundaries):
     Declare  = plan()/attention_plan() is the single resolver  ·  explain_dispatch prints it
     Observe  = infer_dispatch_kernel_total{op,variant} at the launch (proven live on H20)
     Assert   = attention_plan equivalence test (CPU) + assert_kernel_fired.sh (/metrics)
     Govern   = kernel-registry (chosen-vs-best-vs-roofline)
```

The "architecture is unclear" because today the **per-layer kernel selection** is
inlined and duplicated. Pulling every selector onto the `oplib` plane (one pure
resolver per op) + the pool's lifecycle behind `KvLayout` makes "what runs on this
layer" a single, named, testable decision. The launch heterogeneity (recurrent / MLA
/ 5 incompatible arms) is **real** and stays where it is — naming it is the
clarification; merging it would be the mess.

---

## The grounded definitions

```rust
// ── oplib::attention (backend-neutral, --no-default-features, CPU-tested) ──

/// Pure host selector input. KVFormat is feature-free (no cudarc in kv_types.rs).
pub struct AttnShape {
    pub qo_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,    // 128 (qwen3 GQA) | 256 (qwen35 full-attn): selects the
                            // AttnHeads arm AND feeds sm_scale = 1/sqrt(head_dim)
    pub kv_format: KVFormat, // selects the read-kernel family
}

pub enum AttnHeads {        // reuses the existing oplib resolvers; the two sets are disjoint
    Hd128(HeadConfigHd128), // Q16Kv8 | Q32Kv8 | Q40Kv8 | Q64Kv8
    Hd256(HeadConfig),      // Q8Kv2  | Q16Kv2 | Q16Kv4
}

/// THE return value — the only thing this design unifies. Family + AOT head config.
/// Each arm names a kernel whose launch body + arg list stays bespoke (5 incompatible sigs).
pub enum AttnKernel {
    Bf16Paged      { heads: AttnHeads },  // tilelang_tc_run_layer / _hd256
    Fp8FusedDecode { heads: AttnHeads },  // decode_attention_fp8_per_channel_k (KIVI)
    Int8FusedDecode{ heads: AttnHeads },  // decode_attention_int8_per_channel_k
    Int4FusedDecode{ heads: AttnHeads },  // (INT4,256) only; (INT4,128) is Err (qwen3 unreachable)
    TurboQuant     { heads: AttnHeads },  // turboquant_fused_decode_attention (+ optional q-rotate)
}

/// Mirrors oplib::linear::plan: pure, returns a neutral value, CPU-testable.
pub fn attention_plan(shape: &AttnShape) -> Result<AttnKernel, String>;

// ── KvLayout (CUDA-bound page lifecycle, deferred tranche) ──
pub trait KvLayout {
    fn budget_bytes_for_tokens(shape: &KvShape, tokens: usize, fmt: KVFormat) -> usize where Self: Sized;
    fn alloc_tokens(&mut self, slot: usize, count: usize) -> Result<Vec<u32>>;
    fn attach_pages(&mut self, slot: usize, pages: &[u32], token_count: usize) -> Result<()>;
    fn alloc_detached_pages(&mut self, count: usize) -> Result<Vec<u32>>;
    fn free_slot(&mut self, slot: usize);
    // NO append / read — those are model-side kernel launches.
}
```

Explicitly **not** introduced (evidence says they would be speculative): an
`AttentionBackend::forward` trait, a recurrent-attention trait, an MLA trait, a unified
`cuda_launch_decode`. Part C of the design: "no recurrent/MLA trait."

---

## Migration — deletion-style, incremental, qwen35-full-attn-first

Order forced by the topology (the only *single*-site, cleanly-foldable consumer is the
qwen35 hybrid's full-attn decode layer; qwen3-mixed's 9-site smear is a later gated
tranche; `KvLayout` is the harder fold, deferred).

- **Step 1 — PURE (no runtime change, CPU-tested, bench-exempt).** Grow `oplib::attention`
  with `attention_plan`, reproducing the `match kv_pool.format` read-kernel selection from
  `decode_batch_full_attn_layer` (`batch_decode.rs:1369`) **verbatim** (incl. the INT4
  reachability split + unsupported-`head_dim` Err). Add the selection-equivalence CPU test.
- **Step 2 — MOVE the selector (runtime change, bench required).** Point that one decode
  site at `attention_plan()`, rename `match kv_pool.format` → `match kernel`; **each arm keeps
  its bespoke body + every arg verbatim** (no unified launcher). **DELETE** the relocated
  inline selection (deletion-style; no old+new).
- **Step 3 — VERIFY bit-identical.** `kv_precision_parity` (BF16 vs INT8/FP8/INT4/TQ
  trajectory) on H20 catches a dropped workspace ptr that Step-1 can't; `bench_guidellm.sh`
  Δ%≈0. wins/ entry (`pending-remote` — no nvcc on Mac).
- **Later (separate, gated):** qwen3-mixed 9-site fold · `KvLayout` lifecycle trait ·
  DSv4/Metal widening · `oplib::kv_dispatch` (quant scheme) · `oplib::deepseek` (grouped MoE).

Each step is one small tranche, deletion-style, behavior-bit-identical, with its own
verify — never a big-bang trait rewrite.
