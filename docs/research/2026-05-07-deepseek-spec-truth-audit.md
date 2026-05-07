# DeepSeek Spec vs DSV4 Truth Audit

## Scope

- Truth source: `docs/projects/2026-05-07-arle-master-strategy.md` section 5.1 (DSV4 architecture).
- Code audited: `crates/deepseek-spec/src/lib.rs`.
- Local HF replica checked: `infer/models/dsv4-mini-1B-init/config.json`.
- Local HF model code checked:
  `infer/models/dsv4-mini-1B-init/code/deepseek_v4/modeling_deepseek_v4.py`.
- Bottom line: `deepseek-spec` is still a V3/MLA scaffold, not a DSV4 replica
  spec.

## Config Coverage

- Truth section 1 lists 45 top-level config keys.
- Local `config.json` has 47 keys because it also includes
  `transformers_version` and `use_cache`.
- `DeepSeekConfig` covers about 21 of the 45 truth keys by exact name or alias.
- Schema coverage estimate: about 47% of truth keys, or about 45% of the local
  HF `config.json`.
- Operational coverage is 0% today: parsing the HF replica config fails before
  validation because required V3 fields are absent.
- Blocking required V3 fields: `kv_lora_rank`, `qk_nope_head_dim`,
  `v_head_dim`, and `first_k_dense_replace`.

## Missing Field Table

| Field/group | Truth value | Current status |
|---|---:|---|
| `architectures` | `DeepseekV4ForCausalLM` | missing |
| `model_type` | `deepseek_v4` | missing |
| `dtype` | `bfloat16` | missing |
| `head_dim` | 64 | missing; V3 code derives from qk/nope/v dims |
| `swiglu_limit` | 10.0 | missing |
| `q_lora_rank` | 384 | present, but optional V3 semantics |
| `o_lora_rank` | 384 | missing |
| `o_groups` | 4 | missing |
| `qk_rope_head_dim` | 32 | present |
| `routed_scaling_factor` | 1.5 | present |
| `scoring_func` | `sqrtsoftplus` | missing |
| `topk_method` | `noaux_tc` | missing |
| `index_n_heads` | 8 | missing |
| `index_head_dim` | 64 | missing |
| `index_topk` | 128 | missing |
| `num_hash_layers` | 2 | missing |
| `sliding_window` | 64 | missing |
| `compress_ratios` | 24 entries | missing |
| `compress_rope_theta` | 160000.0 | missing |
| `hc_mult` | 4 | missing |
| `hc_sinkhorn_iters` | 20 | missing |
| `hc_eps` | 1e-6 | missing |
| `num_nextn_predict_layers` | 1 | present |
| `rope_parameters` / YaRN | factor 16 | missing nested struct |
| `initializer_range` | 0.02 | missing |
| `attention_bias` / `attention_dropout` | false / 0.0 | missing |
| `pad_token_id` | null | missing |
| `use_cache` / `transformers_version` | local HF only | missing |

## Tensor Name Coverage Gaps

| DSV4 tensor family | HF replica names | `deepseek-spec` status |
|---|---|---|
| Top-level embed/head | `embed.weight`, `head.weight`, `norm.weight` | wrong names: expects `model.embed_tokens.weight`, `lm_head.weight` |
| Top-level mHC head | `hc_head_{base,fn,scale}` | missing |
| Layer norms | `layers.L.attn_norm.weight`, `ffn_norm.weight` | wrong names: `input_layernorm`, `post_attention_layernorm` |
| Layer mHC streams | `layers.L.hc_attn_*`, `hc_ffn_*` | missing |
| Q-LoRA attention | `attn.wq_a`, `q_norm`, `wq_b` | semantically close but wrong names |
| Single-KV GQA | `attn.wkv`, `attn.kv_norm` | missing; current map is V3 MLA KV |
| O-LoRA grouped output | `attn.wo_a`, `attn.wo_b` | missing; current map has only `o_proj` |
| Attention sink | `attn.attn_sink` | missing |
| CSA/HCA compressor | `attn.compressor.{wkv,wgate,ape,norm}` | missing |
| Lightning Indexer | `attn.indexer.{wq_b,weights_proj,compressor.*}` | missing |
| Router | `ffn.gate.{weight,bias,tid2eid}` | partial; no bias or hash table |
| Experts | `ffn.experts.J.{w1,w2,w3}.weight` | wrong names: `gate_proj/up_proj/down_proj` |
| Shared expert | `ffn.shared_experts.{w1,w2,w3}.weight` | wrong names |
| MTP module | `mtp.K.*` with inner attn/ffn/mHC/head | V3-style `model.layers.N.*`; wrong shape |
| Dual RoPE path | config-only `rope_theta` + `compress_rope_theta` | missing compressed theta |

## Shard Adjustment Suggestions

- Add a DSV4 tensor-name layer separate from `DeepSeekMlaTensorNames`; do not
  retrofit V4 into the V3 MLA structs.
- Treat `num_key_value_heads=1` as a first-class shard case.
- For the single KV tensor `attn.wkv`, replicate across TP ranks unless a
  dedicated sub-head split policy is introduced.
- Q-LoRA: shard `wq_b` by query heads on dim 0; keep `wq_a` and `q_norm`
  replicated unless TP support for rank splitting is explicitly added.
- O-LoRA: shard `wo_a` along group/rank output only when `o_groups` divides the
  TP layout; shard `wo_b` as row-parallel over the low-rank input.
- MHC params, norm weights, `attn_sink`, compressor `ape`, and head mHC params
  should be replicated.
- Lightning Indexer `wq_b` and `weights_proj` can split by `index_n_heads` only
  when divisible by TP size; otherwise replicate.
- MoE router and hash table should be replicated; routed experts keep
  `ExpertParallel`, with expert `w1/w3` column and `w2` row only inside a local
  TP-over-expert policy.
- MTP should reuse the DSV4 layer shard policy, not `model.layers.N` V3 naming.

## Deprecated MLA References

- `DeepSeekMlaTensorNames` should be marked V3-only.
- `kv_a_proj_with_mqa`, `kv_a_layernorm`, and `kv_b_proj` are V3 MLA-only.
- `RawDeepSeekConfig.kv_lora_rank` and `DeepSeekConfig.kv_lora_rank` block DSV4
  parsing and should be deprecated for V4.
- `qk_nope_head_dim`, `v_head_dim`, `qk_head_dim()`, `q_proj_dim()`, and
  `kv_b_proj_dim()` are V3 MLA shape helpers.
- `validate()` still enforces "MLA rank and head dimensions must be non-zero".
- `layer_tensor_names()` emits `self_attn.kv_a_proj_with_mqa` and `kv_b_proj`.
- Tests are anchored to `DEEPSEEK_V3_CONFIG`; keep them as V3 legacy fixtures
  and add a DSV4 truth-config fixture.
- `nano()` is documented as MLA/no-MoE/no-MTP; it is not a DSV4-mini truth SKU.

## Assessment

- Config schema coverage: about 47% of truth section 1.
- Actual ability to parse the HF replica config: 0%.
- Tensor-name coverage for DSV4 layer weights: under 30% by family, and the
  covered pieces mostly have wrong prefixes or V3 names.
- Shard policy is usable for generic MoE expert slicing, but not for DSV4's
  single-KV-head GQA, O-LoRA grouping, mHC streams, Lightning Indexer, or MTP.
- Required next step: introduce explicit `DeepSeekV4Config` and
  `DeepSeekV4TensorNames`, then mark the current MLA structs as V3 legacy.
