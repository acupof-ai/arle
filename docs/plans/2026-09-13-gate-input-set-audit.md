# Gate input-set audit — does each gate's input set contain what it claims to cover?

Date: 2026-09-13. Read-only census; **no code changed**. Method: for every
`*_parity.rs` example, every `*_gate.{sh,py}`, the `parity_gpu_batch.sh`
roster (derived from `operators/registry.toml`), and the numeric-bound
train examples, read the gate's own header for its claim, read its
constants/generators for its actual input set, and read the production
configuration for that quantity from `docs/support-matrix.md`, model
configs, or production code (not memory).

**The column that matters is "Input contains production shape".** `yes`
= the gate exercises the real shape/dtype/TP/weights of the thing it
names; `no` = a green run cannot speak to the named production path;
`unknown` = the production geometry could not be named from anything
committed (itself a finding).

Legend: **RNG** = synthetic random/synthetic weights, no checkpoint;
**real** = loads/uses production checkpoint or production-derived
tensors.

## The motivating population (found by accident this week)

Five input-set-excludes-the-target defects, none found by the gate:

1. `dspark_drafter_batch_invariance` — synthetic RNG, GATE_VOCAB 2048,
   TP1; does not load Qwen3.8-27B.
2. `marlin_fp8_parity` — samples a fixed 1024 columns regardless of
   matrix width (per the row's own note).
3. `dsv4`-era oracle fed the gate's own latent instead of the device's.
4. Vulkan device tests — no runner provides an ICD, so the input set is
   never instantiated.
5. 16 device-requiring `cuda-kernels` functions never executed.

## Infer-cuda kernel/model parity gates

| Gate | Claims | Actual input set | Production config (source) | Contains? |
|---|---|---|---|---|
| argmax_parity | device argmax vs host across vocab/batch | RNG-normal logits; VOCABS 7…151936 incl real Qwen vocab 151936 | Qwen vocab 151936 (const matches) | **yes** (dtype/width only; argmax is weight-independent, so RNG is sufficient by nature) |
| deepgemm_grouped_prefill_parity | per-expert FP8 grouped prefill GEMM vs f64 | RNG e4m3 operands, real-magnitude per-block f32 scales; M_CAP 128 one tile; 256-expert band | DSv4/Qwen3.8 grouped FP8 MoE at prefill; exact per-expert M/K geometry not enumerated here | **partial** — dtype/quant path realistic; full production M sweep and real weight distribution not covered |
| dspark_draft_attn_parity | single-vs-batched drafter ring attention at DSpark geometry | hash/RNG synthetic fill, 40q/8kv/hd128/block7 (matches DSpark config dims); B=1 vs 8 | Qwen3.8-27B-DSpark dims 40q/8kv hd128 block7 | **no** — dims match but synthetic weights; green covers kernel arithmetic, not the real drafter path |
| dspark_dsa_parity | DSA (deepseek sparse attn) kernel vs oracle | RNG bf16/fp8, synthetic | DSv4 DSA production dims | **partial/unknown** — production DSA geometry not cross-checked line-by-line here |
| dspark_sampler_parity | DSpark speculative sampler filters/accept vs oracle | RNG logits; vocab-parametrized synthetic incl spike rows; MAX_DEPTH 4 | sampler is weight-independent given logits | **yes** (sampler logic; RNG logits are the natural input) |
| dsv4_decode_moe_parity | batched decode MoE pack/router/combine | RNG; TOPK6, HIDDEN4096, ep 32/64/128/256, tokens 1/8/32; TP_WORLDS 2/4/8 | DSv4 256-expert TP8/EP8 decode (support-matrix) | **partial** — EP256/TP8 present, weights RNG; combine bound derived, real distribution not |
| dsv4_parity | multi-rank DSv4 first-token greedy parity | **real** `INFER_DSV4_MODEL_PATH` checkpoint, TP=N NCCL, 8 ranks; gates first of 16 oracle tokens | DSv4-Flash TP8/EP8 | **yes for token 1 / no for tokens 2-16** (incremental bails; verdict now scoped via #445); never run (no model+cards) |
| dsv4_tp_oproj_parity | TP1-8 o-proj DeepGEMM vs scalar reference | RNG fp8; TPS 1/2/4/8, decode B 1/8; production C=1024 o_lora_rank named | DSv4 TP o-proj shapes | **partial** — TP sweep and dtype real; RNG weights |
| elementwise_parity | fused elementwise kernels vs scalar per model | RNG bf16; width table qwen36-27b 5120×17408, dsv4 4096×2048; KV_DIM 1024 synthetic table | those named widths | **partial** — production widths listed, contents synthetic (elementwise is data-independent, so largely sufficient) |
| fa2_sm70_parity | FA2 sm70 attention vs host on V100 | small synthetic seq; sm70-only | sm70 is **not** the production box (H20 is sm90) | **n/a device** — targets hardware the fleet doesn't run; skips on sm80+; keep/drop is a decision |
| fa3_hd256_shim_parity | Qwen3.6 paged FA3 shim (D256 head) quant path | RNG; D=256; descale/requant e4m3 device formulas | Qwen3.6 full-attn head_dim 256 | **partial** — head dim matches; synthetic Q/K/V |
| flashmla_hca_decode_parity | HCA ratio128 batched decode attention | synthetic latent pool, K=V=0.5 dominant row + ±0.05; B8, 584B/token FP8 packed, start 127-384 | DSv4-Flash HCA decode geometry | **partial** — production geometry/dtype/packing, synthetic KV; output negative tooth bound underived (decode-bound row) |
| flashmla_prefill_parity | CSA/HCA prefill index + sparse fwd vs f64 | synthetic latent ±0.02 + per-case dominant for the negative; chunks 128/2048/4096, 584B-equivalent BF16 pool | DSv4-Flash prefill CSA/HCA geometry | **partial** — geometry real, KV synthetic; numerically tight (rel ~7e-4), which is what an arithmetic gate needs |
| flashmla_sparse_decode_parity | CSA ratio4 batched decode attention | same synthetic dominant/random construction; starts 63-256 | DSv4-Flash CSA decode | **partial** (as HCA; output bound underived) |
| gdr_decode_parity | GDN linear-attention decode recurrence vs f64 | RNG BF16; KEY_DIM128, K4, geom (kh,vh)×B sweep | Qwen3.5/3.8 GDN head_dim128, 40q/8kv | **partial** — geom sweep includes (16,48)/(8,24); RNG state |
| gdr_varlen_parity | varlen vs chunked-FlashQLA GDR prefill, FQ-vs-f64 anchor | RNG; KEY_DIM128 K4; lengths 5/17/64; geoms (16,48)/(8,24)/(4,12)/(2,6) | multi-row GDN prefill at TP1-8 | **yes for the routing question** (it is a path-equivalence gate; RNG sufficient), pending GPU |
| marlin_fp8_parity | Marlin FP8 GEMM vs f64 over Qwen3.8-27B shapes | RNG fp8; SHAPES fixed incl q/k/v_proj 1024×5120; M sweep 1-256; **3 fixed seeds**; **samples fixed 1024 columns** | Qwen3.8-27B per-channel FP8: hidden 5120, 24h/4kv/hd256, vocab 248320 | **partial** — some production shapes, but fixed-1024-column sampling means wide-matrix behaviour (the 248320/17408 dims) is never exercised; the known defect #5 |
| marlin_w8a16_parity | Marlin W8A16 GEMM vs f64/fallback | RNG; GROUP128; SHAPES + declined 96×5120; M 1-32; 3 seeds | Qwen3.6-27B dense w8a16 shapes | **partial** — production-aligned shapes listed, RNG weights, bounded M |
| moe_routing_parity | MoE route/count/pack/m-indices/weights/combine vs f64 | RNG; 256 experts TOPK6 HIDDEN4096, tokens 1/8/32, ep 32-256 | GLM-5.2/DSv4 256-expert routing | **partial** — EP256/TOPK6 match, RNG logits; verified first GPU run this session (#436) |
| paged_quant_attn_parity | quantized FA3 paged attention vs host | RNG; H16/HK4/D256/PAGE16/QLEN8 | Qwen3-family INT8/FP8 paged quant FA3 | **partial** — head/head-dim match; RNG KV; single short QLEN |

## Train parity examples (multi-rank/training)

| Gate | Claims | Actual input set | Production config | Contains? |
|---|---|---|---|---|
| cp_hidden_parity | context-parallel hidden-state equivalence | tiny synthetic cfg vocab16, CP2 SEQ16, name-seeded tiny model | CP on 27B-scale training | **no (scale)** — algebra identity at toy size; does not exercise production dims/weights |
| cp_ring_transport_parity | CP zigzag ring KV transport | HEADS4 hd128 SEQ16 CP2 synthetic | production CP transport | **partial** — hd128 real, seq/world toy |
| moe_tp_parity | MoE expert TP sharding identity | name-seeded tiny MoE, world 2 (env), finite-diff eps1e-2 | Qwen3.6 MoE at TP2-8 training | **no (scale/TP)** — TP2 only by default, tiny experts; proves sharding algebra not production |
| nd_parallel_parity | ndim-parallel grad reduction equivalence | deterministic vocab-cycling input, CP2, 4 layers (named 27B GDN count), WINDOW64 | 27B ND training | **partial** — GDN layer count matches, width/scale toy |

## End-to-end / script gates

| Gate | Claims | Actual input set | Production config | Contains? |
|---|---|---|---|---|
| lever_gate.sh | license-or-kill model correctness | drives needle ladder ×3 against a **real served model** (`MODEL` env) | whichever model is pointed at; CI Metal uses mlx-community/Qwen3.5-0.8B-MLX-4bit | **yes if run with the production model**; the gate itself is model-neutral so coverage depends on the `MODEL` the runner chooses |
| needle_gate.py | needle-in-haystack exact retrieval | real model, synthetic haystack at chosen lengths/depths | served model | **yes** (tests retrieval behaviour on the real model; haystack is legitimately synthetic) |
| sampling_gate.py | sampling-parameter correctness vs live serve | real serve, parametrised API probes | served model | **yes** |
| longctx_numerical_gate.py | long-context numerical cross-server equality | compares arle vs sglang completions on the same prompts at long length | production long-ctx model | **yes if both serves up**; gate_invoke=manual, not wired to CI (runs ad hoc) |
| qwen35_tq4_dense_parity.py | TQ4 quantized-checkpoint dense tensor/module parity | **real** checkpoint — `--source-model` required (no machine default), opens safetensors, compares packed quant tensors vs a torch/transformers reference, Qwen3.5-9B ModelScope layout | the quantized 9B checkpoint under test | **yes (weights), manual** — real quantized weights; run manually against the named checkpoint, no CI runner |
| spec_parity.py | spec decode zero token-id mismatch | Metal, `models/Qwen3.5-0.8B-MLX-4bit`, r3lax DSpark draft | Metal spec on the 0.8B test model | **yes for that pairing** — and it found the pairing is not correctness-preserving (prereg: 7/8 drift); it does NOT cover DSpark on 27B |
| dspark_flashqla_verify.sh | chunked-FQ routing + DSpark gate across TP1-8 | wraps the kernel gates on pod; real routing observation | production GDR at TP | **yes for routing**; inherits the drafter gate's synthetic-weight limit for acceptance |
| parity_gpu_batch.sh | roster + PASS/FAIL/SKIP aggregation | derives gate list from registry; runs examples + dsv4 model gate | — | meta-runner; its coverage is exactly the union of the rows above, so it inherits every gap and additionally SKIPs the model gate without model+cards |

## Cross-cutting findings

1. **Kernel-arithmetic gates are appropriately synthetic.** Argmax,
   sampler, elementwise, routing, and the GDR/FlashMLA geometry-equivalence
   gates are data-independent or path-equivalence checks: deterministic RNG
   at the production shape is a sound input, and demanding real weights
   would add cost without coverage. These are **not** the defect.
2. **The defect concentrates in gates whose claim is model-level
   correctness but whose input is synthetic.** `dspark_draft_attn_parity`,
   both marlin gates' wide-matrix sampling, the train CP/TP/ND examples at
   toy scale, and `#374` share one shape: the headline names a 27B/production
   path; the input set excludes its weights or its width. A green there is
   necessary (kernel arithmetic) and not sufficient (model correctness).
3. **Real-weight coverage is two gates, and automated real-weight parity
   that has ever run is zero.** `dsv4_parity` loads a checkpoint and gates
   the first of 16 tokens, but it has never executed (no model+cards).
   `qwen35_tq4_dense_parity.py` loads real quantized weights but is a
   hand-driven torch diagnostic with no CI runner. So the number of
   automated real-weight parity gates that have ever run green is **0** —
   and there is no real-weight 27B DSpark gate at all, the precise gap
   behind `dspark-accept-collapse`.
4. **"unknown" is minimal.** Every gate's target could be named except the
   exact production DSA per-expert geometry (the gate sweeps but the
   production M/K is not enumerated in-tree) — one genuinely under-specified
   target. A gate whose target configuration nobody can state is covering
   something unspecified.
5. Device instantiation is a separate axis from shape/weights: Vulkan and
   the 16 cuda-kernels device functions have correct shapes but no runner
   that ever executes them.

## Recommended convention (not applied here)

Every registered gate's `gate_gap` should state its input set in four
fields — shapes, dtype, TP/EP degree, weights (RNG vs checkpoint) — and the
registry should carry the production target so "contains" is mechanically
checkable rather than rediscovered per incident.

## The smallest executable real-weight gate (priced option, not a plan to build)

Facts read from the pod on 2026-09-13, with the inventory command attached
(the earlier draft searched only the /host model dir and got this wrong):

- **A complete production checkpoint is already on the box, no
  provisioning needed.** `Qwen3.6-35B-A3B-FP8` is present and complete:

  Verified on the pod (pod-side model root + checkpoint name; absolute
  machine root omitted to avoid hard-coding a machine path):
  - size 35 GB, 42 `.safetensors` shards on disk;
  - the index `weight_map` names exactly 42 distinct shards and all 42 are
    present (checked by loading the JSON and diffing the shard set against
    disk);
  - 663 GB free on the model volume.

  The earlier draft's mistake was searching only the other mounted model
  tree (whose copy of this model has just 4 shards and whose 27B-FP8 dirs
  are config-only). There is **no 27B checkpoint anywhere on the box**;
  the complete model is the 35B-A3B MoE.
- **H20 = ~96 GB each** (97871 MiB), so the 35 GB checkpoint fits TP1 on
  one card with room for KV. This checkpoint has a recorded **1×H20
  single-GPU eager baseline** (`docs/baselines.md:150`) and multiple wins
  entries bench it at 1×H20 — no bring-up risk.
- **DSv4-Flash-FP8 is 294 GB → minimum 4 H20 by memory**, served at
  TP8/EP8, and is not on the box.

Two priced options:

| Option | Checkpoint | Cards | What it covers beyond the synthetic gates | Cost |
|---|---|---|---|---|
| **A (smallest; provisions nothing)** | **Qwen3.6-35B-A3B-FP8**, TP1 | **1 H20** | Real FP8 MoE weights through the full decode/prefill stack — DeepGEMM/Marlin lanes, quant KV, the real hidden/vocab widths, end-to-end greedy correctness on a MoE | **a card window + minutes of run; zero transfer, zero approval** — weights present, TP1 already baselined |
| C | **DSv4-Flash-FP8**, TP4 minimum (serves TP8) | **4-8 H20** | The only gate that exercises real NCCL multi-rank, EP8 MoE, FlashMLA on real DSv4 tensors — what `dsv4_parity` already targets | provision **294 GB** (not present; cold read dominates) + 4-8 cards; unique multi-rank coverage |

(An NVFP4 option was dropped: no NVFP4 checkpoint exists on the box, so it
would price a model we do not have against one we do.)

**Recommendation:** Option A changes the "automated real-weight gates ever
run = 0" number at the lowest possible cost — one card, minutes, no
provisioning decision for anyone to approve (the only gate is GPU
availability, already blocked for other reasons). It needs no new harness:
the lever/needle/sampling gates already drive a real served model; the work
is pointing the GPU batch's model phase at this checkpoint when a window
opens, not building code.

Option C stays separately queued for the multi-rank question and the 15
ungated dsv4 tokens; it is not the smallest path to a nonzero count.
