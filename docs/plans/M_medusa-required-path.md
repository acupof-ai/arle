# M_medusa — Medusa multi-head spec-decode REQUIRED path (post-classical-DEAD)

> Master §7.4 P1.1 promoted from "preferred" to **REQUIRED** by **4 classical
> KILLs**: `5f26675` (4k self α=7%), `3ac5f4d` (4k ext α=19%),
> `8f2b227` (32k self α=23%), `aa00c6a` (W3 c=4 production-shape α=19%).
> Pattern: classical Leviathan α ≤ 0.25 across all 4 tested workloads on
> Qwen3-4B + sm_89 + ARLE current = structural ceiling. Medusa shared-target
> trained heads is the architectural change required to break α ceiling.
> The 4th KILL specifically refutes the "high prefix hit will save spec"
> hypothesis — prefix_hit_rate ≠ accept_rate (independent axes).
>
> Codex own (training + substrate). Plan ready for pickup.

> **2026-05-10 scope update**: Qwen3-specific execution is frozen by user
> direction. The Qwen3.5 path is not a direct transplant of this plan:
> Step 0 audit found a recurrent-state rollback blocker in the CUDA
> Qwen3.5 hybrid model. Do not expose Qwen3.5 Medusa until the verifier
> has a model-owned accepted-length commit/rollback contract. See
> `docs/research/2026-05-10-medusa-phase1b-qwen35-step0-audit.md`.

## Phase 1 — Target

| Field | Value |
|---|---|
| Metric | tok/s on agent W3 / W4 production shape (master §2.1) |
| Baseline | W4A16 Marlin no-spec at agent shape — TBD (gated on `a672b08` admission fix) |
| **License** | tok/s ≥ 1.5× vs no-spec at W3/W4 (master §7.4 threshold) |
| Soft win | tok/s ≥ 1.2× — proceed but flag for K-tuning |
| Kill | tok/s < 1.0× at any agent shape — Medusa training failed |
| Wall-clock budget | 1 week training + 1 day integration + 1 day bench |

## Phase 2 — Hardware constraints

sm_89 RTX 4070 Ti SUPER, 16 GB VRAM. Same as production W4A16 setup.

Medusa-specific:
- Training: needs target model frozen + 4 Medusa heads trained (~8-16M params each)
- Memory at training: target 4B BF16 (8 GB) + 4 heads (~32 MB) + activations
  + optimizer state. Tight on 16 GB; may need gradient checkpointing or
  smaller global batch.
- Inference: target + 4 Medusa heads coexist in VRAM. ~150 MB extra.

## Phase 3 — Binding constraint (formula-grounded)

Medusa speedup uses tree-attention over top-T candidates per head + target
verification on accepted prefix. Standard Leviathan formula does NOT apply
directly (that's classical 1-draft-per-step). Correct derivation:

```
E[accepted tokens / step] = 1 + Σ_{i=1..K} Π_{j=1..i} α_j
  where α_j is per-position acceptance probability for j-th Medusa head.

For uniform α (worst case — heads decay slower with proper training):
  α=0.7: E = 1 + 0.7 + 0.49 + 0.343 + 0.24 = 2.78 tokens/step
  α=0.85: E = 1 + 0.85 + 0.72 + 0.61 + 0.52 = 3.71 tokens/step

Per-step cost ratio (Medusa K=4 vs no-spec):
  cost_ratio = 1 + K × (head_trainable_params / target_params) + tree_attn_overhead
  Qwen3-4B + 4 heads (each ResBlock + reused lm_head):
    head_trainable = ~6.5M (single ResBlock) — lm_head shared with target
    cost_ratio = 1 + 4 × (6.5M / 4B) + 0.05 ≈ 1.06× (heads are essentially free)

Throughput speedup = E[accepted/step] / cost_ratio:
  α=0.7:  2.78 / 1.06 = 2.62×
  α=0.85: 3.71 / 1.06 = 3.50×
```

Literature claims α 0.7-0.85 (Medusa-2 paper, Vicuna-7B baseline). For
Qwen3-4B production W3/W4 (structured agent shape), predict α 0.6-0.8
range based on similar coding-fine-tuned model behavior. Predicted
throughput speedup: **2.0-3.0× tok/s**.

Why classical α 0.07-0.25 → Medusa α 0.6-0.85? Architectural difference:
- Classical:single fixed model self-predicting K positions ahead → entropy
  compounds geometrically → α^K decay
- Medusa:K *trained* heads each specialized to position-i prediction →
  per-position α stays high (paper §3.2 shows heads learn position-specific
  features that vanilla self-spec cannot capture)

## Phase 4 — Formula prediction (concrete numbers)

For Qwen3-4B production W3/W4:
- W3 c=4 baseline established: ITL p50 8.5 ms = ~117 tok/s/session × 4 conc
  ≈ 468 tok/s aggregate, or per-session ~117 tok/s (`370a267`)
- W3 c=16 baseline: BLOCKED on `cb087c7` deadlock(codex `369292f`
  page_budget hypothesis fix in flight)
- Medusa K=4 prediction(α=0.7):2.62× → ~306 tok/s/session,~1226 tok/s aggregate
- Medusa K=4 prediction(α=0.85):3.50× → ~410 tok/s/session,~1638 tok/s aggregate
- K=8 doesn't help if heads decay below α=0.5 at i≥5 (need empirical α curve);
  K=4 is the safe starting point per Medusa-2 paper.

## Phase 5 — Implementation outline (codex own)

### 5.1 Training data (~1 week)

- Source: agent W3/W4 traces from `scripts/data/agent_trace_default.jsonl`
  + master §2.1 representative workload (system prompt + tool calls)
- Augmentation: Qwen3-4B as teacher model generating 100k+ token sequences
- Format: per-token logits + last-N hidden states (input for Medusa heads)

### 5.2 Medusa head architecture (~200 LOC PyTorch)

- 4 heads on top of Qwen3-4B last hidden state(d=2560,vocab=151936)
- Each head:**1× ResBlock(d → d)** with skip,reuse target's lm_head
  for d → vocab projection (per Medusa paper §3.2 "shared lm_head")
- Per-head trainable params:~6.5M(ResBlock weights only)
- Total trainable across 4 heads:~26M(<1% of 4B target — heads are tiny)
- Initialize ResBlocks from random + bias toward identity(start as
  "predict same as target")
- Loss:cross-entropy at each Medusa position(1, 2, 3, 4 ahead)

### 5.3 ARLE integration (~300 LOC)

- `infer/src/speculative/medusa.rs` — new module:
  - Load Medusa head weights alongside target Qwen3-4B
  - During decode: run target → 4 Medusa head logits in parallel
  - Tree-attention over top-K candidates per head (per Medusa paper)
  - Verify with target run on accepted prefix, accept ≤ K+1 tokens per step
- Sampler integration: replace `--spec-draft-model self/external` with
  `--spec-draft-model medusa:<path-to-heads>`
- TileLang verify kernel reuse from existing `infer/src/speculative.rs`

### 5.4 Tests (~50 LOC)

- `infer/tests/medusa_consistency.rs` — output token-equivalent to no-spec
  (Medusa is verified spec-decode = bit-exact greedy when properly impl)
- Acceptance rate counter in `/v1/stats` (already plumbed)

## Phase 6 — Combinational A/B

Post-license, combine with W4A16 + xgrammar:

| Stack | Expected tok/s |
|---|---|
| BF16 baseline | 150 (4k W3) |
| W4A16 Marlin | 200 (1.33×) |
| W4A16 + Medusa K=4 | 400-500 (2.5-3.3×) |
| **W4A16 + Medusa + xgrammar** | 380-470 (slight grammar overhead) |

## Phase 7 — Tradeoffs

| Axis | Status | Note |
|---|---|---|
| LOC | ⚠ ~500 substrate + training data prep | codex own |
| HW specificity | ✅ none | Medusa is general |
| Memory | ⚠ +150 MB head weights + tree-attn buffers | acceptable on 16 GB |
| **Training risk** | ❌ ~1 week + data quality | the main cost |
| Numerical correctness | ✅ verified spec — bit-exact greedy | matches no-spec |
| Generality | ⚠ workload-trained | retrain per major model update |
| Acceptance ceiling | predicted 0.7-0.85 | breaks classical α ≤ 0.25 |

## Phase 8 — License decision

| Result | Action |
|---|---|
| tok/s ≥ 2× vs no-spec | LAND HARD — production default for agent path |
| tok/s 1.5-2× | LAND incremental + tune K |
| tok/s 1.2-1.5× | hold for retraining or larger heads |
| tok/s < 1.0× | KILL Medusa axis on Qwen3-4B + sm_89 |

## Pre-execution gates

1. **W3+W4 admission deadlock fix**(codex page_budget hypothesis
   `369292f` + `infer/src/scheduler/cuda/execution.rs` in flight)— required
   for production-shape c=16/c=8 baseline pre-Medusa。**Currently blocking;
   c=4 workaround licensed at `370a267`.**
2. **Training environment** — Qwen3-4B target + heads fit on 16 GB GPU
   (heads are 26M params total = <100 MB, easily fits even with target +
   activations + gradients)
3. **bench_agent_trace.py harness** — production-shape measurement(ready,
   used by `370a267` and `aa00c6a`)
4. **W4A16 production decode default** — LICENSED at `f6f3af3` 1.64× ITL,
   stable comparison baseline for Medusa A/B

## Cross-references

- Master §7.4 P1.1 update: [`docs/projects/2026-05-07-arle-master-strategy.md`](../projects/2026-05-07-arle-master-strategy.md) (`5acbe94`)
- **4 classical KILL evidence**(α range 0.07-0.25):
  - `5f26675` self-spec K=5 4k random α=0.07
  - `3ac5f4d` ext-draft Qwen3-0.6B K=5 4k random α=0.19
  - `8f2b227` self-spec K=5 32k random α=0.23
  - **`aa00c6a` self-spec K=5 W3 c=4 production-shape α=0.19**(refutes "high
    prefix hit saves spec" — prefix_hit_rate 93% but accept_rate 19%
    independent)
- W3+W4 admission deadlock evidence:
  - `cb087c7` W3 c=16 deadlock initial(harness retry-backoff insufficient)
  - `e3669d4` W4 c=8 deadlock confirms substrate(workload-dependent)
  - `369292f` codex page_budget root-cause hypothesis(in-flight fix)
- W4A16 baseline: `f6f3af3` LICENSED 1.64× ITL(production decode default)
- W3 c=4 baseline: `370a267` 384 turns OK,99% prefix hit
- Medusa paper: <https://arxiv.org/abs/2401.10774>
- Medusa-2(improved): <https://arxiv.org/abs/2402.04968>
- ARLE spec substrate: `infer/src/speculative.rs` (721 LOC)
- Skill v1.3.0: `.claude/skills/kernel-optimization/SKILL.md` (`faffcb0`)

## Rule

Medusa is **mandatory** for spec-decode axis on Qwen3-4B + ARLE current.
Three classical KILL evidences proved α ≤ 0.25 ceiling is structural.
The 1-week training cost is now the cheapest path; classical alternatives
all benched dead.

If user/codex wants to defer Medusa training cost, the alternative is
"spec-decode axis OFF for production" — which loses the master §6.1
5-cap moat capability 4 entirely. Master strategy should reflect this
tradeoff explicitly when prioritizing.
