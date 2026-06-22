# Metal DFlash spec-decode: top-k acceptance (`--spec-accept-topk`)

## Context

Metal DFlash (MTP self-spec) verify accepts a draft token only when it equals the
target's argmax (exact greedy). ckl wants a relaxed accept: accept the draft if it
lies in the target's **top-k** for that position — raises acceptance rate (longer
accepted prefixes → fewer target forwards/token → faster) at the cost of a lossy
deviation from exact greedy. Draft generation and the single target verify forward
are UNCHANGED; only the accept comparison changes; verify cost is unchanged (the
logits are already computed, top-k is a free membership test).

## What Worked

**Change (option A — commit the draft token on top-k hit):**
- `mlx_qwen35_model.cpp:qwen35_compiled_verify_block_summary` — `accept_topk` param;
  when `>1`, accept iff the draft's logit rank (count of strictly-greater logits) is
  `< k` (tie-robust), else the original `== argmax`. `logits` reshaped to
  `[block_size, vocab]` for shape safety.
- Threaded as a **CLI flag** (not a user env var, per `feedback_runtime_config_cli_flags_not_env`):
  `--spec-accept-topk K` (`cli/args.rs`, default 1) → `serve.rs` exports
  `INFER_METAL_DFLASH_ACCEPT_TOPK` (internal, confined to the resolve boundary like
  `--speculative-tokens`→`INFER_METAL_DFLASH_TOKENS`) → `executor.rs:resolve_dflash`
  reads it into `MetalDflashOptions.accept_topk` → `MetalDflashRuntime.accept_topk`
  field → verify call site. No env read in the hot path.

**Default k=1 is bit-identical**: routed to the original `== argmax` branch; the env
is only exported when speculation is active. Opt-in; baseline unchanged.

**Algorithm validated** (`scripts`-style MLX check vs numpy top-k reference):
200/200 random trials match; tie-at-threshold accepted; k=1 == argmax membership.
Compiles clean (mlx-sys C++ + infer-metal); `cli` env-flag test passes (flag→"2",
`--no-speculative`→absent, auto-resolve→default "1").

## e2e measured on Qwen3.6-27B (2026-06-22)

Ran on `mlx-community/Qwen3.6-27B-OptiQ-4bit` (target) + auto-resolved
`Qwen3.6-27B-MTP-4bit` draft, depth-2 MTP, `arle run --no-tools`, 256 tok, same
prompt, `RUST_LOG=infer_metal=info INFER_METAL_DFLASH_TRACE=1` for per-block accept:

| | mean accepted/block | mean matched/block | accepted hist |
|---|---|---|---|
| topk=1 (exact) | 1.700 | 0.700 | {1:45, 2:105} |
| topk=2 | **1.861** | **0.861** | {1:19, 2:118} |

**topk=2 raises accepted/block +9.5% (matched +23%)** — "reject at first draft"
dropped 45→19 blocks. This is the only robust number (a count). The model is
**DENSE** (`Qwen3.6-27B-OptiQ`: `model_type=qwen3_5`, `num_experts=None`, dense FFN +
hybrid linear/full attention — NOT MoE), so per-block forward cost is
content-independent → +9.5% acceptance ≈ **~+9.5% expected throughput**.

**Two caveats (SOLID):**
- **Bounded lever — verified `block_size=2` = ONE draft token/step.** (Load log:
  `block_size=2`; trace `matched ∈ {0,1}` always.) So `accepted ∈ {1,2}` by
  construction (draft + posterior), NOT because "a 2nd draft fails" — there is no 2nd
  draft. top-k's whole effect is raising that single draft's hit-rate: topk=1 70%
  (105/150 matched=1) → topk=2 86% (118/137), i.e. mean 1.70→1.86. Per-forward ceiling
  is 2 tokens. Bigger speedup needs a DEEPER draft (`block_size>2`, speculate 2-3
  tokens) — only then does top-k get to rescue later drafts. (An earlier version of
  this entry wrongly said "depth-2 / 2nd draft never survives" — corrected.)
- **NO trustworthy real tok/s was obtained.** Both attempts were noise: the
  diff-method (192/(wall₂₅₆−wall₆₄)) gave −11% (cold-process / thermal run-to-run
  jitter); the trace-derived "eff tok/s" (12.5→17.4) is an ARTIFACT — `INFER_METAL_
  DFLASH_TRACE=1` forces a per-block sync that breaks the async pipeline and inflates
  `total_ms` to ~135 ms/block (not real decode). On a DENSE model the verify_ms swing
  (123 vs 97 ms) cannot be content — it is run-to-run/thermal noise. Clean tok/s needs
  a warm, trace-off, same-binary repeated matched A/B
  ([[feedback_matched_ab_for_small_bench_effects]]); not done.

Correctness: outputs stayed coherent (valid technical answers) at topk=2; lossy, so a
needle gate is the proper check (not run; no degeneration observed).

**Process error logged:** earlier in this session the analysis (a) asserted the model
was MoE without reading its config and (b) reported trace-inflated per-block time as
"tok/s" — both violate case-as-fact (theorize before measuring). Corrected here.

## Rule

A relaxed/lossy decode accept (top-k, typical-acceptance) cannot be gated on
byte-identity vs baseline — it is designed to deviate. Gate on the correct-inference
suite (needle + self-consistency). Keep the default the exact path so the baseline
stays bit-identical and the knob is pure opt-in.
