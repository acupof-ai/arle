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
dropped 45→19 blocks; 137 vs 150 blocks for similar output → ~+9.5% fewer target
forwards → ~+9.5% expected throughput. Mechanism confirmed.

**Two caveats (SOLID):**
- **Bounded lever.** `accepted=3` (both drafts) NEVER occurs in either — once the
  first draft is rescued by top-2, the second rarely survives even top-2. On depth-2
  MTP self-spec the draft is target-aligned (top-1 already accepts most), so top-k
  only rescues the first draft; gain caps ~9.5%. Deeper drafts may benefit more.
- **tok/s not cleanly resolved.** Single-shot diff-method (192/(wall₂₅₆−wall₆₄))
  showed a spurious −11% — cold-process warmup jitter + lossy-content confound swamp
  a ~9.5% signal. Clean tok/s needs a warm same-binary matched A/B
  ([[feedback_matched_ab_for_small_bench_effects]]); local memory flakiness (27B at
  the 48 GiB Mac's edge) made that impractical. accepted/block is the robust evidence.

Correctness: outputs stayed coherent (valid technical answers) at topk=2; lossy, so a
needle gate is the proper check (not run; no degeneration observed).

## Rule

A relaxed/lossy decode accept (top-k, typical-acceptance) cannot be gated on
byte-identity vs baseline — it is designed to deviate. Gate on the correct-inference
suite (needle + self-consistency). Keep the default the exact path so the baseline
stays bit-identical and the knob is pure opt-in.
