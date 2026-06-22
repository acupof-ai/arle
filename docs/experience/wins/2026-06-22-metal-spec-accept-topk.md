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

## Pending (pending-remote)

**e2e DFlash gate NOT run** — the draft model `z-lab/Qwen3.6-35B-A3B-DFlash` is not
cached locally (no DFlash draft → `self.dflash=None` → verify path never executes).
Needs a DFlash-enabled Metal serve: `--spec-accept-topk 2` vs `1`, measure
acceptance-rate↑ / tok/s, and a **correct-inference needle gate** (NOT byte-identity
vs baseline — top-k accept deliberately deviates). Cross-link this entry on landing.

## Rule

A relaxed/lossy decode accept (top-k, typical-acceptance) cannot be gated on
byte-identity vs baseline — it is designed to deviate. Gate on the correct-inference
suite (needle + self-consistency). Keep the default the exact path so the baseline
stays bit-identical and the knob is pure opt-in.
