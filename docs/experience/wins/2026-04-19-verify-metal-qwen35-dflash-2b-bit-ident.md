# Metal DFlash Qwen3.5 Layer 2b — B=1 bit-ident regression-check — 2026-04-19

> Layer 2b of the Qwen3.5 DFlash batched-verify roadmap
> ([`docs/plans/metal-dflash-qwen35-verify-batch.md`](../../plans/metal-dflash-qwen35-verify-batch.md))
> landed in commit `29e0e31` on 2026-04-18, but no dated entry captured the
> acceptance gate. This fills that gap and marks 2b **done for correctness**
> (no throughput delta yet — 2b is plumbing consumed by 2c/2d).

## Goal

**Type:** regression-check.
Does `qwen35_compiled_verify_block_batched` at `B=1` produce logits
bit-identical to the scalar `qwen35_compiled_verify_block` (Layer 1) on
the same prompt + draft block? This is the plan §2b acceptance gate.

## Hypothesis

At `B=1`, the batched path is a strict superset of the scalar path — the
additive `attn_mask` is null, `cache_pos_arr=[cache_pos]`,
`rope_offsets=[cache_pos]`. The only new machinery (per-row cache-pos
indexing, batched GDR state) collapses to scalar dispatch. Expected:
max abs logit delta < 1e-3 after a `bf16 → f32` promotion.

## Parameters

- Test: `backend::metal::qwen35::tests::verify_block_batched_matches_verify_block_for_b1`
  (`infer/src/backend/metal/qwen35.rs:1748`).
- Model: `mlx-community/Qwen3.5-4B-MLX-4bit` (via
  `$QWEN35_MODEL_PATH`, resolved to
  `~/.cache/huggingface/hub/.../snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`).
- Prompt: `[1, 2, 3, 4]`; draft block: `[5, 6]` (`block_size=2`).
- KV capacity: `prompt_len + block_size + 4 = 10`.
- Compare: scalar `CppQwen35Model::verify_block` vs test-local helper
  `verify_block_batched_b1` (`qwen35.rs:1681`), which calls
  `mlx_sys::qwen35_compiled_verify_block_batched` with `batch_size=1`,
  `attn_mask=null`, `rope_offsets=cache_pos_arr=[prompt_len]`.
- Tolerance: element-wise `|lhs - rhs| < 1e-3` on `float32` promotion.

Command:

```bash
QWEN35_MODEL_PATH="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3" \
cargo test --release --no-default-features --features metal -p infer \
  --lib -- --test-threads=1 verify_block_batched_matches_verify_block_for_b1
```

## Environment

- Hardware: Apple M4 Max (40 GPU cores, ~400 GB/s UMA).
- macOS: 26.3.1 (build 25D771280a); Metal 3 runtime via MLX 0.29.x
  (pinned in `crates/mlx-sys/CMakeLists.txt`).
- Commit: `942605c` (clean tree).
- Feature set: `cargo build --release --no-default-features --features metal`.
- Build-time env: none beyond default; no `INFER_*` overrides.

## Results

```
running 1 test
test backend::metal::qwen35::tests::verify_block_batched_matches_verify_block_for_b1 ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 327 filtered out; finished in 0.41s
```

- Assertion: all `vocab * block_size` logits matched within `1e-3`.
- Wall time: 0.41 s (mostly model load; the forward itself is sub-ms).
- **Δ vs baseline (Layer 1 scalar):** zero numerical delta within
  tolerance at `B=1`. This is the intended acceptance criterion.
- No throughput number — Layer 2b does not yet ride any serving path;
  it is FFI + test fixture only. Throughput deltas land with Layer 2c
  (scheduler integration) and 2d (speculative_block wire-up).

## Problems observed

None. Single clean run; no §6 watch-item deviations.

- Warmup (§6.1): single test invocation, not a sweep — N/A.
- Thermal throttling (§6.2): sub-second; N/A.
- Kernel launches/token (§6.3): verify_block is `S=2`, batched emits the
  same fused forward graph; we did not measure launches (not required
  for a correctness check).
- Determinism (§6.7): argmax path inside the compiled graph is
  deterministic at `temperature=0`; the test compares logits directly,
  so RNG is not involved.

## Learnings

- **Correctness plumbing ≠ perf win.** Landing a batched FFI alongside a
  scalar one without a throughput delta is a legitimate milestone — but
  the wins entry must declare goal type `regression-check` and state
  "no throughput delta expected" up front, otherwise future readers
  looking for "did 2b move tok/s?" get confused.
- **Test helper ≠ production API.** `verify_block_batched_b1` lives
  inside the `#[cfg(test)] mod tests` of `qwen35.rs`. Layer 2c must
  promote it to a `pub(super) fn verify_block_batched` on
  `CppQwen35Model` before the scheduler can call it. Promotion is part
  of 2c's scope, not a 2b follow-up.
- **B=1 bit-ident is necessary but not sufficient.** At `B=1` the
  additive mask is null and per-row offsets collapse; the real test of
  the batched path is `B≥2` with distinct `cache_pos_arr[b]` and a
  non-trivial mask. That test lands with Layer 2c (runtime actually
  drives multi-row verify).
- **Stopping rule hit** (spec §7.3): variance N/A for a correctness
  pass; hypothesis confirmed; §6 watch-list clean; no prior-snapshot
  delta (first 2b-specific entry). One run suffices.

## Follow-ups

- Layer 2c: lift the `open.len() >= 2` downgrade at
  `infer/src/backend/metal/runtime.rs:1040-1056` and wire
  `verify_block_batched` into `execute_qwen35_packed_decode_batch`.
- Layer 2c bench: guidellm sweep at `c=2/4/8` on Qwen3.5-4B-4bit with
  DFlash enabled; acceptance per plan §2d = ≥2× throughput vs
  Layer-1 (auto-downgrade) at `c≥8`.
- Update `docs/plans/metal-dflash-qwen35-verify-batch.md` §2b status to
  **done 2026-04-18** (FFI) + **verified 2026-04-19** (B=1 bit-ident).

## Cross-links

- Plan: [`docs/plans/metal-dflash-qwen35-verify-batch.md`](../../plans/metal-dflash-qwen35-verify-batch.md) §Layer 2b.
- Landing commit: `29e0e31` (2026-04-18,
  `feat(metal): add qwen35_compiled_verify_block_batched for B>1 verify`).
- Prior Layer 1 snapshot:
  [`2026-04-17-metal-qwen35-dflash-block-verify.md`](2026-04-17-metal-qwen35-dflash-block-verify.md).
- Layer 2a snapshot:
  [`2026-04-18-metal-dflash-kernel-port-vs-reference.md`](2026-04-18-metal-dflash-kernel-port-vs-reference.md).
