# Metal MTP Bench Parity Gate

## Context

The 2026-06-02 MTP depth sweep showed prompt-sensitive speedups and regressions.
The next optimization step needed a token-id gate before changing packed verify
or scheduler behavior. Text hashes and snippets were not enough because both
baseline and MTP can be deterministic while still following different token
trajectories.

`metal_serve` already accepted `--mtp-draft-model`, but `metal_bench` did not
have a direct MTP draft flag. That made local verification harder than needed
and kept the benchmark tool from exercising the same MTP request-state path.

## What Worked

- Added `metal_bench --mtp-draft-model` and `--mtp-draft-tokens`, mutually
  exclusive with DFlash.
- Kept MTP bench execution on `--use-step-driver`, matching the request-state
  path where MTP is actually wired.
- Added `metal_bench --mtp-parity`, which runs baseline target decode and MTP
  decode on the same prompt IDs, reports generated token IDs and the first
  divergence, and exits non-zero on mismatch.
- Extended the step-driver bench helper to capture generated token IDs from
  both terminal prefill emission and decode steps.
- Left block-internal draft-token trace out of the hot path. That trace still
  needs an explicit diagnostic switch because materializing every draft block
  would add synchronization overhead.

## Verification

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features metal --bin metal_bench`
- `cargo test -p infer --no-default-features --features metal --bin metal_bench -- --nocapture`
- `cargo build --release -p infer --no-default-features --features metal --bin metal_bench`
- `git diff --check`

Local smoke, MTP only:

```text
QWEN35_MTP_PROFILE=1 ./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --use-step-driver \
  --mtp-draft-model mlx-community/Qwen3.6-35B-A3B-MTP-4bit \
  --prompt-tokens 20 --generation-tokens 16 --warmup 0 --runs 1 --json
```

Result: `blocks=15`, `block_size=3`, `avg_accepted_inputs=1.0`,
`acceptance_rate=0.0`, `total_time_ms=8438.2`, `ttft_ms=5617.8`.

Local parity smoke:

```text
./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --mtp-draft-model mlx-community/Qwen3.6-35B-A3B-MTP-4bit \
  --mtp-parity \
  --prompt-tokens 20 --generation-tokens 8 --json
```

Result: `matched=true`; both paths emitted
`[27502, 61610, 27502, 61610, 27502, 61610, 27502, 61610]`.
MTP still had `acceptance_rate=0.0`, `total_time_ms=12392.3`, while baseline
was `total_time_ms=6772.6`.

These are smoke checks only (`warmup=0`, `runs=1`) and do not license a
performance claim.

## Rule

MTP optimization must pass token-id parity and report acceptance before packed
verify or scheduler changes can be interpreted as speedups.
