# Metal Spec Serve Fails Closed

## Goal

- Type: regression / contract.
- Make Metal `arle serve` carry speculative-decode CLI options into the unified
  serve layer and refuse the request until the rewrite executor consumes them.

## Hypothesis

- `--spec-type mtp` and `--mtp-*` should no longer be silently ignored on Metal.
- Non-Metal backends should keep rejecting speculative options at CLI config
  resolution.
- Standard decode with no speculative options should keep the same serve path.

## Command

```bash
cargo check -p cli --release --no-default-features --features metal,no-cuda
cargo test -p cli --release --no-default-features --features metal,no-cuda serve::tests -- --nocapture
cargo run --release --no-default-features --features metal,no-cuda -- \
  serve --backend metal --model-path /tmp/agent-infer-no-such-model --spec-type mtp
```

Earlier same-diff CPU surface checks:

```bash
cargo test -p infer-api --release --no-default-features --features cpu,no-cuda
cargo test -p cli --release --no-default-features --features cpu,no-cuda serve::tests -- --nocapture
```

## Environment

- Backend: Metal serve config path.
- Model: none loaded; the fail-closed check runs before backend/model load.
- Hardware: Apple Silicon local Mac.
- Feature set: `--release --no-default-features --features metal,no-cuda`.
- Non-default flags / env vars: none.
- Commit: current tranche; see git history for the commit containing this entry.

## Results

| Check | Result |
|---|---|
| Metal feature compile | PASS |
| Metal `serve::tests` | PASS, 16/16 |
| CPU `infer-api` tests | PASS, 11/11 |
| CPU `cli serve::tests` | PASS, 16/16 |
| Runtime fail-closed smoke | PASS, exits before model load |

Runtime smoke emitted:

```text
[ARLE serve] error: speculative decode is not wired into the rewrite serve path yet: requested spec_type=mtp, mtp_draft_model=none, mtp_draft_tokens=none; refusing to silently run standard decode
```

## Guidellm

- Not run for this tranche. The changed behavior is an invalid/unsupported
  config path that exits before model load; no inference workload executes.
- No TTFT / ITL / throughput claim is made here. A future Metal MTP executor
  implementation must run the normal guidellm gate before any default or
  performance claim.

## Problems

- This does not implement Metal MTP. It only prevents the CLI from accepting
  speculative flags and then running ordinary target-only decode.

## Learnings

- Experimental decode flags should fail closed at the unified serve boundary
  until a backend consumes them; accepting a flag without changing execution is
  a correctness bug in the control plane.

## Delta vs baseline

- Baseline: prior rewrite serve accepted Metal spec flags at CLI config
  resolution but did not carry them into `infer-api`.
- Delta: Metal spec options now flow through `ServeHttpOptions`; any requested
  speculative decode exits with an explicit error instead of silently using
  standard decode.
