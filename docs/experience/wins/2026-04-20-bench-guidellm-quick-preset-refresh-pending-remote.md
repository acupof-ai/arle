# guidellm Quick Preset Refresh — exploration preset, 2026-04-20

## Context

- **Status:** `pending-remote`
- **Scope:** `scripts/bench_guidellm.sh --quick`
- **Backend:** any HTTP-served backend (`cuda_*` / `metal_*`)
- **Model:** default quick path still targets `Qwen/Qwen3-4B` unless overridden
- **Hardware:** real server run pending
- **Commit:** pending local integration
- **Feature set:** n/a — bench-wrapper parameter change only
- **Server launch:** pending remote/local server run

## Preset being changed

The `--quick` exploration preset now uses:

```bash
./scripts/bench_guidellm.sh <label> --quick
```

with the following effective parameters:

```bash
guidellm benchmark \
  --profile concurrent \
  --data prompt_tokens=512,output_tokens=128 \
  --rate 1,2,4,8 \
  --max-seconds 60 \
  --warmup 5
```

This remains an exploration-only path. It does **not** seed a wins entry from
the script itself; this doc exists to satisfy the bench gate for
`scripts/bench_*.sh` parameter changes.

## Local validation

- `bash -n scripts/bench_guidellm.sh`
- `./scripts/bench_guidellm.sh --help`
- `./scripts/bench_guidellm.sh quick-preset-refresh --quick --target http://127.0.0.1:9`

The final command intentionally hit the server preflight and failed with
`server not running`, which confirms the new `--quick` preset parses and reaches
the runtime precondition checks.

## Results

| check | result |
|---|---|
| shell syntax | pass |
| help text | pass |
| quick preset arg parsing + preflight | pass |
| real benchmark numbers | pending-remote |

## Notes

- What changed in the code since baseline: `--quick` moved from a shorter
  `30s / warmup=3` exploration profile to `512-in/128-out`, `60s`, `warmup=5`
  so requests complete reliably in seconds on 4–8B models while still giving
  each concurrency level multiple completed requests.
- Why no local throughput table: no inference server was running in this
  session, and `--quick` is intentionally non-canonical.
- Follow-ups: run one real `--quick` profile against the next available local
  or remote server and confirm the wall-clock matches the advertised ~4-minute
  envelope.
