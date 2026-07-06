# OPD rollout arm: `--rollout-engine {infer,train}` replaces the env toggle

`bench-exempt` — CLI-surface refactor, CUDA rollout paths unchanged. cuda build
`pending-remote` (no nvcc); non-cuda build + train test suite green.

## Context
The OPD rollout arm was env-only (`ARLE_OPD_INFER_ROLLOUT`). Per "只保留
`--rollout-engine`, 其他全部删除" — one flag, delete the rest.

## What Worked
- `--rollout-engine {infer,train}` on `TrainOpdArgs`; `apply_opd_rollout_engine`
  sets a `OnceLock<bool>` read by `infer_rollout_flag_enabled()` (default `infer`).
- Deleted: the `ARLE_OPD_INFER_ROLLOUT` branch, the never-landed `--engine-offload`
  flag + `OpdEngineOffloadArg` + `set_engine_offload_override`, the
  `LazyLock<OnceLock<Option<_>>>` (→ plain `OnceLock<bool>`). `engine_offload_mode()`
  reverts to its original env-only form.
- Non-cuda accepts the flag but it's inert (no infer engine on CPU).

## Rule
When a flag replaces an env toggle, delete the env path — one entry, one default,
no fallback layer.
