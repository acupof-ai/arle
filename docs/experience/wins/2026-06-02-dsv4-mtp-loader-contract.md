# DSv4 MTP Loader Contract

## Context

The DSv4-Flash target is the SGLang-style single-node TP8 + EAGLE path. SGLang
executes the checkpoint's internal next-token prediction model as a frozen-KV
draft model; ARLE previously recognized `num_nextn_predict_layers` in config and
manifest only, but the CUDA model object had no loaded `mtp.N` weights.

Target workload remains the hot GPU-cache DSv4-Flash TP8 + EAGLE 256K/1500
shape: TTFT ~0.44s, TPOT ~4.85ms, E2E ~7.7s, output throughput ~196 tok/s.

## What Worked

- Added GPU-resident `DeepseekMtpLayer` storage for `mtp.N` weights.
- Made `sglang-best-practice` profile load full DSv4 layer weights by default,
  with `ARLE_DSV4_LOAD_LAYER_WEIGHTS` still available as an explicit override.
- Added `ARLE_DSV4_LOAD_MTP_WEIGHTS` override; by default, MTP weights load only
  under the high-performance profile.
- Matched SGLang `deepseek_v4_nextn.py` for `e_proj`/`h_proj`: both are
  replicated, not tensor-parallel column shards.
- Updated the startup contract so logs distinguish "MTP not loaded" from
  "MTP loaded, but frozen-KV EAGLE draft forward/graph is still missing".

## Verification

- `cargo fmt --check && git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
- `cargo test -p deepseek-spec v4::tests::parses_hf_flash_alias_config_fields`
- Remote pod `/data01/build/arle` at commit `dfca260e`:
  `scripts/dsv4_fast_build.sh` used prebuilt CUDA artifacts and completed in
  22.15s.
- Remote tensor-index check on `/data01/models/DeepSeek-V4-Flash` confirmed
  `mtp.0.enorm.weight`, `mtp.0.hnorm.weight`, `mtp.0.e_proj.weight`,
  `mtp.0.h_proj.weight`, `mtp.0.norm.weight`, and `mtp.0.hc_head_fn` exist in
  `model-00046-of-00046.safetensors`.
- Remote high-perf startup contract probe:
  `/tmp/dsv4_mtp_contract_20260602_151015.log`. All 8 ranks loaded
  `layers=43` and `mtp_layers=1`, then failed closed at the expected remaining
  missing executable path:
  `CUDA frozen-KV EAGLE draft forward/graph capture is not implemented yet`.

`cargo test -p deepseek-spec` was not a valid local gate on this Mac because
three existing tests require `infer/models/dsv4-mini-1B-init/config.json`, which
is not present locally.

## Rule

For DSv4 EAGLE work, "config knows MTP exists" is not enough. The high-perf
profile must load the actual `mtp.N` weights and the startup contract must say
which executable piece is still absent before any performance claim is made.
