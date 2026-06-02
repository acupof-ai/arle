# DSv4 Internal MTP Draft Mode Contract

## Context

The DSv4-Flash target is the warm single-node TP8 + EAGLE shape:
256K/1500, TTFT about 0.44 s, TPOT about 4.85 ms, E2E about 7.7 s,
and output throughput about 196 tok/s. ARLE previously only exposed
`self-spec` and `external:<path>` draft modes. That mixed two different
semantics:

- `self-spec` is MagicDec-style sparse self speculation.
- DSv4 EAGLE is an internal checkpoint `mtp.N` draft head over frozen target KV.

Using `self-spec` for DSv4 EAGLE would hide the real missing path and could
silently run the wrong runtime structure.

## What Worked

- Added `DraftMode::InternalMtp` with CLI aliases `internal-mtp`, `mtp`,
  `eagle`, and `internal-eagle`.
- Added a model trait hook for batched internal MTP draft proposals:
  `forward_internal_mtp_draft_batch`.
- Added startup validation for `--spec-draft-model internal-mtp/eagle` so
  unsupported models fail before serving opens instead of falling back silently.
- Added a CUDA scheduler branch that routes internal MTP draft proposals through
  the existing target verifier and commit logic.
- Added DSv4-specific fail-closed messages that distinguish "MTP weights not
  loaded" from "frozen-KV MTP draft is eager-only; graph capture is still
  missing".
- Added the first real DSv4 internal MTP draft forward:
  - target pre-head HC stream is captured after target prefill/decode,
  - MTP seed uses shared embedding plus `enorm/e_proj` and
    `hnorm/h_proj` over the captured HC lanes,
  - MTP decoder runs one frozen-SWA layer that reads target layer-0 SW KV
    without writing it,
  - greedy draft tokens are produced from MTP `hc_head + norm + lm_head` and
    routed into the existing target verifier.

This is still not a performance win yet. It is the first real eager
frozen-KV MTP draft path. The DSv4 best-practice profile remains fail-closed
until full decode CUDA graph capture/replay, graph-safe DeepEP/NCCL, and
SGLang-style top-k tree drafting are implemented.

## Verification

- `cargo test -p infer internal_mtp --no-default-features --features no-cuda`
  - passed, including `internal_mtp_allows_multi_token_without_sparse_kv`
    and `request_spec_internal_mtp_honors_aliases_and_opt_outs`.
- `cargo check -p infer --no-default-features --features no-cuda`
  - passed.
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
  - passed with pre-existing DSv4 warnings.
- `git diff --check`
  - passed.

Remote DSv4 fast-build and startup contract verification are pending for the
next tranche.

Follow-up verification at commit `471acc9c` on remote pod
`/data01/build/arle`:

- `scripts/dsv4_fast_build.sh` used the prebuilt CUDA archive and completed in
  26.17 s without nvcc / TileLang AOT.
- High-performance TP8 probe with `--spec-enabled --spec-draft-model eagle`
  loaded `mtp_layers=1` on all ranks, then failed closed before serving opened.
  Artifact: `/tmp/dsv4_internal_mtp_contract_20260602_153458.log`.
- The failure was still the expected executable-path gap:
  `CUDA frozen-KV EAGLE draft forward/graph capture is not implemented yet`.

The first remote probe also showed that the speculative decode config log was
behind the DSv4 best-practice contract failure. The constructor now logs
`Speculative decode config` before model-specific contract validation so
operators can see `draft_model=InternalMtp` even when the high-performance path
fails closed during startup.

Follow-up verification at commit `f266b9fd` on remote pod
`/data01/build/arle`:

- `scripts/dsv4_fast_build.sh` used the DSv4 prebuilt CUDA archive and
  completed in 18.20 s without nvcc / TileLang AOT.
  Artifact: `/tmp/dsv4_fast_build_f266b9fd_20260602_161632.log`.
- High-performance TP8 + EAGLE startup with
  `ARLE_DSV4_PERFORMANCE_PROFILE=sglang`, FP8 KV,
  `--cuda-graph-max-bs 16`, and `--spec-draft-model eagle` still fails closed
  before serving opens. Artifact:
  `/tmp/dsv4_eagle_contract_373d0b1e_20260602_161225.log`.
- The explicit missing best-practice pieces are still:
  full-decode CUDA graph capture/replay, token-owned DP/EP request routing,
  owner-group NCCL/token-sync subgroups, DeepEP/NCCL graph replay,
  graph-captured FlashMLA/SWA/C4/C128 metadata replay, and batched attention
  planning without host start-pos loops.
- Debug-fallback TP8 EAGLE with allreduce MoE and DeepGEMM experts now serves
  and returns real tokens on a 32-token cap request:
  `137 + 269 = 406.</think><|end_of_text|>`. Usage:
  `prompt_tokens=17`, `completion_tokens=16`, `total_tokens=33`; request_trace
  reported `error=null`, `ttft_ms=99.04`, and total latency 3.53 s.
  Artifact:
  `/tmp/dsv4_eagle_debug_f266b9fd_20260602_161711.log`.
- The correctness fix was to route MTP FFN through learned-bias MoE routing.
  The checkpoint has `mtp.0.ffn.gate.bias` and no `mtp.0.ffn.gate.tid2eid`;
  using target layer 0's hash routing made the draft path fail with
  `hash-routed DeepSeek V4 MoE layer missing tid2eid`.

This remains a correctness milestone, not a performance win. The measured
debug-fallback decode steps were hundreds of milliseconds and are not
comparable to the DSv4-Flash TP8 + EAGLE + 256K/1500 hot-cache target
(`~0.44 s TTFT`, `~4.85 ms TPOT`, `~7.7 s E2E`, `~196 tok/s`).

## Rule

Do not overload `self-spec` for DSv4 EAGLE. Internal checkpoint MTP draft,
external draft models, and sparse self speculation are separate runtime
structures and need separate startup contracts.
