# DSv4 SGLang Path Research Before Profiling

## Scope

This is a source/config research note only. No SGLang or ARLE performance test was
run for this note, and the existing remote service process was not used as a
baseline.

Goal: understand what a valid SGLang DSv4 comparison must mean before running a
PyTorch profiler trace or making more ARLE changes.

## Correction: Runtime SGLang Tree

The first pass inspected the wrong SGLang tree.

Two SGLang source trees exist on the pod:

| Path | Commit | Runtime relevance |
|---|---|---|
| `/workspace/sglang` | `0d51db3` (`feat: add SGLANG_APPLY_CONFIG_BACKUP=auto and default to it (#38)`) | actual Python import path |
| `/sgl-workspace/sglang` | `232982a` (`Fix NPU docker release workflow (#16253)`) | stale for DSv4 runtime analysis |

Live Python import metadata resolves DSv4 modules from `/workspace/sglang`:

```text
sglang.srt.models.deepseek_v4 -> /workspace/sglang/python/sglang/srt/models/deepseek_v4.py
sglang.srt.layers.attention.deepseek_v4_backend -> /workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py
sglang.srt.arg_groups.deepseek_v4_hook -> /workspace/sglang/python/sglang/srt/arg_groups/deepseek_v4_hook.py
sglang.srt.layers.moe.mega_moe -> /workspace/sglang/python/sglang/srt/layers/moe/mega_moe.py
```

Therefore the earlier statement that the readable SGLang source had no
`DeepseekV4ForCausalLM` support is false for the runtime tree. It only describes
the stale `/sgl-workspace/sglang` checkout. All path-comparison work must use
`/workspace/sglang` as the source reference unless the serving environment is
changed and re-verified.

## Current Evidence

- Runtime SGLang source inspected: `/workspace/sglang`.
- Stale non-runtime SGLang source inspected: `/sgl-workspace/sglang`.
- Remote DSv4 model inspected: `/data01/models/DeepSeek-V4-Flash`.
- The model config says `architectures=["DeepseekV4ForCausalLM"]` and
  `model_type="deepseek_v4"`.
- The runtime tree registers `DeepseekV4ForCausalLM` via
  `python/sglang/srt/models/deepseek_v4.py` and `EntryClass =
  [DeepseekV4ForCausalLM]`.
- The runtime tree registers the CUDA `dsv4` attention backend in
  `python/sglang/srt/layers/attention/attention_registry.py`.
- The runtime tree applies DSv4 defaults through
  `python/sglang/srt/arg_groups/deepseek_v4_hook.py`.
- The DSv4 model directory includes a standalone reference implementation under
  `inference/`, with V4-specific attention compression/indexing and a simple
  MoE path.

This means the DSv4 SGLang implementation has been identified at source level.
The remaining gap is not "find the implementation"; it is "lock the exact
launch contract and run a fresh warm control profile from that implementation."

## DSv4 Model Shape

Remote `config.json` headline fields:

| Field | Value |
|---|---:|
| layers | 43 |
| hidden size | 4096 |
| heads | 64 |
| head dim | 512 |
| window size | 128 |
| routed experts | 256 |
| active experts/token | 6 |
| shared experts | 1 |
| MoE intermediate | 2048 |
| scoring | `sqrtsoftplus` |
| route scale | 1.5 |
| topk method | `noaux_tc` |
| hash layers | 3 |
| quant | FP8 e4m3, UE8M0 block scales `[128,128]` |
| MTP layers | 1 |

The standalone `inference/model.py` path implements:

- sliding-window + compressed sparse attention;
- indexer top-k over compressed KV (`index_topk=512`);
- hyper-connection pre/post mixing;
- first `n_hash_layers=3` MoE layers with token-id routing;
- normal routed layers using `sqrtsoftplus` scores plus bias;
- per-rank local experts plus all-reduce in the simple reference MoE.

Important correction: `o_groups=8` is attention output grouping in the reference
implementation, not proof of MoE expert-group routing. The generic SGLang
DeepSeek V3 path uses `n_group/topk_group` grouped top-k, but the V4 config read
here does not contain those fields.

## SGLang DSv4 Runtime Path

The actual `/workspace/sglang` DSv4 path has model-specific defaults:

- `attention_backend = "dsv4"`;
- `page_size = 256`;
- `kv_cache_dtype = "fp8_e4m3"` when the user leaves it on `auto`;
- `max_running_requests = 256` when not specified;
- speculative decoding, if enabled, must be EAGLE with
  `speculative_eagle_topk == 1`;
- `swa_full_tokens_ratio = 0.1` by default for DSv4.

The H200 FP8 cookbook path uses `sgl-project/DeepSeek-V4-Flash-FP8`, not the raw
`/data01/models/DeepSeek-V4-Flash` directory. It sets
`SGLANG_DSV4_FP4_EXPERTS=0`.

Manual test recipes in `/workspace/sglang/test/manual/dsv4/` show the intended
lanes:

| Lane | Key launch contract |
|---|---|
| Low latency | `--tp 4`, EAGLE, `--speculative-num-steps 3`, `--speculative-num-draft-tokens 4` |
| Balanced | `--tp 4 --dp 4 --enable-dp-attention --moe-a2a-backend deepep`, EAGLE step 1, `--cuda-graph-max-bs 128`, `--max-running-requests 128` |
| Max throughput | `--tp 4 --dp 4 --enable-dp-attention --moe-a2a-backend deepep`, `--cuda-graph-max-bs 128`, `--max-running-requests 256` |
| CP | `--tp 4 --moe-a2a-backend deepep --enable-nsa-prefill-context-parallel --nsa-prefill-cp-mode round-robin-split --chunked-prefill-size 16384` |
| TP8 sanity | `--tp 8 --max-running-requests 8`, no explicit DeepEP or speculation |

DeepEP recipes use:

```text
--deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
```

or `1024` for the CP recipe.

Attention is DSv4-specific, not a generic FlashMLA call:

- `DeepseekV4AttnBackend` asserts `page_size == 256` and `head_dim == 512`;
- metadata is built around SWA, C4, and C128 page indices;
- `MQALayer` fuses Q norm/RoPE through DSv4 JIT helpers;
- K/V norm/RoPE writes directly into the FlashMLA paged cache;
- compressor and C4 indexer work are first-class model-layer operations;
- the backend calls `flash_mla.flash_mla_with_kvcache(...,
  is_fp8_kvcache=True, indices=..., topk_length=..., extra_k_cache=...)`.

MoE is also DSv4-specific:

- `DeepseekV4DecoderLayer` constructs `DeepseekV2MoE(...,
  is_deepseek_v4=True)`;
- hash top-k can use the DSv4 JIT `hash_topk` path;
- `mega_moe.py` contains an SM90 FP8 MegaMoE path using
  `deep_gemm.fp8_mega_moe`;
- `moe_runner/deep_gemm.py` uses grouped masked DeepGEMM and the DSv4 JIT
  `silu_and_mul_masked_post_quant` activation+quant path.

## Generic SGLang MoE Path

The generic SGLang MoE stack still matters because DSv4's DeepEP and DeepGEMM
paths are selected through the same server arguments and runner framework:

1. `--moe-a2a-backend`
   - default: `none`;
   - choices include `deepep`;
   - when set to `deepep`, SGLang forces `ep_size = tp_size`;
   - DeepEP dispatch group is `get_tp_group().device_group`.

2. `--moe-runner-backend`
   - default: `auto`;
   - for FP8 MoE, `auto` chooses `deep_gemm` only when JIT DeepGEMM is enabled
     and A2A is `deepep` or `mooncake`;
   - otherwise FP8 `auto` falls back to Triton in the generic path.

3. `--deepep-mode`
   - default: `auto`;
   - resolves to `normal` for extend/prefill batches;
   - resolves to `low_latency` for decode batches.

So a valid SGLang DeepGEMM+DeepEP comparison cannot be inferred from a launch
command that omits `--moe-a2a-backend deepep` unless the selected DSv4 lane is
the TP8 sanity lane or another documented no-DeepEP recipe. For the Balanced and
MaxThroughput lanes, DeepEP is explicit.

## DeepGEMM Warmup

SGLang enables JIT DeepGEMM by default on SM90 when the `deep_gemm` Python module
is importable.

Key mechanics:

- `SGLANG_ENABLE_JIT_DEEPGEMM=True` by default.
- `SGLANG_JIT_DEEPGEMM_PRECOMPILE=True` by default.
- cache directory comes from `SGLANG_DG_CACHE_DIR`, defaulting to
  `~/.cache/deep_gemm`.
- `python3 -m sglang.compile_deep_gemm ...` starts a server and sends a warmup
  request to trigger compilation.
- without precompile/cache, the first matching DeepGEMM execution may enter a
  "DeepGEMM warmup" loop over many M sizes. Source warning says this can take
  10-20 minutes.
- the DSv4 manual tests force `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1` because the
  exhaustive warmup grid is too broad for routine DSv4 cookbook runs.

Profiling must exclude this warmup. A profile that includes DeepGEMM compile or
first-use warmup is not a steady-state SGLang baseline.

## DeepEP `num_sms`

SGLang `DeepEPConfig` loads `--deepep-config` if provided. The config has
`normal_dispatch` and `normal_combine`, and both must agree on `num_sms`.

If no config is supplied, SGLang uses DeepEP's default `Buffer.num_sms`.

`num_sms` affects normal-mode communication resources and also participates in
DeepEP buffer initialization:

- normal mode: `num_qps_per_rank = DeepEPConfig.num_sms`;
- low-latency mode: `num_qps_per_rank = num_experts / group.size()`;
- auto mode: `num_qps_per_rank = max(num_sms, num_experts / group.size())`.

SGLang warns when normal-mode DeepEP uses fewer than half the GPU SMs and TBO is
off. Therefore `num_sms=80` is not a random tuning knob: it is part of the
communication resource and queue-pair sizing contract.

## SGLang Profiler Path

SGLang already has a PyTorch profiler API:

```bash
python3 -m sglang.profiler \
  --url http://127.0.0.1:30000 \
  --num-steps N \
  --profile-by-stage \
  --output-dir <dir> \
  --merge-profiles
```

The `/start_profile` request with `profile_by_stage=true` records prefill
(`EXTEND`) and decode (`DECODE`) separately:

- first prefill batch starts the EXTEND profile;
- first decode batch stops EXTEND and starts DECODE;
- after the target decode count, SGLang stops and flushes DECODE.

Caveat: CUDA graph visibility is separate. `--enable-profile-cuda-graph` profiles
graph capture paths and adds top operator tables there. It does not automatically
make steady-state CUDA graph replay equivalent to a fully expanded per-kernel
decode trace. If we disable CUDA graph for visibility, that is a different
execution path and must be labeled as such.

## Current ARLE Comparison Risks

The current ARLE DSv4 path and the intended SGLang fast path are not yet proven
to be isomorphic.

Known ARLE state from local source/docs:

- `ARLE_DSV4_MOE_BACKEND` default is `allreduce` for the current replicated-token
  path.
- `native-deepep` is guarded because DeepEP dispatch/combine assumes token-sharded
  EP ownership; passing replicated full-token rows creates a fanout/transport
  mismatch.
- `infer/src/tensor_parallel.rs` contains SGLang-style axis math, but comments
  state it is a diagnostics/contract input, not proof that DP/CP/MoE-DP execution
  is fully wired.
- `ARLE_DSV4_SGLANG_PATH=1` is fail-closed unless the DSv4 SGLang path contract
  is complete.

Script risk:

- `scripts/dsv4_beat_sglang_bench.sh` launches SGLang with only
  `--tp 8 --kv-cache-dtype fp8_e4m3`; it does not force DeepEP, does not
  precompile DeepGEMM, and previously assumed `/sgl-workspace/sglang` was the
  relevant source tree.
- That script is therefore not a valid SGLang-best-practice profiler harness for
  the Balanced or MaxThroughput lanes yet. It may approximate the TP8 sanity lane
  only after the runtime tree, model package, warmup state, and correctness are
  logged.

## What Must Be Locked Before Running Perf

1. Keep the SGLang DSv4 implementation identity attached to every run:
   - source tree `/workspace/sglang`;
   - commit `0d51db3` unless re-verified;
   - model package `sgl-project/DeepSeek-V4-Flash-FP8` for the H200 FP8
     cookbook lanes;
   - attention backend `dsv4`, page size 256, FP8 KV cache.

2. Lock the SGLang launch contract:
   - TP/DP/EP sizes;
   - whether `--enable-dp-attention` is required;
   - `--moe-a2a-backend deepep` or DSv4-specific override;
   - `--moe-runner-backend auto` versus explicit `deep_gemm`;
   - `--deepep-mode auto`;
   - `--deepep-config` and its `num_sms`;
   - CUDA graph settings;
   - speculative decoding on/off.

3. Exclude warmup from profiler windows:
   - run `sglang.compile_deep_gemm` or an equivalent warmup until DeepGEMM cache
     is populated;
   - verify there is no "DeepGEMM warmup" in the measured window.

4. Separate metrics:
   - no-spec raw target-step TPOT;
   - speculative output-token TPOT with accept rate;
   - TTFT and TPOT should not be mixed across these definitions.

5. Match correctness before performance:
   - same prompt, tokenizer, `ignore_eos`, max tokens, and output sanity;
   - decode actual generated text when output is suspicious.

## Research Conclusion

The actual SGLang DSv4 source path is now identified:

```text
/workspace/sglang @ 0d51db3
```

The earlier negative finding came from inspecting `/sgl-workspace/sglang`, which
is not the runtime import tree for DSv4.

The next correct step is still not to run an unqualified trace. The correct next
step is to update the SGLang harness to select a documented lane, record the
runtime tree and launch contract, warm DeepGEMM with fast warmup/cache, verify
normal output, then run a warm stage-split PyTorch profile.

For optimization planning, the source-level SGLang best-practice deltas are now
clear:

1. match DSv4 defaults first: `dsv4` backend, page size 256, FP8 KV, model
   package, and request/concurrency lane;
2. for Balanced/MaxThroughput, match TP4/DP4 with DP attention and DeepEP
   rather than ARLE's replicated-token TP/EP fallback;
3. move ARLE's MoE path toward token-owned DeepEP/MegaMoE-style routing plus
   grouped DeepGEMM and fused `silu_mul_quant`, not BF16 materialization plus
   post-FFN all-reduce;
4. move ARLE attention toward the SGLang DSv4 metadata/cache/indexer contract
   instead of isolated per-row CSA selector tuning;
5. keep raw target TPOT and speculative/effective TPOT in separate metric lanes.
