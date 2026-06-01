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
- Runtime SGLang commit inspected read-only on 2026-06-01:
  `0d51db344d3f` (`feat: add SGLANG_APPLY_CONFIG_BACKUP=auto and default to
  it (#38)`). The checkout is dirty, so any future numeric baseline must record
  the exact diff state, not only the commit.
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
- Current already-running SGLang process on the pod is explicitly not used as a
  best-practice baseline for this note. Its command uses TP8, disables CUDA
  graph, forces `--moe-runner-backend marlin`, and enables HICache/EIC:

```text
python3 -m sglang.launch_server --model-path /data01/models/DeepSeek-V4-Flash \
  --host 0.0.0.0 --port 30000 --tp-size 8 --trust-remote-code \
  --mem-fraction-static 0.8 --disable-cuda-graph --watchdog-timeout 1800 \
  --moe-runner-backend marlin --enable-hierarchical-cache --enable-eic-cache \
  --hicache-io-backend kernel
```

This means the DSv4 SGLang implementation has been identified at source level.
The remaining gap is not "find the implementation"; it is "lock the exact
launch contract and run a fresh warm control profile from that implementation."

This update intentionally does not run that profile. It is pure source/artifact
research. All SGLang-vs-ARLE per-stage claims below are therefore structural
unless explicitly backed by an existing ARLE artifact.

## Existing ARLE Artifacts Used

No new benchmark, trace, or profiler run was executed for this section. The
following existing artifacts are the only numeric evidence used:

| Artifact | What it can support | What it cannot support |
|---|---|---|
| `docs/trace-artifacts/2026-05-27-allreduce-nsys/summary.json` | ARLE replicated-token all-reduce path has decode wave p50 about 94.8 ms under the captured window; top kernels include NCCL all-reduce, FP8 GEMV, hybrid attention, route, CSA select, and frequent CUDA API alloc/free/launch overhead | It is not a matched SGLang baseline and predates later path fixes |
| `docs/trace-artifacts/2026-05-27-allreduce-nsys/decode-only-kernel-top.csv` | Per-kernel ordering for the old ARLE all-reduce path | It cannot be added to newer request-trace numbers as if it were same-binary evidence |
| `docs/experience/wins/2026-06-01-dsv4-operator-request-trace.md` | Existing ARLE request-level operator trace: `ffn_total` and `attn_total` both accumulate about 13 s of trace-on phase time in a 32-token smoke; top phases include `ffn_routed_local`, `attn_swa_all_reduce`, `ffn_all_reduce`, `attn_hybrid_kernel`, and `ffn_expert_loop` | Trace synchronizes CUDA around phases and is diagnostic only |
| `docs/experience/wins/2026-06-01-dsv4-csa-select-topk-cover-fastpath.md` | Same-binary warm p2047/o8 and p2047/o32 behavior after CSA top-k-cover fast path: 91.1 ms and 117.1 ms TPOT after first token | Not a SGLang-comparable or SLO-shape result |
| `docs/experience/errors/2026-06-01-dsv4-native-deepep-replicated-token-kill.md` | Native DeepEP on current ARLE replicated-token route over-transports token rows; observed fanout 4.46 matches the top-6-over-8 rank fanout model | Does not quantify the future token-owned DeepEP path |

Remote artifact paths named by older docs under `/sgl-workspace/bench-artifacts`
were not re-used here as live files. On the current pod, that directory was not
present during the read-only directory check. Treat those paths as historical
doc references unless the files are copied into the repo or re-verified later.

## Metric Lanes Are Not Interchangeable

There are two different SGLang metrics that must not be collapsed:

| Lane | Meaning | Status |
|---|---|---|
| no-spec raw target-step TPOT | wall time per target model decode forward | Pending validation |
| spec-on output-token TPOT | wall time per accepted output token with EAGLE/MTP | Pending validation |

The user supplied two SGLang reference fragments during the investigation:
approximately `18 ms/token`, and a speculative setup with acceptance about
`2.94` and about `258 output tok/s`. These fragments are not enough to know
whether the 18 ms number is raw target-step TPOT or effective output-token TPOT.

If the number is output-token TPOT under acceptance `A`, then:

```text
raw target-step TPOT ~= output-token TPOT * A
```

If the number is raw target-step TPOT, then:

```text
output-token TPOT ~= raw target-step TPOT / A
```

So the same wall-clock run can look very different depending on accounting.
Until a matched SGLang baseline records both raw target steps and accepted output
tokens, every SGLang per-stage number in this report is marked as
`待验证`.

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

Important SGLang defaults from `/workspace/sglang@0d51db3`:

| Component | Source | Default / behavior |
|---|---|---|
| DSv4 hook | `python/sglang/srt/arg_groups/deepseek_v4_hook.py` | forces `attention_backend="dsv4"`, `page_size=256`, FP8 KV on `auto`, EAGLE-only spec if spec is enabled, `swa_full_tokens_ratio=0.1` |
| DSv4 env defaults | `python/sglang/srt/environ.py` | fused hash top-k, JIT EP activation, fused WQA/WKV, fused SwiGLU clamp, fused store cache, multi-stream overlap, and `SGLANG_PREP_IN_CUDA_GRAPH=True` are all default-on |
| H200 FP8 cookbook | `test/manual/dsv4/test_h200_fp8_flash.py` | low-latency TP4+EAGLE; balanced/max-throughput TP4/DP4 + DP attention + DeepEP + CUDA graph max BS 128; CP lane uses DeepEP plus context-parallel prefill |
| Parallel groups | `python/sglang/srt/distributed/parallel_state.py` | builds separate TP, attention-TP/DP/CP, MoE-EP/DP/TP groups inside the TP axis |
| DeepEP config | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` | `--deepep-config` feeds normal dispatch/combine `num_sms`; auto mode reserves resources for both normal and low-latency paths |
| DeepGEMM warmup | `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` | JIT DeepGEMM precompile is default-on; without cached precompile it can spend 10-20 minutes warming kernels, which must be outside measured windows |

## Stage Mapping: SGLang vs Current ARLE

This is the core structural answer. "Same" means the contracts are meaningfully
isomorphic. "Different" means both systems have an implementation for the stage
but the data layout or runtime contract differs. "Missing" means ARLE has no
equivalent SGLang fast-path mechanism in the current executable route.

| Stage | SGLang `/workspace/sglang@0d51db3` | Current ARLE route | Classification | Structural consequence |
|---|---|---|---|---|
| Process model | one process per rank under SGLang distributed runtime | ARLE has multiproc scaffolding, but request relay still broadcasts the same logical request to all ranks | Different | Native DeepEP can boot, but token ownership is still replicated |
| Rank layout | TP axis is subdivided into attention DP/CP/TP and MoE EP/DP/TP groups | `MultiAxisConfig` parses SGLang-style axes, but `validate_current_axis_support` rejects anything beyond global TP/EP | Missing in hot path | ARLE cannot express SGLang TP4/DP4+DP-attention+EP semantics yet |
| Request ownership | DP/attention and MoE groups own distinct row shards where configured | `DistributedSchedulerGroup` submits cloned full requests to every rank | Different | Every rank routes the same rows, so EP transport sees duplicate sources |
| DSv4 defaults | hook fail-forces `dsv4`, page 256, FP8 KV | ARLE defaults are its own replicated-token DSv4 path; SGLang-path claim fails closed | Different | A short smoke can be correct while still not SGLang-comparable |
| Attention prep | fused WQA/WKV, fused Q norm/RoPE, fused KV norm/RoPE direct to FlashMLA paged cache | ARLE has fused prep pieces but still runs current route through explicit per-row cache/indexer work | Different | Metadata and cache traffic stay visible as independent phases |
| Attention metadata | raw decode/verify metadata can be upgraded inside CUDA graph; C4/C128/SWA page indices feed FlashMLA | ARLE has `attn_csa_project`, `attn_csa_select_kernel`, and separate FlashMLA/CSA prep work | Different | Selector work remains per-layer/per-row enough to dominate some shapes |
| Attention core | one `flash_mla.flash_mla_with_kvcache` call consumes SWA plus optional C4/C128 extra KV with FP8 cache and top-k lengths | `forward_decode_batch` batches FFN but explicitly loops attention per row; `attn_hybrid_kernel` is separate | Different | Attention cost scales like many small launches and per-row work, not a batched SGLang FlashMLA replay |
| MoE top-k | SGLang can use fused/hash/JIT top-k and DeepEP-aware remap | ARLE route/count uses local kernels plus host-visible metadata in several paths | Different | Routing overhead is not just top-k math; it is orchestration and movement |
| EP dispatch | DeepEP dispatch receives token-owned rows for the configured EP group | native DeepEP on ARLE receives replicated full token rows unless unsafe trace flag is set | Different and currently blocked | Observed 4.46x fanout is expected, not a mysterious DeepEP slowness |
| Expert GEMM | grouped masked DeepGEMM or MegaMoE-class fused path, plus JIT activation+quant | ARLE has DeepGEMM-auto for local experts, but still under replicated-token transport and separate scratch/materialization | Different | Even a fast expert GEMM is surrounded by the wrong transport/data contract |
| SwiGLU + quant | `silu_and_mul_masked_post_quant` fuses activation, clamp, and FP8 quant for DeepGEMM | ARLE has `dsv4_deepgemm_swiglu_quantize_w13_cuda` and scratch-zero work; newest skip-zero change is pending remote perf validation | Different | User-supplied SGLang/vLLM trace points here as a large gap, but ARLE exact delta is pending |
| Combine | DeepEP combine returns token-owned rows; MegaMoE avoids the same post-FFN all-reduce shape | default ARLE local experts call `post_moe_expert_all_reduce_hidden_states` | Different | `ffn_all_reduce` is a fallback reconciliation tax, not a target SGLang stage |
| CUDA graph | DSv4 metadata path explicitly supports `SGLANG_PREP_IN_CUDA_GRAPH=True` and graph replay buckets | ARLE has graph support generally, but current DSv4 route still exposes many small kernels, alloc/free, D2H/H2D events in artifacts | Different | Launch and allocation overhead remains first-order in the current ARLE trace |
| Spec decode | H200 low-latency and balanced lanes use EAGLE/MTP in documented configs | ARLE DSv4 path has no equivalent accepted-token accounting in the current comparison | Missing | Spec-on output-token TPOT can explain a large apparent gap without any kernel difference |

## Gap Waterfall, Without New Runs

A numeric waterfall from SGLang TPOT to ARLE TPOT is not SOLID yet because the
matched SGLang no-spec/spec traces do not exist in the repo. The waterfall below
is therefore a structural waterfall. Each row says whether the gap is already
licensed by existing evidence or remains `待验证`.

| Waterfall item | Evidence status | Why it matters | Required later trace, not run now |
|---|---|---|---|
| Speculative decoding accounting | `待验证`; only user-supplied accept/tok/s fragments exist | If SGLang TPOT is output-token TPOT under EAGLE/MTP and ARLE is raw target-step, the apparent gap includes an acceptance-rate multiplier | Same SGLang workload with spec off and spec on; record target steps, accepted tokens, acceptance rate, output-token TPOT |
| Topology and token ownership | Structurally confirmed by ARLE code and DeepEP fanout artifact | ARLE current route clones the full request to every rank, then sums full hidden states; SGLang intended lanes use TP/DP/EP subgroup ownership | Per-rank request/token counters and DeepEP `num_recv` on a token-owned ARLE candidate |
| MoE orchestration | Confirmed as current-route cost by ARLE request trace; SGLang exact delta `待验证` | ARLE visible `ffn_routed_local` + `ffn_all_reduce` is not the SGLang DeepEP/MegaMoE contract | SGLang PyTorch profile of the same no-spec target-step workload, plus ARLE request trace on same workload |
| Expert compute and activation quant | SGLang source confirms DeepGEMM/JIT fused activation path; ARLE delta `待验证` | User-supplied vLLM/SGLang trace says fused SwiGLU+quant and expert GEMM dominate; ARLE still has scratch/materialization around local/DeepGEMM path | Operator trace with `dsv4_deepgemm_*`, scratch memset, and activation quant phases split out |
| Attention metadata and selector | Confirmed current-route cost by ARLE trace and CSA fast-path artifact; SGLang exact delta `待验证` | SGLang feeds sparse/recent page indices into FlashMLA, while ARLE still has explicit CSA selection and per-row attention | Matched SGLang profile showing DSv4 metadata inside/outside graph and FlashMLA kernel timing |
| CUDA graph coverage | SGLang source default confirmed; current running SGLang server disables graph and is incomparable | Graph coverage changes launch count and metadata location; disabling graph for visibility changes the path | Two SGLang traces, graph-on for wall clock and graph-profile/expanded trace for attribution, both labeled |
| Warmup | SGLang source confirmed DeepGEMM precompile can dominate cold startup | A trace including JIT warmup is not steady-state | Log grep for DeepGEMM warmup outside measured window and cache/precompile artifact |

## Why The Current ARLE Stages Look Unreasonable

Assuming the current ARLE stage table is true, the stages are unreasonable
because they belong to the fallback route, not because each kernel is merely
poorly tuned.

| ARLE stage | Existing evidence | Structural diagnosis |
|---|---|---|
| `ffn_routed_local` | Request trace lists it as a top FFN phase; code calls `forward_local_routed_gpu` on the default route | This is replicated-token local expert work. It should disappear from the SGLang-candidate path as a top-level reconciliation pattern, replaced by token-owned DeepEP/MegaMoE work |
| `ffn_all_reduce` | Request trace and nsys both expose it; code calls `post_moe_expert_all_reduce_hidden_states` immediately after local routed experts | This is the cost of the fallback data contract. Optimizing this collective cannot make the route SGLang-equivalent because SGLang's intended combine is part of token-owned EP/MoE |
| `attn_hybrid_kernel` | nsys top kernels and request trace show it as a visible stage; code launches ARLE hybrid attention after separate metadata/selector work | SGLang's DSv4 core call consumes SWA plus C4/C128 extra KV in one FlashMLA API. ARLE's separate hybrid stage is not isomorphic |
| `attn_csa_select_kernel` | CSA fast-path artifact shows p2047/o8 improves, but p2047/o32 still high; code allocates selected blocks and runs selector separately | It is real current-route work, but SGLang-path optimization should first move selector/index metadata into the FlashMLA/graph contract, not only tune the standalone selector |
| `ffn_expert_loop` | Request trace still shows expert loop under current path | This is evidence that transport/topology/orchestration still exposes local expert iteration. It is not the DeepGEMM/MegaMoE fused steady path |
| `attn_all_reduce` / `attn_swa_all_reduce` | Request trace shows large attention all-reduce in the trace-on smoke | This is tied to current TP/attention layout. SGLang DP-attention lanes have different groups and row ownership |
| Runtime API overhead | Old nsys trace shows hundreds of thousands of kernel launches and large `cuMemAllocAsync`/`cuMemFreeAsync` totals in the filtered decode window | Even if each kernel is small, the path is graph/materialization poor compared with SGLang's DSv4 graph metadata design |

Therefore the current best explanation is not "DeepEP is slow" or "FlashMLA is
slow". The current best explanation is:

```text
ARLE is measuring a replicated-token fallback route with per-row attention and
post-FFN all-reduce, while SGLang's fast DSv4 path is a token-owned multi-axis
route with DSv4 FlashMLA metadata/cache contracts, DeepEP/MegaMoE-class MoE,
CUDA graph metadata coverage, and often speculative output-token accounting.
```

The exact millisecond contribution of each structural item is `待验证` until a
matched SGLang no-spec/spec artifact exists.

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

## What Must Be Locked Before Any Future Perf Run

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

This is a future-run checklist only. It was not executed for this report.

## Research Conclusion

The actual SGLang DSv4 source path is now identified:

```text
/workspace/sglang @ 0d51db3
```

The earlier negative finding came from inspecting `/sgl-workspace/sglang`, which
is not the runtime import tree for DSv4.

The structural reason SGLang is faster is now clear at source level:

1. SGLang's intended H200 DSv4 lanes use a multi-axis TP/DP/EP contract with DP
   attention and token-owned MoE rows. ARLE's current executable route still
   submits replicated full-token requests to every rank and reconciles local
   expert output with all-reduce.
2. SGLang's attention path is a DSv4 FlashMLA contract: fused WQA/WKV, fused
   norm/RoPE/cache write, graph-aware C4/C128/SWA metadata, and one
   `flash_mla_with_kvcache` core call. ARLE's current route still exposes
   per-row attention plus standalone CSA selector/hybrid work.
3. SGLang's MoE path has DeepEP/MegaMoE-class transport plus grouped DeepGEMM
   and JIT fused activation/quant. ARLE has pieces of DeepGEMM, but they are
   embedded in the replicated-token route and still carry local routing,
   materialization, and post-FFN all-reduce costs.
4. SGLang's DSv4 defaults keep metadata and prep closer to CUDA graph replay.
   Existing ARLE artifacts still show heavy launch, alloc/free, D2H/H2D, and
   explicit per-stage overhead.
5. SGLang cookbook lanes can also use EAGLE/MTP. If the reference metric is
   output-token TPOT under speculative decoding, ARLE's no-spec path is being
   penalized by metric definition before any kernel comparison starts.

The exact numeric waterfall from SGLang TPOT to ARLE TPOT remains `待验证`
because there is no matched SGLang no-spec/spec trace artifact in the repo. The
report therefore rejects any precise percentage attribution such as "X ms from
spec decode, Y ms from topology" until a future trace supplies those controls.

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

The next investigation, when running is allowed again, should start with a
matched SGLang no-spec target-step baseline and a separate spec-on EAGLE/MTP
baseline. This report intentionally does not start either run.
