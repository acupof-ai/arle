# M6 CUDA vLLM Gap Follow-ups

## Context

The first M6 CUDA P0 snapshot on `48d31ace09f8` completed all four ARLE-vs-vLLM
workloads on the local RTX 4070 Ti SUPER, but ARLE won only 1 of 8 score cells.
This plan tracks the losing cells without blocking the benchmark snapshot.

Primary snapshot:
[2026-05-07-m6-world-rank-snapshot-cuda.md](../experience/wins/2026-05-07-m6-world-rank-snapshot-cuda.md)

## Targets

| workload | gap | target |
|---|---:|---|
| prefill-heavy out tok/s | ARLE -5.4% | close >= 6% without regressing TTFT |
| decode-heavy out tok/s | ARLE -4.9% | close >= 6% and stabilize TTFT p99 |
| longctx-32k out tok/s | ARLE -34.9% | remove 16 GiB pool bottleneck or mark hardware-limited with remote H100/Ada result |
| high-conc out tok/s | ARLE -62.9% | raise effective active concurrency beyond 14 or prove memory-bound limit |

## Work Items

1. Prefill-heavy:
   - Compare ARLE TileLang paged prefill launch shape against vLLM Triton
     attention at 4096-in / 16-out.
   - Confirm whether chunking at 2048 is optimal for single request prefill on
     sm_89.

2. Decode-heavy:
   - Measure per-token decode phase with CUDA graph on and deterministic BF16
     GEMM enabled.
   - Separate model math time from sampling / host response streaming overhead.

3. Longctx-32k:
   - Re-run on a larger GPU before drawing architecture conclusions.
   - On the local card, test whether KV tier promotion/demotion can keep c=4
     progressing without the `active=2 waiting=2` residency ceiling.

4. High-conc:
   - Compare ARLE `max_slots=14` against vLLM `max_num_seqs=64` and an
     equal-slot vLLM run.
   - Profile scheduler admission and decode batch width under c=64; the first
     goal is to increase output tok/s while preserving ARLE's TTFT p50 lead.

## High-conc Phase 1 investigation (2026-05-07)

Scope: evidence-only. No ARLE code changes. `nsys`, `perf`, and `pidstat` were
not installed on this host, so the profile used GuideLLM, ARLE `/v1/stats`,
`nvidia-smi dmon/pmon`, `top -H`, and `ps -L` samples.

### Server config diff

| field | ARLE-CUDA | vLLM |
|---|---|---|
| launch shape | `--max-seq-len 5120` shared with prefill/decode M6 runs | `--max-model-len 2048` for high-conc |
| high-conc request shape | 1024 in / 256 out, c=64 | 1024 in / 256 out, c=64 |
| max active sequences | `max_slots=14` auto | `--max-num-seqs 64` |
| max batched tokens | `max_num_batched_tokens=16384` | internal; not exposed in OpenAI API logs |
| chunked prefill | `chunked_prefill_size=2048`, `max_prefill_tokens=16384` | `enable_chunked_prefill=True` |
| KV cache | FP8E4M3 paged pool, 51,520 tokens / 3,220 pages | `--kv-cache-dtype fp8`, GPU KV cache 75,968 tokens in the Phase 1 profile run |
| capacity log | `max_slots=14`, `TokenKVPool: 51520 max tokens`, `mem_fraction_static=0.85` | `Maximum concurrency for 2,048 tokens per request: 37.09x`, `max_num_seqs=64` |
| CUDA graph | ARLE captures B=1..14 | vLLM captures up to 128, including FULL decode largest=64 |
| attention backend | ARLE TileLang/custom CUDA Qwen3 path | vLLM `TRITON_ATTN` |

Important mismatch: the M6 high-conc ARLE server was still sized for 5120-token
contexts, while vLLM was sized for 2048-token contexts. The workload itself
needs only 1280 tokens/request. That makes ARLE's auto slot count a likely
first-order limiter before any kernel tuning.

### Short profile results

Short runs used the same 1024-in / 256-out c=64 shape with `--max-seconds`
30-35 and `--warmup 3`.

| backend | run label | TTFT p50 | ITL p50 | out tok/s | req/s | conc p50 |
|---|---|---:|---:|---:|---:|---:|
| ARLE | `m6-highconc-phase1-arle` | 1051.5 ms | 30.69 ms | 363.80 | 1.531 | 16 |
| ARLE | `m6-highconc-phase1-arle-cpu` | 1051.3 ms | 30.67 ms | 332.86 | 1.556 | 16 |
| vLLM | `m6-highconc-phase1-vllm` | 7927.4 ms | 45.50 ms | 792.98 | 3.281 | 64 |
| vLLM | `m6-highconc-phase1-vllm-cpu` | 1201.7 ms | 45.66 ms | 1019.44 | 4.296 | 64 |
| ARLE full M6 reference | `m6-arle-high-conc-r2` | 1059.0 ms | 30.81 ms | 414.66 | 1.678 | 16 |
| vLLM full M6 reference | `m6-vllm-high-conc-r2` | 1606.2 ms | 44.65 ms | 1114.71 | 4.530 | 64 |

The warmed short vLLM CPU-profile run is close enough to the full M6 baseline
to use for qualitative comparison. The first short vLLM profile had a high
TTFT tail because it was immediately after server start.

### ARLE telemetry during high-conc

Parsed from `bench-output/2026-05-07-m6-highconc-phase1-arle-profile/`.

| metric | median | peak | mean |
|---|---:|---:|---:|
| active | 13 | 14 | 10.6 |
| waiting | 49 | 50 | 38.6 |
| batch_width | 13 | 14 | 10.6 |
| decode_tokens | 12 | 14 | 9.8 |
| prefill_tokens | 0 | 11275 | 842.8 |
| engine_batch_occupancy | 0.2895 | 0.3431 | 0.2254 |
| kv_util | 29.0% | 34.3% | 22.5% |

Step phase samples:

| phase | median | peak | mean |
|---|---:|---:|---:|
| decode | 19.5 ms | 198.7 ms | 57.2 ms |
| prefill | 1.0 ms | 100.4 ms | 26.2 ms |
| loop total | 94.5 ms | 211.0 ms | 83.6 ms |

Interpretation: under c=64, ARLE quickly fills all 14 slots and leaves roughly
50 requests waiting. The scheduler is not stuck, but the effective decode batch
width is capped at 14. `kv_util` is low for this workload, so the cap is not a
KV pool exhaustion signal.

### Hot-path proxy profile

`nsys` / `perf` were unavailable, so this is a proxy profile rather than kernel
symbol attribution.

| backend | GPU evidence | CPU evidence | read |
|---|---|---|---|
| ARLE | `nvidia-smi dmon`: active samples SM median 100%, mean 98.2%; memory-util median 93%, mean 59.3%; power mean 270 W | `top -H`: one `infer` thread median 99.8% CPU across 33 active samples; Tokio threads near idle | GPU is saturated, but each step only carries up to 14 decode rows. CPU has one hot scheduler/driver thread, not broad runtime contention. |
| vLLM | `nvidia-smi dmon`: active samples SM median 100%, mean 98.8%; memory-util median 30.5%, mean 52.8%; power mean 243 W. `pmon` showed `VLLM::EngineCore` at 98% SM / 29% mem in an active sample | `top -H`: one `VLLM::EngineCore` worker thread median 98.4% CPU; EngineCore main median 17.9%; API Python thread median 5.0% | vLLM is also GPU-saturated, but keeps c=64 concurrency in the benchmark. Python/GIL is not the observed limiter. |

The visible gap is therefore not "ARLE has idle GPU"; both backends can hit
near-100% SM. The gap is more likely "vLLM does more useful decode rows per
GPU-saturated step".

### Root-cause hypotheses

1. **Most likely / highest impact: ARLE slot sizing is mismatched for high-conc.**
   - Evidence: ARLE high-conc uses `--max-seq-len 5120` and auto-sizes to
     14 slots; vLLM high-conc uses `--max-model-len 2048` and admits
     `--max-num-seqs 64`. ARLE telemetry shows active/batch_width peaks at 14
     while waiting stays near 50. The workload only needs 1280 tokens/request.
   - Falsification experiment: rerun ARLE high-conc with a dedicated
     high-conc server envelope, starting with `--max-seq-len 2048` and then a
     `--num-slots` sweep (`14`, `28`, `42`, `56`) if memory allows. If out
     tok/s scales mainly with slots, this is the primary root cause.

2. **Likely / high impact: ARLE per-step useful work is capped by one-token
   decode over too few rows.**
   - Evidence: ARLE `decode_tokens` median is 12 and peak 14; vLLM maintains
     GuideLLM concurrency p50=64. ARLE has lower ITL per request, but total
     output tok/s is lower because fewer rows are decoded per step.
   - Falsification experiment: run an equal-slot vLLM baseline with
     `--max-num-seqs 14` and the same 2048 max model length. If vLLM throughput
     drops near ARLE, admission width dominates; if vLLM remains far ahead,
     kernel/graph lowering dominates.

3. **Possible / medium impact: ARLE mixed prefill/decode steps are too costly
   during high-conc admission.**
   - Evidence: ARLE phase samples show loop-total median 94.5 ms and prefill
     peaks at 100.4 ms while filling the 64-request queue. Full-run ITL is
     stable after admission, but high-conc throughput includes continuous
     admission as requests complete.
   - Falsification experiment: replay c=64 with a prefilled/prefix-hit workload
     or a two-phase bench that admits all requests then measures decode-only
     steady state. If output tok/s jumps only after removing admission, tune
     mixed prefill/decode scheduling; otherwise focus on slot count and decode
     width.

### Next smallest validation actions

1. **Config-only bench commit / doc entry:** add a `high-conc-2048` bench
   variant that launches ARLE with `--max-seq-len 2048` and sweeps
   `--num-slots`. This validates hypothesis 1 without touching runtime code.

2. **Equal-width control:** run vLLM with `--max-num-seqs 14` against the same
   high-conc shape. This isolates scheduler width from kernel implementation
   and gives an apples-to-apples per-row efficiency baseline.

3. **If slot-width wins:** implement the smallest runtime change that decouples
   high-conc admission capacity from the conservative 5120-token server
   envelope. The first code change should be a guarded scheduler/config path,
   not kernel work.

### Phase 1 verify: slot sweep and equal-width vLLM control

Scope: evidence-only scout runs. No ARLE code changes. All runs used the
high-conc shape `1024 in / 256 out, c=64`, one run per point,
`--max-seconds 45`, `--warmup 5`.

ARLE command shape:

```bash
RUST_LOG=info RUST_BACKTRACE=full \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
target/release/infer \
  --model-path /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --port 8000 \
  --max-seq-len 2048 \
  --num-slots <16|32|48|64>
```

One initial `--num-slots 16` launch immediately after stopping a stale longctx
T1 run failed with `CUDA_ERROR_OUT_OF_MEMORY`; the later s16 rerun below
started cleanly and is the valid datapoint.

| ARLE config | KV pool tokens | active peak | running_batch peak | decode peak | kv_util peak | TTFT p50 | ITL p50 | out tok/s | req/s | client conc p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5120 / auto 14 ref (`m6-arle-high-conc-r2`) | 51,520 | 14 | 14 | 14 | 35.4% | 1059.0 ms | 30.81 ms | 414.66 | 1.678 | 16 |
| 2048 / 16 (`m6-highconc-phase1-arle-2048-s16-rerun`) | 51,440 | 16 | 16 | 16 | 38.6% | 1058.5 ms | 33.84 ms | 365.53 | 1.600 | 18 |
| 2048 / 32 (`m6-highconc-phase1-arle-2048-s32`) | 50,336 | 32 | 32 | 32 | 80.0% | 1088.9 ms | 54.65 ms | 408.86 | 1.775 | 34 |
| 2048 / 48 (`m6-highconc-phase1-arle-2048-s48`) | 49,648 | 47 | 38 | 38 | 95.4% | 3877.6 ms | 65.43 ms | 442.09 | 1.900 | 47 |
| 2048 / 64 (`m6-highconc-phase1-arle-2048-s64`) | 48,544 | 46 | 37 | 37 | 94.5% | 3881.8 ms | 62.16 ms | 393.82 | 1.850 | 42 |

vLLM equal-width control:

```bash
PATH=/tmp/arle-vllm-venv/bin:$PATH \
NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-14' \
CC=/usr/bin/gcc-14 \
CXX=/usr/bin/g++-14 \
CUDAHOSTCXX=/usr/bin/g++-14 \
CUDA_VISIBLE_DEVICES=0 \
/tmp/arle-vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --served-model-name Qwen/Qwen3-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-seqs 14 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --trust-remote-code \
  --no-enable-log-requests \
  --uvicorn-log-level warning
```

vLLM s14 logs: `max_num_seqs=14`, GPU KV cache 69,440 tokens, maximum
concurrency for 2,048-token requests 33.91x, CUDA graph capture sizes up to 24.

| vLLM config | TTFT p50 | ITL p50 | out tok/s | req/s | client conc p50 | note |
|---|---:|---:|---:|---:|---:|---|
| 2048 / max-num-seqs 14 (`m6-highconc-phase1-vllm-s14`) | 17608.6 ms | 20.24 ms | 614.55 | 2.650 | 51 | server-side active seq telemetry unavailable |
| 2048 / max-num-seqs 64 ref (`m6-vllm-high-conc-r2`) | 1606.2 ms | 44.65 ms | 1114.71 | 4.530 | 64 | full M6 reference |

Conclusion: the slot-width hypothesis is **only partially true**.

- Raising ARLE slots from 16 to 48 raises active rows and reduces waiting, so
  the original `max_slots=14` envelope was indeed too narrow for high-conc.
- Throughput does **not** scale with slot count: best scout was only 442 out
  tok/s at s48, and s64 regressed to 394 out tok/s. This is still far from
  vLLM s64 at 1115 out tok/s.
- At s48/s64, ARLE hits `kv_util` around 95%, but effective `running_batch`
  tops out at 38/37 instead of 48/64. More slots mostly turn into higher TTFT
  and ITL rather than useful decode throughput.
- vLLM constrained to `--max-num-seqs 14` still reaches 615 out tok/s, faster
  than every ARLE scout point. Therefore equal-width control rejects the
  "admission width alone explains the gap" hypothesis.

Updated root-cause read:

1. **Confirmed partial:** the default 5120-token envelope leaves high-conc too
   narrow, but a high-conc-specific 2048 envelope is not sufficient by itself.
2. **Likely primary next bottleneck:** ARLE mixed prefill/decode scheduling at
   high slot counts. The service trace shifts toward `mixed` plans, prefill
   queues reach 40+, and decode rows cap around 37-38 while the token budget is
   shared with 1025-token prompt chunks.
3. **Still possible:** decode kernel / graph lowering is less efficient than
   vLLM at the same effective width; vLLM s14 outperforms ARLE best scout by
   39%.

Next validation actions:

1. Run ARLE s32/s48 controls with prefill throttled:
   `--max-prefill-tokens 2048` and/or `--prefill-max-requests 1|2`. If out
   tok/s rises while TTFT remains acceptable, the fix belongs in high-conc
   scheduler policy rather than attention kernels.
2. If prefill throttling does not move throughput, profile decode kernels with
   Nsight on a machine that has `nsys`/`ncu`, comparing ARLE s32/s48 against
   vLLM s14. That is the point to enter the M3.5 / M_b path.
3. Do **not** implement a server-envelope-only change as the Phase 1 fix. It
   increases active rows but does not close the 2.7x throughput gap.

### Phase 1 step 3: prefill admission controls

Scope: evidence-only scout runs. No ARLE code changes. The fixed workload
remained `1024 in / 256 out, c=64`, one run per point, `--max-seconds 45`,
`--warmup 5`. ARLE was held at the best previous scout envelope:
`--max-seq-len 2048 --num-slots 48`.

Default server command:

```bash
RUST_LOG=info RUST_BACKTRACE=full \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
target/release/infer \
  --model-path /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --port 8000 \
  --max-seq-len 2048 \
  --num-slots 48
```

Throttled points added:

```bash
  --max-prefill-tokens 2048 --prefill-max-requests <1|2|4>
```

The default point resolved `max_prefill_tokens=16384,
prefill_max_requests=none`; the three throttled points resolved
`max_prefill_tokens=2048` and `prefill_max_requests=<N>`. Raw server logs are
under
`bench-output/2026-05-07-m6-highconc-phase1-step3-prefill-sweep/`.

`prefill_rows share` below is computed from sampled `/v1/stats` rows as
`sum(prefill_rows) / sum(prefill_rows + decode_rows)` over scheduled samples.
It is a row-count share; each prefill row can still carry a 1025-token prompt
chunk in this workload.

| ARLE s48 config | TTFT p50 | ITL p50 | out tok/s | req/s | plan labels | prefill_rows share | active peak | running_batch peak | avg running/active | kv_util peak | artifact |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| default (`max_prefill_tokens=16384`, no request cap) | 4204.1 ms | 72.13 ms | 402.33 | 1.750 | `idle=2 decode=435 prefill=2 split=0 mixed=112` | 9.2% | 48 | 43 | 78.5% | 96.5% | `m6-highconc-phase1-step3-arle-s48-default` |
| `--max-prefill-tokens 2048 --prefill-max-requests 1` | 5132.4 ms | 77.05 ms | 374.29 | 1.625 | `idle=2 decode=8 prefill=2 split=0 mixed=531` | 2.5% | 48 | 47 | 83.5% | 94.5% | `m6-highconc-phase1-step3-arle-s48-prefill1` |
| `--max-prefill-tokens 2048 --prefill-max-requests 2` | 4169.2 ms | 72.24 ms | 408.37 | 1.775 | `idle=2 decode=7 prefill=2 split=0 mixed=540` | 2.6% | 48 | 44 | 79.6% | 94.9% | `m6-highconc-phase1-step3-arle-s48-prefill2` |
| `--max-prefill-tokens 2048 --prefill-max-requests 4` | 4057.9 ms | 72.24 ms | 408.74 | 1.775 | `idle=2 decode=8 prefill=2 split=0 mixed=540` | 2.6% | 48 | 44 | 78.3% | 95.9% | `m6-highconc-phase1-step3-arle-s48-prefill4` |

Comparison against equal-width control:

| backend / envelope | TTFT p50 | ITL p50 | out tok/s | read |
|---|---:|---:|---:|---|
| ARLE 5120 / auto 14 ref | 1059.0 ms | 30.81 ms | 414.66 | Lower TTFT, but only 14 useful decode rows. |
| vLLM 2048 / `--max-num-seqs 14` | 17608.6 ms | 20.24 ms | 614.55 | Same nominal max seq count still 1.48x ARLE throughput. |
| ARLE 2048 / s48 best Step 3 point | 4057.9 ms | 72.24 ms | 408.74 | Wider admission worsens ITL and does not approach vLLM s14. |

Conclusion: this **does not confirm simple mixed prefill/decode admission as
the throughput unlock**.

- The throttle works mechanically: `prefill_rows share` drops from 9.2% to
  about 2.5-2.6%.
- Throughput does not improve. `prefill-max-requests=1` regresses to 374 out
  tok/s; `2` and `4` sit around 408 out tok/s, which is scout-noise distance
  from the default 402 out tok/s and below the previous s48 scout at 442 out
  tok/s.
- `running_batch` peak improves for `prefill=1` (47/48) but useful output
  throughput still drops, so filling rows is not sufficient if the step shape
  is mostly tiny mixed work.
- Throttled runs shift almost every scheduler step into `mixed` labels
  (`~98% mixed`) while carrying only one or two prefill rows. That is a useful
  scheduler-policy signal, but not a validated performance fix.
- The equal-width vLLM control remains the stronger bottleneck evidence:
  vLLM at 14 max sequences is still 1.48x faster than ARLE's high-conc M6
  reference and 1.50x faster than this Step 3 best point.

Updated next action:

1. Move to Nsight kernel / CUDA graph profiling for decode-heavy effective
   widths (`ARLE s32/s48` vs `vLLM --max-num-seqs 14`). This is the next
   falsification point for decode kernel, graph replay, and launch/driver
   overhead.
2. Do not land a prefill admission rate-limit as the Phase 1 fix from this
   evidence. A future scheduler policy change should be tested as
   decode-priority / mixed-step collapse, not as a static
   `prefill-max-requests` cap alone.

## **Priority 0A: re-run longctx with T1 host-pinned KV overflow enabled**

The M6 baseline run did NOT explicitly enable the T1 host-pinned tier.
`SchedulerConfig::t1_host_pinned_capacity_bytes` defaulted to `None`
(the constructor's conservative auto-size, see `scheduler/types.rs:177`).
The local box has **31.3 GB system RAM** vs the **16 GB GPU**; the
infrastructure to spill GPU KV to host-pinned RAM **already exists**
(`infer/src/kv_tier/host_pool.rs` + `KvTierAdapter`, M2 of unification)
and the `infer` binary already exposes the CLI flags
(`infer/src/main.rs:203-219`):

- `--t1-host-pinned-capacity-mb <N>` (target ≈ 16384 — leaves ~10 GB for OS / build / driver)
- `--t1-host-pinned-high-water <FRAC>` (default 0.85)
- `--t1-host-pinned-low-water <FRAC>` (default 0.70)

Re-run the M6 canonical sweep with explicit T1 sizing **before** going
deep on long-context combo plan implementation. Hypothesis: longctx-32k
-34.9% shrinks materially because kv_util-100% scenarios spill to host instead
of stalling at `active=2 waiting=2`.

Update after the high-conc Phase 1 investigation: do not treat T1 as the
primary high-conc fix. The high-conc profile showed `kv_util` peaking at only
34.3% while active slots were capped at 14 and waiting stayed near 50, so
high-conc is currently a slot-envelope / useful-decode-width problem before it
is a KV overflow problem. Keep T1 as a cheap control run, but prioritize
`high-conc-2048` and equal-slot vLLM controls for the 2.7x throughput gap.

```bash
RUST_LOG=info NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run --release -p infer --no-default-features --features cuda -- \
  --model-path /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --port 8000 \
  --max-seq-len 5120 \
  --t1-host-pinned-capacity-mb 16384

PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
scripts/bench_guidellm.sh cuda-m6-with-t1 \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor /home/ckl/projects/arle/infer/models/Qwen3-4B
```

Expected delta:
- longctx-32k: gap closes (host-pinned absorbs the prefix overflow that currently stalls at 2 active).
- high-conc: likely small or no change unless the server envelope also raises
  `max_slots`; current evidence says `kv_util` is not the limiter.
- prefill-heavy / decode-heavy: unchanged (no overflow happening in baseline).

If the longctx expected delta lands, publish it as the first configuration-only
win. Do not claim high-conc closure until the slot-width controls land.

**Acceptance**: re-run produces a `cuda-m6-with-t1` wins entry with
direct delta-vs-baseline rows. Closing longctx-32k on local hardware counts as
a gap closed for this plan's acceptance criterion.

## Strategic alignment with combo plan (manager note 2026-05-07)

Three of the four gaps overlap directly with already-spec'd combo plan
sub-plans. Do NOT attack them in isolation — they are downstream of work
already in flight:

| M6 gap | Combo plan that addresses it | Relationship |
|---|---|---|
| decode-heavy out tok/s -4.9% | [`M_b.2`](longctx-spec-tilelang-combo.md) sparse-self-spec fusion | spec-decode is the canonical decode-throughput multiplier; M_b.2 targets +10-15% on top of vanilla decode without retraining. Likely closes this gap on its own. |
| longctx-32k out tok/s -34.9% | [`M_d`](M_d-tier-kv-spec-decode-coordination.md) Tier-KV × spec coordination | Long-context throughput is gated by KV residency, exactly what M_d's eager-prefetch + scratch-page commit barrier address. Plus the 16 GB pool ceiling means H100/L4 retest is the right path before architecture changes. |
| high-conc out tok/s -62.9% | [`M3.5`](M3.5-collapse-scheduler-loops.md) shared CPU policy + [`M_b`](M_b-tilelang-fused-draft-verify-kernel.md) batched verify | High-conc throughput is gated by scheduler decisions per tick + per-row verify cost. M3.5 unifies decisions; M_b/M_c add multi-token-per-step credit on the verify path. |
| prefill-heavy out tok/s -5.4% | (no combo plan; isolated tile-shape tuning) | The only gap that's a pure single-shape prefill tuning question. Worth its own focused micro-optimization. |

Update after high-conc Phase 1: for the high-conc 2.7x gap, run the
configuration controls first (`high-conc-2048` ARLE slot sweep and equal-slot
vLLM). If those controls show the gap is mostly admission width, fix the server
envelope before spending a sprint on M3.5/M_b implementation work. For the
decode-heavy and longctx gaps, the combo-plan mapping still stands.

## Acceptance

- Publish a follow-up wins or errors entry with at least one closed gap.
- Do not count hardware-limited longctx as a runtime regression until a larger
  CUDA runner repeats the workload.
- Keep the M6 raw command shapes unchanged unless the change is explicitly
  documented as a new benchmark variant.
- For the three gaps that map to combo plan sub-plans, gap closure happens
  when the combo plan sub-plan lands — this plan tracks the bench delta
  per sub-plan, not duplicate implementations.
