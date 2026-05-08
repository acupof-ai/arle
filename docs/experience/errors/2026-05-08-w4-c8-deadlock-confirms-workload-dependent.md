# W4 c=8 deadlock — confirms ARLE admission deadlock is workload-dependent,not c=16-specific

> Companion to [`cb087c7`](2026-05-08-w3-c16-deadlock-not-just-admission.md)
> W3 c=16 deadlock。This run tests **canonical W4 spec(c=8,8K prompt,
> 256-token resume)against same W4A16 Marlin server build**。Same deadlock
> signature reproduces:active=8,prefill_queue=7,prefill_rows=0,tokens_out=0。
> 0/256 turns succeed。
>
> **Implication**:deadlock is NOT c=16-specific —— it triggers at
> **prefill-load × concurrency × max-seq-len product**。Both W3(1K
> prompt × c=16)AND W4(8K prompt × c=8)hit the same threshold。
> ARLE admission/scheduler substrate fix is now **critical path for
> ALL master §2.1 production-shape baselines**。

## Setup

```bash
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer \
  --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --port 8000 --num-slots 16 --max-seq-len 9216

PATH=.venv/bin:$PATH \
python scripts/bench_agent_trace.py \
  --workload agent-w4-tool-resume \
  --num-concurrent 8 \
  --label arle-w4-c8-canonical
```

W4 spec:
- 128 sessions × 2 turns(warmup + resume)= 256 total turns
- Base prompt 8192 tokens(±64 jitter)
- Tool output 256 tokens(±16 jitter)
- Resume max_tokens 256
- **Concurrency 8**(canonical,below W3 c=16 deadlock threshold)

## Result — IDENTICAL deadlock signature to W3 c=16

```
turns OK: 0 / 256 (0%)
all 256 errors: "HTTP 503 after 5 503 retries: Server is at capacity"
elapsed per failing turn: ~31 s(retry-backoff exhausted then bail)
```

Server log timeline:
- t=0:8 sessions admitted(`chat/completions: messages=2, prompt_bytes=~31KB`)
- t=0+:All subsequent submissions immediately 503'd("Scheduler at capacity")
- t=502s(after warmup):Resume turns submitted with prompt_bytes ~50KB(10K tokens)
  but server still showed `Scheduler at capacity` → never recovered

## /v1/stats — Same deadlock signature as W3 c=16

```
active=8, waiting=0, scheduled=0, prefill_queue=7
decode_rows=0, prefill_rows=0, running_batch=0
batch_width=0, decode_tokens=0, prefill_tokens=0, tokens_out=0
step_last=0.0ms
step_phase_us=adm:111, prefill:435027, decode:0, emit:0, total:435138
                       ↑↑↑ 435 sec wall-time on "prefill phase" but ZERO actual rows
plan_label=idle:0, decode:0, prefill:2, split:0, mixed:0
peak_mem=14279.6MB
engine_active_requests=8, engine_batch_occupancy=0.0200
session_affinity_miss=8
resume_prefill_tokens=8293  ← only 13% of expected 65536 (8 × 8192)
```

Three deadlock fingerprints:
1. **`prefill_queue=7, prefill_rows=0`**:7 waiting,0 actively prefilling
2. **`step_phase_us prefill:435027 + tokens_out=0`**:435 sec on prefill phase semantically,but no token output → schedule label vs actual work mismatch
3. **`resume_prefill_tokens=8293 / 65536`**:13% partial then halt

## Comparison — W3 c=16 vs W4 c=8

| Workload | Prompt size | Concurrency | Slots | Expected total prefill | Actual prefill | tokens_out | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| W3 c=16 | 1024 | 16 | 16 | 16384 | 1084 (7%) | 0 | DEADLOCK |
| **W4 c=8** | **8192** | **8** | **16** | **65536** | **8293 (13%)** | **0** | **DEADLOCK** |

Two production-spec workloads,**different concurrency,different prompt
size,SAME deadlock signature**:
- active = max scheduler-admit threshold(8 or 16)
- prefill_queue = active - 1 always
- prefill_rows = 0 forever
- tokens_out = 0
- partial prefill_tokens completed(7-13%)then halt

## Hypothesis refinement(was Hypothesis A in W3 entry)

**Hypothesis A — chunked prefill chunk-size × concurrency × prompt interaction
fails admission rotation**:strongly supported now。

Evidence:
- W3 1K prompt × c=16:total expected prefill 16K tokens → fails
- W4 8K prompt × c=8:total expected prefill 64K tokens → fails
- Both partial(7-13%)then halt

**Likely root cause**:`max_num_batched_tokens` envelope or chunked-prefill
admission policy treats partial prefill as "in progress" and refuses to
schedule subsequent waiting items even when the active session has stalled。
Rough triggering condition:
```
admitted_sessions × prompt_tokens > max_num_batched_tokens × 2-3
```

For W3:16 × 1024 = 16384,exceeds default 16384 envelope by 0-2x。
For W4:8 × 8192 = 65536,exceeds default 16384 envelope by 4x。

**Hypothesis B — KV slot reservation HOL block**:still possible,
secondary。Reserved KV slots（8 × 8K = 64K KV entries）approach
peak_mem=14.3GB which is most of GPU。Resume_prefill_tokens stall at 13%
suggests KV pressure may interact with prefill rotation。

## Master §7.1 P0.0 escalation

W3 AND W4 production-spec baseline are now BOTH blocked on substrate fix:

- W3 c=16 baseline:**BLOCKED**(`cb087c7`)
- W4 c=8 baseline:**BLOCKED**(this entry)
- W3 c=4 workaround:WORKING(`f6f3af3`,`370a267`)
- W4 c=4 workaround:**UNTESTED**(this run was c=8;diagnostic-c override
  allows c=4 if needed)

**Codex substrate priority elevation**:was "fix W3 c=16 deadlock" ↔ now
"fix W3+W4 production-shape admission deadlock — affects two-thirds of
master §2.1 binding workloads"。

Likely fixes(in priority order based on this evidence):
1. **Increase `max_num_batched_tokens` envelope** to handle 8 × 8K = 65K
   prompt budget(currently 16K? need verify)
2. **Round-robin prefill rotation** instead of HOL — let stalled session
   defer to next slot's prefill chunk
3. **Active vs prefill_queue counter race fix** — investigate
   `infer/src/scheduler/cuda/core/scheduler.rs` invariants

## Workaround for spec-decode axis re-test

Until substrate fix lands:
- Use **c=4** for both W3 AND W4 via diagnostic override
- W4 c=4:128 sessions × c=4 = 32 sequential turns,~10 min wall
- This unblocks Medusa training data collection / spec-axis re-test

W3 c=4 already validated(`370a267`)— W4 c=4 next on Claude actionable list。

## Skill v1.3.0 methodology validation

Per anti-pattern #13(NULL elimination):
- W3 c=16 KILL alone could have been "transient capacity" → harness fix
- W4 c=8 KILL with same fingerprint **proves it's substrate** → harness fix is NOT sufficient
- Cross-workload reproduction is the gold standard for substrate-bug claims

Per skill v1.3.0 §0 SOLID:multi-workload evidence is required to claim
"substrate bug",single-workload could be workload-specific quirk。This
entry is the second evidence piece making the claim SOLID。

## Cross-references

- W3 c=16 deadlock: `cb087c7`(`docs/experience/errors/2026-05-08-w3-c16-deadlock-not-just-admission.md`)
- W3 c=4 baseline:`370a267`(`docs/experience/wins/2026-05-08-w3-c4-baseline-first-valid.md`)
- Master §7.1 P0.0 baseline mandate
- Master §2.1 W3/W4 spec
- ARLE scheduler core: `infer/src/scheduler/cuda/core/scheduler.rs`
- ARLE admission: `infer/src/scheduler/cuda/runtime/admission.rs`
- W4 bench script: `scripts/bench_agent_trace.py` line 70+ `WORKLOAD_W4`
- Bench logs: `/tmp/w4-bench.log`,`/tmp/infer-w4.log`(local only)

## Rule

When a substrate hypothesis is suspected,**always reproduce on a second
workload with different parameters**(prompt size,concurrency,seq_len,
slot count)。Single-workload deadlock could be workload tuning;
multi-workload deadlock with same fingerprint is substrate。

For ARLE specifically:**any future scheduler invariant fix MUST verify
both W3 c=16 AND W4 c=8 unblock**(not just one),since the two trigger
the same deadlock via different parameter products。
