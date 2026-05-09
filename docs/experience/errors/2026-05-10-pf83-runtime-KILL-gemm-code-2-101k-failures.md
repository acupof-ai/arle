# PF8.3 RUNTIME KILL — gemm_w4_fp8_marlin_cuda fails code 2 (cudaErrorMemoryAllocation) on EVERY request under bench load (101,380/101,380 = 100% failure)

## Context

PF8.3 substrate landed (`11763ba`) + cargo build PASS + greedy_consistency
PASS (4.33s on hybrid checkpoint). PF8.5 license bench attempts v3-v10
all hit "0 successful requests" but I assumed it was bench infrastructure
issues (PATH, --backend-kwargs, --outputs html, absolute path,
pre-mkdir). Direct ps verify of server log THIS tick reveals real
root cause: **the PF8 GEMM kernel fails 100% under sustained load**.

## Root cause (raw evidence per skill v1.11.0+ #28+#31)

```bash
$ grep -c "gemm_w4_fp8_marlin_cuda failed with code 2" /tmp/pf83-treatment-fp8-direct.log
101380

$ grep "gemm_w4_fp8_marlin_cuda failed" /tmp/pf83-treatment-fp8-direct.log | head -1
2026-05-10T07:34:10.488658+08:00 ERROR infer::scheduler::cuda::prefill:627
  Request 1: prefill batch failed: gemm_w4_fp8_marlin_cuda failed with code 2

$ grep "gemm_w4_fp8_marlin_cuda failed" /tmp/pf83-treatment-fp8-direct.log | tail -1
2026-05-10T07:44:00.030529+08:00 ERROR infer::scheduler::cuda::prefill:627
  Request 101380: prefill batch failed: gemm_w4_fp8_marlin_cuda failed with code 2
```

**Every PF8 request failed**. First failure was at Request 1. Last
failure (before server kill) was Request 101380. **100% failure rate
across ~10 min of sustained load**.

CUDA error code 2 = `cudaErrorMemoryAllocation` per CUDA runtime
docs. The PF8 GEMM kernel allocation fails the moment it's invoked.

## Why bench v3-v10 "didn't show this"

Each bench attempt hit the same 100% PF8 failure but:
- v3 baseline INT8 (INFER_MARLIN_W4_FP8_PREFILL=0) → PF8 path NOT
  hit → real numbers captured for INT8
- v4 killed before PF8 ran
- v5 cleanup wedged before PF8 ran
- v6 404 /health blocked validation
- v7/v8/v9 SAVE crash but bench tool counted "0 successful requests"
  (per v3 conc=8 row pattern)
- v10 SAVED files but every concurrency = 0 latency = 0 successful

The savesuccess at v10 + 0-latency table SHOULD have been the
unmistakable signal but I wasted v6-v10 on guidellm CLI debugging
when the real issue was always the kernel.

**Anti-pattern lesson**: when bench reports 0 successful requests,
ALWAYS check server log FIRST before debugging bench infrastructure.

## Why greedy_consistency PASSED (codex's earlier validation)

greedy_consistency runs at conc=1 with small batches. The exact
shape that triggers the kernel allocation failure may not occur:
- batch_size = 1 (single request) vs batch_size > 1 in bench
- prompt_length might differ
- specific tile + thread_m_blocks selection differs

**Anti-pattern lesson**: single-request greedy_consistency PASS is
NOT sufficient validation for a new GEMM kernel. Sustained-load
bench must be part of the gate sequence.

## Why ace3cbe Bug #2 fix didn't catch this

ace3cbe documents codex review caught 3 bugs INCLUDING Bug #2:
"max_par/lock workspace contract underrun". Codex fixed the wrapper
to honor caller's workspace contract. But the fix didn't address the
ALLOCATION failure mode — fix was for buffer-size mismatch, not OOM.

Possible deeper causes:
1. Per-call scratch allocation in `run_marlin_w4_fp8_prefill`
   (linear.rs:1631+) hitting OOM under sustained load
2. CudaSlice allocation pool fragmentation
3. Workspace size estimation wrong for actual prompt+batch combo
4. Marlin template instantiation requiring more shared memory than
   available

## PF8.3 KILL decision per a66d99a §2

| Gate | Result |
|------|--------|
| greedy_consistency | PASS (4.33s, conc=1) |
| TTFT p50 Δ% | UNDEFINED (0 successful requests) |
| ITL p50 | UNDEFINED |
| Throughput | 0 req/s actually completed |
| **Kernel runtime** | **100% FAILURE under sustained load** |

**KILL** per a66d99a §2 KILL threshold "any regression" — going
from working baseline to 100% failure is the worst possible
regression.

## What stays + what changes

**STAYS in tree** (substrate is fine, just kernel needs fix):
- All marlin_pf8/* vendored files
- marlin_w4_fp8_kernel.cu wrapper
- FFI binding gemm_w4_fp8_marlin_cuda
- DeviceMatrix sidecar buffer
- linear.rs dispatch enum
- PF8.5 prep tooling (3fa5e74 + 84d61eb + c382fba + bf47413 + e99e5a5
  + a6cf5ac + 9bb3843 + c6ccd24 + 172c311 + 45579c0)

**CHANGES**:
- `INFER_MARLIN_W4_FP8_PREFILL=1` is OPT-IN-ONLY (already the
  default behavior per db063ff)
- Don't recommend setting env=1 in prod without kernel fix
- Update Task #44 status: PF8.1+2+3+4 LANDED + PF8.5 KILL = PF8 chain
  closed at substrate (subsubsubstep work complete, but value-add
  KILLed by kernel runtime bug)

## Pivot per 2e1e73a decision matrix → KILL branch

Per `docs/research/2026-05-10-post-pf83-next-axis-decision-matrix.md`
KILL branch:
1. ✅ Errors entry naming WHICH gate failed: gemm_w4_fp8_marlin_cuda
   code 2 = cudaErrorMemoryAllocation, 100% failure, root cause TBD
2. ✅ PF8 substrate stays in tree, dispatch stays opt-in (db063ff
   default off)
3. **PIVOT**: #28 Medusa Phase 1.A (Phase 1.A unblocked per
   `8735361`, dataset download verified `arle data download --repo
   lmsys/lmsys-chat-1m --file data.jsonl`)

Codex follow-up:
- Investigate code 2 root cause (per CudaSlice allocator + workspace
  size + shared memory budget)
- Add a mandatory "sustained-load smoke" to PF8.3 fix verification
  (greedy_consistency PASS alone is insufficient — see anti-pattern
  lesson above)

## Skill v1.12.0+ candidate strengthening (proposed #34)

Per session anti-pattern accumulation (now extends ace3cbe + this
KILL):

> **#34 (proposed)**: greedy_consistency single-request PASS is
> NECESSARY but NOT SUFFICIENT for new GEMM kernel substrate. ALWAYS
> follow with sustained-load bench (≥30s, multiple concurrencies)
> before declaring license. PF8.3 (`11763ba` + `ace3cbe`) shipped
> with greedy PASS but failed 100% under sustained load — kernel
> bug only surfaced at conc>=1 + sustained ≥10min.

> **#34b (proposed)**: when bench reports "0 successful requests",
> CHECK SERVER LOG FIRST before debugging bench tool. v6-v10 wasted
> ~30 min on guidellm CLI quirks when the real issue was the kernel
> rejecting every request.

## Cross-references

- `11763ba` PF8.3 substrate (the LANDED commit being KILLed at runtime)
- `ace3cbe` codex review caught 3 bugs (didn't catch this one — single-
  request greedy validates differently from sustained load)
- `2e1e73a` post-PF8.3 next-axis decision matrix (KILL branch path)
- `7f7a58e` PF8.5 v3-v5 cascade failures (red herring; real issue was
  kernel)
- `860ed91`+`596eb51` v6-v10 chain (wasted on bench infrastructure
  while kernel was the actual blocker)
- `8735361` Medusa Phase 1.A pickup chain (next axis)
- a66d99a §2 license matrix (KILL threshold "any regression")
- Server log: /tmp/pf83-treatment-fp8-direct.log (101380 failures,
  10min sustained run)

## Status

**🚫 PF8.3 KILLed at runtime** despite substrate landing successfully.
The compile-clean + clippy-clean + greedy_consistency PASS was a
false-positive license signal. Sustained-load bench reveals 100%
kernel failure (gemm_w4_fp8_marlin_cuda cudaErrorMemoryAllocation).

PF8.5 license sequence DEFERRED indefinitely until kernel fix.
Pivot to #28 Medusa Phase 1.A per 2e1e73a decision matrix.

Per skill v1.11.0+ #28+#31: every claim grounded in raw evidence
(server log grep + count + first/last timestamps + ps cleanup verify
— all THIS tick).
