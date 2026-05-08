# M_pf-graph v2 — Phase 0 KILL postmortem build-on plan

> Supersedes the single-bucket Phase 0 substrate (KILLED in
> [`docs/experience/errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md`](../experience/errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md)).
> Per user directive 2026-05-08 "持续积累才能突破 + 速度第一",
> this plan accumulates Phase 0's lesson into a corrected v2 instead of
> abandoning the graph-capture path. The 415 LOC substrate was reverted
> and must be re-implemented; pattern is in the KILL entry's by-file
> review (`344f606`).
>
> Master strategy: [§7.2 P0.1 reframe](../projects/2026-05-07-arle-master-strategy.md#72-p01).

## Why Phase 0 failed (root cause from `8b4a03b`)

- 4097-token request decomposes to 2048+2048+1 → tail forces eager
- single-key graph cache → recapture on alternating start_pos
- `INFER_PREFILL_GRAPH=1` clamped prefill envelope → serialized c=4 admission
- BF16-forced KV (graph compat) ≠ production auto-FP8 → KV pressure +
  prefix-cache fallbacks
- Result: -0.8% TTFT vs ARLE pre-Phase 0 (well under +10% license)

## Phase 0v2 fix list (codex's own)

1. Multi-key graph cache (HashMap<PrefillGraphKey, CudaGraph>) replaces
   single-key `Option<PrefillGraphKey>`
2. Tail handling — detect short-tail prefill, either:
   - (a) merge into prior chunk if scheduler shape allows, OR
   - (b) capture a separate small-bucket graph (e.g. 256, 512 buckets), OR
   - (c) skip graph for sub-bucket tails (eager fallback)
3. FP8 paged-KV graph support — match production auto-FP8 baseline so
   the A/B is apples-to-apples without forcing BF16
4. Remove `INFER_PREFILL_GRAPH=1` prefill envelope clamp — let
   scheduler admit c=4 normally; graph capture should not change
   admission policy
5. **nsys evidence prerequisite**: dispatch / launch overhead must
   be measured to be ≥ 30% of 4k prefill step time before substrate
   investment licenses Phase 0v2.B-D

## Phase 0v2.A — nsys baseline (PREREQUISITE, ~30 min)

**Owner**: codex 0:0. Output: nsys trace + launch-density analysis.

```bash
# Start ARLE with production defaults (auto-FP8 KV) — match real
# user shape, not BF16-forced.
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer --model-path infer/models/Qwen3-4B \
  --port 8000 --num-slots 8 --max-seq-len 5120

# In a second terminal:
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/profile_nsys_guidellm.sh m_pf_graph_v2_baseline \
  --concurrencies 4 --max-seconds 60 \
  --data 'prompt_tokens=4096,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_min=256,output_tokens_max=256'
```

If `scripts/profile_nsys_guidellm.sh` also has stale syntax (mirror of
the ncu wrapper bug found in the E2 errors entry), fix it OR drop to
direct `nsys profile --capture-range=cudaProfilerApi --output=baseline
./target/release/infer ...` + cuProfilerStart/Stop signal trigger
(per master §4.2 M_nsys P0 substrate).

**License gate (Phase 0v2.A → Phase 0v2.B)**:

- Total CUDA launch density across 4k prefill ≥ 200 launches/sec
- Cumulative cudaLaunchKernel host time ≥ 30% of step time
- Identify top 5 launches by host time (likely norm + QKV proj +
  RoPE + KV write + attention + output proj per layer × 36 layers)

Document in `docs/experience/wins/2026-05-08-m_pf_graph_v2-nsys-baseline.md`
or `errors/` if dispatch is < 30% (then Phase 0v2 is also KILLED — pivot
to algorithmic restructure or accept 2× slower than SGLang).

## Phase 0v2.B — multi-key cache + tail handling (~6-8 hr, ~500 LOC)

Trigger: A passes license. Owner: codex 0:0.

Files (mirror Phase 0 structure per `344f606` review):
- `infer/src/model/qwen3/prefill.rs`: `Option<PrefillGraphKey>` →
  `HashMap<PrefillGraphKey, GraphResources>`. LRU bound (e.g. 8 entries).
- Tail strategy: (a) prefer merge into prior chunk via scheduler hint;
  fallback to (c) eager for sub-256 tails. (b) multi-bucket graph
  deferred to v3 — keep v2 scope tight.
- Drop `INFER_PREFILL_GRAPH=1` envelope clamp from `infer/src/main.rs`.
- Keep opt-in env (`INFER_PREFILL_GRAPH=1`) gating; default OFF.

## Phase 0v2.C — FP8 paged-KV graph support (~4-6 hr)

Trigger: B implementation green. Owner: codex 0:0.

The current FP8 paged decode kernel (`batch_decode_paged_hd128_fp8.py`)
is already production-default. Phase 0v2.C extends graph capture to
work over auto-FP8 KV path — `kv_cache_to_paged_fp8` + scale tensor
plumbing must be reachable from the captured graph.

If the FP8 prefill path uses the same BF16 prefill kernel + post-write
quant (likely), this is just plumbing — kv tier code paths under graph
capture mode.

## Phase 0v2.D — license bench (~30 min)

Trigger: C done. Owner: codex 0:0.

```bash
# Default (production auto-FP8, no graph)
scripts/bench_guidellm.sh m_pf_graph_v2_default --concurrencies 4 \
  --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,...,output_tokens=256,...'

# Graph on (production auto-FP8 + graph)
INFER_PREFILL_GRAPH=1 scripts/bench_guidellm.sh m_pf_graph_v2_on \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,...,output_tokens=256,...'
```

License: TTFT p50 Δ ≥ +10% (Phase 0 license, unchanged); ITL Δ
within ±2% noise; tok/s no regression. nsys re-trace shows launch
density drop on graph-on arm.

## Kill criteria (any → KILL Phase 0v2 entirely)

1. Phase 0v2.A nsys evidence shows dispatch < 30% of step → kernel
   time is binding after all → kernel-side or algorithmic work, not graph
2. Phase 0v2.B implementation > 1000 LOC (vs Phase 0's 415) — scope
   creep means architecture mismatch
3. Phase 0v2.D bench Δ < +5% TTFT — same noise band as Phase 0,
   accept the lesson and pivot to algorithmic restructure

## Cross-references

- Phase 0 substrate by-file review: `docs/research/2026-05-08-mpgc-phase0-415loc-review.md`
  (commit `344f606`) — has the structural pattern to re-implement
- Phase 0 KILL entry: `docs/experience/errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md`
- E2 BN=32 KILL entry: `docs/experience/errors/2026-05-08-e2-prefill-bn32-failed-kernel-time-not-binding.md`
- master strategy §3.3 + §6.2 + §7.2

## Rule

Phase 0v2 must NOT regress production auto-FP8 baseline at any step;
matched A/B is FP8-vs-FP8, never BF16-forced. Dispatch overhead is
the explicit binding-constraint hypothesis being tested — Phase 0v2.A
nsys is the gate, not optional.
