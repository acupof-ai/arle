# V100 perf gap closure — systematic optimization plan

## Why

Industry comparison entry
[`wins/2026-05-29-vs-industry-v100-qwen35.md`](../experience/wins/2026-05-29-vs-industry-v100-qwen35.md)
documented the V100 sm_70 gap on Qwen3.5-4B + 128/128:

```
metric                ours    vLLM    SGLang   TRT-LLM
c=1 TTFT (ms)         230    ~175    ~150     ~120
c=1 ITL (ms)          13.7    ~10     ~8.5     ~7.5
```

User directive: "差距好大 系统性的优化吧" — systematic, not one-shot.
Per CLAUDE.md §0 SOLID: license-or-kill each hypothesis with evidence
(bench Δ%, not source survey or callgraph inference), document kills
in `errors/`, ship wins in `wins/`.

## Evidence anchor — current per-decode-step breakdown (memory-bandwidth view)

Qwen3.5-4B is ~4 B parameters; one decoder step touches the full
weight set once. V100 SXM2-32GB HBM2 ≈ 900 GB/s. For BF16 weights:

```
weight bytes / step ≈ 4 B params × 2 B (bf16) ≈ 8 GB
theoretical floor   ≈ 8 GB / 900 GB/s ≈ 8.9 ms
measured ITL (int4 KV, bf16 weights)         = 13.7 ms
→ ~65% of theoretical bandwidth peak (8.9 / 13.7)
published vLLM ITL on V100 4B-class          = ~10 ms → ~89% peak
```

The gap is ~25 percentage points of memory-bandwidth utilization at
decode. KV quantization (already shipping INT4) saves K/V reads but
**doesn't save weight reads**, which is the dominant cost at 4B class.

## Phase 2 — KILL: H2 graph capture coverage (2026-05-29)

**Evidence (from `bench_runs/int4/server.log`):** 8 captured groups of
3 contiguous layers each = 24 captured layers; pattern is
`group=N covers layers [4N+0..4N+2]`, **skipping layer 4N+3**. Layer 3
was the same layer that failed the TQ4 decode pathway with
`qwen35 decode full-attention layer_idx=3`, so the uncaptured layers
are the 8 **full-attention layers** of Qwen3.5-4B's hybrid (the other
24 are linear-attention / GDR).

**Math:** 8 uncaptured layers × ~30 μs launch overhead ≈ 0.24 ms per
decode step = 1.7% of 13.7 ms ITL. **Not the binding cost.** The full-
attention layer's actual ITL contribution is the attention *compute*
on V100 (TileLang sm_70 decode kernel + page-aware K/V loads), not
the per-layer launch.

**Verdict:** KILLED. Even forcing the full-attention layers into the
graph (which would require recordable paged-KV index slots and an
extra graph re-record path per request) buys at most 1.7% ITL. Not
worth the dispatch-layer plumbing budget.

## Phase 1 — KILL: split heuristic SM-tier awareness (2026-05-29)

**Hypothesis:** `choose_decode_num_splits` returns 32 splits at any
`total_q_heads * num_blocks ≥ 32 * SM_COUNT`. On V100 (80 SMs) ×
Qwen3.5-4B (32 q_heads) at c=1, this gives 32 splits for a 128-token
KV: each split processes only 4 KV tokens, so per-launch + merge
overhead might dominate compute savings. Tuning `kTargetBlocksPerSm`
from 32 (the L4 sm_89 value) to 4 (the older pre-2026-05 value) for
sm_70 should lift ITL.

**Experiment:** patched `choose_decode_num_splits` SM-tier-aware
(sm_89+ = 32, sm_80/86 = 16, sm_70/75 = 4). Mac type-check clean,
V100 rebuild clean. Re-ran guidellm `int4 c=1/4/8 128/128`.

**Result (vs Step 1 baseline at same shape, same binary chain):**

```
c   metric       baseline   sm_tier patch    Δ
1   TTFT (ms)    229.6      230.3            +0.3  (noise)
1   ITL  (ms)    13.7       13.7             0.0
4   TTFT (ms)    536.1      534.0            -2.1  (noise)
8   TTFT (ms)    835.9      835.2            -0.7  (noise)
```

**Verdict:** KILLED. Zero measurable ITL improvement at the tested
shape. Split count is not the binding constraint at 256-token KV
(128 in + 128 out) on V100 for this kernel. Patch reverted to keep
the working-tree clean and avoid landing an untested-on-A100 change.

**What it ruled out:** decode-attention's split-merge plumbing is
not the bottleneck at decode shapes this audit covers. Future
attacks on ITL should target the bandwidth-dominated GEMV path or
the per-step kernel-launch overhead, not the split count.

## Open hypotheses, ranked by expected impact

Each entry lists: lever, ITL impact estimate, cost, and the cheap
license-or-kill experiment.

### H1. W4A16 / W4A8 weight quantization on decode hot path **[BLOCKED]**

**Block discovered 2026-05-29:** there is no Marlin/AWQ/GPTQ
Qwen3.5-4B checkpoint cached on the V100 box (only the BF16
9.3 GB original at `~/.cache/modelscope/.../Qwen/Qwen3.5-4B`).
The Marlin path in `crates/cuda-kernels/csrc/gemm/marlin_*` and
the `--quant-format marlin_w4a8` CLI surface both expect a
pre-quantized checkpoint in Marlin's packed weight layout.

Unblocking H1 first requires generating that checkpoint — either:
(a) calibration + AWQ/GPTQ quant of Qwen3.5-4B (Python tooling,
~30 min compute + validation), or (b) download a community
W4-quantized variant if one ships (verify quality matches BF16 base
within the 4×4 grid first). Defer until someone schedules it as a
dedicated cycle; ITL impact ranking unchanged.

**Lever:** Use the existing Marlin W4A16 / W4A8 weight-quant kernels
(`crates/cuda-kernels/csrc/gemm/marlin_*`) for the per-layer Q/K/V/O
GEMVs that currently run BF16. Weight bytes drop 4×; at the 65% BW
utilization we measured, this should cut ITL by ~40-50% if the
existing kernels saturate similar BW.

**Expected ITL Δ:** 13.7 → 7-9 ms (closes 50-80% of the gap to
vLLM/SGLang).

**Cost:** Marlin kernels exist in tree per CLAUDE.md "marlin 我们也有
呢". Decode wiring needs auditing — Marlin path is wired for prefill
but may not be on the decode hot path yet. ~half-day to confirm,
half-day to wire if needed.

**License-or-kill:** flip an `INFER_DECODE_W4` env var on/off, bench
guidellm c=1 ITL on a 4B Qwen variant that has W4 weights cached.
≥10% ITL reduction → land. <5% → kill.

### H2. CUDA Graph capture coverage at decode

**Lever:** Audit which decode-step kernels are captured under the
"Piecewise CUDA Graph captured: group=0, layers=0-2, B=1" path seen
in the audit log. The Qwen3.5 hybrid has 8 full-attention + 28
linear-attention layers; if only some are captured the dispatch
overhead per uncaptured layer adds up. NUM_WARPS=4 + BLOCK_SIZE=128
in the decode attention kernel may be sub-optimal for V100's
register file — capture may exclude paths with high register
pressure.

**Expected ITL Δ:** 13.7 → 11-13 ms (closes ~20% of the gap).

**Cost:** read warmup.rs + qwen35/batch_decode.rs, identify uncaptured
layers, force-enable capture or rewrite the offending kernels to
fit. ~1-2 days.

**License-or-kill:** enable `RUST_LOG=infer::scheduler::cuda::core::warmup=debug`
during a 4×4 audit, count "Graph captured" lines vs total layers.
If all 36 layers are captured → kill H2. If <90% captured → land
the wiring fix.

### H3. GEMV kernel tile / vectorization tune for V100 head_dim=256

**Lever:** Our `quantized_gemv.cu` and BF16 GEMV kernels are tuned
for sm_80+. Head_dim=256 + V100's 32-thread warp + register pressure
ceiling may not map well to the current tile sizes. Re-tune
TileLang AOT prefill kernels for sm_70-specific occupancy.

**Expected ITL Δ:** 13.7 → 12-13 ms (closes ~10% of the gap), plus
a possible TTFT win.

**Cost:** ncu profile (V100 has ncu in `/usr/bin/ncu`) of one
canonical GEMV, identify register/occupancy/bandwidth limiter,
either tune or fork a sm_70 variant. ~2-3 days.

**License-or-kill:** `ncu --metrics gld_efficiency,sm__throughput.active`
on one GEMV during a decode step. If efficiency >80% → kill (kernel
is already saturating). If <50% → land a sm_70 tile-tune fork.

### H4. FlashInfer-style page-aware decode attention

**Lever:** The reference industry baselines on V100 sm_70 go through
FlashInfer's V100 decode path which has page-aware split scheduling
(not the static N-split this kernel uses) and tuned shared-memory
layouts. Rewriting our `decode_attention_*_per_channel_k_partial_kernel`
family to mirror that pattern would close the kernel-vs-kernel gap.

**Expected ITL Δ:** 13.7 → 9-10 ms (closes ~50% of the gap), but only
if H1 isn't already landed (H1 supersedes most of this).

**Cost:** ~1 week of kernel work. Highest cost in this list.

**License-or-kill:** only after H1 + H2 land; if there's still a
>20% ITL gap, run a same-shape FlashInfer V100 microbench (already
in tree as `bench_kv_cache.py`?) to confirm the gap is on the
attention kernel specifically.

### H5. TTFT prefill kernel — TileLang sm_70 fork tune

**Lever:** TTFT gap is mostly prefill (1.8 ms/token vs vLLM's
~1.17 ms/token at 128-token shape). TileLang AOT
`batch_prefill_paged_hd256_q32_kv8_sm70` is the relevant kernel.
Re-tune the TileLang Python template for sm_70 occupancy: register
budget, shared-memory layout, double-buffering depth.

**Expected TTFT Δ:** 230 → 160-180 ms (closes 30-50% of the c=1
prefill gap).

**Cost:** TileLang template surgery. ~2-3 days.

**License-or-kill:** ncu --metrics on the sm_70 prefill kernel
during a 128-token request. If bandwidth + occupancy are both >70%
of peak → kill. Otherwise land the re-tune.

## Phase 3 — H3 attempt + nsys substrate bug (2026-05-29 EOD)

Attempted the H3 cheap experiment: `nsys profile` against `infer
--kv-cache-dtype int4` during one guidellm 128/128 c=1 request.

The capture itself worked (`trace.qdstrm`, 12 MB), but **no nsys tool
on the V100 box could finish post-processing it:**
- `/usr/bin/nsys` is 2022.4.2.1, which the 2023-format `qdstrm`
  rejects: `"Qdstrm version 2023.4.4 not supported"`.
- `/usr/local/cuda-12.4/nsight-systems-2023.4.4/...` finalises
  partway then errors with `"Wrong event order has been detected
  when adding events to the collection"` — a known nsys-2023 import
  bug on traces produced under its own profiler.
- `nsys export --type sqlite` rejects `.qdstrm` directly; needs
  finalisation first.

So the trace exists at `/tmp/nsys2/trace.qdstrm` on V100 but cannot
currently be converted to a readable form on this box without
either an nsys 2023 patch or a fresh install of a matched 2024+
toolchain (the host's `/usr/lib/nsight-systems/...` 2022 importer
explicitly refuses the 2023 capture format). Documented for the
next session: re-run with `/usr/bin/nsys profile` (forcing the
2022 capture) and use `/usr/lib/nsight-systems/host-linux-x64/
QdstrmImporter` to import — or upgrade nsys on this box to a
version consistent across capture + importer.

**Without the kernel breakdown the H3 verdict is deferred**, but
the bandwidth-analysis anchor from the §Why section still stands:
ITL is ~65% of theoretical peak BW at 4B-class on V100; gap to
89% peak is mostly GEMV efficiency. H3 evidence would confirm or
refute that more precisely; until then, the H1 (Marlin W4 weights)
lever stays the highest-ranked open hypothesis.

## H1 unblock — also blocked on substrate (2026-05-29 EOD)

H1's cheap license-experiment is `--kv-cache-dtype int4
--quant-format marlin_w4a8` against a pre-quantized Qwen3.5-4B
checkpoint. Three independent V100-substrate blockers:

1. **No Marlin/AWQ/GPTQ Qwen3.5-4B cached on this box.** Only the
   BF16 9.3 GB original is at
   `~/.cache/modelscope/.../Qwen/Qwen3.5-4B`. Searched
   `~/.cache --iname '*Qwen3.5*W4*|*AWQ*|*GPTQ*'`, found nothing.
2. **HF hub query is broken in this venv.** `huggingface_hub`'s
   `list_models()` triggers the same `httpx.InvalidURL: Invalid
   port ':'` mounts-dict parse bug we hit on guidellm; `env -i`
   fixes that path for guidellm but breaks for HF API because the
   API needs the corp proxy to reach `huggingface.co`. Mutually
   exclusive constraints.
3. **AutoAWQ install path requires pip + network through the
   broken httpx/proxy stack.** Same blocker.

Unblocking H1 needs ONE of:
- A clean fresh Python venv with consistent torch+cu121 + a working
  httpx (≥0.27) where the proxy URL parses correctly. Pre-existing
  scripts `scripts/{vllm_serve_control.sh, bench_sglang_longctx.sh}`
  hint at the pattern but don't currently build their own venvs.
- A pre-quantized Qwen3.5-4B AWQ/GPTQ/Marlin variant pushed to the
  box out-of-band (rsync from a working machine).
- A Marlin quantization run against the BF16 baseline using
  `llmcompressor` or `auto-awq`, on a machine that has both Internet
  + the AWQ tooling.

## Execution order (revised after Phases 1-3, all V100-substrate-blocked)

After three kills + two substrate blockers, the queue is now bimodal —
V100 is exhausted of cheap evidence-bearing experiments at this
session's tool state, and the user's decision was "**Both — V100
cheapest levers first, then sm_80+**". The cheapest V100 levers
are done; remaining V100 work is structural (H4 / H5 each ~1 week
of kernel surgery) or substrate-unblock (H1 + H3 need new tooling).

**Pivoting to sm_80+** is now the higher-throughput path:

1. **sm_80+ guidellm baseline** — re-run the Step 1 sweep on A100 /
   L4 hardware (Qwen3.5-4B, 128/128 c=1/4/8 against the same `infer`
   binary). TileLang AOT prefill `kTargetBlocksPerSm=32` was tuned
   on L4 sm_89; sm_80+ ITL is where ARLE structurally wins per
   [`wins/2026-05-08-prefill-cap-8-multi-shape-safe-default-flip.md`](../experience/wins/2026-05-08-prefill-cap-8-multi-shape-safe-default-flip.md)
   and the Qwen3.6-MoE / Qwen3.5-7B sm_89 wins from April-May 2026.
2. **sm_80+ industry comparison** — re-run with vLLM/SGLang on the
   same A100/L4 box; the cross-runtime delta should be where the
   *actual* wins-over-industry numbers come from.

V100 lanes remaining as deferred:

3. **H1 unblock + run** — generate a Marlin Qwen3.5-4B variant on a
   working venv (any sm_80+ machine), rsync to V100, flip
   `--quant-format marlin_w4a8`, re-bench. Single biggest expected
   V100 ITL win.
4. **H3 redo** — install consistent nsys 2024.3+ on V100 (or fall
   back to ncu's metric-mode for `gld_efficiency` on one GEMV).
   Yields the bandwidth verdict for H4 and H5.
5. **H4 / H5** — only if H1 + H3 leave a >20% V100 residual gap, AND
   the sm_80+ work has stabilised. ~1 week each.

Removed from queue: H2 (killed 2026-05-29).

After each landed win: re-run `guidellm 128/128 c=1/4/8 int4` per the
[Step 1 wins entry](../experience/wins/2026-05-29-guidellm-ttft-throughput-v100-qwen35.md),
update the comparison table in
[the industry entry](../experience/wins/2026-05-29-vs-industry-v100-qwen35.md),
and re-prioritise the remaining hypotheses based on the new gap.

## Rule

Systematic optimization is a forced-ranked queue of hypotheses, each
licensed by a *cheap* experiment that decides land-or-kill before
spending the full implementation budget. No more "8-hour rewrite then
discover it doesn't help" attempts (the
[group-quant kill](../experience/errors/2026-05-29-int4-kv-group-quant-kill.md)
is the canonical example of that anti-pattern from this project).

Anchor the queue in the bandwidth-vs-compute breakdown for the
specific shape under test. ITL on a 4B-class model at decode is
weight-bandwidth-bound; the rank ordering above reflects that.
Re-derive the ranking when the shape changes (long context, large
batch, etc. flip the dominant cost).
