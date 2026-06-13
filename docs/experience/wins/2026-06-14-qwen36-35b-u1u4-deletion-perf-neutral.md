# Qwen3.6-35B-A3B — U1-U4 deletion (#88) is perf- and correctness-neutral

**Date:** 2026-06-14. **Backend:** CUDA, Qwen3.6-35B-A3B (67 GB bf16), H20
GPU 0, TP=1, `--num-slots 8`. **Binary:** `23d6a0b8` (the U1-U4 deletion HEAD),
fresh build in an isolated worktree (`/data01/build/arle-u1u4-verify`, 5m13s,
sccache-warm). **Baseline:** `36b12bc4` from
[`2026-06-12-qwen36-35b-head-csweep.md`](2026-06-12-qwen36-35b-head-csweep.md).

## Goal

Empirically confirm that deleting the validated-losing #88 SGLang kernel-align
lanes (U1 triton-AOT infra, U2 GDN decode, U3 fused_moe, U4 fused add-RMSNorm —
all opt-in, default-OFF; see
[`2026-06-13-qwen36-sgl-kernel-align-validate-bistability.md`](2026-06-13-qwen36-sgl-kernel-align-validate-bistability.md))
did NOT regress the default Qwen3.6-35B path.

## Hypothesis

Perf-neutral **by construction**: every deleted lane was an
`if <flag> { <sgl> } else { <hand> }` collapsing to `<hand>`; flag-OFF was the
default and byte-identical to the current path. The deletion only removed the
unreachable `<sgl>` arms + their scaffolding → post-deletion default == prior
default. The empirical run must reproduce the baseline envelope within noise.

## Params · Env

Verbatim baseline harness (`q36_sweep.sh`, BIN repointed at the worktree):
256-token essay completions, `temperature=0`, 1 warmup + 3 timed reps per c,
c∈{1,2,4,8}, batched (default) vs `ARLE_QWEN35_BATCHED_DECODE=0` sequential,
same-binary same-shell side-by-side. Build env identical to `arle_build.sh`
(`cuda,nccl`, `TORCH_CUDA_ARCH_LIST=9.0`, DeepGEMM-native on) minus the
unrelated DSv4 serve WIP patch. CUDA 12.9, sm_90a. 8×H20 idle, no contention.

## Results

Aggregate decode tok/s (batched = default-ON path):

| c | U1-U4 batched | baseline batched | Δ | U1-U4 seq | baseline seq |
|---|---|---|---|---|---|
| 1 | 93.1 / 93.0 / 93.6 | 93.5 | −0.3% | 94.9 / 94.8 / 95.5 | 93.9 |
| 2 | 151.8 ×3 | 152.3 | −0.3% | 96.0 / 94.9 / 94.4 | 94.3 |
| 4 | 204.8 / 206.6 / 207.5 | 207.5 | ~0% | 96.4 / 95.5 / 96.4 | 95.0 |
| 8 | bimodal {207, 257} | 255.6 | see below | 94.5 / 95.4 / 94.5 | 93.6 |

- **c=1/2/4 land within 0.6% of baseline; seq arm flat ~95 (identical compute
  path).** batched-vs-seq structure holds: +60/+118/+170% at c=2/4/8 — the
  load-bearing same-binary signal the baseline doc names.
- **c=8 is bimodal, identically in both builds.** A focused 8-rep
  characterization on a clean GPU 0 gave **208.5, 256.5, 208.7, 206.2, 210.1,
  256.2, 256.3, 257.8** — two clean clusters (~207 / ~257), 4 reps each. The
  baseline doc recorded the *same* bimodal split inverted (rep1 205.8, then
  256.3/254.9). My peak **257.8 ≥ baseline 255.6**, proving the binary reaches
  the baseline ceiling. A concurrent `nvidia-smi` sampler showed **SM clock
  pinned at 1980 MHz, 100% util, 36-39 °C** across all reps → **zero clock or
  thermal throttling**; the ~1.8 s wall gap (8.0 s vs 9.8 s) is host/scheduler
  packing of the 8 concurrent streams at the slot boundary, a pre-existing
  8-slot ramp artifact present in both builds, not a deletion effect.

## Correctness (needle gate)

`GATE_PROFILE=generic MODEL=Qwen3.6-35B-A3B RAW=1` on the worktree binary —
**byte-identical to the baseline envelope:**

| len | class ×3 | det | output |
|---|---|---|---|
| 115 | miss | DET | `<think>\nHere's a thinking process` (budget think-burn) |
| 300 | miss | DET | same think-burn |
| 446 | partial | DET | `The secret access code is 73829` (budget ran out mid-digit) |
| 2000 | exact | DET | `738291` |
| 8000 | exact | DET | `738291` |

Retrieval exact wherever the budget reaches the answer (2000/8000), secret code
`738291` matches baseline exactly, every class deterministic ×3. The 115/300
misses are the documented `<think>`-burn artifact (coherent prose, not
corruption) — same class as baseline. No correctness flag.

## Problems

- Pod is offline (GitHub TLS recv error). Transferred the 9 unpushed commits
  via a 665K `git bundle` (prereq `7f305a1e`, a commit the pod object store
  had) → `tn push` → `kubectl cp` → `git fetch <bundle>` → `git worktree add
  23d6a0b8`. Isolated worktree keeps ckl's dirty `/data01/build/arle` (DSv4
  WIP) untouched.
- c=8 single-sweep first looked like a −19% drop until the 8-rep + GPU-clock
  characterization showed bimodality with full-clock in both modes — the
  framing trap the c=8 baseline note already warned about.

## Rule

A deletion of opt-in default-OFF arms that collapse to the existing default is
perf-neutral by construction, but "再次验证性能" still means run the sweep:
confirm the default envelope (c=1/2/4 dead-on, ±0.6%) AND the within-binary
batched-vs-seq structure, AND resolve any single-sweep concurrency anomaly with
a multi-rep distribution + GPU-clock check before calling it a regression
(c=8 {207, 257} bimodal, clock pinned 1980 MHz = scheduler packing, not compute).
