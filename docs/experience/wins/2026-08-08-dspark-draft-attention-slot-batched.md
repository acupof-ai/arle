# DSpark draft attention was launched per slot at 192 blocks — CUDA, 2026-08-08

> Status: **licensed on the decode-shaped workload.** Serve A/B survives a
> device swap, needle gate clean on both arms. **ITL mean 31.05 → 27.81 ms,
> −10.4%**, which is 57% of what the tick decomposition projected — the
> residual is unexplained, see below.
>
> **This does not update the `baselines.md` champion row.** That row tracks one
> workload, the multi-turn long-agent 32K anchor, and a dataset change is a
> fingerprint change under its rule 3. All decode together is 2.1–6.1% of GPU
> time on the anchor, so the win there is expected to be much smaller or
> absent. Anchor A/B pending; a null result on it would not retract this entry,
> it would bound the claim.

## Problem

`nonpaged_prefill_attention_kernel` is 30.5% of a decode tick on the
decode-shaped c=16 capture — the largest single line
([the re-anchor entry](2026-08-08-decode-shaped-reanchor-draft-attention-is-30pct.md)).
Two 2026-08-01 rewrites of this kernel were reverted after failing to transfer
to the serve.

## Root cause

The cost was never in the kernel's arithmetic.

Every one of the 39,690 launches in the 60 s window carries grid
**(32, 6, 1) = 192 blocks**, on one stream, 7.5 µs apart — 16 slots × 5 draft
layers, one launch per slot. 192 blocks is ~2.5 per SM on an H20's 78.

`dspark_draft_blocks` batched the draft GEMMs and left the ring kernels
per-slot, which its own doc comment stated (`dspark.rs:1266`, "Only the ring
kernels stay per-slot"). Each slot owns a separate `df.k_ctx[li]` ring
allocation, so one launch could not address them all.

**This is why the two 08-01 rewrites missed.** Both were tuned by `ncu` at
**3072** blocks, from a harness sweeping rows 12→96 out of the model config. At
3072 blocks the kernel is genuinely ALU-bound (SM 80.15%, ALU 61.9%, L2 hit
99.58%, DRAM 0.06%). At 192 it is occupancy-starved. Two different kernels as
far as optimization goes.

## Fix

`nonpaged_prefill_attention_kernel` takes an optional array of per-slot ring
base pointers and promotes `blockIdx.z` to a slot axis; the new
`nonpaged_prefill_attention_ring_varlen_batched_cuda` entry launches grid
(heads, block, slots). `dspark.rs` stages each layer's k/v ring bases inside the
existing per-slot loop and runs attention once after it. Grid becomes
(32, 6, 16) = 3072 blocks, one launch per layer instead of 16.

Commit `3a8f99b1f`.

## Measured — pinned shape

`crates/cuda-kernels/tools/nonpaged_attn_bench.cu batch 16 6 <kv_len>`, 16 slots
× block 6, 20 iterations after 3 warmups, H20 GPU 0. **Output bit-identical in
every arm.**

| kv_len | per-slot | batched | Δ |
|---:|---:|---:|---:|
| 512 | 3.545 ms | 1.023 | −71.1% |
| 1024 | 7.203 | 2.057 | −71.4% |
| 1376 | 9.013 | 2.766 | −69.3% |
| 2048 | 13.283 | 4.197 | −68.4% |

Flat across the window, so the win is structural rather than window-dependent.

**The harness sits on the serve's operating point.** At `kv_len` 1376 a
192-block launch costs 563 µs against the serve's 558 µs mode (+0.9%). The
serve duration histogram is bimodal — 47% of launches in the 550–600 µs bin
(full window), the rest spread down as slots fill.

## Results — serve A/B

Matched A/B, two binaries built serially from the same tree — NEW `3a8f99b1f`
(sha `9906b136…`) and BASE `d42c7afc2` = its parent (sha `44a4b874…`). Decode-
shaped c=16, 32 requests, `max_tokens` 4096, temp 0, arms swapped across GPU 0
and GPU 1. Artifacts `/host/draftattn/ab/`.

| arm | GPU | out tok/s | total tok/s | ITL mean | TTFT p50 | accept_rate | gen tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| BASE | 0 | 394.8 | 616.2 | 31.53 ms | 1403.7 ms | 0.4602 | 63181 |
| NEW | 0 | 413.0 | 702.8 | **27.47** | 1436.9 | 0.4430 | 50470 |
| BASE | 1 | 390.2 | 651.4 | 30.57 | 1142.7 | 0.4404 | 52942 |
| NEW | 1 | 457.8 | 743.0 | **28.15** | 1226.7 | 0.4541 | 56874 |

| | GPU 0 | GPU 1 |
|---|---:|---:|
| ITL mean | −12.9% | −7.9% |
| total tok/s | +14.1% | +14.1% |
| out tok/s | +4.6% | +17.3% |
| TTFT p50 | +2.4% | +7.3% |

**NEW wins on both devices on both ITL mean and total tok/s**, so the swap rule
is satisfied. Quote **ITL mean, 31.05 → 27.81 ms, −10.4%**: it is
work-normalized, whereas `out tok/s` spans +4.6% to +17.3% because the arms
generated 50.5k–63.2k tokens (temp-0 generations diverge under MoE
non-determinism). TTFT is a wash-to-slightly-worse, as expected for a
decode-side change.

BASE reproduces the documented decode baseline — ITL mean 31.53 / 30.57 against
31.08 ms recorded on 08-08, and 394.8 / 390.2 out tok/s against 407.97.

`accept_rate` symptom check: BASE {0.4602, 0.4404}, NEW {0.4430, 0.4541}. The
cross-arm gap (≤0.017) is smaller than BASE's own device-to-device spread
(0.0198), so the arms agree.

`BENCH_EXIT=1` in all four arms — the script's `complete != requests` rule
firing on the 4 requests that hit the 4096 cap, plus its repetition detector
(8/6 on BASE, 4/7 on NEW). It fires identically on BASE, so it is a property of
the prompt set at 4096 tokens.

## Only 57% of the projected win transferred

The tick decomposition projects more than was measured, and the gap is not
explained.

```
draft attention   29.57 ms of 96.88 ms GPU-busy, in a 112.33 ms tick span
-69.3%         ->  9.07 ms, saving 20.5 ms
new busy           76.4 ms; idle held at 15.45 ms -> span 91.9 ms  = -18.2%
measured ITL                                                        -10.4%
```

Candidates are that the 13.8% intra-tick idle does not shrink with the work, and
that ITL mean carries time outside the decode tick. Neither is measured. **Cause
unknown** — settle it with an `nsys` capture on NEW at the same shape,
confirming the draft-attention line actually fell to ~9 ms and locating the
residual. Recorded as the follow-up rather than guessed at.

## Correctness gate

`scripts/needle_gate.py`, ladder ×3 same-config repeats, per arm, at depth 0.0
and 0.5. Routing `RAW=1 TEMPLATE=qwen3_nonthink NEEDLE_MAX_TOKENS=64` —
ThinkingCap-Qwen3.6 is a thinking model and the default chat route spends the
token budget inside the reasoning block, which false-fails every length.

All four arms, every length, both depths: **3/0/0 exact, deterministic.**
Decoded output was the needle in all 72 runs. NEW is exactly on BASE's envelope.

**Scope of what that gate covered.** The ladder run was `1000,2000,4000,8000`,
which I specified — `lever_gate.sh` defaults to `115,300,446,2000,8000` and
`needle_gate.py` spans a 241-token boundary. **The rungs below 1000 were not
covered**, and they are the ones that matter most here: at short context the
draft ring runs a small `kv_len` and `ctx_base` clamping engages, which is
where a slot-indexing defect would show. Pending. Capability was not checked
either; an MMLU smoke on both arms is pending.

This is the check that mattered. The harness bit-identity covers the kernel math
only; the failure mode this change can produce lives in the caller — a wrong
slot index, a wrong window-table offset, or the k/v pointer-array halves swapped
— any of which makes one draft slot attend another slot's ring. Well-formed
arithmetic on the wrong data, invisible to a math check.

## Learnings

**Pin a kernel microbench's shape from the trace's grid dims, not from the
model config.** The config gave 3072 blocks and a 2048-token window; the serve
runs 192 blocks and behaves like a 1376-token window. Three rewrites were sized
against the config.

**A microbench's bit-identity is not the correctness gate.** I treated it as
one, because in the harness I filled the per-slot pointer array myself, in the
right order. The gate had to be added to the pod brief after the fact; a
perf-only brief produces a perf-only answer.

**"Only X stays per-slot" in a doc comment is an unpriced cost.** It was written
when the batched draft landed and stayed true for seven weeks, through a
root-cause note ([`project_decode_attention_throws_away_batch`]) that predicted
exactly this symptom and named an 8-second check for it — count attention
launches per step at 8 concurrent decoders. The check was never run on the draft
lane.
