# Qwen3.8-Flash-Next first token on Vulkan: five oracle-gated stages, one adversarial audit, one wrong premise

## Context / Goal

Qwen3.8-Flash-Next (`qwen4_exp`, ~180B total / ~6B active) is architecturally
unlike anything the Vulkan lane served: hyper-connections widen the residual to
4×hidden with grouped-norm gated mixing, a 47.68 GiB n-gram/PLE lookup table
feeds layer 1, the MoE routes 512 experts top-10 at NVFP4, attention alternates
36 gated-delta linear layers with 12 full layers at partial rotary 0.25, and
the checkpoint is 206 safetensors shards — no GGUF exists. Goal: a first token
with every stage's arithmetic pinned, on a 74.43 GiB-heap Strix Halo.

## Approach

Five stages (S0 guards → S1 dtypes/router → S2 reader/classifier/slabs → S3
NVFP4 GEMV + host oracles → S4 fused kernels), each landing with an oracle or a
negative control before the next started. Then, concurrently with S5 (config /
upload / n-gram gather / forward), a five-auditor adversarial audit of S0-S4
whose brief centered on one question: **if the oracle is wrong, the kernel
matches it perfectly and everything is wrong together** — so audit the oracles
against the reference first, then the kernels against the oracles.

## Env

- Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151), 128 GB LPDDR5X @ 256 GB/s
- Windows 11, AMD 26.7.1 (LLPC); Armoury Crate Performance, on AC
- Checkpoint: `qwen3.8-flash-next-nvfp4` (125.96 GiB, 296,475 tensors)
- Date: 2026-08-27

## Results

**`The capital of France is` → top-1 ` Paris` (15.76)**, ` located` also in
the top-5. Full 48-layer forward, load 216.9 s, 0.68 s/token on a
deliberately synchronous bring-up path (per-stage submits, host dense) —
an existence proof, not a baseline row.

Per-layer parity, layers 0/1/3 with all 512 experts, 21 stages device-vs-host:
≤ 8.7e-5 per-element everywhere except the linear chain downstream of its bf16
conv quantizer (4.8e-3 = one bf16 ulp; 3.3e-3 on the vector-scale metric).
Mutations caught at distance: identity head-permutation → rel 280; dropped
`weight_scale_2` fusion → rel 3.5e13.

### The audit's three landmines, and where they died

| # | landmine | outcome |
| --- | --- | --- |
| 1 | 72.00 GiB slab plan vs **70.71 GiB driver budget** (`ensure_fits` checks heap *size* 74.43; UMA fails by silent page demotion, not OOM) | routed around: `HybridExperts` drops the F16 dense tier (no GEMV consumer) + lm_head → 65.9 GiB, checked at load against `VK_EXT_memory_budget` budget−usage |
| 2 | the `1 + w` RMSNorm fold is per-family — PLE gate norms apply the bias **in-shader** and must upload raw | already right in `qwen4_upload::folds_norm_bias` (which also caught `linear_attn.norm`, un-flagged by the audit); pinned as a living parity assert |
| 3 | `norm_topk_prob` absent from config.json; HF default `true`; a false default attenuates MoE ~2.5×/layer, finite and coherent | already right in `qwen4_config` (`unwrap_or(true)`); device router weights assert sum-to-1 ± 1e-5 |

The audit's other yield: five tests-that-cannot-fail, worst being the NVFP4
repack fixture — periodic with period 8 bytes, so dropping the sub-block index,
dropping the block index, or pinning the source row all passed. Rewritten
aperiodic (`(5g + g/16) mod 16` over the global index); each mutation re-run
and each now fails.

### The wrong premise: GGUF and HF do not share a K/V head map

The S5 brief asserted the linear-attention shape "is exactly the form ARLE
already consumes." The *shape* is (kd=vd=128, nk=16, nv=48). The *head map* is
not: `qwen35_gated_delta_net.comp` tiles key heads over value heads
(`k = v % nk`, matching GGUF's converter layout), while this HF checkpoint
needs `repeat_interleave` (`k = v / 3`). No K-side permutation reconciles the
two — one key slot would need three different key heads. The fix is a
load-time **value**-head permutation (`v_slot_perm(s) = (s%nk)·(nv/nk) + s/nk`)
over qkv/z/a/b rows, conv channels, `A_log`/`dt_bias`, and `out_proj` columns;
the kernel is reused verbatim. This is the forward-convention trap of
[[reference_gguf_vs_hf_forward_conventions]] in a new costume: the converter
bakes in a layout, and a weight consumer built against converted checkpoints
inherits it invisibly.

### Measurements that changed decisions

- **Budget ≠ size**: heap 1 reports 74.43 GiB but budgets 70.71
  (`memory_budgets()` via `VK_EXT_memory_budget`, now a device test).
- **GPU reads the host heap at 0.5% cost** (204.9 device / 203.9 host-WC /
  198.9 host-cached GB/s, 512 MiB streaming): on UMA, spilling read-only
  weights off the device heap is a residency-tag change, not a requant. This
  is what makes the full-residency replan (return the F16 tier to device
  bindings or host spill, either way GEMV-consumable) cheap.

## Verification

| check | result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `clippy -p infer-vulkan --all-targets -D warnings`, feature on AND off | PASS |
| lib suite | 133/133 (vulkan), 119/119 (without) |
| parity harness re-run at the landed commit | PASS, same error profile |
| full first-token forward re-run at the landed commit | PASS, ` Paris` top-1 |

## Learnings

**Audit the oracles, not just the kernels — and time-box it to run
concurrently with the next stage.** The audit confirmed every numeric claim it
could check against a primary source (NVFP4 nibble order, FP8 over all 256
codes, S4 goldens re-derived from raw shards), which is precisely what made
the first token *mean* something. Its three landmines were all conventions the
next stage was about to guess at, not defects in shipped code — the highest
value came from injecting them into the in-flight S5 forward brief before it
started debugging, not from re-reviewing what had landed.

**Two of three landmines were defused by agents that never saw the audit.**
The upload and config briefs stated the contracts narrowly ("hc_norm: store
1+w"; "read the file, do not guess the shape"), and the agents read the
reference themselves. Precise briefs beat post-hoc review.

**A parity harness converts "reviewed carefully" into "measured"** — and it
needs two error metrics. Downstream of a discrete quantizer (the conv's bf16
round), a 1e-4 input difference legally flips a boundary channel by a full
ulp; per-element relative error reads 4e-3 while the vector-scale metric stays
at 3e-3 and upstream stages stay at 1e-7. One metric alone either false-alarms
or hides.

**On UMA, plan against the driver's budget and treat heaps as fungible for
read-only weights.** Size-based planning over-commits by gigabytes and fails
by silent bandwidth collapse; meanwhile the "device vs host" placement
question for streamed weights is worth 0.5%, measured.

## Rule

Before wiring a weight consumer to an existing kernel, diff the **layout
conventions**, not just the shapes: converter-era assumptions (head maps,
nibble order, norm-bias folds) live in the weights, not the code, and a
shape-compatible checkpoint can still need a load-time permutation. The test
that catches this class is a parity harness against a host transcription of
the *reference* — run per-stage, with a mutation pass to prove each assert can
fail.
