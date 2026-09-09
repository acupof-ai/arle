# Memory is the product

Design note 4 of 5 ([plan](../plans/2026-09-02-design-theses.md)). Metal
and CUDA, Qwen3.5 / Qwen3.6 / Qwen3.8. Date: 2026-09-09.

## Problem

A decoder is bandwidth-bound: every emitted token re-reads the weights, so
decode throughput is set by bytes moved per token divided by bus bandwidth.
On the same hardware, two memory numbers decide what the runtime can sell:
**resident bytes** — what fits alongside the OS and the KV pool — and
**bytes read per token** — what each token costs. A coding-agent server on a
48 GB unified-memory Mac keeps a 35B MoE resident and a multi-turn KV pool;
on a 32 GB H20, a 23.4 GB 4-bit checkpoint once did not fit while the 30.9 GB
8-bit one did. Every feature is judged by both numbers, and a feature that
improves one while quietly degrading the other is a regression.

## Standard practice and where it fails

Two heuristics dominate. The first is the fraction: give the KV pool 90% of
what is left after weights (SGLang `mem_fraction_static = 0.9`) and treat the
rest as undifferentiated headroom. The fraction hides which bytes are fixed
and which are optional, so when the fixed set grows — a repack's source, a
prefill working set — the KV pool shrinks silently and the symptom shows up
as recompute, not as a memory line. The second is the repack that keeps its
source: quantized layouts are built at load from the checkpoint bytes, and
keeping the pre-repack tensors around is the easy path for any dispatch lane
that still reads them. The second copy is invisible in every weights-only
accounting — the file is 23 GB, the resident set is 42 GB, and nothing in
between names the difference.

## The design

**The solve is printed, not applied silently.** `plan_resource_budget`
([`resource.rs:252`](../../crates/infer-metal/src/resource.rs)) computes the
memory limit as `min(working set, total − system reserve, available −
anti-swap reserve)` (`:308`), subtracts the fixed set — weights, runtime
headroom, static state — and budgets KV as `floor(limit ×
mem_fraction_static) − fixed` (`:349`), clamping the page count to what the
budget holds (`:375`). `arle --doctor` prints the whole solve — weights,
headroom, anti-swap reserve, KV budget, planned slots — so the two numbers
are readable before a server is started, and the rejection path prints the
itemized fixed requirement instead of a verdict.

**One resident layout per weight.** The Marlin repack frees its source
inline, per weight — load, repack, free, next — so peak is one weight's two
copies, not the model's
([`loader.rs:2781`](../../crates/infer-cuda/src/loader.rs)), and a
final-state gate (`validate_quant_linear_storage`) fails the load if any
dispatch lane would read a released layout.

**Derived operands live in scratch, rebuilt per call.** The DeepGEMM
prefill path widens NVFP4 to E4M3 per call from Marlin's resident layout;
the widened copy is never resident
([wins 2026-08-20](../experience/wins/2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md)).

**The working set is itemized, not fractional.** The DSv4 slot solve
enumerates the prefill transient — MoE and attention scratch — in
`prefill_transient_reserve_bytes`
([`budget.rs:290`](../../crates/infer-cuda/src/dsv4/budget.rs)) and
subtracts it before solving slots (`:520`), which is what took the 27B from
18 slots plus an OOM to 17 slots 16/16
([wins 2026-08-24](../experience/wins/2026-08-24-dsv4-budget-prefill-reserve.md)).

**Bytes read per token is a first-class axis.** NVFP4 moves 56% of the
weight bytes per layer that FP8 moves (150.4 MB vs 267.5 MB) and decodes
40% faster at c=1 on the same H20
([baselines](../baselines.md)); INT8/FP8 paged KV runs on tensor cores
(`paged_attention_quantized_fa3.cu`), cutting the attention read the same
way.

## The failure

The Marlin repack stored the model twice
([wins 2026-08-20](../experience/wins/2026-08-20-marlin-source-freed-18gb.md)).
Qwen3.8-27B-NVFP4, a 23.42 GB file, sat at 42.08 GB resident; its KV pool
was 281,577 tokens against the FP8 build's 593,995 on the same card, and a
16-conversation × 8-turn workload paid 24× full recompute of a 33K prefix.
The source was kept on purpose: `QWEN_MARLIN_MAX_M = 1024` sent 2048-token
prefill chunks to a dequant→BF16 GEMM that reads the pre-repack bytes, so
the model was stored twice to keep one GEMM 12–21% faster. The fix freed the
source per weight, claimed every M for the repacked layout, and repaired
the second dispatch lane — the single-row GEMV path, which had no Marlin at
all and was found by a crash, not by reading. The mechanism was violated
again within hours: the same day's prefill work needed those pre-repack
bytes back for the BF16-rate GEMM, and the chosen path derived the widened
operand per call into scratch rather than holding a second resident copy.

## The number

`arle --doctor` on Qwen3.5-0.8B-MLX-4bit, M4 Pro 48 GB, prints: weights
0.6 GiB, runtime headroom 4.0 GiB, static state 4,770 MiB, KV budget 2.7 GiB
(131,072 tokens, 8,192 pages) at a 13.2 GiB memory limit. The printed
fixed-plus-KV sum is 12 GiB. Measured `phys_footprint` after a short parity
workload (8 prompts, 96 tokens each) is 1.2 GiB on the plain serve and 1.5
GiB on the draft serve — the sum overestimates resident by ~10 GiB. The gap
is allocation laziness, not accounting error: the KV pool and the GDR static
state (4,770 MiB, one recurrent state per slot) are MLX arrays that
materialize on first touch, and a short workload touches a fraction of one
slot's state. The printed sum is a capacity ceiling, not a steady-state
resident prediction; resident approaches the sum as the workload fills the
KV pool and activates more slots. The measurement was taken on a box under
heavy swap pressure (13 GiB swap used), so even the weights are partially
compressed and the resident figure is a lower bound. The 35B solve prints
from the rejection path today — fixed requirement 38 GiB (weights 19 +
headroom 4 + static state 15,720 MiB) — and the resident-byte comparison on
the 35B is pending a machine without swap pressure, the same deferral as
[note 1](hybrid-prefix-cache.md).

The number that motivates the bytes-read axis: the same 27B checkpoint in
NVFP4 decodes at 84.5 tok/s against FP8's 60.2 on one H20
([baselines](../baselines.md)), and the difference is the 56% weight-byte
ratio, not a faster kernel.

## What would be done differently

Make the two numbers part of every runtime change's commit message: delta
resident bytes and delta bytes read per token, measured or stated as
unmeasured. The Marlin regression lived as long as it did because neither
number was printed anywhere — the KV pool line in the log was the only
signal, and it took a 7× wall-clock regression to make anyone read it. The
doctor solve is the cheap half of that fix; the other half is refusing to
land a layout change without the resident set before and after.
