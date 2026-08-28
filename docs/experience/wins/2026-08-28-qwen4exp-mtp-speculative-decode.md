# qwen4_exp MTP speculative decode: greedy-lossless at full scale, +10-15% at k=1, and the k the model actually wants is 1

> Status: Shipped (opt-in API; not a default flip — see Learnings)

## Context / Goal

Decode is bandwidth-bound: 50.6-56.8 ms/token at the Q4_K W4A16 default, and
the roofline says the wall is the weight bytes, not the math. Speculative
decode is the one lever that divides those bytes by the accepted length. This
checkpoint ships its own drafter — `mtp.*`, 4.86 GiB of BF16 that had been
sitting in the residency plan unread since the S7 prep.

Two deliverables, in priority order:

1. **Correctness**: greedy speculative output must EQUAL plain greedy decode
   token for token. Under greedy acceptance speculation is lossless BY
   CONSTRUCTION, so any mismatch is a rollback bug — and on a model whose
   state is a gated-delta recurrence + conv rings + a PLE ring + an n-gram
   window, rollback is the entire difficulty.
2. **Measurement**: acceptance per draft depth, effective tok/s against the
   same-sitting baseline, and the rollback cost — with a calibrated
   expectation (the vendor's own MTP deployments of this arch family cap
   draft depth at 2, see 50-70% per-step acceptance, and net +13-15%
   throughput, NOT 2x).

## What Worked

**The MTP head (`crates/infer-vulkan/src/qwen4_mtp.rs`, new).** One full-
attention + MoE + hyper-connection layer with four fusion tensors and its OWN
`hyper_connection_mixer`. No reference forward exists — transformers ignores
`^mtp.*` on load — so the semantics are pinned two ways: every
`mtp.layers.0.*` submodule reuses the text decoder's exact suffix vocabulary,
and the four `mtp.*`-only tensors are pinned by SHAPE.
`pre_fc_norm_hidden` is `[10240]`, which is the load-bearing fact: the MTP
consumes the target's **PRE-mixer** 4-stream residual (a `[2560]` norm would
have meant the post-mixer state), fused with the next token's embedding as
`h_in[s] = fc_hidden @ norm(h)[s] + fc_embedding @ norm(e)` — per-stream and
broadcast respectively. Drafting is recurrent (DeepSeek-V3 / vLLM style):
step 1 conditions on the target's `h`, step `j>1` on the MTP's own 10240-wide
output and its own previous draft.

**The upload.** The MTP experts are the quant-EXCLUDED stacked layout
(`experts.gate_up_proj [512,1280,2560]`, fused `[gate; up]`). The plan slices
them per expert into 1536 ordinary dense-GEMV suballocations
(`Qwen4Source::Bf16Slice`), so a drafted token's 10-of-512 experts record as
plain `record_dense_at` GEMVs and the cold 98% spills without split logic.
`spill_rank` was re-ordered cold-first: MTP slices (read only while
speculating, 10/512) rank ahead of the NVFP4 stacks, which rank ahead of the
per-token dense tier. `MTP_HC_LAYER = usize::MAX` addresses the head's three
hyper-connection sites through the existing `(layer, site)` plumbing, so the
device HC kernels drive the MTP with no second code path.

**Rollback = one recorded buffer copy each way.** The verify chunk advances
GDN S, the conv rings, the PLE ring, the KV rows and the n-gram window for
every chunk position, accepted or not. The restore contract:

- GDN S + conv rings are device-resident: `DevResidentLinAttn::
  record_state_save/restore` copy the whole 117 MB state device-to-device.
  The SAVE rides the verify chunk's own submit (free); only a REJECTION pays
  a restore submit. Reading that state to the host instead would have been
  the documented write-combined ~0.10 GB/s trap — over a second per snapshot.
- The PLE ring and the n-gram window are host-side clones.
- KV and RoPE are POSITIONAL: winding `seq_len` back IS their rollback, since
  rows past it are never read and the next chunk rewrites them.
- A rejected suffix is never replayed alone — the rolled-back tokens ride the
  FRONT of the next cycle's chunk, so the cost is always one weight sweep per
  cycle.

**The verify pass reuses prefill wholesale.** `forward_verify` runs
`pending ++ drafts` as ONE `Qwen4Prefill::run_chunk` — grouped MoE experts,
sequential recurrences, batched flash — which is exactly why the equivalence
gate is trustworthy: the prefill=decode bit-exact gate (0.000e0, already
shipped) says that chunk computes decode's values. Its tail is new: the
stream mixer per position, then `lm_head` as NUM_COLS-batched GEMVs, so the
1.2 GiB projection is read once per chunk instead of once per position.

**`NUM_COLS` batching generalized** (`Kernel::gemv_cols_spec`, and
`record_dense_chunk`'s new cols arm): the vendored shader's own batch axis at
`NUM_COLS <= 8`, weight row read once for k columns. Per-column arithmetic is
a `temp[NUM_COLS][NUM_ROWS]` unroll of the single-column loop, so it is
bit-identical to the per-token loop — confirmed on device (the cols and loop
arms report the SAME 2.44e-7..2.86e-7 error vs an f64 oracle at every k in
{1,2,4,8,16}), and the prefill=decode gate still reads 0.000e0 with the arm
live. `ARLE_QWEN4_GEMV_COLS=1` disables it for a matched A/B.

Because it lands on the shipping prefill path, it was measured there too —
matched A/B **in the same load** (the cap is read per recorded dispatch, so
the env flip is a real arm switch), 512 tokens at chunk 256:

| arm | tok/s |
|---|---|
| `NUM_COLS <= 8` (new default) | **68.2** |
| `ARLE_QWEN4_GEMV_COLS=1` (per-token GEMV loop) | 57.6 |

**+18.4% prefill, bit-exactness preserved** — an unlooked-for win from the
machinery the verify pass needed anyway.

## Results

### The gate (THE deliverable)

Full scale, hybrid residency, Q4_K dense default, 3 prompt classes × 5 draft
depths × 40 tokens: **speculative output == plain greedy decode, token for
token, in all 15 configurations.** Truncated-model gate (4 layers,
`SubsetF32`) additionally drives synthetic always-right / always-wrong /
mixed drafters at k in {1,2,3}: equal in all of them, with the always-wrong
arm confirming 0 full-accepts and mixed confirming both lanes fired.

**The gate is proven able to fail.** `ARLE_QWEN4_SPEC_FAULT` skips one piece
of the restore, and the state comparison catches exactly that piece:

| fault | GDN S + rings equal | PLE ring equal | n-gram equal |
|---|---|---|---|
| (none) | true | true | true |
| `skip-gdn` | **false** | true | true |
| `skip-ple` | true | **false** | true |
| `skip-ngram` | true | true | **false** |

### Acceptance and effective throughput

Same sitting, same load. Baseline is plain greedy decode measured immediately
before the sweep (steady state, first 4 tokens dropped).

| prompt | k | accept/step | mean L | cycles(full) | draft ms/cyc | verify ms/cyc | rollback ms/cyc | eff ms/tok | baseline | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| factual-qa | 1 | 85.7% | 0.86 | 21(18) | 10.0 | 77.3 | 0.19 | 46.0 | 50.6 | **1.10x** |
| factual-qa | 2 | 71.9% | 1.44 | 16(10) | 18.3 | 98.9 | 0.56 | 47.1 | 50.6 | 1.07x |
| factual-qa | 3 | 61.9% | 1.86 | 14(6) | 26.8 | 131.9 | 1.05 | 55.9 | 50.6 | 0.90x |
| factual-qa | 4 | 56.2% | 2.25 | 12(4) | 56.8 | 190.8 | 1.16 | 82.5 | 50.6 | 0.61x |
| factual-qa | 6 | 46.7% | 2.80 | 10(2) | 50.3 | 193.1 | 1.32 | 73.1 | 50.6 | 0.69x |
| code | 1 | 85.7% | 0.86 | 21(18) | 11.6 | 76.4 | 0.20 | 46.3 | 51.1 | **1.10x** |
| code | 2 | 71.9% | 1.44 | 16(11) | 21.5 | 94.1 | 0.45 | 46.5 | 51.1 | 1.10x |
| code | 3 | 66.7% | 2.00 | 13(8) | 30.3 | 123.1 | 0.71 | 50.1 | 51.1 | 1.02x |
| code | 4 | 72.5% | 2.90 | 10(6) | 41.5 | 137.1 | 0.68 | 44.8 | 51.1 | **1.14x** |
| code | 6 | 64.6% | 3.88 | 8(4) | 57.8 | 175.6 | 0.96 | 46.9 | 51.1 | 1.09x |
| chat | 1 | 77.3% | 0.77 | 22(17) | 10.8 | 79.0 | 0.32 | 49.6 | 56.8 | **1.15x** |
| chat | 2 | 57.9% | 1.16 | 19(9) | 20.9 | 111.7 | 0.85 | 63.4 | 56.8 | 0.90x |
| chat | 3 | 52.1% | 1.56 | 16(4) | 30.1 | 147.8 | 1.27 | 71.7 | 56.8 | 0.79x |
| chat | 4 | 35.9% | 1.44 | 16(1) | 41.5 | 178.2 | 1.65 | 100.6 | 56.8 | 0.56x |
| chat | 6 | 25.6% | 1.53 | 15(0) | 56.4 | 208.1 | 1.90 | 116.4 | 56.8 | 0.49x |

**Acceptance lands in the vendor's band.** At k=2, per-step acceptance is
57.9-71.9% (mean 67%) against the vendor's 50-70%. At k=1 it is 77-86%. The
head is drafting correctly; nothing here says the wiring is wrong.

**Best k = 1** (`+10%` / `+10%` / `+15%`), the only depth that wins on all
three classes. k=2 is a win on code (1.10x), a small win on factual-qa
(1.07x), and a loss on chat (0.90x). That is the vendor's "+13-15%, NOT 2x"
band, arrived at one depth shallower than the vendor's cap.

**Rollback is not the cost.** 0.19-1.90 ms/cycle against a 50 ms baseline
token — under 4% at the worst point and under 0.5% at the best k. The
snapshot is free by construction (it rides the verify submit). Anyone
tempted to optimize rollback should read this row first.

### Where the win goes instead: the verify pass under-amortizes

At k=1 the verify chunk computes 2 positions in 77.3 ms where 2 baseline
decode tokens cost 101.2 ms — a **1.31x** saving on a chunk that doubled the
work. The dense tier is amortizing exactly as designed; what does not
amortize is everything else in a token:

- The **expert tier does not amortize at k≤8** — this was predicted and is
  now measured. At 512 experts top-10, two tokens activate ~20 distinct
  experts at ~1 row each; the union grows nearly linearly in k, so the NVFP4
  bytes are ~k× regardless of grouping.
- The **dispatch floor is per-position**: hyper-connection sites and the MoE
  router still record per token WITHIN the chunk (~5400 dispatches/token ×
  3.37 µs ≈ 34% of wall, already the known decode ceiling), so a k-position
  chunk pays k× that floor.
- The **draft costs 10-11 ms/cycle at k=1** — the MTP is 1 layer of 49, but a
  full-attention one with its own 512-expert MoE, so ~20% of a decode step
  per draft step is structural, not a bug.

Net: the verify saves ~24 ms per cycle at k=1 and the draft spends ~10 of it.
Deeper k buys more accepted tokens per cycle but the verify grows faster than
acceptance does, which is exactly the shape of the table above.

### MTP device-route parity

`ARLE_QWEN4_MTP_F32=1` (1-layer subset + F32 MTP tier) isolates the wiring
from quantization: per-expert slice GEMVs (experts 0 / 7 / 511 × gate/up/down)
match the host slice views at **1.7e-5..8.4e-5**, and a full forward's
attention K/V — which is expert-selection-free — matches at **6.5e-4**. The
stacked-expert slice addressing (the one piece of the MTP route no
text-stream test exercises) is therefore correct.

At the shipping Q4_K tier the same comparison reads `h_out` max rel 1.8e2 on a
single element with mean abs 9.1e-2: a razor-thin top-10 router margin flips
one expert between the host BF16 view and its Q4_K twin. Reported, not gated
— drafts are proposals, and the number that judges the device MTP is the
acceptance column above, which is in band.

## Problems

**A premise of my own, corrected by measurement.** I built the k-position
verify expecting the dense amortization to carry the whole token, and sized
`QWEN4_VERIFY_MAX_TOKENS` at 16 on that basis. The measured 1.31x at k=1
(against a naive 2x expectation for 2 positions) is the expert tier and the
dispatch floor refusing to amortize. The table's shape — peaking at k=1 —
falls out of that, and it is the reason the honest recommendation is k=1
rather than the vendor's k=2.

**The first full-scale run reported a 1e2 MTP parity "failure" that was not
one.** The parity probe fed the head uniform-random conditioning, which is
far out of distribution for a 512-expert router: the host and device routes
selected different experts wholesale. Fixing it took two changes, both
correct on their own merits — feed it the REAL pre-mixer `h` and a real
embedding row, and move the probe AFTER the correctness gate so a numeric
report can never pre-empt the verdict. The F32 lane above is what actually
answers the wiring question.

**Two benchmark numbers in this arc are contaminated and were re-measured.**
A sibling agent was running the full-scale prefill bench in another worktree
on the same 63.6 GB box; the first sweep read baselines of 104.4 / 72.4 /
97.8 ms/token and k=3 at "1.07x" — noise, not signal, with both processes
thrashing. The table above is from an exclusive run (baselines 50.6 / 51.1 /
56.8, consistent with the shipped 55.5 ms/token Q4_K figure). Ratios from one
sitting; absolutes on this box depend on the power mode.

**One real driver bug, found by the gate.** After a rollback the accepted
tokens are re-verified in the next chunk, so their `h` rows arrive at the MTP
KV canon a SECOND time — the canon's contiguity assert fired at
`[factual-qa] k=4`. The fix is to skip a catch-up position behind the canon
(the replay is deterministic, so the entry already built stands) and keep the
assert for a genuine gap ahead of it. Worth noting the gate caught this as a
hard error, not as silent draft degradation.

## Learnings

**PASS on correctness, PASS on the vendor's calibration, and NOT a default
flip.** Greedy speculative decode is lossless here — proven at full scale over
3 prompts × 40 tokens × 5 depths, with a fault-injection knob demonstrating
the gate can fail. Acceptance is in the vendor's 50-70% band at k=2 and the
best measured k is **1, at +10-15%**. That is the vendor's own throughput
result, so it is success, not disappointment — but +10-15% at one depth on
three prompts is not the evidence bar for flipping a default, and chat
regresses at every k>1. It ships as an opt-in API
(`generate_speculative` + `Qwen4DraftSource`) with the sweep as its harness.

**The next wall is named and measured, and it is not the drafter.** The
verify pass amortizes the dense tier 1.31x and nothing else. Two levers, in
order:

1. **Batch the per-position dispatch floor inside a chunk** — the ~97
   hyper-connection sites and the MoE router still record per token. This is
   the same lever the prefill round already identified (hc.pre was 147k tiny
   dispatches there) and it is worth more to speculation than to prefill,
   because a verify chunk is 2-4 positions where prefill is 256.
2. **A cheaper drafter.** 10-11 ms/cycle to draft one token is ~20% of the
   budget. The MTP layer's own MoE is the bulk of it; drafting with the
   shared expert only (or a top-2 router) is a quality-vs-cost knob that
   costs acceptance rate and nothing else — the equivalence gate holds for
   ANY draft source, which is exactly why `Qwen4DraftSource` is a trait.

**Rollback on a recurrent model is cheap if you never read the state to the
host.** The whole snapshot/restore is two `vkCmdCopyBuffer`s, the save rides
a submit that was happening anyway, and the measured cost is under 4% of a
token at the worst depth. The HOST_CACHED/write-combined trap that would have
made this expensive is documented in the memory index; the device-to-device
copy sidesteps it entirely.

## Reproduce

```bash
# The prefill NUM_COLS A/B (matched, one load).
ARLE_QWEN4_PREFILL=1 ARLE_QWEN4_PREFILL_COLS_AB=1 ARLE_QWEN4_PREFILL_TOKENS=512   cargo test -p infer-vulkan --features vulkan --release   --test qwen4_prefill full_scale_prefill_tok_s -- --exact --nocapture --test-threads=1

# The gate + the sweep (full ~68 GiB hybrid load, minutes).
ARLE_QWEN4_SPEC=1 ARLE_QWEN4_SPEC_KS=1,2,3,4,6 ARLE_QWEN4_SPEC_PARITY=1 \
  cargo test -p infer-vulkan --features vulkan --release \
  --test qwen4_speculative full_scale -- --nocapture --test-threads=1

# The truncated gates (4 layers, ~20 s load each) — these run by default.
cargo test -p infer-vulkan --features vulkan --release \
  --test qwen4_speculative -- --nocapture --test-threads=1

# The MTP wiring parity with quantization out of the picture.
ARLE_QWEN4_MTP_F32=1 cargo test -p infer-vulkan --features vulkan --release \
  --test qwen4_speculative mtp_device_route -- --nocapture --test-threads=1
```

Env: `ARLE_QWEN4_MTP=0` drops the head from a full load;
`ARLE_QWEN4_GEMV_COLS=1` disables the NUM_COLS dense batching;
`ARLE_QWEN4_MTP_WARMUP` bounds the KV-canon warmup;
`ARLE_QWEN4_SPEC_FAULT=skip-gdn|skip-ple|skip-ngram` injects a rollback fault.

## Post-merge re-measure (same day): the speedup did not survive the faster baseline

Merged onto the fence-free + resident-PLE + pipelined mainline (decode 43.5
ms/token) and re-run at full scale, exclusive box: LOSSLESSNESS HOLDS
(sequences token-for-token equal, gate green), but the throughput verdict
inverts —

| prompt | k | accept/step | verify ms/cyc | eff ms/tok | baseline | speedup |
|---|---|---|---|---|---|---|
| factual-qa | 1 | 85.7% | 119.4 | 66.6 | 43.7 | 0.66x |
| code | 1 | 85.7% | 67.9 | 40.0 | 43.5 | **1.09x** |
| chat | 1 | 77.3% | 79.4 | 49.8 | 43.8 | 0.88x |

The decode loop got the fence-free MoE, the resident PLE ring and depth-2
submit pipelining; `forward_verify` — built on the grouped-prefill machinery
— got none of them and still pays the per-layer ids fence, so a 2-position
verify chunk costs 68-119 ms against an 87 ms two-token baseline. Speculation
is shelved as a perf feature until the verify chunk inherits the same
structure, and the three open walls are now ONE wall: the chunked path's
per-(layer,chunk) ids fence (80% of prefill, the whole verify overhead, and
the reason k=1 no longer pays). Device-side expert grouping — router, top-k
and group planning on GPU with indirect dispatch, ids never visiting the
host — is the structural fix that pays all three at once.
