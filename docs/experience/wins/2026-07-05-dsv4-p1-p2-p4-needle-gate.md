# DSv4 KV three-layer P1/P2/P4 — pod needle gate PASS + P4 guard coverage gap found and fixed

**Date:** 2026-07-05. **Backend:** CUDA, DeepSeek-V4-Flash-FP8, TP=4/EP=4,
4×H20 (GPUs 3,4,6,7; GPU 1 excluded — foreign process holding VRAM). Commit
under test: `7b89fe32` (includes P1 `611d18cd`, P2+P4 `2dd9d07c`). Harness
`scripts/needle_gate.py`, `RAW=1`, greedy, needle `738291`, 3 same-config
repeats. `INFER_DSV4_MAX_SEQ_LEN=16384` (TP=4 halves rank count vs the
TP=8 baseline this repo has previously validated at 32K — see
[2026-06-10-dsv4-longctx-closeout-needle-matrix.md](2026-06-10-dsv4-longctx-closeout-needle-matrix.md)
— so per-rank weight shard roughly doubles, leaving less headroom for KV).

## Goal

Close the "pod needle gate 2K/8K/32K" verification the P1 (`611d18cd`) and
P2+P4 (`2dd9d07c`) commit messages deferred: prove the `Dsv4BlockMap`
comp-row single-sourcing (P1/P2) and the CSA select-boundary shape guard (P4)
preserve correct inference on the >2048 comp-row-addressing path this
refactor targets, per
[the plan doc](../../plans/2026-07-04-dsv4-dsa-kv-three-layer.md).

## Results — needle matrix (non-degenerate filler, depth 0)

| length (target) | prompt_tokens | exact | partial | miss | deterministic? |
|---|---|---|---|---|---|
| 1000 | 922 | 3/3 | 0 | 0 | NONDET |
| 2000 | 1828 | 3/3 | 0 | 0 | NONDET |
| 4000 | 3621 | 3/3 | 0 | 0 | NONDET |
| 8000 | 7218 | 3/3 | 0 | 0 | NONDET |
| 8500 | 7661 | 1/1 | 0 | 0 | DET |

All-exact, zero partial/miss, no garbage — stronger than the 2026-06-10
baseline envelope (which had NONDET partial/miss blur past 115 tokens on an
earlier commit). "NONDET" is the documented MoE non-determinism floor, not a
defect (see 2026-06-10 entry, Rule §2). Every length ≥2000 crosses the 2048
comp-row page boundary this refactor targets.

## Finding — P4's guard didn't cover the default decode path (fixed same session)

Code review (before the pod run landed) found the P4 CSA select-boundary
guard (`indexer_rows == value_compressor_rows`, attention.rs) was wired into
only 1 of 3 live call sites that compute `indexer_rows_after` for
`CompressedSparse` mode:

| call site | role | had guard before this fix? |
|---|---|---|
| `mla_attention_prepare` | eager single-row decode + chunked-prefill | yes (2dd9d07c) |
| `mla_attention_prepare_compressed_only` | **batched-decode lane (#60), default-on** (`dsv4_flashmla_decode_batched_enabled()` → `dsv4_flashmla_decode_enabled()` defaults true when FlashMLA compiled) | **no** |
| `mla_attention_decode_graph` | CUDA-graph decode (opt-in, `ARLE_DSV4_DECODE_GRAPH_CSA`) | **no** |

The batched-decode lane is what a normal serve actually runs — a hypothetical
row-count drift there would have stayed silent, undermining P4's stated
purpose ("turns a silent Shape drift past 2048 into a loud boundary fail").
**Consequence for this pod run**: "the P4 guard never fired" is expected and
uninformative for the two ungated call sites — their absence, not a clean
pass, is why they stayed silent. Fixed in this session: the identical
`ensure!` was added to both missing sites (attention.rs, right after each
site's `indexer_rows_after` computation); mac typecheck (`cuda,no-cuda`)
clean after the fix. The needle-gate PASS above validates output *content*
independent of this gap; it does not by itself validate that the tripwire
would fire on a real drift in the batched lane (no drift was injected to
test it — the gate is a correctness gate, not a guard-fires fuzz test).

## Problems — orthogonal multiproc lockstep hang (not this diff), now root-caused

Prompts with `prompt_tokens` ≳8106 froze the entire multiproc engine
indefinitely (`/v1/stats` `steps` frozen, zero progress, reproducible on the
very first request — not a leak). Bisected: 7661 pt OK, 8106 pt hangs.
**Controlled A/B**: reproduced the identical hang at the identical length on
a second pod tree built at `c89c26ae` (immediately before P1), same
TP=4/EP=4/`INFER_DSV4_MAX_SEQ_LEN=16384` config — confirms this is
**pre-existing, not a regression from this refactor**.

The initial "VRAM squeeze" guess from that A/B alone was **measured and
refuted** in a follow-up investigation: GPU memory stays flat at
96999/97871 MiB for the entire hang (no growth, no OOM, no fresh Xid), and
gdb backtraces on a symbol build show the real mechanism is a livelock in
the multiproc coordinator's lockstep ack-wait (`coordinator.rs:88-109`
`wait_for_ack_window` has no timeout — see
[errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md](../errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md)
for the full mechanism). 12K–32K therefore unreachable under TP=4 in this
session, independent of the DSv4 KV-storage refactor either way.

## Verdict

- **P1 (`611d18cd`) + P2 (`2dd9d07c` P2) comp-row single-sourcing: PASS.**
  Correct, coherent, exact-match inference at every reachable length crossing
  2048 (2000/4000/8000/8500 tokens).
- **P4 guard: coverage gap found and fixed this session** (see above) — now
  wired into all 3 live CompressedSparse call sites.
- **32K leg of the original "2K/8K/32K" ask: still open**, blocked by the
  orthogonal TP=4 hang above — recommend re-running at TP=8/EP=8 (the
  previously-validated 32K-headroom config) rather than chasing the TP=4
  liveness bug inside this ticket.

## Rule

- A guard added at one call site of a duplicated code path (eager vs. batched
  vs. graph decode "twins") is not verified until it is added at every twin —
  "the guard never fired" is only informative where the guard exists.
- TP=N changes VRAM headroom non-linearly (fewer ranks → bigger per-rank
  shard) — a max-seq-len ceiling validated at one TP is not portable to
  another without re-deriving the budget (see the plan doc's own budget-chain
  section).
