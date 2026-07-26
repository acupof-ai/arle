# DSpark: one batched argmax per tick — mechanism confirmed, serving delta a wash

## Context

After the ragged B×T batched verify landed (`f4f419629`), 45% of the c=8 plateau
was still strictly O(B). A phase probe attributed it, and the accept scan looked
like the culprit: `dspark_accept_commit` called `argmax_hs_row` per chain row per
slot, and that helper is launch + `ctx.sync()` + D2H. At `block=16`, c=8, a tick
drained the pipeline up to 128 times.

Goal: diagnosis, then throughput. Model ThinkingCap-Qwen3.6-27B-FP8, 1×H20
GPU 0, `--spec-type dspark --mtp-draft-model Qwen3.6-27B-DFlash
--dspark-block-size 16 --spec-max-batch 16 --max-running-requests 16`, greedy.
Driver `conc_drive.py` (short prompts — the tick split is prompt-length
independent, so this is a §2 diagnosis, NOT a serving claim; see Problems).

## What Worked

`argmax_rows()` takes the whole verify output in one `argmax_batch_cuda` launch
and one D2H — the shape the DSv4 lane has always used (`mtp_argmax_batch`) — so
the accept scan is host arithmetic and the per-slot loop adds no syncs. The draft
tail gets the same treatment when no markov head is present, where every block
row's logits are final before the scan.

Matched A/B, both arms built from the same tree differing only in
`qwen35/dspark.rs` + `executor/qwen35.rs`, run back to back in one shell on the
same GPU (before `eb78fbbe…`, after `322ce59f…`):

| metric | BEFORE | AFTER | Δ |
|---|---:|---:|---:|
| draft `argmax`, per slot | 0.56 ms | **0.04 ms** | −14× |
| draft total, per slot | 4.28 ms | 3.76 ms | −12% |
| commit @ c=8 | 17.92 ms | 16.23 ms | −9% |
| tick sum @ c=8 | 107.84 ms | 103.26 ms | −3.9% |
| aggregate tok/s, c=1 / 4 / 8 | 64.9 / 89.5 / 84.7 | 65.6 / 90.6 / 84.4 | +1.1% / +1.2% / −0.4% |

The pre-fix split, stable across concurrency (per-slot draft cost 4.46 ms at
c=1 vs 4.28 ms at c=8 — a serial loop, not contention):

| phase | c=1 | c=8 |
|---|---:|---:|
| draft | 4.53 (14%) | 24.91 (23%) |
| snap | 0.42 (1%) | 2.81 (3%) |
| verify | 24.76 (76%) | 62.20 (58%) |
| commit | 3.01 (9%) | 17.92 (17%) |

## Problems

- **The prediction was wrong and is recorded as such.** From "the accept loop
  contains a sync per row" I predicted commit 17.6 → ~1 ms and −18% on the tick.
  Measured: commit −9%, and c=1 commit barely moved (3.01 → 2.93 ms). The argmax
  D2H was a small slice of commit; its cost is the rejection path —
  `restore_trunk` (48 gated-delta states) plus `replay_linear_only`. Presence of
  a sync in a loop is not evidence that the loop's cost *is* the sync.
- **No serving win.** The three tok/s points are +1.1% / +1.2% / −0.4%, inside
  run variance; sign unresolved at one trial per arm. The change is kept because
  it strictly removes launches, syncs and host round-trips and the phase delta is
  unambiguous — not because it made the server faster.
- Phase numbers come from `ARLE_DSPARK_PHASE=1`, which syncs at phase
  boundaries; both arms carry it identically, and the clean tok/s pass ran
  without it.
- Short-prompt driver: legitimate for the tick split, not for an SLO verdict.
  The serving re-measure belongs on the multi-turn long-agent dataset
  (bench spec §3.3).

## Rule

Attribute *within* a phase before optimizing it. A per-slot cost that is
identical at c=1 and c=8 proves the loop is serial; it says nothing about which
line inside the loop is expensive. The phase timer already split draft into
embed/prep/attn/mlp/head/argmax and showed argmax at 13% — that number was
available before the code was written, and reading it would have predicted the
wash. Where the remaining O(B) actually sits, now measured: draft `attn`
1.54 ms/slot (5 layers × 16 single-row launches), draft `mlp` 1.08, `head` 0.70,
and commit's rollback path — not the argmax.
