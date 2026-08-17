# Prefix restored-length cross-rank min-reduce — engine, 2026-08-17

> Status: pending-remote

## Goal

Keep `prefill_start_pos` identical across TP ranks after prefix attach. A
rank-local attach or sidecar-restore failure degrades that rank to full
recompute (restored length 0) while its peers restore the full prefix; without
a cross-rank reduce the planner then emits mismatched rows and the TP
collectives desync (garble or hang).

## Hypothesis

Min-reducing the restored length across ranks (via the existing
`BackendExecutor::tp_sync_min`) after `attach_prefix_restore` aligns every rank
to the shortest successful restore. In the common case (all ranks restore the
same length) the reduce is a no-op; on divergence the lagging rank's peers
truncate their attach and recompute from the same position. The extra
all-reduce-min runs once per admission, not per step, so it is off the hot
path.

## Parameters

```bash
# A/B: baseline = parent of this commit, treatment = this commit
# Correctness gate (common case is a wash; the reduce is a no-op when all
# ranks restore equally):
python3 scripts/needle_gate.py --url <url> --model <model> --runs 3
```

- Baseline: parent of the T3.2a commit (no restored-length min-reduce)
- Treatment: T3.2a commit (restored-length min-reduce)
- Trials: 3 (needle ladder ×3, same config)

## Environment

- Host / GPU: 8×H20 pod (sm_90), TP ≥ 2
- Driver / CUDA: TBD
- Model / dtype: Qwen3.5/3.6 hybrid (sidecar restore active), BF16
- TP / EP / slots / KV: TP=2+, prefix cache on
- Server flags: prefix cache enabled (tier path on or off — both covered)

## Results

| arm | needle ladder | errors | garble | delta |
|---|---|---:|---:|---|
| baseline | | | | — |
| treatment | | | | |

Raw artifacts: TBD.

## Problems

None yet.

## Learnings

pending-remote. The tier path already min-reduced the *matched* length
(`lookup_prefix_for_attach`, prefix.rs:707) but the *restored* length — after
`attach_pages`, sidecar restore, and the grow/truncate clamp — could still
diverge: a rank whose sidecar restore missed, whose attach alloc failed, or
whose grow alloc failed fell back to 0 while peers kept the full prefix. The
fix restructures `attach_prefix_to_request` so all rank-local failures return
`Ok(0)` from a helper, then one unconditional `tp_sync_min` aligns the result.
A peer that falls back to 0 truncates this rank's attach (free + release) and
sets `prefill_start_pos = 0`; the resulting fresh prefill rewinds the
recurrent sidecar state in `submit_prefill_row` (`row.start_pos == 0` →
`release_recurrent` + reacquire), so no stale GDN state survives the undo.
