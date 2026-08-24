# The writeback cap was spent on groups the update discards

## Context

First FP4 RL round on `Qwen3.8-27B-NVFP4`, DAPO, 4 tasks x 8 samples,
`--writeback-cap 8`. Round 0 ran all 32 rollouts and burned about 70 minutes of
GPU. Every update row read:

    update r0  groups=1  trajectories=0  tokens_trained=0  update_secs=4e-6

Four updates, no gradient step in any of them. Nothing in the log said so —
`--max-update-seq` logged no skip, `cc-convert` reported one window (an eval
task), and 933 dumps with 928 token sidecars were on disk.

## Root Cause

`cap_left` is a per-round budget introduced as "Round-scoped budget of
TRAINABLE trajectories". The comment states the intent; the code charged it
against every trajectory in the batch, trainable or not:

    let group_trained = update_preset.planned_training_count(&batch) > 0;
    if args.writeback_cap.is_some() {
        ...
        batch.truncate(cap_left);
        cap_left -= batch.len();
    }

DAPO filters with `DropZeroAdvGroup`: a trajectory trains only if some other
sample in its group scored differently. Round 0's groups were

    [0.75 x8]                        zero variance -> all dropped
    [0.75 x8]                        zero variance -> all dropped
    [0.0  x8]                        zero variance -> all dropped
    [0.2, 0, 0, 0.2, 0.2, 0.2, 0.2, 0]   the only group with signal

The first group is untrainable and was charged all 8 of the cap. `cap_left`
reached 0 before the fourth group, which was then `truncate(0)`-ed to empty.
The one group carrying a learning signal was the one that got nothing.

`group_trained` was already computed one line above — for the refill path,
which correctly treats a dead group as dead. The cap accounting ignored it.

## Fix

Spend the cap only on a group the preset will actually train; clear a dead
group's batch instead, which is what `update` would do with it anyway. Behavior
inside the update is unchanged — only the budget accounting moves.

## Rule

A budget must be charged at the point where the thing it pays for is decided,
not before the filter that throws it away. When a comment names the unit
("trainable trajectories") and the code counts something else, the code is
wrong even when nothing errors.

The failure was invisible: 32 rollouts, a plausible `reward_mean=0.406`, and
`trajectories=0` printed once per update in a metrics file nobody was reading.
The guard added the same day fires on `trained > 0 && tokens == 0` and cannot
see `trained == 0` — the strictly worse state.
