# The cleanup script deleted its own salvage directory

## Context

Reclaiming disk on the H20 pod: 94 stale ARLE build/source trees, 116 GB, on a
box at 80% full. The keep set was reasoned about carefully — the active tree,
`/host/spec-phase` (the archived champion binaries `docs/baselines.md` rule 4
reuses for two-arm runs), the bench-artifact tree, and another engineer's tree.

Because deleting an archived binary is irreversible and had just proven its
worth — running `/host/spec-phase/arle-mk` back to back against HEAD settled a
regression question in 90 minutes with no rebuild and no bisect — the script
first copied every loose binary out of the doomed trees into a salvage
directory.

## Root cause

```bash
SALVAGE=/host/arle-archive
mkdir -p "$SALVAGE"                    # created here

for d in /host/arle-* /host/phase1-*; do   # ...and matched by this glob
  basename "$d" | grep -qE "$KEEPRE" && continue
  echo "$d" >> /tmp/delset.txt
done
```

`/host/arle-archive` matches `/host/arle-*`. The keep-regex listed the four
directories reasoned about; the script's own output directory was not among
them, because it did not exist when that list was written.

Order of events: create the salvage dir, sweep it into the delete set, copy
three binaries into it, print `salvaged=3`, `rm -rf` it.

The script reported success. `salvaged=3 -> /host/arle-archive` is printed
before the delete loop runs, so it was true when printed and false by the time
the script exited.

## Fix

Not re-run; the deletion already happened. For the next one:

- The salvage destination goes **outside** the namespace the delete set is
  globbed from, or is added to the keep set in the same statement that creates
  it — a name is not exempt from a glob just because the script owns it.
- Verify after the destructive step, not before. `ls "$SALVAGE"` as the last
  line would have caught this; `salvaged=3` did not.

## Impact

Bounded, and checked rather than assumed:

| | |
|---|---|
| `arle-mk` (DSpark champion row) | intact, `/host/spec-phase` |
| `arle-fa3b2` (MoE champion row) | intact, same |
| loose binaries at `/host` top level | intact — files, never in the delete set |
| 3 binaries inside doomed trees | **lost** |

The lost three were needle-gate A/B binaries from 07-10 that no current
baseline row names. Disk went 406 GB → 518 GB free.

Separately and not caused by this: the `b8d390bf3` binary that measured the
c≥4 regression was overwritten by the next build at the same
`target/release/arle` path, so it is no longer available as a control arm.

## Rule

**A cleanup script's own output must be excluded from its delete set at the
point of creation, not by a keep-list written earlier.** The keep-list encodes
what the author thought about; a directory created by the script is invisible
to that reasoning because it did not exist yet.

**A self-reported success count is not verification.** `salvaged=3` was emitted
by the step that succeeded, then invalidated by a later step in the same
script. The same failure shape appeared twice today: a gate that counted absent
symptoms and passed on code that never ran
([`errors/2026-08-06-rollout-engine-reprofiles-its-kv-pool-after-the-student-lands.md`](2026-08-06-rollout-engine-reprofiles-its-kv-pool-after-the-student-lands.md)),
and this. Check the end state, not the operation's own report.

**Overwriting `target/release/arle` retires the binary that measured the last
number.** Copy it out before the next build whenever it is a control arm.
