# A row marked active under an owner who never worked it

Date: 2026-09-13.

## Context

`parity-fix-flashmla-pack` sat in `docs/agenda.jsonl` with status `active` and an
owner for several days. On asking that owner for a status, the answer was that
they had never worked the row and had no action on it. The work the row existed
for had happened on two other rows in the meantime: the measurement that settled
it came from a different session, and the fix for the two output teeth it was
opened for is a bound change on a third row.

The same day, a second session reported its lane pushed with commits while the
agenda showed the lane created and empty. Two instances, opposite directions:
one row claimed work that was not happening, one hid work that was.

## Root Cause

Status and owner are written when a row is opened or routed, and nothing writes
them again when the work moves. Routing happens in conversation between
sessions; the ledger is updated only if whoever routed it remembers. So the
ledger drifts toward whatever was true at the moment of assignment, and every
later reader — including the controller deciding what is staffed and what needs
someone — reads a stale claim as current state.

This is the documented-gate defect applied to process rather than code. The row
asserts a wiring, the wiring does not exist, and nothing checks. The reason it
survived is the same: `active` is not falsifiable by reading the file, only by
asking the named owner, and nobody asks a row that looks handled.

## Fix

The row was closed as folded, with the verdict naming where the work actually
happened. Two mechanical checks worth having, neither implemented here:

1. A row `active` for more than N days whose owner's lane has no commits is
   either stale or blocked, and hygiene can say so.
2. Routing a task in conversation should update the row in the same action.
   Nothing enforces it; the controller has to treat "I told someone to do X" and
   "the row says someone is doing X" as two separate writes.

## Rule

Ask the named owner before counting a row as staffed. An agenda that reports a
row as active when nobody is on it is worse than one that reports nothing,
because it consumes the attention that would otherwise go to finding an owner.
When a status is only falsifiable by asking a person, it decays at the rate
people forget to write it down.
