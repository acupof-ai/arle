# Tiered KV Tier A coordinator local

## Context

Tier A wires the first scheduler-visible `StageTicket` control flow through the
local coordinator stub so staged hits can pause admission, complete on a ticket,
and re-enter once the radix metadata says they are GPU-ready again.

## What Worked

Coordinator stage commands now emit queued/completed events in order, the CUDA
scheduler tracks staged admissions on tickets, and the page-lifecycle stub is
updated at publish/evict boundaries without pretending the real transport path
exists yet.

## Rule

For early staged-promotion work, land the ticketed control flow and lifecycle
bookkeeping first, keep completion semantics explicit, and defer the real copy
transport until the scheduler can already re-admit correctly.
