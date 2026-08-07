# Named a call site whose gate was off, and killed the graph lever on 3% — CUDA, 2026-08-07

## Context

Chasing the host-side time left in the DSpark decode window after the day's two
accepted wins. The tick decomposition (`ARLE_DSPARK_PHASE`) put 94% of the tick
inside the two forwards, so the remaining cost had to be inside a forward or
between ticks. An nsys capture over 19.92 s of decode settled it:

```
window 19.92 s | GPU busy 10.07 s (50.5%) | idle 9.86 s
gaps n=430584   total 9.86 s   mean 22.9 us
  0-5 us     n=410485   0.59 s    6.0%
  5-20 us    n= 18045   0.14 s    1.5%
  20-50 us   n=   643   0.02 s    0.2%
  50-200 us  n=  1328   0.12 s    1.2%
  >1 ms      n=    79   8.98 s   91.1%
```

Of that 8.98 s, **7.45 s sat in no CUDA API call at all** — host CPU.

> **Correction (measured after this was written).** The 45%-of-window figure is
> true of this window and does not generalize: the window was *aimed* at decode,
> where the cost concentrates. Over a full 512 s bench the same mechanism is
> **9.4% of wall**. See
> [`../wins/2026-08-07-prefix-sidecar-serialize-bulk-copy.md`](../wins/2026-08-07-prefix-sidecar-serialize-bulk-copy.md).
> A window selected for a phenomenon reports that phenomenon's share of the
> window, not of the run.

## Error 1 — the graph lever was killed on the wrong 3%

Earlier the same day I had written that CUDA graph capture was "the only lever
left that's big", reasoning that 605 kernels per forward at ~15 µs each in a
~40 ms forward meant the gaps were the wall. The gap histogram says the
opposite: **all 430k launch gaps together are 0.59 s, 3% of the window.** Graph
replay collapses exactly those. The 91% lives in 79 stalls averaging 114 ms,
which no graph touches.

The mistake was arithmetic-by-average: kernel count × mean kernel time versus
wall gives a plausible per-launch gap, and a plausible number is not a measured
distribution. **Bin the gaps before costing a launch-overhead fix.** One query
against the nsys sqlite would have answered it at any point that day —
`nsys` itself was gone from the box, but `CUPTI_ACTIVITY_KIND_KERNEL` in the
`.sqlite` is plain SQL and needs no profiler binary.

## Error 2 — named a source whose gate was off in the measured config

The stalls were bracketed by 3145728-byte transfers, D2H 1488 and H2D 816,
each paired 1:1 with a 61440-byte copy. 3145728 = 48 × 3 MiB is the gated-delta
recurrent state per linear layer and 61440 the conv ring, so the payload was
certain. I then named the producer as `Qwen35SlotImage` / `swap_out_image` —
the whole-slot park — wrote it up, and committed a fix for it (`a546ba80a`).

Both park routes are unreachable in the measured serve:

| route | gate | state in the run |
|---|---|---|
| `try_park_for_oversubscription` | `slot_oversubscription` | `--kv-oversubscription` defaults false, not passed |
| `requeue_preempted_decode_with_bias` | `kv_tier_capacity() == 0` | L2 tier on, 830 GB budget |

The serve log also carried none of the `log::warn!` lines every KV-overflow
preempt path emits. The actual producer is `Qwen35RecurrentSnapshot`, written
at every stride boundary of every prefill so a later conversation can restore
the hybrid prefix — a second struct with a copy-pasted serializer of the same
shape (`d626a1b03`).

**The payload identified the data, not the code path.** Two call sites can move
byte-identical payloads; the transfer signature cannot distinguish them. Before
naming a call site as the source of a measured cost, check that its gate is
satisfiable in the configuration that was measured — the flag's default, the
CLI line actually used, and whether the path's own log lines appear.

This is the mirror image of the morning's lesson. There, a flag defaulting **on**
silently made a minority branch the only path. Here, a flag defaulting **off**
made a branch I named unreachable. Same question, asked in neither case: *what
is this flag's value in the run I am looking at?*

## The cost that was actually there

`to_bytes` walked the state one element at a time — `extend_from_slice(&x.to_le_bytes())`
per f32, 37M calls per snapshot — and `from_bytes` rebuilt it with
`chunks_exact(4).map().collect()`. 147 MiB serialized four bytes at a time, on
the tick, at every stride boundary of every prefill. Both are bulk byte copies
now, the idiom `attention/prefix_state.rs::push_bf16` already used.

Neither had a counter or a log. `kv_tier_stats.demoted_slots` existed and was
never surfaced; the sidecar had nothing at all. **Third instance today of a
large cost with no instrument that could notice it** — after the two per-row
loops. Both commits now log size and elapsed per event.

## Rules

**Bin the distribution before costing a fix aimed at its mean.** "N events ×
mean cost" and "where the time actually is" are different questions, and only
the second one prices a lever.

**A payload signature identifies data, not a call site.** Confirm the gate of
the path you name is satisfiable in the measured configuration before writing
it down as the source — and prefer a path that emits a log line you can see in
the run over one you inferred.

**A profiler binary is not required to read its own database.** `nsys` was
uninstalled from the box; every finding here came from `sqlite3` over the
`.sqlite` the earlier captures had already written.
