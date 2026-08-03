# The DSpark verify body is not the decode body — 304 host-coupled nodes, 2026-08-03

> Status: **built, measured, reverted.** The tranche captured but the graph
> validator rejected it; measured result was a wash (121.3 vs 121.4 tok/s) with
> the treatment provably never engaged. Reverted rather than left dormant.
> Issue #198 stays open with a corrected scope.

## Context

Verify is 86% of a DSpark block step and fixed-shape at c=1
(`chain_rows == block_size` every tick), so #198 proposed reusing T4's
whole-step decode-graph machinery over `n` query rows instead of one:
a fixed-capacity `PageMeta`, a persistent `[vocab, rows]` logits slot, and
a split of `dspark_verify_logits` into a pure-GPU body plus an
outside-capture tail.

All of that worked. The capture ran and was then rejected:

```
Qwen3.6 dspark verify graph failed (slot 0, rows 16), downgrading to eager:
captured graph is host-coupled: 304 host-side memcpy node(s),
0 host-callback node(s) of 1817 total
```

Matched A/B, empty box, temp 0, both arms default-on binaries differing only
by the tranche, 2 reps:

| arm | tok/s r1 | tok/s r2 | captures | downgrades |
|---|---:|---:|---:|---:|
| base | 121.4 | 123.1 | 0 | 0 |
| verify graph | 121.3 | 121.5 | **0** | **1** |

Identical md5 across all four arms — the eager path ran in every one.

## Root cause

**The premise was that the multi-row forward is as capture-clean as the
one-row forward. It is not, by 304 nodes.** T4 established that
`forward_hidden_staged` performs no H2D at `seq_len == 1`, and I generalized
that to "no H2D", when what was actually verified was "no H2D *on the
one-row path*". The multi-row body reaches per-layer staging the decode path
never touches — the linear-attn snapshot's pointer tables and the routing
scratch — each of which uploads from pageable host memory.

A second, self-inflicted error rode along. I added a cache to `batched_copy`
to skip re-uploading a table the device already holds, reasoning that the
snapshot addresses are executor-lived and therefore constant. They are — but
**one `Qwen35CopyScratch` is shared across all layers**, so consecutive calls
within a step always carry different tables and the cache never hits. The
idea is sound and worth ~2 pageable H2D per linear layer per step; the
implementation needs a per-layer cache slot, and it belongs in its own
tranche with its own measurement, not folded into a graph change.

Also sloppy: the failure handler cleared `decode_graph_armed`, disarming T4's
decode lane and not just the verify lane. It cost nothing here — under DSpark
the `rows == 1` lane never runs, and both arms show 0 decode captures — but a
per-lane failure must disable only its own lane.

## The census

One diagnostic build answered where the 304 live. Only variable: `capture:
None` on the verify rows, disabling the linear-attn snapshot.

| build | host memcpy nodes | total nodes |
|---|---:|---:|
| tranche as committed | 304 | 1817 |
| snapshot disabled | 256 | 1721 |

Qwen3.6-27B is **64 layers — 48 `linear_attention` + 16 `full_attention`**
(`full_attention_interval: 4`), which makes both terms exact:

- **48 nodes = the linear-attn snapshot**, one H2D per linear layer
  (`batched_copy`'s pointer table). This is the part a per-layer cache slot
  would remove.
- **256 nodes = 64 × 4 — four H2D per layer, in every layer.** Not the
  snapshot, not MoE-only, not attention-only: the common layer body at
  multi-row shape.

That second term is the verdict. The host coupling is not one localized
helper to fix; it is a per-layer pattern spanning all 64 layers, so
unblocking the verify graph means de-host-coupling the multi-row layer body
itself. Against ~1.8 ms of launch overhead on a ~36.5 ms block step (~5% of
DSpark throughput), that is not the next thing to do.

Worth noting the same work would also make the rows>1 batched-decode path
capturable, which is where its value actually sits — the payoff should be
argued on that axis, not on DSpark's 5%.

## Rule

**"No host coupling" is a property of a code path at a shape, not of a
function.** Before reusing a capture-safety result at a new shape, re-derive
it at that shape: enumerate the H2D sites the new shape reaches, or capture
once and read the node census. The census is cheap and it is the ground
truth — one rejected capture told me more than the code reading that preceded
it.

Corollary for #198: the work is not "reuse T4's machinery". It is "make the
multi-row layer body device-derived" — 256 of the 304 nodes are four H2D per
layer across all 64 layers.

The census cost one build and one request, and it converted "unknown scope"
into "four per layer, everywhere". **Run the census before writing the
implementation, not after it fails.**
