# The MoE routing gate compared a nondeterministic order and an unproduced buffer state

## Context

`moe_routing_parity` checks eight families against an f64 host oracle. Two of
them were comparing something other than the kernel contract, in opposite
directions: one was red for a reason production does not have, the other was
green by accident.

## Root Cause

**m-indices.** The harness allocated the fill kernel's output with
`alloc_zeros`, leaving the aligned padding at 0. Production pre-fills -1 —
`neg1_filled` in `moe/qwen.rs` and `alloc_neg1_i32` in `moe/dsv4.rs`, both
backed by `memset_d8_async(0xFF)`. `dsv4_fill_m_indices_from_counts_kernel`
writes only rows below each expert's count and never touches the padding, so
the padding keeps whatever the caller left there. The oracle builds
`vec![-1i32; aligned_total]`, so the comparison put oracle -1 against device 0
on every expert whose count is not a multiple of 128. With TOPK 6 and 32 to
256 experts per rank that is nearly every expert. The gate was testing a
buffer state production never produces, and a 0 in the padding means those
rows route to expert 0.

**Pack.** `dsv4_pack_local_experts_with_slots_kernel` launches one block per
route and takes its slot from `atomicAdd` on the expert's cursor, so the order
within an expert's span is an arbitrary permutation of the routes assigned to
it. The comparator required sequence equality against an oracle that fills
each span in ascending route order. That check passes only while block
scheduling happens to run ascending; it is a property the kernel does not
promise, and every consumer downstream — the weight array, the packed hidden
states, and the combine oracle, which reads the kernel's own slot map — is
permutation-tolerant by construction.

## Fix

Pre-fill the m-indices buffer with -1 so the harness matches production.
Compare each expert's pack span as a set rather than a sequence.

Changing the pack comparator forced a matching change to its negative control.
The sabotage was `packed_route_slot.swap(0, 1)`, a positional swap inside one
span, which a set comparison cannot see — keeping it would have left the tooth
dead. It now flips a route value, which changes the span's multiset.

## Rule

Loosening a comparator to match a real contract invalidates any negative
control that was exploiting the stricter one. Re-derive the sabotage from the
new comparison in the same change, or the loosening silently removes the
check. And when a harness allocates a buffer the kernel only partially writes,
the initial state is part of the contract under test: copy production's, do
not pick a convenient one.
