# File imbalance came from orthogonal axes expanded into method names

> Status: Confirmed

## Context

The observation that started this: the codebase looked unbalanced, some files far
too large and others too small. Line counts confirmed the imbalance but named the
symptom rather than the cause. The cause is that orthogonal axes were expanded
into method names instead of being taken as parameters, which produces one
2000-4000 line impl block of near-duplicate methods per file.

Measured before any edit:

| File | Monolithic impl | Share of file |
|------|-----------------|---------------|
| `infer-cuda/src/executor/dsv4.rs` | `impl Dsv4CudaExecutor` 2109 | 84% of 2506 |
| `infer-cuda/src/dsv4.rs` | `impl Dsv4Model` 4348 | 72% of 6018 |
| `train/src/qwen35.rs` | `Qwen35Layer` 2365 + `Qwen35Model` 2319 | 67% of 7025 |
| `cli/src/train_cli.rs` | eight `run_*` drivers, 2889 | 44% of 6545 |
| `autograd/src/backend_cuda.rs` | `impl Backend for CudaBackend` 3080 | 32% of 12875 |

`dsv4.rs` had 29 top-level items for 6018 lines: 207 lines per item.

## What Worked

**Naming the axes, not the line count.** `impl Qwen35Layer`'s 27 methods are a
cross-product:

```
{full_attention, linear_attention}
  x {plain, capture_prefix, gen_segment, with_kv_cache}
  x {plain, profiled}
{mlp, sparse_mlp} x {plain, collecting_routes, frozen_routes}
```

Three booleans became 27 methods. Once stated that way the fix is a parameter,
not a file split.

**Measuring similarity instead of eyeballing it.** Extracting each method body by
brace-matching and diffing line by line gave the numbers that made the collapse
safe to attempt: `forward_sparse_mlp` ~ `_collecting_routes` 91.6%,
`_collecting_routes` ~ `_with_frozen_routes` 87.1%,
`forward_full_attention_with_kv_cache` ~ `_profiled` 84.8%.

**Counting call sites.** Every one of the 13 variant methods had exactly one call
site (two had two). A variant with one call site is an expanded call tree, not an
API — that single number justified the collapse better than the similarity
percentages did.

**Letting the measurement veto the plan.** The MoE-route axis collapsed cleanly,
8 methods to 3 behind `MoeRouteMode{Free, Collect, Frozen}`. `Qwen35Layer::forward`
did not: it wraps the MLP in `checkpoint_seq_chunked` with a `move` closure that
cannot capture the `&mut` sink, and chunking would split route signatures across
chunks. That is a real semantic difference. The two layer-level route drivers were
85% identical to each other and merged; `forward` stayed separate.

**Adding the field the parameter was standing in for.** Four of these methods
threaded `layer_index` by hand. A layer does have a position in the stack, so
`Qwen35Layer` gained an `index` field, set at its single construction site. The
frozen-signature cursor then moved into the sparse-MLP path, where only sparse
layers consume a signature.

**Verifying in a clean worktree.** Another session was mid-refactor on
`infer-cuda/src/attention.rs` in the same tree, and the shared tree did not
compile. A `git worktree` at HEAD with only the changed files overlaid separated
the two: 51 errors were theirs, and after overlaying the files this change also
touched, zero were this change's. Attribution inside a shared dirty tree is
guesswork.

Result: `qwen35.rs` 7024 to a 279-line facade over 16 modules, net -124 lines
despite 16 new module headers. `dsv4.rs` 6013 to 95 lines over 14 modules.
`executor/dsv4.rs` 2551 to 7 modules.

## Rule

A file is rarely too long on its own; it is long because an axis that should be a
parameter was written into method names. Name the axes and count the call sites
before proposing a split — a method with one call site is a step in an expanded
call tree, and moving it to another file preserves the duplication while hiding
it. Splitting by "this chunk compiles independently" is what leaves a codebase
with a few monoliths and a scatter of 90-150 line fragments and nothing in
between; split on one concept per module instead.
