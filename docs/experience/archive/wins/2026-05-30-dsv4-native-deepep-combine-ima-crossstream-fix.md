# DSv4 native-deepep combine IMA — root-caused + fixed (cross-stream race)

## Context

native-deepep (DeepEP CUDA-IPC dispatch/combine MoE backend, the +46%-over-NCCL
lever) **IMA'd on every request** at the combine step (`sync after combine: illegal
memory access`), 500'ing all traffic and blocking the lever. Three prior debug
rounds (see [`../errors/2026-05-29-dsv4-native-deepep-combine-ima-regression.md`])
ruled out DeepEP version, runtime config, env knobs, scratch sizing, and metadata
zeroing, but could not pinpoint it — the fault was a Heisenbug (compute-sanitizer
memcheck made it vanish) and looked "cross-rank/intermittent".

User reframed it correctly: *"deepep 有这么复杂吗 别的框架都正常接入了"* — DeepEP is
a standard library others integrate fine, so the bug is almost certainly OUR
integration deviating from official usage, not DeepEP complexity.

## What Worked

**Decisive control experiment first.** Ran DeepEP's OWN `tests/test_intranode.py`
(the flat `/sgl-workspace/DeepEP` build SGLang uses) on this same 8×H20 →
**EXIT=0**: full intranode dispatch+combine correctness check + tuning sweep, zero
illegal address. Proof: **DeepEP works on this hardware via the official API; the
bug is 100% our integration.**

**Arg-by-arg audit vs official runtime methods.** Our `arle_deepep_buffer_dispatch`/
`_combine` (deepep-sys/csrc/deepep_buffer.cpp) match DeepEP's
`Buffer::intranode_dispatch`/`intranode_combine` (csrc/legacy/buffer.hpp) exactly —
all 28 dispatch args, the combine args, both num_memset values, nvl_chunked,
num_sms/num_channels, every scratch size, 512MB NVL (needs ~210MB). The orchestration
was correct.

**The real deviation — the missing `stream_wait`.** DeepEP's official dispatch and
combine BOTH begin with `stream_wait(comm_stream, compute_stream)` (comm stream waits
on an event recorded on the compute stream). Our wrapper creates its **own private
CUDA stream** (`deepep_buffer.cpp:212`), runs every DeepEP op on it, host-syncs it
*after* each op, but **never orders it before against the model compute stream
`ctx.stream`**:
- dispatch reads `topk_idx_i64` (written by the route/top-k kernel on `ctx.stream`)
  before that kernel finishes → garbage routing → garbage dispatch metadata →
  garbage index in combine → **IMA**;
- combine reads `expert_out` (scatter kernel output on `ctx.stream`) before it
  finishes → wrong values.

This explains every observation: memcheck serializes the streams (masks it); the
round-3 `ARLE_DEEPEP_COMBINE_DEBUG` D2H `cudaMemcpy` itself synchronizes (so the
"metadata looks sane" dumps were also Heisenbug artifacts); layer-intermittency =
top-k-kernel-vs-dispatch timing. The "cross-rank" framing was a misread of a LOCAL
cross-stream race.

**Fix (f30043af)**: `ctx.stream.synchronize()` before dispatch and before combine in
`mlp.rs` — the minimal correct ordering (the wrapper is already host-serialized; an
event-based `stream_wait` mirroring DeepEP is a later perf tranche).

## Validation (8×H20, TP=8 multiproc)

`ARLE_DSV4_MOE_BACKEND=native-deepep ARLE_DSV4_EXPERT_BACKEND=native`, bf16 KV,
DeepEP d4f41e4, num-slots 1, max-seq 4096:
- all **8 ranks boot** (`native-deepep rank N/8 booted (peer_handles=8)`);
- requests return **HTTP 200 with full completions** (16-token greedy);
- **zero `illegal memory access` / `combine failed` / `unspecified launch failure`**
  in the server log. The IMA that 500'd every request is GONE.

## Caveat — a SEPARATE correctness bug is now exposed (not a perf win yet)

With the crash gone, native-deepep output is **numerically wrong** (`': is? is?'`
vs allreduce `'**Paris**'` on the identical config, flip-only A/B). So this fix is a
**stability/crash win, not yet a throughput win** — the +46% perf bench is blocked
until the native-deepep MoE value bug is root-caused (open issue in the errors doc).
No guidellm sweep is meaningful while output is garbage; bench deferred (`pending —
correctness-blocked`), not skipped.

## Rule

- **For a "library is too complex / buggy" hypothesis, run the library's OWN test on
  the target hardware FIRST.** DeepEP's `test_intranode.py` passing on the 8×H20 in
  one shot collapsed three rounds of "is DeepEP broken?" into "our integration omits
  one `stream_wait`."
- **A library that runs its ops on a private stream needs an explicit
  cross-stream wait against the caller's compute stream** — host-syncing the private
  stream *after* the op does not order it *before* against producer kernels on
  another stream. Mirror the library's own `stream_wait(comm, compute)` at op entry.
- **Any debug probe that does a D2H copy (or memcheck) SERIALIZES streams and will
  mask a cross-stream race** — the "metadata looks correct" dumps were the race being
  hidden by the dump's own sync. Suspect this whenever a fault vanishes under
  observation.
