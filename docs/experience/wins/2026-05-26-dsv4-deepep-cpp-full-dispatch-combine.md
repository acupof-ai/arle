# DSv4 native DeepEP — pure-C++ 8-rank intranode dispatch + combine round-trip PASS (phase 1.0a-iv)

## Context

Phase 1.0a-iii
([`./2026-05-26-dsv4-deepep-cpp-full-dispatch.md`](./2026-05-26-dsv4-deepep-cpp-full-dispatch.md))
landed pure-C++ 8-rank intranode dispatch with byte-deterministic output
but combine deadlocked on `channel_tail_idx > expected_head=0` waits even
after fixing the `num_tokens / num_recv_tokens` inversion. Phase 1.0a-iv
roots and fixes that deadlock and closes the full dispatch+combine
round-trip from a torch-free binary.

## What worked — root cause + fix

`intranode::combine` deadlocked because the spike was passing the
**wrong** `channel_prefix_matrix` to it:

- **Dispatch input** `channel_prefix_matrix` (written by `notify_dispatch`):
  INCLUSIVE prefix; `[d][i]` = cumulative count of tokens this rank
  sends to dst `d` through channels `[0..i]`. Used only by the dispatch
  SENDER to populate `channel_start_offset` / `channel_end_offset`.
- **Dispatch output** `recv_channel_prefix_matrix` (written by the
  dispatch RECEIVER into the param the kernel signature names
  `recv_channel_offset`): EXCLUSIVE prefix in the recv-side rank's local
  recv_x; `[s][i]` = number of tokens this rank received from src `s`
  through channels `[0..i-1]`.

Upstream `Buffer::intranode_combine` reuses the parameter name
`channel_prefix_matrix` for the *recv-side exclusive* matrix. Python's
handle unpack at `deep_ep/buffer.py:424` is the smoking gun:

```python
rank_prefix_matrix, _, channel_prefix_matrix, src_idx, ... = handle
```

The `_` discards the dispatch input, and the third handle slot
(`recv_channel_prefix_matrix`) gets rebound to `channel_prefix_matrix`
for the combine call.

Fix: pass `d_recv_channel_prefix` instead of `d_channel_prefix_matrix`
into `intranode::combine`. Single-variable change; combine PASSes
immediately.

### Diagnostic confirming the prefix swap

For rank 0 with my symmetric routing (rank R sends to ranks `(R..R+5) mod 8`):

| matrix | row 0 (cols 0..9) | shape |
|---|---|---|
| dispatch input `channel_prefix_matrix` | `[1,1,1,1,1,1,1,1,1,1]` | inclusive cumsum |
| dispatch output `recv_channel_prefix_matrix` | `[0,1,1,1,1,1,1,1,1,1]` | exclusive cumsum |

Combine sender computes
`num_channel_tokens = channel_prefix[c+1] - channel_prefix[c]`. With the
inclusive matrix, channel 0 yields `1 - 1 = 0` (sender does nothing).
With the exclusive matrix, channel 0 yields `1 - 0 = 1` (sender works).

### Run output (8 × H20, no torch, no Python, no pybind, no nvshmem)

```
[parent] cuda devices=8, ranks=8, shape=(tokens=1, hidden=4096, topk=6, experts=256)
[parent] gathered handles
[parent] rank 0 num_recv_tokens=6 sha256=d7b803d2…0abb1 first8={0.0000,0.0006,0.0012,0.0018,0.0024,0.0030,0.0036,0.0042}
[parent] rank 1 num_recv_tokens=6 sha256=f145ea86…cc3f1 first8={6.0000,6.0000,…,6.0000}
[parent] rank 2 num_recv_tokens=6 sha256=f7313a3f…75e0c first8={12.0000,…,12.0000}
[parent] rank 3 num_recv_tokens=6 sha256=a4769df0…baa4d2 first8={18.0000,…,18.0000}
[parent] rank 4 num_recv_tokens=6 sha256=c37dc5a2…197a79 first8={24.0000,…,24.0000}
[parent] rank 5 num_recv_tokens=6 sha256=b192fa2a…b5fdbd first8={30.0000,…,30.0000}
[parent] rank 6 num_recv_tokens=6 sha256=41909788…f206f0 first8={36.0000,…,36.0000}
[parent] rank 7 num_recv_tokens=6 sha256=2da30636…68c912f first8={42.0000,…,42.0000}
[parent] PASS
```

Math check: rank R's synthetic input is `bf16(R + j*1e-4)`. Six dest
ranks each receive one copy; in this spike there is no expert step, so
combine sums six identical copies of `bf16(R + j*1e-4)` back to rank R.
Output[j] ≈ `6R + 6j*1e-4`. Rank 0 first 8 values are `{0, 0.0006,
0.0012, 0.0018, ...}` = `6 * j * 1e-4` — exact match. Ranks 1..7 saturate
the bf16 mantissa on the `6R` integer term, dropping the `6j*1e-4`
contribution into rounding; values are `{6, 6, ...}`, `{12, 12, ...}`,
... up to `{42, 42, ...}`.

### Byte-determinism

Two consecutive re-runs of the same binary produce **identical sha256
on all 8 ranks** — combine output is reproducible at the byte level
across runs.

## Why this is the architecture gate

Dispatch + combine round-trip from a pure C++ binary, no torch / no
Python / no NVSHMEM, with byte-deterministic output, is the architecture
license for native DeepEP integration via process-per-rank sidecars:

1. The DeepEP kernel layer is fully driveable from C++.
2. The 8-rank fork+IPC handshake is stable.
3. Dispatch and combine semantics are now mechanically replicable from
   `Buffer::intranode_{dispatch,combine}` after capturing the two
   parameter-naming traps:
   - `num_tokens` / `num_recv_tokens` are inverted from intuition (phase
     1.0a-iii;
     [`feedback_deepep_kernel_api_inverted_naming.md`](../../../../.claude/projects/-Users-user-code-agent-infer/memory/feedback_deepep_kernel_api_inverted_naming.md)).
   - `channel_prefix_matrix` in `combine` is the dispatch OUTPUT
     `recv_channel_prefix_matrix`, NOT the dispatch INPUT
     `channel_prefix_matrix` (this phase;
     [`feedback_deepep_combine_uses_recv_channel_prefix.md`](../../../../.claude/projects/-Users-user-code-agent-infer/memory/feedback_deepep_combine_uses_recv_channel_prefix.md)).

Phase 1.1 production sidecar (the actual `arle-deepep-sidecar` binary,
shared-memory IPC, and `ARLE_DSV4_MOE_BACKEND=native-deepep` runtime
wiring) is unblocked.

## What is NOT yet validated

- Perf — pure C++ binary did **not** measure dispatch+combine latency,
  bandwidth, or per-layer wall-clock contribution. Phase 1.5 SLO bench
  A/B vs the existing NCCL DeepEP-style path is the next quantitative
  gate.
- Expert step — spike skipped the expert FFN between dispatch and
  combine; output is `sum(6 * recv_x)`, not the real
  `sum(weight_k * expert_k(recv_x))`. ARLE's real integration plugs the
  DeepGEMM/native expert path between dispatch and combine.
- Numerical parity with upstream Python — sha256s here are reproducible
  intra-binary but were not compared byte-for-byte with the
  `tests/test_intranode.py` reference run on the same input pattern.
  Phase 1.1 will wire a side-by-side smoke before production sidecar
  ships.
- Internode (RDMA) path — `intranode` only.
  `internode::{dispatch,combine}` and `internode_ll::*` are separate
  kernel families with their own host-side wrappers; the same
  parameter-naming traps probably apply but need a dedicated spike.

## Artifacts

- Spike source: `/tmp/phase1a_iii_spike.cpp` (24.6 KB).
- Build: `nvcc -DDISABLE_NVSHMEM --expt-relaxed-constexpr
  --expt-extended-lambda -gencode arch=compute_90,code=sm_90 …
  intranode.cu layout.cu runtime.cu phase1a_iii_spike.cpp -lcudart`
  (build script `/tmp/phase1a_iii_build.sh`).
- Memories landed:
  `feedback_deepep_combine_uses_recv_channel_prefix.md`,
  `feedback_deepep_kernel_api_inverted_naming.md` (companion).

## Rule

When porting `Buffer::intranode_combine` to a torch-free C++ caller,
**read the Python handle unpack at `deep_ep/buffer.py` line ~424
before the C++ kernel signature** — DeepEP re-uses the
`channel_prefix_matrix` name for two semantically different tensors
(send-side inclusive vs recv-side exclusive prefix). The combine kernel
wants the recv-side. Same lesson applies anywhere DeepEP's torch
wrapper renames an output tensor mid-flow: the kernel's parameter name
is not authoritative.
