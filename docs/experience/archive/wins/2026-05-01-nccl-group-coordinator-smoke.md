# nccl group coordinator smoke

## Context

F0 single-node multi-GPU first landing tranche:
`feat(cuda): add nccl group coordinator smoke behind nccl feature`.

The goal is a foundation proof, not a throughput claim: two rank threads create
CUDA contexts on two devices, exchange one NCCL unique id through ARLE's TCP
rendezvous, run one `all_reduce(sum)` over `[f32; 3]`, and verify both ranks
receive `[5.0, 7.0, 9.0]`.

## What Worked

- `infer` feature `nccl` now enables `cudarc/nccl`.
- `infer/src/distributed/nccl.rs` adds `NcclGroup` with `TcpStore` and
  `EnvBootstrap` init methods.
- `infer::distributed::smoke_2_thread_all_reduce()` spawns two rank threads and
  asserts the all-reduce result on both ranks.
- `infer/tests/distributed_nccl_smoke.rs` runs the smoke when cudarc reports at
  least two CUDA-visible GPUs, otherwise it skips.
- `infer --nccl-smoke` runs the same smoke and exits before server startup.

Validation:

```bash
ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig \
  cargo check -p infer --features nccl

ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig \
  cargo test -p infer --features nccl distributed_nccl_smoke
```

Result:

```text
cargo check -p infer --features nccl: passed
distributed_nccl_smoke: passed, 1 test, not skipped
```

Clippy note:

```bash
ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig \
  cargo clippy -p infer --features nccl -- -D warnings
```

Result: failed on existing crate-wide pedantic warnings outside the F0 NCCL
change path. After fixing this patch's own `needless_pass_by_value` warning,
the remaining errors are in existing model, scheduler, ops, weight-loader, and
P2.3 speculative files.

## Rule

For foundation runtime work with no throughput claim, record the operational
smoke result as the evidence artifact instead of creating a GuideLLM benchmark.
