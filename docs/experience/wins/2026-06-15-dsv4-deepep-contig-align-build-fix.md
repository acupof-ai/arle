# DSv4 DeepEP grouped expert build fix — contig alignment argument restored

## Context

While preparing the pre-P4 DSv4 regression baseline for the CUDA quant subsystem,
the H20 pod build failed in `infer-cuda/src/moe.rs`: the DeepEP grouped-expert path
called `deepgemm_grouped_experts` with one fewer argument than its current
signature.

## What Worked

The missed call site is the DeepEP receive path. It builds offsets with the
plain `dsv4_exclusive_scan_i32` and packs exactly `recv_slots` rows, so the
contiguous grouped DeepGEMM alignment for that path is `1`. Passing `1` restores
the compile contract without changing the aligned non-DeepEP path, which already
passes its computed `contig_align`.

Verification:

```bash
CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12060 cargo clippy -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib -- -D warnings
```

Both passed locally. Full remote DSv4 build/regression continues under the CUDA
quant subsystem work.

## Rule

When a grouped DeepGEMM helper takes an alignment parameter, the value must match
the scan/pack layout that produced the offsets. Unaligned compact rows use `1`;
128/64-aligned scans pass their explicit alignment.
