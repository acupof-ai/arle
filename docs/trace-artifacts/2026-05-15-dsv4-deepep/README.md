# DSv4 DeepEP Trace Artifacts

Date: 2026-05-15

These artifacts were captured on the remote 8xH20 host against the real
`/root/DeepSeek-V4-Flash` checkpoint. The run uses the default DSv4 DeepEP MoE
path with FP8 KV cache and incremental KV enabled.

Current trace set:

- [`nsys-one-token-current/`](nsys-one-token-current/) isolates one generated
  decode token with Nsight Systems.
- [`shared-expert-scratch/`](shared-expert-scratch/) records the shared expert
  scratch cleanup, including trace-off math/writing smoke and a single-token
  Nsight comparison.
