# Qwen3.6 NextN-MTP Speculative Decode — CUDA Port Plan

**Status:** scoped 2026-06-22 (parallel 4-reference Workflow); implementation in progress.
**Lever:** ~2-3x decode on Qwen3.6-27B-FP8, FP8-preserving. MTP head present in the
27B-FP8 checkpoint (mtp.fc + mtp.layers.0.*; mtp_num_hidden_layers=1).
**References:** Metal MTP draft (infer-metal/dflash.rs), DSv4 CUDA orchestration
(infer-cuda/executor/spec_decode.rs), core seam (infer-seam/infer-core), Qwen35
CUDA primitives (infer-cuda/qwen35.rs).

---


