# docs/resources — reference notes

Status of each entry vs the 2026-06-04 rewrite (`e81b98fb`, monolith `infer/`
deleted; single `arle` binary):

| Entry | Status |
| --- | --- |
| `profiling-guide.md`, `infer-cuda-profiling-wrappers.md` | Current (`arle serve` + native benchmark profiling wrappers) |
| `kv-cache-quantization.md` | Concept current; re-port of the parity gate tracked in #58 |
| `metal-dflash.md`, `metal-dflash-params.md` | **Historical** — written against the deleted `metal_request`/`metal_bench`/`metal_serve` binaries. DFlash survives only as the `mlx-sys` draft-model FFI substrate; the rewrite Metal serve has no DFlash route (its spec path is MTP). |
| `eli-integration.md` | **Historical** — eli drove the deleted `metal_serve`; the rewrite entry point is `arle serve --backend metal`. |
