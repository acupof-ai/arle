# DSv4 native-deepep — combine IMA regression at HEAD (blocks the +46% lever)

## SLO-shape probed? — partial (smoke 137+269 fails before any throughput shape)

## Context

After the RoPE long-context fix shipped, the chosen next optimization was
native-deepep (the +46% lever from
[`../wins/2026-05-27-dsv4-native-deepep-perf-ab.md`]). Built infer with
`ARLE_DEEPEP_DIR` + the deepep-sys static archive, served multiproc, and hit a
hard runtime failure: native-deepep boots and **dispatch** works, but **combine
hits an illegal memory access** on every request — even a 16-token smoke
(`Compute 137 + 269`). HTTP 500:
`native-deepep combine failed: deepep call returned status -2: sync after combine: an illegal memory access was encountered`.

## What works / what's confirmed

- **Boot OK** (multiproc): `ARLE_MULTIPROC_SERVE=1` → coordinator forks 7
  workers, relay accepts 7 connects, `native-deepep rank N/8 booted
  (peer_handles=8)` for all 8 ranks. CUDA-IPC handle exchange succeeds.
  (Single-process `INFER_CUDA_DEVICES` mode fails earlier at boot:
  `cudaIpcOpenMemHandle: invalid device context` — DeepEP needs separate
  processes; multiproc is mandatory.)
- **Dispatch OK** with `ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1` (toolchain default).
  Without it, dispatch IMAs ("sync after dispatch"); with it, dispatch passes
  and the failure moves to combine.

## Ruled out (the combine IMA is NOT any of these)

1. **DeepEP version.** Tested BOTH `/sgl-workspace/DeepEP` (flat
   `csrc/kernels/api.cuh` layout) AND `/data01/build/DeepEP @ d4f41e4` (the
   `csrc/kernels/legacy/` refactored layout the deepep-sys wrapper expects AND
   the version the +46% A/B used). deepep-sys build.rs auto-probes both layouts
   and compiles cleanly against either — combine IMAs at BOTH. (Use
   `/data01/build/DeepEP` for builds regardless — it's the wrapper-matched one.)
2. **Runtime config.** Reproduced the +46% A/B config exactly — `--kv-cache-dtype
   bf16`, `--num-slots 1`, `--mem-fraction-static 0.10`, FlashMLA prefill+decode
   OFF — combine still IMAs. So it is NOT my fp8 / FlashMLA / num-slots-4 config.
3. **Env knobs.** `ARLE_DSV4_PADDED_DISPATCH=1` does not help.

## Root cause (localized, not yet pinpointed)

A **genuine combine regression at HEAD**, introduced sometime in the **233
commits** between the +46% build (main `04938e85`, 2026-05-27) and now
(`7ae45ce0`). The combine call site `mlp.rs:5340-5362` passes
`num_input_tokens=num_recv`, `num_output_tokens=hidden.seq_len`,
`d_recv_channel_prefix=rchan_ptr`, `d_send_head=shead_ptr` — the params are
built from the **dispatch output**, and a commit that changed the dispatch
output layout would leave these stale → out-of-bounds read in the combine
kernel → IMA. Consistent with the DeepEP combine API hazards in
[[feedback_deepep_kernel_api_inverted_naming]] /
[[feedback_deepep_combine_uses_recv_channel_prefix]].

**Suspect commits** (touch native-deepep / deepep-sys / mlp.rs dispatch+combine
since `04938e85`), ranked:
1. `67ac6400` feat(cuda): B-3.3.5 wire DeepGEMM grouped expert dispatch into
   native-deepep — **top suspect**; changed the dispatch output layout the
   combine consumes.
2. `07305fe9` / `7ea63e83` / `b5f20f12` — A3 phase 1 "skip counts D2H + host
   scan on decode hot path" (and its bug-fix + default-OFF). Changed how
   route counts/offsets (→ recv_channel_prefix) are computed.
3. `ac1f0ccc` fp8 grouped GEMM kernels; `5bd83267` DispatchPolicy knob refactor.

## Next (focused debug — a fresh effort, not tail-of-session)

- **git-bisect** the 6 suspects: build each (`ARLE_DEEPEP_DIR=/data01/build/DeepEP`),
  multiproc-serve, smoke `137+269`. ~15-20 min/test. First-bad commit pinpoints it.
- OR **compute-sanitizer** the combine on a 2-rank run to get the exact OOB
  address + kernel, then diff the dispatch→combine data-flow (num_recv,
  rchan_ptr, shead_ptr provenance) vs `04938e85`.
- OR diff `mlp.rs` dispatch+combine and `crates/deepep-sys` between `04938e85`
  and HEAD; the stale-layout arg will be visible.

## Rule

- **A "+X% proven" lever is only proven at the commit it was measured on.**
  Before re-investing in native-deepep (or any backend last benched weeks +
  hundreds of commits ago), smoke-test it at HEAD FIRST — it may have silently
  regressed. The +46% was main `04938e85`; HEAD combine is broken.
- native-deepep run-reqs (so the boot/dispatch stages aren't re-discovered):
  `ARLE_MULTIPROC_SERVE=1` + `ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1` +
  `ARLE_DEEPEP_DIR=/data01/build/DeepEP` (the d4f41e4 legacy layout) +
  `ARLE_DSV4_EXPERT_BACKEND=native`. See [[project_h20_pod_access]].
