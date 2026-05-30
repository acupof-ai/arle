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
since `04938e85`):
- `67ac6400` B-3.3.5 DeepGEMM grouped expert dispatch — **LIKELY EXONERATED**:
  its FFN change is gated on `use_deepgemm_experts` (we run
  `ARLE_DSV4_EXPERT_BACKEND=native`, so that branch never executes); the only
  always-on change is `scratch.expert_out.seq_len = num_recv` (mlp.rs:117 of the
  diff), and it does NOT touch the combine's recv_channel_prefix / send_head /
  num_recv dataflow. Verify, but deprioritize.
- `07305fe9` / `7ea63e83` / `b5f20f12` — A3 phase 1 "skip counts D2H + host scan"
  (+ bug-fix + default-OFF). Changed route counts/offsets (→ recv_channel_prefix).
  A3 is **default OFF** (b5f20f12), so only the culprit if the OFF path didn't
  fully revert. Check `dsv4_a3_phase1_enabled()` gating around the
  count/offset/recv_channel_prefix computation.
- `ac1f0ccc` fp8 grouped GEMM; `5bd83267` DispatchPolicy knob refactor;
  plus anything in the big `87089f2d` dispatch-governance merge.

Because all named suspects are either deepgemm-gated or A3-default-OFF, pure
code-reading did NOT pinpoint it — **compute-sanitizer (exact OOB address +
kernel) or a proper git-bisect is the decisive next tool**, not more grepping.

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

## Update 2026-05-30 — confirmed GENUINE at CLEAN true-HEAD (not a stale-tree artifact)

Critical caveat on the earlier "ruled out" work: the pod `/data01/build/arle`
was a stale scp snapshot, and full-tree syncs kept getting SIGTERM-interrupted
(kubectl-exec kills long `tar -x`), so the prior native-deepep tests ran on
inconsistent PARTIAL trees — untrustworthy. After a CLEAN sync
(`rm -rf infer && tar -xf infer-head2.tar` IN TMUX, then build; BUILD_EXIT=0,
record_linear_kernel present) the pod is verified true-HEAD (the default
allreduce decode now passes needle 4/4 to 2272 tok — see
[[../wins/2026-05-30-dsv4-oplib-linear-decode-fix-true-head]]).

Re-ran native-deepep (multiproc + FUSED_DISPATCH_PAYLOAD default + DeepEP
d4f41e4) on this CLEAN true-HEAD: **combine STILL IMAs** (`sync after combine:
illegal memory access`). So the combine bug is GENUINE at HEAD, not an artifact.

compute-sanitizer (memcheck) attempt FAILED to boot: memcheck slows the 8-proc
NCCL TCP-store rendezvous past its hardcoded 30s timeout
(`SOCKET_TIMEOUT = Duration::from_secs(30)`, infer/src/distributed/init_method.rs:36)
→ rank-0 "accept rank 7 timed out after 30s". 0 errors reported (forward never ran).

**Precise next steps (fresh session, reliable pod):**
1. Make `SOCKET_TIMEOUT` env-configurable (init_method.rs:36) → e.g. 600s → re-run
   `compute-sanitizer --target-processes all --tool memcheck` → get the exact OOB
   buffer + access in the combine kernel.
2. OR a 2-rank repro harness (world_size=2) — small enough for memcheck to boot.
3. Source narrowing already done (all RULED OUT at the d4f41e4 contract):
   combine host/kernel signature matches the deepep-sys wrapper exactly; nvl_chunked
   (6,256) consistent dispatch↔combine; num_sms=20/num_channels=10 consistent
   dispatch↔combine↔scratch; kNvlBytes=512MB generous; dispatch passes, combine
   OOBs. The OOB is a runtime value/buffer issue only compute-sanitizer will pinpoint.

## Update 2026-05-30 (round 2) — Heisenbug: uninitialized-read, but not fully pinpointed

Breakthrough diagnosis + partial fixes (all committed), combine STILL faults:

- **Heisenbug confirmed**: under `compute-sanitizer --tool memcheck` (which
  zero-fills every allocation) the native-deepep combine WORKS (HTTP 200); WITHOUT
  it, IMA. So the root cause is an UNINITIALIZED-MEMORY READ (memcheck's zero-fill
  masks it). [Needed `ARLE_RENDEZVOUS_TIMEOUT_SECS` (new env, commit) for memcheck
  to boot the 8-proc NCCL rendezvous past 30s.]
- **Fixes applied** (defensive, correctness-safe, committed): zero-init the
  native-deepep dispatch-metadata scratch (1e578148: send_head / rank_prefix /
  recv_channel_prefix / channel_prefix_matrix / recv_src_idx), the DeepEP NVL IPC
  buffer + workspace (cfce577f: cudaMemset local_buf 512MB + workspace), and ALL
  remaining state.rs scratch (d8776131: alloc_traced→alloc_zeros_traced).
- **Symptom progression**: IMA ("illegal memory access") → after metadata zeroing
  → "unspecified launch failure" → after NVL + all-scratch zeroing → still
  "unspecified launch failure". So zeroing every REACHABLE buffer is NOT enough.
- **`--tool initcheck`** (detects uninit reads WITHOUT zero-masking) flagged uninit
  reads only in `cublasLt::splitKreduce_kernel` (a cuBLAS split-K GEMM workspace —
  the expert GEMM via cublasLtMatmul), a known mostly-benign cuBLAS initcheck
  false-positive, NOT the combine. The combine request TIMED OUT (504) under
  initcheck, so it didn't reproduce there either.

**Conclusion**: the remaining uninitialized read is in memory NOT reachable by the
Rust/wrapper zeroing — most likely a DeepEP-internal allocation, OR an IPC-peer
buffer ordering issue (each rank zeros its own local_buf at create, but the combine
may read a peer's buffer region the peer's dispatch never wrote), OR a true race
that memcheck's serialization masks. Tools available can't cleanly pinpoint it
(memcheck masks it; initcheck times out + flags only benign cuBLAS).

**Next (fresh focused effort):**
1. `--tool racecheck` to rule in/out a race (the other thing memcheck serializes away).
2. Instrument the DeepEP intranode combine kernel directly (add bounds asserts /
   printf on the index it derives from send_head/recv_channel_prefix at the faulting
   thread) — source is at /data01/build/DeepEP/csrc/kernels/legacy/intranode.cu:706.
3. Check the IPC-peer-buffer write/read ordering: does the combine read a region of
   a PEER's local_buf that that peer's dispatch leaves unwritten? Zero is applied
   per-rank-local; a cross-rank unwritten region is the prime remaining suspect.

The defensive zero-init commits are kept (uninit reads are real bugs regardless);
native-deepep remains non-functional at HEAD until the cross-rank/DeepEP-internal
uninit (or race) is found.

## Update 2026-05-30 (round 3) — send_head RULED OUT; fault is cross-rank/intermittent

Instrumented the deepep-sys wrapper (host-side D2H dumps of the metadata, env
`ARLE_DEEPEP_COMBINE_DEBUG`) — decisive:

- **send_head was a RED HERRING.** DeepEP indexes `send_head[token_idx * kNumRanks
  + rank]` (intranode.cu:359/670, cached_notify_combine:33/44) → shape
  `[num_recv_tokens × kNumRanks]`, so the original `× ep_world` allocation was
  CORRECT. My debug dump mis-sized it (`× num_channels`) and the resulting
  `cudaMemcpy invalid argument` mis-led me. Reverted.
- **dispatch-output == combine-input.** Dumping `recv_channel_prefix` + `rank_prefix`
  right after `dispatch` AND right before `combine` shows IDENTICAL values every
  layer — so there is NO ordering / overwrite / routing bug; rank-0's local
  metadata is correct and sane (e.g. L0 num_recv=48 chan=0,1,2,3 rank=6,0,0,0).
- **Fault is layer-intermittent.** Combine succeeds for L0 (num_recv=48) and L1
  (num_recv=0 — legitimate: no tokens routed to rank-0's experts that layer), then
  faults at L2 (num_recv=32) with "sync after combine: unspecified launch failure".
  No local-metadata difference explains why L2 faults but L0 doesn't.

**Refined conclusion**: combine is a CROSS-RANK kernel (each rank reads peers'
`local_buf` via CUDA IPC, indexed by its own prefixes). All RANK-LOCAL inputs are
verified correct. The intermittent (layer-specific) + memcheck-masked fault now
points to a **cross-rank race or a peer-buffer region rank-0 reads that the peer's
dispatch left unwritten** — NOT a local sizing/uninit bug (those are ruled out).

**Next (focused DeepEP-kernel debug session):**
1. `compute-sanitizer --tool racecheck` (the one thing memcheck serializes away).
2. Dump a PEER's metadata (not just rank-0 local) at the faulting layer — does the
   peer rank-0 reads from have valid prefixes/send_head for that layer?
3. Audit the cross-rank barrier between dispatch and combine in cached_notify_combine
   vs the [[feedback_deepep_combine_uses_recv_channel_prefix]] /
   [[feedback_deepep_kernel_api_inverted_naming]] contracts — a barrier_signal
   mismatch would let rank-0 combine before a peer finished its dispatch write.

The defensive zero-init commits (metadata + NVL + all-scratch) are kept but did NOT
fix it (they changed IMA→launch-failure via the send_head red-herring zeroing).
native-deepep remains non-functional at HEAD.
