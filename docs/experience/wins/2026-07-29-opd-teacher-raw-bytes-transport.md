# OPD API-teacher raw-bytes transport — kill the multi-GB JSON parse, 2026-07-29

> Status: pending-remote (typecheck + CPU tests pass; H20 wall-clock pending)

## Context

`arle train opd --teacher-runtime api` ran at ~13-20 min/step on H20: one CPU
core pinned at 100%, GPU 0-6% idle, RSS oscillating 8.2↔9.5 GB. A 2000-step
curve would take ~4 weeks — infeasible. Stack sampling + RSS oscillation
(repeated ~1.3 GB alloc/free) localized it to the teacher-logits client path,
not rollout or LoRA re-merge.

## Root cause

The teacher returns `[seq, vocab=248320]` logits (~1 GB bf16). The old wire
format base64-encoded that to a ~1.35 GB string, wrapped it in JSON, and the
client parsed the multi-GB JSON string on one core (`ureq .into_json()`), then
base64-decoded, then element-by-element bf16→f32. The JSON string parse was the
dominant cost — `O(seq·vocab)` single-core, and it looked rollout-length-
independent because vocab dwarfs the rollout-length delta in `seq·vocab`.

## What Worked

Raw-bytes transport. The route (`infer-api/.../raw_logits_route.rs`) now returns
`application/octet-stream`: the raw bf16-LE `[seq, vocab]` block with shape in
`x-logits-rows`/`x-logits-cols` headers — no base64, no JSON body. The client
(`train/teacher_infer.rs`) reads the byte body directly (`into_reader().
read_to_end`) and bulk-decodes bf16→f32. Dropped `base64` from both crates'
`Cargo.toml` and deleted the dead JSON response struct + f32/base64 decode
helpers. Numerically identical (same bf16 values on the wire), so no correctness
gate beyond the existing parity.

## Cost

None on quality — same bf16 payload, just without the string round-trip. The
server still does one `for v in &host` f32→bf16 encode loop (`O(seq·vocab)`,
single-core); if that shows up in the H20 profile it's the next lever (bulk
`half` slice cast). The bigger remaining lever is engine-side `teacher_topk`
(payload 248320→k), still hard-disabled at `train/opd.rs` pending engine work.

## Rule

A localhost HTTP hop that moves a ~GB tensor must not serialize it as a JSON
string — the multi-GB string parse dominates and pins one core. Use a raw byte
body + headers for the shape; keep JSON only for the small request.

## Verification (pending-remote)

Mac has no nvcc; CI Lint mirror typecheck + clippy pass
(`CUDARC_CUDA_VERSION=12080 cargo {check,clippy} -p infer-api -p train
--no-default-features --features cuda,no-cuda`). Train CPU tests pass, including
`api_teacher_fetches_http_logits_into_tensor_store` (real TCP mock exercising the
raw-bytes + header protocol end-to-end). H20 gate: re-run the OPD step and
confirm step wall-clock drops from ~13 min to seconds-scale via
`ARLE_OPD_STEP_PROFILE=1` (`teacher_forward_seconds` should collapse).
