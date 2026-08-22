# Flag deletion hardcoded decode-graph OFF — the winning literal was `false`

> Status: Fixed `cfcc5d4d9`-era follow-up. Introduced in `1864ddac5` (flag
> deletion wave), caught by the SOTA-defaults audit the same day.

## Context

Wave 1 deleted `--qwen35-decode-graph` (off-arm measured −58.7 % TPOT,
`cb6b3389d`). The seam field was kept for the OPD rollout engine, whose own
flag defaults off.

## Root Cause

The ServeArgs→flags mapping hardcoded `qwen35_decode_graph: false` — the OPD
default, not serve's. The seam doc comment said "Serve hardcodes on"; the
literal said off. The wave also fixed the runtime-flags static `false→true`,
which masked the bug in any path that reads the static before
`apply_runtime_flags`; serve calls apply, so every CUDA serve on main ran
decode-graph-less. The stale startup hint still advertised the deleted flag.

## Fix

Mapping hardcodes `true`; the disabled-branch hint now names the OPD flag.

## Rule

When hardcoding a deleted flag, the literal is the verdict — diff the mapping
site, not just the flag definition. A hardcode that contradicts the nearby
doc comment is wrong by construction; the comment states the intended value.
