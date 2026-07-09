# DSpark partial-ctx drafting (P2.5) — prefix-restore no longer degrades to plain decode

> Status: pending-remote — pod A/B (accept split by `base>0` vs `==0`) in a later round.

## Context

[P2.5](../../plans/2026-07-09-dspark-dflash-spec-decode-qwen36.md): prefix-cache-hit
requests degraded to plain decode forever — the draft-ctx append gate required
`ctx_len == start_pos` (fresh slot 0 vs restored start_pos>0) and `pending`
required coverage from 0. At OPD's ~91% hit rate DSpark was near-inert.

## What Worked

`Qwen35DsparkSlotState` gains `ctx_base`/`ctx_end` (buffer row = abs − base;
RoPE/attention positions stay absolute via a `ctx_base`-row offset into the
cos/sin tables + buffer-relative kernel start_pos). Prefill/warm-decode rebase
the empty ctx at the gap position instead of bailing; sliding draft layers are
exact once the tail ≥ window (2048), only the 1 full-attention layer is
approximate. `ctx_base==0` reduces to the prior byte-identical arithmetic.
Telemetry: `[dspark-phase] ... accept={k} base={ctx_base} ...` splits accept
by rebased vs full-ctx chains for the pod A/B.

## Rule

Kill gate per plan §P2.5: rebased-chain accept collapsing toward ~1/16 →
KILL, go to the sidecar-the-draft-KV fallback.
