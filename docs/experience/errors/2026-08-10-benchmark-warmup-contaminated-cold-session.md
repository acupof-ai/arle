# Benchmark warmup contaminated one cold session

> Status: Fixed

## Context

The first point of the re-anchored Qwen3.6-27B sweep reported 113 prefix hits
over 128 requests. With 16 sessions and 8 turns, exactly 112 requests should
reuse a prior measured turn.

## Root Cause

`scripts/bench_throughput.py` warmed the server with `prompts[0]`. That inserted
the first session's turn-0 prefix into the cache before measurement, creating
the extra hit. Later ascending concurrency points were already fully warm and
did not expose the contamination.

## Fix

The warmup now prepends a dedicated marker to the same production-length
prompt. It exercises the same shape without sharing the measured prompt's
prefix. The aborted baseline is retained only as diagnostic evidence; the
current fingerprint requires a fresh sweep.

## Rule

Benchmark warmup inputs must be disjoint from measured cache keys.
