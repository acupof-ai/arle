# Fixed-output benchmark allowed early EOS

> Status: Fixed

## Context

The Qwen3.6-27B baseline uses 128 requests per concurrency point with a fixed
214-token output shape. On the current corrected model, dataset index 11 ended
at EOS on the first generated token and aborted the c=1 point at 127/128.

## Root Cause

`scripts/bench_throughput.py` sent `max_tokens=214` without `ignore_eos=true`.
The request was valid model behavior but did not have the benchmark's fixed
output shape. The streaming and non-streaming endpoints both reproduced the
empty decoded output, and disabling DSpark did not change it.

## Fix

The canonical runner now always sends and records `ignore_eos=true`. On the
same binary, model, prompt, and GPU, that request completed 214 tokens with a
non-empty 877-character output. The empty-output correctness gate remains
unchanged.

The aborted sweep is invalid for performance comparison. A fresh full sweep is
required because the request fingerprint changed.

## Rule

A fixed-output benchmark must force the output cap; EOS-sensitive evaluation
must use an evaluation runner.
