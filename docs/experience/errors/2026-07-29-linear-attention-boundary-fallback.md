# Linear-attention boundary lost its reference fallback

## Context

Frozen writeback parity failed after boundary capture became CUDA-only.

## Root Cause

CPU and CUDA prefixes shorter than the convolution window cannot use the
streaming boundary kernel.

## Fix

Use the streaming kernel only with a full window; otherwise reuse the host
reference. Long prefixes stay device-only.

## Rule

An optimized boundary op keeps a small-shape correctness path.
