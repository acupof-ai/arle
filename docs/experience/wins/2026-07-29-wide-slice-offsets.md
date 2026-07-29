# CUDA slice addresses 256K gated queries

## Context

At 256K, each gated-q half has 1.61B elements but indexes a 3.22B-element
`q_full`. The slice kernels used signed 32-bit flattened offsets.

## What Worked

Use 64-bit totals, strides, and offsets while keeping rank and dimensions
32-bit. CUDA validation is pending the current remote gate.

## Rule

Tensor dimensions may fit 32-bit while their flattened address does not.
