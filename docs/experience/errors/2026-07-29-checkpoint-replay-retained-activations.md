# Checkpoint replay retained completed activations

## Context

Exact `b5f078ae0` completed 64K forward and CE, then OOMed in the first
full-attention replay. Driver memory peaked at 97,483/97,508 MiB.

## Root Cause

Target-only backward removed intermediate gradients but kept every replay
activation until the whole checkpoint returned. A gated-q slice then needed a
3,072 MiB full-shape gradient with 25 MiB free.

## Fix

Free each non-target forward output after its backward entry completes.

## Rule

Checkpoint replay must release activations at their reverse last use.
