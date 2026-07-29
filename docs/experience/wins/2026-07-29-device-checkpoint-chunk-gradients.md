# Device-resident checkpoint chunk gradients

## Context

Sequence-chunked checkpoint replay copied every chunk gradient to host and built
the full input gradient in host memory. At 256K, that buffer alone is about
3.5 GiB for `[1, 262144, 3584]`.

## What Worked

Preallocate the input gradient on device, write each chunk into its slice, and
accumulate parameter gradients with the existing device operation. This removes
per-chunk device-to-host copies and the full host gradient buffer.

Remote 64K/128K/256K validation is pending.

## Rule

Checkpoint chunking must bound both device memory and host traffic.
