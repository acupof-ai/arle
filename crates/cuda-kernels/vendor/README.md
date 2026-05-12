Vendored upstream kernels used as references for ARLE-native DeepSeek V4 work.

- `deepgemm/`: copied from `deepseek-ai/DeepGEMM` at
  `714dd1a4a980f7937a74343d19a8eba4fe321480`.
- `tilekernels/`: copied from `deepseek-ai/TileKernels` at
  `36d9e45d38e204ebb87e6f6e833821eee0482fe5`.

These sources are not linked into the default CUDA build yet. The current ARLE
path ports the required raw FP8/FP4 kernels behind C ABI entry points first, then
can replace selected kernels with direct DeepGEMM/TileKernels integrations.
