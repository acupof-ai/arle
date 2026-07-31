Vendored upstream (adopt-official-first; see each entry's pin).

- `xgrammar/`: `mlc-ai/xgrammar` v0.1.34 (`d68df627908376f3ed5e0a989395a03cc41894cd`)
  — `include/`, `cpp/`, and the two headers `build.rs` validates
  (`3rdparty/dlpack/include`, `3rdparty/picojson/picojson.h`). No Python
  bindings, no tests, no kernels. Compiled only under
  `xgrammar-sys --features real`; `XGRAMMAR_SOURCE_DIR` still overrides.
