# Rule

Do not add repo-facing binaries or APIs with `smoke`-specific naming just to support one-off validation flows.

# Why

If a Rust-side helper is needed for tokenizer generation or test fixtures, keep it as a generic library helper or fold it into existing test support. Temporary validation convenience should not become permanent user-facing CLI surface.
