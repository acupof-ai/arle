# Rule

The project brand is `ARLE`. User-facing CLI/docs/site/tooling should default
to `ARLE` / `arle`, with `agent-infer` retained only where a deliberate
compatibility fallback is needed.

# Why

The user explicitly corrected the project name during the PR-7 merge tranche.
Leaving mixed branding in binaries, scripts, exported filenames, or the landing
page creates avoidable DX friction and makes the rename feel half-finished.

# Preventive action

- Prefer `ARLE` in headings, banners, status text, and default filenames.
- Prefer `arle` for the root CLI binary and shell examples.
- When an old env var / file path is still widely used, support it as a legacy
  fallback but document the new `ARLE`-first path.
