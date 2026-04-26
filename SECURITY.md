# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| `main` | Yes |
| Latest tagged release | Yes |
| Older releases | Best effort |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT open a public issue.**
2. Use [GitHub Security Advisories](https://github.com/cklxx/arle/security/advisories/new) to report privately.
3. If the advisory flow is unavailable, open a minimal public issue requesting a private contact channel **without** including exploit details or reproduction steps.
4. Include: description, reproduction steps, affected surface, and potential impact.

We will acknowledge receipt within 48 hours and provide a fix timeline within 7 days.

## Scope

This policy covers ARLE, including:

- HTTP API endpoints (`/v1/completions`, `/v1/chat/completions`)
- Agent tool execution (`shell`, `python` tools)
- CUDA kernel memory safety
- Model weight loading and deserialization

## Agent Tool Execution

The built-in agent tools (`shell`, `python`) execute arbitrary code by design.
These tools are intended for local/trusted use only. **Do not expose the
tool-enabled local agent to untrusted input.**

Use:

```bash
arle --doctor
arle --model-path /path/to/model run --no-tools --prompt "..."
```

`arle --doctor` reports the active tool sandbox backend:

- `nsjail`: preferred Linux sandbox path when installed.
- `sandbox-exec`: macOS fallback when available.
- `bare`: no supported sandbox backend was detected.

If the backend is `bare`, use `--no-tools` for untrusted prompts or install a
supported sandbox before enabling tool execution.
