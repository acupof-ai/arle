# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT open a public issue.**
2. Email the maintainers or use [GitHub Security Advisories](https://github.com/cklxx/agent-infer/security/advisories/new) to report privately.
3. Include: description, reproduction steps, and potential impact.

We will acknowledge receipt within 48 hours and provide a fix timeline within 7 days.

## Scope

This policy covers the `agent-infer` inference engine, including:

- HTTP API endpoints (`/v1/completions`, `/v1/chat/completions`)
- Agent tool execution (`shell`, `python` tools)
- CUDA kernel memory safety
- Model weight loading and deserialization

## Agent Tool Execution

The built-in agent tools (`shell`, `python`) execute arbitrary code by design. These tools are intended for local/trusted use only. **Do not expose the agent binary to untrusted input without sandboxing.**
