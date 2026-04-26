#!/usr/bin/env bash
set -euo pipefail

ARLE_BASE_URL="${ARLE_BASE_URL:-http://127.0.0.1:8000}"

curl "$ARLE_BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello from ARLE"}],"max_tokens":64}'
