#!/usr/bin/env bash
# Stream a chat reply from a running LLM Chatbot Template server over SSE.
# Start the server first:  llm-chatbot-template serve   (or: docker compose up)
# Then run:                ./examples/curl_stream.sh "Hello from curl"
# Part of LLM Chatbot Template by Viprasol Tech (https://viprasol.com).
set -euo pipefail

HOST="${CHATBOT_HOST:-http://127.0.0.1:8000}"
SESSION="${CHATBOT_SESSION:-curl-demo}"
MESSAGE="${1:-Hello there!}"

curl -N -s -X POST "${HOST}/chat/stream" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"${SESSION}\", \"message\": \"${MESSAGE}\"}"
echo
