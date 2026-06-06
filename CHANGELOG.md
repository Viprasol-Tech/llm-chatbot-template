# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[SemVer](https://semver.org/).

## [0.1.0] - 2025-06-06

### Added
- Initial release of **llm-chatbot-template**: a streaming LLM chatbot backend
  template built on FastAPI.
- `POST /chat/stream` — Server-Sent Events (SSE) token-by-token streaming reply.
- `POST /chat` — blocking JSON reply with running token estimate.
- In-memory, per-`session_id` conversation memory (`SessionStore`) with `POST /reset`.
- Pluggable `LLM` protocol with `complete()` and `stream()`; deterministic offline
  `FakeLLM` (aliased `EchoLLM`) and an optional, lazily-imported `OpenAIClient`.
- `ChatSession` with system prompt, history, prompt assembly, and oldest-first
  token-budget trimming.
- Self-contained static HTML chat page served at `/`.
- Typer CLI: `demo` (offline, optionally `--stream`), `serve` (uvicorn), `version`.
- Settings via `pydantic-settings` (`CHATBOT_*` env vars / `.env`).
- Dockerfile + docker-compose, runnable `examples/`, and `.env.example`.
- Tooling: ruff, mypy (strict), pytest, and GitHub Actions CI across
  Ubuntu/macOS/Windows on Python 3.11–3.13.

[0.1.0]: https://github.com/Viprasol-Tech/llm-chatbot-template/releases/tag/v0.1.0
