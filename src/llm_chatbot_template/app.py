"""FastAPI application: a streaming chat backend with in-memory session memory.

Endpoints
---------
* ``GET  /``            — a self-contained static HTML chat page.
* ``GET  /health``      — liveness probe.
* ``POST /chat``        — blocking JSON reply: ``{"session_id", "message"}``.
* ``POST /chat/stream`` — Server-Sent Events token stream of the reply.
* ``POST /reset``       — clear a session's memory.

The active provider is chosen from :class:`~config.Settings` and defaults to the
offline :class:`~providers.FakeLLM`, so the server (and its tests) run with no API
keys and no network. Build the app with :func:`create_app` for testing or custom
wiring; ``app`` is a module-level instance for ``uvicorn``.

Part of LLM Chatbot Template by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from llm_chatbot_template import __version__
from llm_chatbot_template.chat import Chatbot
from llm_chatbot_template.config import Settings, get_settings
from llm_chatbot_template.providers import LLM, FakeLLM, OpenAIClient
from llm_chatbot_template.store import SessionStore

_STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    """A single chat request: a message scoped to a session id."""

    session_id: str = Field(default="default", min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8192)


class ChatResponse(BaseModel):
    """A blocking chat reply plus a running token estimate for the session."""

    session_id: str
    reply: str
    history_turns: int
    estimated_tokens: int


class ResetRequest(BaseModel):
    """Request to clear a session's memory."""

    session_id: str = Field(default="default", min_length=1, max_length=128)


def build_provider(settings: Settings) -> LLM:
    """Construct the configured :class:`~providers.LLM` provider.

    Args:
        settings: Application settings selecting the provider.

    Returns:
        A ready-to-use provider (offline :class:`FakeLLM` by default).
    """
    if settings.provider == "openai":
        return OpenAIClient(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            system_prompt=settings.system_prompt,
        )
    return FakeLLM()


def _sse(data: str, *, event: str | None = None) -> str:
    """Format a single Server-Sent Events frame."""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


def create_app(settings: Settings | None = None, llm: LLM | None = None) -> FastAPI:
    """Build and return a configured FastAPI application.

    Args:
        settings: Optional settings override (defaults to :func:`get_settings`).
        llm: Optional provider override (defaults to :func:`build_provider`).

    Returns:
        The wired FastAPI app.
    """
    settings = settings or get_settings()
    provider = llm or build_provider(settings)
    store = SessionStore(system_prompt=settings.system_prompt)

    app = FastAPI(
        title="LLM Chatbot Template",
        version=__version__,
        description="Streaming LLM chatbot backend template — by Viprasol Tech.",
    )
    app.state.store = store
    app.state.provider = provider

    def _chatbot_for(session_id: str) -> Chatbot:
        session = store.get_or_create(session_id)
        return Chatbot(provider, max_tokens=settings.max_tokens, session=session)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """Serve the static chat page."""
        html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/health")
    def health() -> dict[str, str | int]:
        """Liveness probe with the active provider and live-session count."""
        return {
            "status": "ok",
            "version": __version__,
            "provider": settings.provider,
            "sessions": len(store),
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        """Return a full (blocking) assistant reply for ``req.message``."""
        bot = _chatbot_for(req.session_id)
        reply = bot.send(req.message)
        return ChatResponse(
            session_id=req.session_id,
            reply=reply,
            history_turns=len(bot.session.history),
            estimated_tokens=bot.session.total_tokens(),
        )

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest) -> StreamingResponse:
        """Stream the assistant reply token-by-token over Server-Sent Events."""
        bot = _chatbot_for(req.session_id)

        def event_stream() -> Iterator[str]:
            for chunk in bot.send_stream(req.message):
                yield _sse(chunk, event="token")
            yield _sse("[DONE]", event="done")

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/reset")
    def reset(req: ResetRequest) -> dict[str, str | bool]:
        """Clear the memory for ``req.session_id``."""
        existed = store.reset(req.session_id)
        return {"session_id": req.session_id, "cleared": existed}

    return app


app = create_app()
"""Module-level app for ``uvicorn llm_chatbot_template.app:app``."""
