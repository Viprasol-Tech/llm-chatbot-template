"""Tests for the FastAPI app: blocking chat, SSE streaming, memory, and reset."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_chatbot_template.app import create_app
from llm_chatbot_template.config import Settings


@pytest.fixture
def client() -> TestClient:
    app = create_app(settings=Settings(provider="fake", max_tokens=2048))
    return TestClient(app)


def test_health_reports_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["provider"] == "fake"


def test_index_serves_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "LLM Chatbot Template" in resp.text


def test_chat_returns_reply_and_token_estimate(client: TestClient) -> None:
    resp = client.post("/chat", json={"session_id": "s1", "message": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "Echo: you said 'hello'"
    assert body["history_turns"] == 2
    assert body["estimated_tokens"] > 0


def test_chat_remembers_session_across_requests(client: TestClient) -> None:
    client.post("/chat", json={"session_id": "mem", "message": "one"})
    second = client.post("/chat", json={"session_id": "mem", "message": "two"})
    # 2 user + 2 assistant turns accumulated for this session.
    assert second.json()["history_turns"] == 4


def test_chat_sessions_are_isolated(client: TestClient) -> None:
    client.post("/chat", json={"session_id": "a", "message": "x"})
    other = client.post("/chat", json={"session_id": "b", "message": "y"})
    assert other.json()["history_turns"] == 2


def test_chat_validation_rejects_empty_message(client: TestClient) -> None:
    resp = client.post("/chat", json={"session_id": "s", "message": ""})
    assert resp.status_code == 422


def test_chat_stream_emits_sse_tokens_and_done(client: TestClient) -> None:
    resp = client.post("/chat/stream", json={"session_id": "stream", "message": "hi there"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    text = resp.text
    assert "event: token" in text
    assert "event: done" in text
    assert "[DONE]" in text

    # Reassembling the token data lines reproduces the full reply.
    tokens = [
        line[len("data: ") :]
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert "".join(tokens) == "Echo: you said 'hi there'"


def test_stream_persists_assistant_turn(client: TestClient) -> None:
    client.post("/chat/stream", json={"session_id": "persist", "message": "remember me"})
    follow = client.post("/chat", json={"session_id": "persist", "message": "again"})
    # 1 streamed user+assistant pair, then 1 blocking pair == 4 turns.
    assert follow.json()["history_turns"] == 4


def test_reset_clears_session_memory(client: TestClient) -> None:
    client.post("/chat", json={"session_id": "del", "message": "one"})
    reset = client.post("/reset", json={"session_id": "del"})
    assert reset.json()["cleared"] is True

    fresh = client.post("/chat", json={"session_id": "del", "message": "two"})
    assert fresh.json()["history_turns"] == 2


def test_reset_unknown_session_reports_false(client: TestClient) -> None:
    resp = client.post("/reset", json={"session_id": "never-existed"})
    assert resp.json()["cleared"] is False


def test_build_provider_defaults_to_fake() -> None:
    from llm_chatbot_template.app import build_provider
    from llm_chatbot_template.providers import FakeLLM

    assert isinstance(build_provider(Settings(provider="fake")), FakeLLM)
