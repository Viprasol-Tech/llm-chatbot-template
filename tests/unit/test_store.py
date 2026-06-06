"""Tests for the in-memory SessionStore."""

from __future__ import annotations

from llm_chatbot_template.store import SessionStore


def test_get_or_create_returns_same_session() -> None:
    store = SessionStore(system_prompt="sys")
    first = store.get_or_create("a")
    again = store.get_or_create("a")
    assert first is again
    assert first.system_prompt == "sys"
    assert len(store) == 1


def test_distinct_ids_get_distinct_sessions() -> None:
    store = SessionStore()
    assert store.get_or_create("a") is not store.get_or_create("b")
    assert len(store) == 2


def test_get_returns_none_for_unknown() -> None:
    store = SessionStore()
    assert store.get("missing") is None


def test_reset_removes_session() -> None:
    store = SessionStore()
    store.get_or_create("a")
    assert store.reset("a") is True
    assert store.get("a") is None
    assert store.reset("a") is False
