"""Tests for ChatSession history, prompt assembly, and budget trimming."""

from __future__ import annotations

import pytest

from llm_chatbot_template.session import ChatSession, estimate_tokens


def test_history_accumulates_in_order() -> None:
    session = ChatSession(system_prompt="sys")
    session.add_user("first")
    session.add_assistant("second")
    session.add_user("third")

    roles = [m.role for m in session.history]
    contents = [m.content for m in session.history]
    assert roles == ["user", "assistant", "user"]
    assert contents == ["first", "second", "third"]


def test_messages_prepends_system_prompt() -> None:
    session = ChatSession(system_prompt="be helpful")
    session.add_user("hi")
    messages = session.messages()

    assert messages[0].role == "system"
    assert messages[0].content == "be helpful"
    assert [m.role for m in messages] == ["system", "user"]


def test_build_prompt_includes_system_and_turns() -> None:
    session = ChatSession(system_prompt="be brief")
    session.add_user("hello")
    session.add_assistant("hi")
    prompt = session.build_prompt()

    assert "System: be brief" in prompt
    assert "User: hello" in prompt
    assert "Assistant: hi" in prompt
    # Ordering: system before user before assistant.
    assert prompt.index("System:") < prompt.index("User:") < prompt.index("Assistant: hi")
    # Ends with a cue for the model to continue.
    assert prompt.rstrip().endswith("Assistant:")


def test_estimate_tokens_heuristic() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0
    # 10 words * 1.3 = 13 tokens.
    assert estimate_tokens(" ".join(["word"] * 10)) == 13


def test_trim_keeps_system_prompt_and_drops_oldest_first() -> None:
    session = ChatSession(system_prompt="keep me")
    # Each message is one word ~ 2 tokens by the heuristic.
    session.add_user("alpha")
    session.add_assistant("beta")
    session.add_user("gamma")
    session.add_assistant("delta")

    before = session.total_tokens()
    dropped = session.trim_to_budget(before // 2)

    assert dropped >= 1
    # System prompt always survives.
    assert session.messages()[0].content == "keep me"
    # Oldest history messages were the ones removed; relative order preserved.
    order = ["alpha", "beta", "gamma", "delta"]
    surviving = [m.content for m in session.history]
    assert "alpha" not in surviving
    assert surviving == [c for c in order if c in surviving]
    # Newest message is always retained.
    assert surviving[-1] == "delta"


def test_trim_respects_budget() -> None:
    session = ChatSession(system_prompt="sys")
    for i in range(20):
        session.add_user(f"message number {i} with some words")

    session.trim_to_budget(40)
    assert session.total_tokens() <= 40
    # System prompt is never dropped even under a tight budget.
    assert session.messages()[0].role == "system"


def test_trim_can_empty_history_but_keeps_system() -> None:
    session = ChatSession(system_prompt="this system prompt has several words")
    session.add_user("a longer user message that exceeds the tiny budget below")
    session.trim_to_budget(0)

    assert session.history == []
    assert session.messages()[0].role == "system"


def test_trim_negative_budget_raises() -> None:
    session = ChatSession()
    with pytest.raises(ValueError):
        session.trim_to_budget(-1)
