"""Tests for the Chatbot orchestration and the FakeLLM provider."""

from __future__ import annotations

from collections.abc import Iterator

from llm_chatbot_template.chat import Chatbot
from llm_chatbot_template.providers import LLM, CallableLLM, EchoLLM, FakeLLM
from llm_chatbot_template.session import ChatSession


def test_fake_llm_is_an_llm() -> None:
    assert isinstance(FakeLLM(), LLM)


def test_echo_llm_alias_points_at_fake_llm() -> None:
    assert EchoLLM is FakeLLM


def test_fake_llm_echoes_last_user_line() -> None:
    session = ChatSession(system_prompt="sys")
    session.add_user("first thing")
    session.add_assistant("ok")
    session.add_user("second thing")
    reply = FakeLLM().complete(session.build_prompt())

    assert reply == "Echo: you said 'second thing'"


def test_fake_llm_stream_concatenates_to_complete() -> None:
    prompt = "User: hello world\nAssistant:"
    fake = FakeLLM()
    streamed = "".join(fake.stream(prompt))
    assert streamed == fake.complete(prompt)
    # More than one chunk is produced for a multi-word reply.
    assert len(list(fake.stream(prompt))) > 1


def test_chatbot_appends_user_then_assistant() -> None:
    bot = Chatbot(FakeLLM(), system_prompt="sys")
    reply = bot.send("hello world")

    assert reply == "Echo: you said 'hello world'"
    roles = [m.role for m in bot.session.history]
    assert roles == ["user", "assistant"]
    assert bot.session.history[0].content == "hello world"
    assert bot.session.history[1].content == reply


def test_chatbot_multi_turn_accumulates() -> None:
    bot = Chatbot(FakeLLM(), system_prompt="sys")
    bot.send("one")
    bot.send("two")
    bot.send("three")

    user_turns = [m.content for m in bot.session.history if m.role == "user"]
    assert user_turns == ["one", "two", "three"]
    assert len(bot.session.history) == 6


def test_chatbot_send_stream_appends_full_reply_once() -> None:
    bot = Chatbot(FakeLLM(), system_prompt="sys")
    chunks = list(bot.send_stream("hi there"))

    assert "".join(chunks) == "Echo: you said 'hi there'"
    roles = [m.role for m in bot.session.history]
    assert roles == ["user", "assistant"]
    assert bot.session.history[-1].content == "Echo: you said 'hi there'"


def test_chatbot_with_callable_provider_adapter() -> None:
    bot = Chatbot(CallableLLM(lambda prompt: "REPLY"), system_prompt="sys")
    assert bot.send("hi") == "REPLY"
    assert list(Chatbot(CallableLLM(lambda p: "X")).send_stream("y")) == ["X"]


def test_chatbot_trims_to_budget_before_calling() -> None:
    seen_prompts: list[str] = []

    class RecordingLLM:
        def complete(self, prompt: str) -> str:
            seen_prompts.append(prompt)
            return "r"

        def stream(self, prompt: str) -> Iterator[str]:
            yield self.complete(prompt)

    bot = Chatbot(RecordingLLM(), system_prompt="sys", max_tokens=12)
    for _ in range(10):
        bot.send("a fairly wordy user message to consume the budget")

    # The session stays within budget after each trimmed turn.
    assert bot.session.total_tokens() <= 12
    # System prompt is always present in every prompt the model saw.
    assert all("System: sys" in p for p in seen_prompts)
