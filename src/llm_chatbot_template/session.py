"""Conversation session: system prompt + message history + token-budget trimming.

A :class:`ChatSession` holds a single system prompt plus an ordered list of
``user``/``assistant`` messages. It can assemble the conversation into a single
prompt string for a language model and trim the oldest turns to fit a token
budget while always keeping the system prompt.

Part of LLM Chatbot Template by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]

# A small multiplier mapping whitespace-delimited words to an estimated token
# count. Real tokenizers split sub-words, so words tend to under-count slightly;
# 1.3 is a widely used rule-of-thumb that errs on the safe (larger) side.
_TOKENS_PER_WORD = 1.3


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in ``text``.

    Uses a deterministic ``word_count * 1.3`` heuristic, rounded up, so the
    estimate never under-counts a non-empty string. Empty/whitespace-only text
    yields ``0``.

    Args:
        text: The text to measure.

    Returns:
        A non-negative estimated token count.
    """
    words = text.split()
    if not words:
        return 0
    return int(len(words) * _TOKENS_PER_WORD + 0.999)


class Message(BaseModel):
    """A single conversation message with a role and content."""

    role: Role
    content: str

    @property
    def tokens(self) -> int:
        """Estimated token cost of this message's content."""
        return estimate_tokens(self.content)


class ChatSession(BaseModel):
    """An ordered conversation: one system prompt plus user/assistant turns.

    The system prompt is stored separately from the mutable ``history`` so it is
    never dropped by trimming. ``build_prompt`` flattens the whole conversation
    into a single string suitable for a text-completion style model.
    """

    system_prompt: str = Field(default="You are a helpful assistant.")
    history: list[Message] = Field(default_factory=list)

    def add_user(self, content: str) -> Message:
        """Append a ``user`` message and return it.

        Args:
            content: The user's message text.

        Returns:
            The appended :class:`Message`.
        """
        message = Message(role="user", content=content)
        self.history.append(message)
        return message

    def add_assistant(self, content: str) -> Message:
        """Append an ``assistant`` message and return it.

        Args:
            content: The assistant's reply text.

        Returns:
            The appended :class:`Message`.
        """
        message = Message(role="assistant", content=content)
        self.history.append(message)
        return message

    def messages(self) -> list[Message]:
        """Return the full message list: system prompt followed by history.

        Returns:
            A new list whose first element is the system message.
        """
        system = Message(role="system", content=self.system_prompt)
        return [system, *self.history]

    def total_tokens(self) -> int:
        """Estimate the token cost of the whole conversation (system + history)."""
        return sum(message.tokens for message in self.messages())

    def build_prompt(self) -> str:
        """Assemble the conversation into a single prompt string for the model.

        Each message is rendered as ``"<Role>: <content>"`` on its own line, in
        order, ending with an ``"Assistant:"`` cue so a completion model knows to
        continue as the assistant.

        Returns:
            The assembled multi-line prompt string.
        """
        lines = [f"{message.role.capitalize()}: {message.content}" for message in self.messages()]
        lines.append("Assistant:")
        return "\n".join(lines)

    def trim_to_budget(self, max_tokens: int) -> int:
        """Drop oldest non-system messages until the budget is met.

        The system prompt is always retained. User/assistant turns are removed
        from the front (oldest first) until the estimated total token count is at
        or below ``max_tokens``, or only the system prompt remains.

        Args:
            max_tokens: The maximum estimated token budget to fit within.

        Returns:
            The number of messages dropped from ``history``.

        Raises:
            ValueError: If ``max_tokens`` is negative.
        """
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")

        dropped = 0
        while self.history and self.total_tokens() > max_tokens:
            self.history.pop(0)
            dropped += 1
        return dropped
