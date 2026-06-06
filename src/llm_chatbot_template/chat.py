"""The Chatbot: drive a :class:`ChatSession` through a pluggable LLM provider.

A :class:`Chatbot` owns a session and an :class:`LLM` provider. Each turn appends
the user's message, optionally trims to a token budget, assembles the prompt,
calls the model, and records the reply. Use :meth:`Chatbot.send` for a single
blocking reply, or :meth:`Chatbot.send_stream` to yield the reply token-by-token
(the assistant turn is appended once streaming completes).

Part of LLM Chatbot Template by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_chatbot_template.session import ChatSession

if TYPE_CHECKING:
    from collections.abc import Iterator

    from llm_chatbot_template.providers import LLM


class Chatbot:
    """A multi-turn chatbot backed by a :class:`ChatSession` and an LLM provider."""

    def __init__(
        self,
        llm: LLM,
        *,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int | None = None,
        session: ChatSession | None = None,
    ) -> None:
        """Create a chatbot.

        Args:
            llm: The pluggable language model (an :class:`~providers.LLM`).
            system_prompt: The system prompt for a freshly created session.
                Ignored if ``session`` is supplied.
            max_tokens: Optional token budget; when set, the session is trimmed
                to this budget before each model call.
            session: An existing session to continue, instead of a new one.
        """
        self.llm = llm
        self.max_tokens = max_tokens
        self.session = session if session is not None else ChatSession(system_prompt=system_prompt)

    def _prepare(self, message: str) -> str:
        """Append the user turn, trim to budget, and return the assembled prompt."""
        self.session.add_user(message)
        if self.max_tokens is not None:
            self.session.trim_to_budget(self.max_tokens)
        return self.session.build_prompt()

    def send(self, message: str) -> str:
        """Send a user message, call the model, and return the assistant reply.

        Args:
            message: The user's input text.

        Returns:
            The assistant's reply text.
        """
        prompt = self._prepare(message)
        reply = self.llm.complete(prompt)
        self.session.add_assistant(reply)
        return reply

    def send_stream(self, message: str) -> Iterator[str]:
        """Send a user message and stream the reply token-by-token.

        Chunks are yielded as they arrive from the provider; the assembled reply
        is appended to the session as a single assistant turn once the stream is
        exhausted.

        Args:
            message: The user's input text.

        Yields:
            Successive chunks of the assistant's reply.
        """
        prompt = self._prepare(message)
        chunks: list[str] = []
        for chunk in self.llm.stream(prompt):
            chunks.append(chunk)
            yield chunk
        self.session.add_assistant("".join(chunks))
