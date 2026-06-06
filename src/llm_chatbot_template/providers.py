"""Pluggable LLM providers — a streaming protocol, an offline fake, and OpenAI.

The chatbot talks to a language model through a tiny :class:`LLM` protocol with
two methods: :meth:`~LLM.complete` (return the whole reply) and
:meth:`~LLM.stream` (yield the reply token-by-token). Any object implementing
them is a valid provider, so wrapping the SDK of your choice takes a few lines.

Two providers ship in the box:

* :class:`FakeLLM` — a deterministic, offline fake. It needs no API key and no
  network, so the demo, the SSE endpoint, and the whole test-suite run locally.
  (``EchoLLM`` is kept as a backwards-compatible alias.)
* :class:`OpenAIClient` — an optional OpenAI-compatible client. The ``openai``
  SDK is imported lazily inside the constructor, so importing this module never
  requires the dependency unless you actually instantiate the client.

Part of LLM Chatbot Template by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llm_chatbot_template.session import Message


@runtime_checkable
class LLM(Protocol):
    """A language model: maps an assembled prompt to a reply, all at once or streamed."""

    def complete(self, prompt: str) -> str:
        """Generate the full reply for ``prompt``."""
        ...

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield the reply for ``prompt`` as a sequence of text chunks (tokens)."""
        ...


def _last_user_line(prompt: str) -> str:
    """Return the content of the last ``User:`` line in an assembled prompt.

    Falls back to the final non-empty line if no ``User:`` line is present.

    Args:
        prompt: An assembled prompt produced by ``ChatSession.build_prompt``.

    Returns:
        The extracted user text, or an empty string if none is found.
    """
    last_user = ""
    last_nonempty = ""
    for raw in prompt.splitlines():
        line = raw.strip()
        if not line:
            continue
        last_nonempty = line
        if line.startswith("User:"):
            last_user = line[len("User:") :].strip()
    return last_user or last_nonempty


class FakeLLM:
    """A deterministic offline LLM that echoes the latest user message.

    Useful for demos, the streaming SSE endpoint, and tests: given a prompt
    assembled by :meth:`ChatSession.build_prompt`, it finds the most recent user
    turn and returns a fixed, reproducible reply referencing it. No network, no
    keys. :meth:`stream` yields that same reply one whitespace-delimited token at
    a time so streaming consumers can be exercised entirely offline.
    """

    def __init__(self, prefix: str = "Echo") -> None:
        """Create the fake provider.

        Args:
            prefix: A label prepended to every reply.
        """
        self.prefix = prefix

    def _reply(self, prompt: str) -> str:
        user_text = _last_user_line(prompt)
        return f"{self.prefix}: you said '{user_text}'"

    def complete(self, prompt: str) -> str:
        """Return a deterministic reply echoing the last user message.

        Args:
            prompt: The assembled conversation prompt.

        Returns:
            A reply of the form ``"<prefix>: you said '<last user text>'"``.
        """
        return self._reply(prompt)

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield the deterministic reply token-by-token (whitespace-delimited).

        Each yielded chunk includes its trailing space (except the last) so that
        concatenating every chunk reproduces :meth:`complete` exactly.

        Args:
            prompt: The assembled conversation prompt.

        Yields:
            Successive chunks of the reply text.
        """
        reply = self._reply(prompt)
        tokens = reply.split(" ")
        for index, token in enumerate(tokens):
            yield token if index == len(tokens) - 1 else token + " "


# Backwards-compatible alias: the offline provider used to be called ``EchoLLM``.
EchoLLM = FakeLLM


class OpenAIClient:
    """An optional OpenAI-compatible chat provider (``openai`` imported lazily).

    The ``openai`` package is imported inside ``__init__`` only, so importing this
    module — or running the offline demo and tests — never requires the SDK. Point
    ``base_url`` at any OpenAI-compatible server (OpenAI, Together, Groq, a local
    Ollama/LM Studio endpoint, …).
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        """Create the client and the underlying OpenAI SDK object.

        Args:
            model: The chat model name to call.
            api_key: API key; falls back to the SDK's ``OPENAI_API_KEY`` env var.
            base_url: Optional OpenAI-compatible base URL.
            system_prompt: System message sent before the assembled prompt.

        Raises:
            RuntimeError: If the optional ``openai`` package is not installed.
        """
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised without the dep
            raise RuntimeError(
                "The 'openai' package is required for OpenAIClient. "
                "Install it with: python -m pip install openai"
            ) from exc

        self.model = model
        self.system_prompt = system_prompt
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def complete(self, prompt: str) -> str:
        """Call the chat completion API and return the full reply text."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt),
        )
        content: str = resp.choices[0].message.content or ""
        return content

    def stream(self, prompt: str) -> Iterator[str]:
        """Call the chat completion API with ``stream=True`` and yield deltas."""
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield str(delta)


class CallableLLM:
    """Adapt a plain ``Callable[[str], str]`` into the :class:`LLM` protocol.

    Streaming is emulated by computing the full reply and yielding it as a single
    chunk — handy for wrapping a legacy function without a native stream.
    """

    def __init__(self, func: object) -> None:
        """Wrap ``func`` (any ``Callable[[str], str]``)."""
        if not callable(func):
            raise TypeError("CallableLLM requires a callable")
        self._func = func

    def complete(self, prompt: str) -> str:
        """Return ``func(prompt)`` as a string."""
        return str(self._func(prompt))

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield the full ``func(prompt)`` reply as one chunk."""
        yield self.complete(prompt)


def messages_to_prompt(messages: Sequence[Message]) -> str:
    """Render a sequence of messages into a single completion-style prompt.

    Args:
        messages: Ordered conversation messages (system first, by convention).

    Returns:
        A multi-line ``"<Role>: <content>"`` string ending with an
        ``"Assistant:"`` cue.
    """
    lines = [f"{m.role.capitalize()}: {m.content}" for m in messages]
    lines.append("Assistant:")
    return "\n".join(lines)
