"""In-memory session store: map a session id to its :class:`ChatSession`.

A tiny, thread-safe-enough-for-a-template registry so the HTTP layer can keep
per-conversation memory across requests. Swap this for Redis/SQLite in
production — the interface is intentionally small.

Part of LLM Chatbot Template by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import threading

from llm_chatbot_template.session import ChatSession


class SessionStore:
    """A process-local registry of :class:`ChatSession` objects keyed by id."""

    def __init__(self, *, system_prompt: str = "You are a helpful assistant.") -> None:
        """Create an empty store.

        Args:
            system_prompt: Default system prompt for sessions created on demand.
        """
        self._system_prompt = system_prompt
        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> ChatSession:
        """Return the session for ``session_id``, creating it if absent."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = ChatSession(system_prompt=self._system_prompt)
                self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> ChatSession | None:
        """Return the session for ``session_id`` or ``None`` if it does not exist."""
        with self._lock:
            return self._sessions.get(session_id)

    def reset(self, session_id: str) -> bool:
        """Delete the session for ``session_id``. Return whether one existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        """Number of live sessions currently held."""
        with self._lock:
            return len(self._sessions)
