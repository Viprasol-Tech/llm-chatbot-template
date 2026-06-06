"""LLM Chatbot Template — a streaming FastAPI chatbot backend by Viprasol Tech.

Public API:

* :class:`~llm_chatbot_template.chat.Chatbot` — multi-turn orchestration.
* :class:`~llm_chatbot_template.session.ChatSession` — history + token trimming.
* :class:`~llm_chatbot_template.providers.FakeLLM` — offline deterministic provider.
* :class:`~llm_chatbot_template.providers.OpenAIClient` — optional OpenAI backend.
* :func:`~llm_chatbot_template.app.create_app` — build the FastAPI app.

Part of LLM Chatbot Template by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
