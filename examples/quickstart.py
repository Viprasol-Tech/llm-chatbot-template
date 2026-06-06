"""Runnable, offline quickstart for LLM Chatbot Template.

Run it with no API keys and no network:

    python examples/quickstart.py

It drives a multi-turn conversation through the deterministic offline
``FakeLLM`` — first a blocking reply, then a token-by-token streamed reply —
exactly the same engine the FastAPI ``/chat`` endpoints use.

Part of LLM Chatbot Template by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from llm_chatbot_template.chat import Chatbot
from llm_chatbot_template.providers import FakeLLM


def main() -> None:
    """Demonstrate blocking and streaming replies against the offline FakeLLM."""
    bot = Chatbot(
        FakeLLM(),
        system_prompt="You are a concise, friendly assistant.",
        max_tokens=128,
    )

    # 1) A single, blocking reply.
    print("User:      Hello there!")
    print("Assistant:", bot.send("Hello there!"))

    # 2) A streamed reply — chunks arrive token-by-token.
    print("\nUser:      Stream the next reply please")
    print("Assistant: ", end="", flush=True)
    for chunk in bot.send_stream("Stream the next reply please"):
        print(chunk, end="", flush=True)
    print()

    print(
        f"\nHistory turns: {len(bot.session.history)} | "
        f"Estimated tokens: {bot.session.total_tokens()}"
    )


if __name__ == "__main__":
    main()
