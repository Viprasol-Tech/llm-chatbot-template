"""Command-line interface for LLM Chatbot Template.

* ``llm-chatbot-template demo`` runs a short scripted multi-turn conversation
  against the offline :class:`FakeLLM` provider — no API keys, no network.
* ``llm-chatbot-template serve`` starts the FastAPI streaming server with uvicorn.
* ``llm-chatbot-template version`` prints the installed version.

Part of LLM Chatbot Template by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import typer
from rich.console import Console

from llm_chatbot_template import __version__
from llm_chatbot_template.chat import Chatbot
from llm_chatbot_template.providers import FakeLLM

app = typer.Typer(add_completion=False, help="LLM Chatbot Template — by Viprasol Tech.")
console = Console()

_SCRIPT = [
    "Hello there!",
    "What can you help me with?",
    "Summarize our chat so far.",
]


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"llm-chatbot-template [bold cyan]{__version__}[/] - by Viprasol Tech")


@app.command()
def demo(
    max_tokens: int = typer.Option(
        64, help="Token budget; oldest turns are trimmed to fit before each reply."
    ),
    stream: bool = typer.Option(False, "--stream", help="Stream replies token-by-token."),
) -> None:
    """Run a scripted multi-turn conversation against the offline FakeLLM."""
    bot = Chatbot(
        FakeLLM(),
        system_prompt="You are a concise, friendly assistant.",
        max_tokens=max_tokens,
    )
    console.print(f"System: [dim]{bot.session.system_prompt}[/]")
    for turn in _SCRIPT:
        console.print(f"User:      {turn}")
        if stream:
            console.print("Assistant: ", end="")
            for chunk in bot.send_stream(turn):
                console.print(f"[green]{chunk}[/]", end="")
            console.print()
        else:
            reply = bot.send(turn)
            console.print(f"Assistant: [green]{reply}[/]")
    console.print(
        f"\nHistory turns: [bold]{len(bot.session.history)}[/] | "
        f"Estimated tokens: [bold]{bot.session.total_tokens()}[/] "
        f"(budget {max_tokens})"
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev)."),
) -> None:
    """Start the FastAPI streaming chat server with uvicorn."""
    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover - requires the optional dep
        raise typer.BadParameter(
            "uvicorn is required to serve. Install with: python -m pip install 'uvicorn[standard]'"
        ) from exc

    console.print(f"Serving LLM Chatbot Template on [cyan]http://{host}:{port}[/]")
    uvicorn.run("llm_chatbot_template.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
