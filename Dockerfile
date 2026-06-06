# syntax=docker/dockerfile:1
# LLM Chatbot Template — by Viprasol Tech (https://viprasol.com)
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the package (with the optional uvicorn server extra) first so the
# dependency layer is cached independently of source changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Liveness probe hits the app's /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "llm_chatbot_template.app:app", "--host", "0.0.0.0", "--port", "8000"]
