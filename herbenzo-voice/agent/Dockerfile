FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

RUN useradd --create-home --uid 10001 herbenzo
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY herbenzo_agent ./herbenzo_agent
RUN uv sync --locked --no-dev && mkdir -p /data && chown herbenzo /data

USER herbenzo
ENV DATABASE_URL=sqlite:////data/herbenzo_voice.db
EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8081/health')"
CMD ["uv", "run", "--no-dev", "uvicorn", "herbenzo_agent.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8081"]
