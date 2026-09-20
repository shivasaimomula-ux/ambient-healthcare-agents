FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

# git: nvidia-pipecat is installed from the pinned ace-controller commit; libgl/libglib: opencv (pipecat dep)
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 herbenzo
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY config.py herbenzo_agent_service.py pipeline_herbenzo.py websocket_transcript_output.py ipa.json ./
COPY configs ./configs

USER herbenzo
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://localhost:7860/health || exit 1
CMD ["uv", "run", "--no-dev", "python", "pipeline_herbenzo.py", "--host", "0.0.0.0", "--port", "7860"]
