FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Install locked production dependencies
RUN uv sync --frozen --no-dev

# Copy application and model artifacts
COPY app ./app
COPY models ./models

EXPOSE 8000

CMD ["sh", "-c", "/app/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]