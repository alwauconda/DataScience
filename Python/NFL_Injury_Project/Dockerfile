FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy project definition files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies (frozen = uses exact versions from uv.lock)
RUN uv sync --frozen --no-dev

# Copy project files
COPY . .

# Default command — override with docker compose run
CMD ["uv", "run", "python", "01_load_data.py"]
