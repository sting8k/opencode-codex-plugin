# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy project files
COPY pyproject.toml ./
COPY codex_proxy/ ./codex_proxy/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Create a non-root user
RUN useradd -m -u 1000 codexuser && \
    chown -R codexuser:codexuser /app

# Create directory for auth file (to be mounted)
RUN mkdir -p /app/config && \
    chown -R codexuser:codexuser /app/config

# Switch to non-root user
USER codexuser

# Expose the default port
EXPOSE 8111

# Set default command
CMD ["codex-proxy"]

