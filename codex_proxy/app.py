"""FastAPI application factory for the Codex proxy."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from aiohttp import ClientSession
from fastapi import FastAPI

from .config import ProxySettings
from .routes import router

logger = logging.getLogger(__name__)

class AuthConfigError(RuntimeError):
    """Raised when the auth.json file is missing or invalid."""

async def _read_auth_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AuthConfigError(f"Auth file {path} does not exist; supply --auth-path")

    try:
        content = await asyncio.to_thread(path.read_text)
    except OSError as exc:
        raise AuthConfigError(f"Failed to read auth file {path}: {exc}") from exc

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AuthConfigError(f"Auth file {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise AuthConfigError(f"Auth file {path} must contain a JSON object")

    tokens_raw = data.get("tokens")
    if tokens_raw is None:
        tokens: dict[str, Any] = {}
    elif isinstance(tokens_raw, dict):
        tokens = tokens_raw
    else:
        raise AuthConfigError(
            f"Auth file {path} needs 'tokens' to be an object when present"
        )

    access_token = tokens.get("access_token")
    api_key = data.get("OPENAI_API_KEY") or data.get("api_key")
    if not access_token and not api_key:
        raise AuthConfigError(
            f"Auth file {path} must include tokens.access_token or OPENAI_API_KEY/api_key"
        )

    normalized = dict(data)
    normalized["tokens"] = tokens
    return normalized


def create_app(settings: ProxySettings | None = None) -> FastAPI:
    """Build the FastAPI app with configured routers and lifespan hooks."""

    settings = settings or ProxySettings.from_env()

    app = FastAPI(
        title="Codex OpenAI Proxy",
        description="FastAPI port of the Rust warp Codex proxy for OpenAI-compatible endpoints.",
        version="0.1.3",
    )

    app.include_router(router)

    app.state.settings = settings
    app.state.http_client = None
    app.state.auth_data = None

    @app.on_event("startup")
    async def on_startup() -> None:  # pragma: no cover - exercised at runtime
        app.state.http_client = ClientSession(
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }
        )
        auth_path = settings.resolved_auth_path()
        app.state.auth_data = await _read_auth_file(auth_path)
        if app.state.auth_data:
            logger.info("✓ Loaded authentication data from %s", auth_path)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:  # pragma: no cover - exercised at runtime
        client: Optional[ClientSession] = app.state.http_client
        if client and not client.closed:
            await client.close()

    return app
