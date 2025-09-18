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


async def _read_auth_file(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        logger.warning("Auth file %s does not exist; proceeding without credentials", path)
        return None

    try:
        content = await asyncio.to_thread(path.read_text)
        return json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse auth file %s: %s", path, exc)
    except OSError as exc:
        logger.error("Failed to read auth file %s: %s", path, exc)
    return None


def create_app(settings: ProxySettings | None = None) -> FastAPI:
    """Build the FastAPI app with configured routers and lifespan hooks."""

    settings = settings or ProxySettings.from_env()

    app = FastAPI(
        title="Codex OpenAI Proxy",
        description="FastAPI port of the Rust warp Codex proxy for OpenAI-compatible endpoints.",
        version="0.1.0",
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
            logger.info("Loaded authentication data from %s", auth_path)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:  # pragma: no cover - exercised at runtime
        client: Optional[ClientSession] = app.state.http_client
        if client and not client.closed:
            await client.close()

    return app
