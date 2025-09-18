"""Entry points for launching the FastAPI proxy via uvicorn."""
from __future__ import annotations
from .app import AuthConfigError, _read_auth_file, create_app
import argparse
import asyncio
import logging
import os

import uvicorn

from .app import create_app
from .config import ProxySettings


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8111
DEFAULT_DEBUG_PATH = "/tmp/debug_codexproxy.log"
DEFAULT_AUTH_PATH = "~/.codex/auth.json"
if os.name == "nt":
    DEFAULT_AUTH_PATH = r"%USERPROFILE%\\.codex\\auth.json"


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _resolve_debug_settings(debug_arg: str | None, current_path: str | None) -> tuple[bool, str]:
    """Return debug enabled flag and chosen path based on CLI input."""

    if not debug_arg:
        return False, current_path or DEFAULT_DEBUG_PATH

    candidate = debug_arg.strip() if isinstance(debug_arg, str) else ""
    path = candidate or DEFAULT_DEBUG_PATH
    return True, path



def _build_settings(args: argparse.Namespace) -> ProxySettings:
    """Construct ProxySettings from CLI arguments with defaults."""

    settings = ProxySettings.from_env()
    settings.host = (args.host or DEFAULT_HOST)
    settings.port = (args.port or DEFAULT_PORT)
    settings.auth_path = (args.auth_path or DEFAULT_AUTH_PATH)

    debug_enabled, debug_path = _resolve_debug_settings(getattr(args, "debug", None), settings.debug_sse_path)
    settings.debug_sse_enabled = debug_enabled
    settings.debug_sse_path = debug_path
    return settings



def _log_configuration(settings: ProxySettings) -> None:
    """Emit a concise summary of the active configuration values."""

    debug_display = settings.debug_sse_path if settings.debug_sse_enabled else "disabled"
    try:
        resolved_auth_path = settings.resolved_auth_path()
        auth_display = str(resolved_auth_path)
    except ValueError:
        auth_display = settings.auth_path

    logger.info("Initializing Codex OpenAI Proxy ...")
    logger.info(
        "✓ Loaded configuration host=%s port=%s auth_path=%s debug=%s",
        settings.host,
        settings.port,
        auth_display,
        debug_display,
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Codex OpenAI proxy server")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Host interface to bind",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port to bind",
    )
    parser.add_argument(
        "--auth-path",
        default=DEFAULT_AUTH_PATH,
        help="Path to codex auth.json (platform default if omitted)",
    )
    parser.add_argument(
        "--debug",
        metavar="PATH",
        default=None,
        help="Enable SSE debug logging and write to PATH",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    settings = _build_settings(args)
    
    try:
        auth_path = settings.resolved_auth_path()
        # Validate auth config before creating the app so Uvicorn stays quiet.
        asyncio.run(_read_auth_file(auth_path))
    except AuthConfigError as err:
        logger.error("[!] Auth configuration error: %s", err)
        raise SystemExit(1)
    
    _log_configuration(settings)

    uvicorn.run(
        create_app(settings),
        host=settings.host or DEFAULT_HOST,
        port=settings.port or DEFAULT_PORT,
        log_level="info",
    )


async def run_async() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, run)


if __name__ == "__main__":
    run()
