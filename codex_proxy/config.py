"""Configuration helpers for the Codex FastAPI proxy."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProxySettings:
    """Runtime configuration values for the proxy service."""

    host: str | None = None
    port: int | None = None
    auth_path: str | None = None
    debug_sse_enabled: bool = False
    debug_sse_path: str | None = None
    streaming_mode: bool = True

    def resolved_auth_path(self) -> Path:
        """Expand user and environment variables to obtain the auth file path."""

        if not self.auth_path:
            raise ValueError("auth_path is not configured; supply --auth-path")

        expanded = os.path.expanduser(os.path.expandvars(self.auth_path))
        return Path(expanded)
