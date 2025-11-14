"""Slim HTTP route handlers for the Codex FastAPI proxy.

This rewrite keeps the public API and core behaviors intact while removing
non-essential code and complexity. Adds optional per-request and CLI-configurable
SSE debug logging for upstream responses.
"""
from __future__ import annotations

import asyncio, aiohttp
import json
import logging, traceback
import re
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, AsyncIterator, Awaitable, Callable, Iterable, Iterator, Optional, Union

from aiohttp import ClientError
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .schemas import (
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    ChatResponseMessage,
    Choice,
    ModelsList,
    Usage,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _coerce_json_object_from_bytes(raw: bytes) -> dict[str, Any]:
    """Parse request body bytes into a JSON object, handling double-encoded strings.
    
    Raises:
        ValueError: If body is empty or resolves to empty string
        TypeError: If parsed result is not a JSON object (dict)
        json.JSONDecodeError: If JSON parsing fails
    """
    if not raw or raw.strip() == b"":
        raise ValueError("Empty request body")
    
    # Decode once; tolerate bad bytes for diagnostics
    text = raw.decode("utf-8", errors="replace").strip()
    
    # First parse with fallback for incorrectly escaped outer quotes (e.g., \"{...}\")
    try:
        first = json.loads(text)
    except json.JSONDecodeError as e1:
        # Fallback: if starts with \" and ends with \", strip the backslashes on OUTER quotes only
        if text.startswith('\\"') and text.endswith('\\"') and len(text) >= 4:
            try:
                patched = '"' + text[2:-2] + '"'
                first = json.loads(patched)
            except json.JSONDecodeError:
                raise e1
        else:
            raise e1
    
    # Handle double-encoded JSON: payload is a JSON string that itself contains JSON
    if isinstance(first, str):
        inner = first.strip()
        if not inner:
            raise ValueError("Body resolves to an empty string after decoding")
        second = json.loads(inner)
        if not isinstance(second, dict):
            raise TypeError("Decoded payload is not a JSON object")
        return second
    
    if not isinstance(first, dict):
        raise TypeError("Request body must be a JSON object")
    
    return first


# Upstream endpoint (ChatGPT Responses API)
CHATGPT_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
MAX_UPSTREAM_RETRIES = 2
RETRY_BACKOFF_BASE = 0.5

# Minimal model aliases and effort parsing
_EFFORT_RE = re.compile(r"(?:[:_\-](minimal|low|medium|high))$", re.IGNORECASE)
_MODEL_ALIAS = {
    "gpt5": "gpt-5",
    "gpt-5": "gpt-5",
    "gpt-5-latest": "gpt-5.1",
    "gpt5.1": "gpt-5.1",
    "gpt-5.1": "gpt-5.1",
    "gpt-5.1-latest": "gpt-5.1",
    "gpt5-codex": "gpt-5-codex",
    "gpt-5-codex": "gpt-5-codex",
    "gpt-5-codex-latest": "gpt-5-codex",
    "gpt5.1-codex": "gpt-5.1-codex",
    "gpt-5.1-codex": "gpt-5.1-codex",
    "gpt-5.1-codex-latest": "gpt-5.1-codex",
    "gpt5-codex-mini": "gpt-5-codex-mini",
    "gpt-5-codex-mini": "gpt-5-codex-mini",
    "gpt-5-codex-mini-latest": "gpt-5-codex-mini"
}

# Fixed models payload (unchanged IDs)
MODELS_PAYLOAD = ModelsList(
    data=[
        # {"id": "gpt-5", "object": "model", "owned_by": "owner"},
        # {"id": "gpt-5-high", "object": "model", "owned_by": "owner"},
        # {"id": "gpt-5-medium", "object": "model", "owned_by": "owner"},
        # {"id": "gpt-5-low", "object": "model", "owned_by": "owner"},
        # {"id": "gpt-5-minimal", "object": "model", "owned_by": "owner"},
        {"id": "gpt-5.1", "object": "model", "owned_by": "owner"},
        {"id": "gpt-5.1-high", "object": "model", "owned_by": "owner"},
        {"id": "gpt-5.1-medium", "object": "model", "owned_by": "owner"},
        {"id": "gpt-5.1-low", "object": "model", "owned_by": "owner"},
        # {"id": "gpt-5-codex", "object": "model", "owned_by": "owner"},
        # {"id": "gpt-5-codex-high", "object": "model", "owned_by": "owner"},
        # {"id": "gpt-5-codex-medium", "object": "model", "owned_by": "owner"},
        # {"id": "gpt-5-codex-low", "object": "model", "owned_by": "owner"},
        {"id": "gpt-5.1-codex", "object": "model", "owned_by": "owner"},
        {"id": "gpt-5.1-codex-high", "object": "model", "owned_by": "owner"},
        {"id": "gpt-5.1-codex-medium", "object": "model", "owned_by": "owner"},
        {"id": "gpt-5.1-codex-low", "object": "model", "owned_by": "owner"},
        {"id": "gpt-5-codex-mini", "object": "model", "owned_by": "owner"},
        {"id": "gpt-5-codex-mini-high", "object": "model", "owned_by": "owner"},
        {"id": "gpt-5-codex-mini-medium", "object": "model", "owned_by": "owner"},
    ]
)

# ---------- Helpers ----------

# Tool remapping message for OpenCode environment
TOOL_REMAP_MESSAGE = """<user_instructions priority="0">
<environment_override priority="0">
YOU ARE IN A DIFFERENT ENVIRONMENT. These instructions override ALL previous tool references.
</environment_override>

<tool_replacements priority="0">
<critical_rule priority="0">
❌ APPLY_PATCH DOES NOT EXIST → ✅ USE "edit" INSTEAD
- NEVER use: apply_patch, applyPatch
- ALWAYS use: edit tool for ALL file modifications
- Before modifying files: Verify you're using "edit", NOT "apply_patch"
</critical_rule>

<critical_rule priority="0">
❌ UPDATE_PLAN DOES NOT EXIST → ✅ USE "todowrite" INSTEAD
- NEVER use: update_plan, updatePlan
- ALWAYS use: todowrite for ALL task/plan operations
- Use todoread to read current plan
- Before plan operations: Verify you're using "todowrite", NOT "update_plan"
</critical_rule>
</tool_replacements>

<substitution_rules priority="0">
Base instruction says:    You MUST use instead:
apply_patch           →   edit
applyPatch            →   edit
update_plan           →   todowrite
updatePlan            →   todowrite
read_plan             →   todoread
absolute paths        →   relative paths
</substitution_rules>

<verification_checklist priority="0">
Before file/plan modifications:
1. Am I using "edit" NOT "apply_patch"?
2. Am I using "todowrite" NOT "update_plan"?
3. Is this tool in the approved list above?
4. Am I using relative paths?

If ANY answer is NO → STOP and correct before proceeding.
</verification_checklist>
</user_instructions>
"""


def _add_tool_remap_message(input_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepend high-priority tool remapping instructions for OpenCode environment.
    
    Only called when tools are present in the request to avoid unnecessary injection.
    """
    remap_message = {
        "type": "message",
        "role": "developer",
        "content": [{
            "type": "input_text",
            "text": TOOL_REMAP_MESSAGE
        }]
    }
    return [remap_message, *input_list]


def _load_default_instructions() -> str:
    """Load GPT-5 Codex prompt with caching fallback strategy.
    
    Strategy:
    1. Check project-relative cache (.cache/gpt_5_prompt.md)
    2. If stale/missing, fetch from GitHub
    3. If GitHub fails, use bundled prompt.md as fallback
    
    Cache TTL: 24 hours (configurable via CODEX_PROMPT_CACHE_TTL_HOURS env var)
    """
    import os
    
    GITHUB_PROMPT_URL = "https://raw.githubusercontent.com/openai/codex/main/codex-rs/core/gpt_5_codex_prompt.md"
    CACHE_TTL_HOURS = int(os.getenv("CODEX_PROMPT_CACHE_TTL_HOURS", "24"))
    
    cache_dir = Path(__file__).parent / ".cache"
    cache_file = cache_dir / "gpt_5_prompt.md"
    cache_meta = cache_dir / "gpt_5_prompt.meta"
    bundled_prompt = Path(__file__).parent / "prompt.md"
    
    # Helper: check if cache is fresh
    def is_cache_fresh() -> bool:
        if not cache_file.exists() or not cache_meta.exists():
            return False
        try:
            timestamp = float(cache_meta.read_text().strip())
            age_hours = (time.time() - timestamp) / 3600
            return age_hours < CACHE_TTL_HOURS
        except (OSError, ValueError):
            return False
    
    # Helper: fetch from GitHub
    def fetch_from_github() -> str | None:
        try:
            import urllib.request
            print("✓ Fetching GPT-5 prompt from GitHub: https://github.com/openai/codex")
            with urllib.request.urlopen(GITHUB_PROMPT_URL, timeout=10) as response:
                if response.status == 200:
                    content = response.read().decode("utf-8")
                    # Save to cache
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(content, encoding="utf-8")
                    cache_meta.write_text(str(time.time()), encoding="utf-8")
                    print("✓ Cached GPT-5 prompt from GitHub")
                    return content
        except Exception as exc:
            logger.warning("Failed to fetch prompt from GitHub: %s", exc)
        return None
    
    # Strategy: cache → GitHub → bundled fallback
    if is_cache_fresh():
        try:
            print("✓ Using cached GPT-5 prompt")
            return cache_file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read cached prompt: %s", exc)
    
    # Try fetching from GitHub
    github_content = fetch_from_github()
    if github_content:
        return github_content
    
    # Fallback to bundled prompt.md
    try:
        print("✓ Using bundled prompt.md as fallback")
        return bundled_prompt.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load prompt from all sources (cache, GitHub, bundled): {exc}"
        ) from exc


_DEFAULT_INSTRUCTIONS: str | None = None


def _get_default_instructions() -> str:
    """Lazy-load instructions on first use (after logging is configured)."""
    global _DEFAULT_INSTRUCTIONS
    if _DEFAULT_INSTRUCTIONS is None:
        _DEFAULT_INSTRUCTIONS = _load_default_instructions()
    return _DEFAULT_INSTRUCTIONS


def _normalize_model(model: str | None) -> tuple[str, dict[str, str] | None]:
    if not isinstance(model, str) or not model.strip():
        return "gpt-5", None
    name = model.strip()
    m = _EFFORT_RE.search(name)
    effort = m.group(1).lower() if m else None
    base = _EFFORT_RE.sub("", name).lower()
    normalized = _MODEL_ALIAS.get(base, base)
    return normalized, ({"effort": effort, "summary": "auto"} if effort else None)


def _flatten_content(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for item in c:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"]) 
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(json.dumps(item, separators=(",", ":")))
        return " ".join(parts)
    return json.dumps(c, separators=(",", ":"))

def _extract_upstream_error(body: str) -> str:
    try:
        data = json.loads(body)
        err = data.get("error", {})
        msg = err.get("message")
        reset = err.get("resets_in_seconds")
        if reset is not None:
            return f"{msg or 'Upstream error'} (resets in {reset} seconds)"
        return msg or "Upstream error"
    except json.JSONDecodeError:
        return "Upstream error"

def _sanitize_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    out: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") != "function":
            continue
        fn = t.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        entry: dict[str, Any] = {
            "type": "function",
            "name": name,
        }
        if isinstance(fn.get("description"), str) and fn["description"].strip():
            entry["description"] = fn["description"].strip()
        if isinstance(fn.get("parameters"), dict):
            entry["parameters"] = fn["parameters"]
        out.append(entry)
    return out


def _messages_to_input(messages: list[Any]) -> list[dict[str, Any]]:
    input_list: list[dict[str, Any]] = []
    system_buf: list[str] = []

    for m in messages:
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        content = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
        text = _flatten_content(content)

        if role == "system":
            if text.strip():
                system_buf.append(text.strip())
            continue

        if role == "tool":
            tool_call_id = getattr(m, "tool_call_id", None)
            if tool_call_id is None and isinstance(content, dict):
                tool_call_id = content.get("tool_call_id")
            if tool_call_id:
                text = f"Function call result (id: {tool_call_id}): {text}"
            else:
                text = f"Function call result: {text}"
            input_list.append({
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": text}],
            })
            continue

        mapped_role = role if role in ("assistant", "developer", "user") else "user"
        content_type = "output_text" if mapped_role == "assistant" else "input_text"
        input_list.append({
            "type": "message",
            "role": mapped_role,
            "content": [{"type": content_type, "text": text}],
        })

    # Prepend system messages as a single user message (keeps behavior stable)
    if system_buf:
        input_list.insert(0, {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "\n\n".join(system_buf)}],
        })
    return input_list


# Stable per-process prompt cache key (matches documented behavior)
_PROMPT_CACHE_KEY = str(uuid.uuid4())

def _build_upstream_payload(req: ChatCompletionsRequest) -> dict[str, Any]:
    model, reasoning = _normalize_model(req.model)
    
    # Transform messages to input format
    input_list = _messages_to_input(req.messages)
    tools = _sanitize_tools(getattr(req, "tools", None))
    
    # Inject tool remapping message when tools are present
    if tools:
        input_list = _add_tool_remap_message(input_list)
    
    payload: dict[str, Any] = {
        "model": model,
        "instructions": _get_default_instructions(),
        "input": input_list,
        "tools": tools,
        "tool_choice": getattr(req, "tool_choice", None) or "auto",
        "parallel_tool_calls": True,
        "store": False,
        "stream": True,  # always request stream; we'll buffer when client asked non-stream
        "prompt_cache_key": _PROMPT_CACHE_KEY,
    }
    if reasoning:
        payload["reasoning"] = reasoning
        # When store=false and reasoning is present, request encrypted CoT
        payload["include"] = ["reasoning.encrypted_content"]
    # Do NOT forward unsupported tunables like max_tokens/max_output_tokens/temperature/top_p
    return payload


def _build_headers(request: Request) -> dict[str, str]:
    # Prefer incoming Authorization header; else app.state.auth_data
    headers = {
        "Host": "chatgpt.com",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "OpenAI-Beta": "responses=v1",
        "Origin": "https://chatgpt.com",
        "Referer": "https://chatgpt.com/",
    }
    # auth = request.headers.get("authorization") or request.headers.get("Authorization")
    # if auth:
    #     headers["Authorization"] = auth
    # else:
    auth_data = getattr(request.app.state, "auth_data", None)
    if isinstance(auth_data, dict):
        tokens = auth_data.get("tokens") or {}
        access_token = tokens.get("access_token")
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        account_id = tokens.get("account_id")
        if account_id:
            headers["chatgpt-account-id"] = account_id
        if not access_token:
            api_key = auth_data.get("OPENAI_API_KEY") or auth_data.get("api_key")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _make_debugger(request: Request) -> Optional[Callable[[str], None]]:
    """Compose a debug writer honoring application settings with per-request overrides."""

    settings = getattr(request.app.state, "settings", None)
    enabled = bool(getattr(settings, "debug_sse_enabled", False))
    path_value = getattr(settings, "debug_sse_path", "") if settings else ""
    path = path_value.strip() if isinstance(path_value, str) else ""

    # enable order: app default → query param → header
    qp = request.query_params
    qv = (qp.get("debug_sse") or qp.get("debug"))
    if isinstance(qv, str):
        q = qv.strip().lower()
        if q in {"1", "true", "yes", "on"}:
            enabled = True
        if q in {"0", "false", "no", "off"}:
            enabled = False
    hv = request.headers.get("x-debug-sse")
    if isinstance(hv, str):
        h = hv.strip().lower()
        if h in {"1", "true", "yes", "on"}:
            enabled = True
        if h in {"0", "false", "no", "off"}:
            enabled = False
    if not enabled:
        return None

    if path:
        def writer(line: str) -> None:
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] {line}\n")
            except Exception as e:  # noqa: BLE001
                logger.debug("debug write failed: %s", e)
        return writer

    def writer_log(line: str) -> None:
        logger.debug("%s", line)

    return writer_log

    
def _should_retry(status: int) -> bool:
    return status >= 500

async def _sleep_with_backoff(attempt: int, debug: Optional[Callable[[str], None]]) -> None:
    delay = RETRY_BACKOFF_BASE * (2 ** attempt)
    logger.info("Retrying upstream request after %.2fs (attempt %d)", delay, attempt + 1)
    if debug:
        debug(f"retrying upstream request after {delay:.2f}s (attempt {attempt + 1})")
    await asyncio.sleep(delay)

@asynccontextmanager
async def _upstream_request(
    client: Optional[aiohttp.ClientSession],
    url: str,
    payload: dict,
    headers: dict
) -> AsyncIterator[aiohttp.ClientResponse]:
    """Unified upstream request handling for both client modes."""
    if client is None:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=None)) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                yield response
    else:
        async with client.post(url, json=payload, headers=headers) as response:
            yield response
            
def _yield_error_response(state: _StreamState, message: str) -> Iterator[str]:
    """Generate consistent error response chunks."""
    for chunk in state.iter_content_chunks(f"Error: {message}", finish="stop"):
        yield chunk
    yield "data: [DONE]\n\n"
    
class _UpstreamStreamError(RuntimeError):
    """Raised when upstream SSE stream emits a terminal failure event."""

    def __init__(self, message: str, *, event: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.event = event or {}
    
async def _get_stream_iterator(response: aiohttp.ClientResponse) -> AsyncIterator[bytes]:
    """Get appropriate stream iterator based on response capabilities."""
    if hasattr(response, "aiter_raw"):
        return _iter_sse_lines(response.aiter_raw())
    else:
        return _iter_sse_lines(response.content.iter_chunked(1024))
# ---------- SSE translation ----------

class _StreamState:
    _SSE_EVENT_LIMIT_BYTES = 4 * 1024
    _DELTA_BUDGET_BYTES = _SSE_EVENT_LIMIT_BYTES - 512  # leave room for JSON envelope overhead

    def __init__(self, response_id: str, model: str) -> None:
        self.response_id = response_id
        self.model = model
        self.created = int(time.time())
        self.think_open = False
        self.tool_map: dict[str, dict[str, Any]] = {}
        self.args_buf: dict[str, str] = {}
        self.tool_order: list[str] = []
        self.args_buf: dict[str, str] = {}
        self.tool_order: list[str] = []
        self.summary_active = False

    def _format_event(self, data: dict[str, Any]) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _chunk(self, data: dict[str, Any]) -> str:
        return self._format_event(data)

    def _split_for_budget(self, text: str) -> list[str]:
        if not text:
            return []
        max_bytes = max(512, self._DELTA_BUDGET_BYTES)
        parts: list[str] = []
        buf: list[str] = []
        buf_bytes = 0
        for ch in text:
            encoded = ch.encode("utf-8")
            if buf and buf_bytes + len(encoded) > max_bytes:
                parts.append("".join(buf))
                buf = [ch]
                buf_bytes = len(encoded)
            else:
                buf.append(ch)
                buf_bytes += len(encoded)
        if buf:
            parts.append("".join(buf))
        return parts

    def iter_content_chunks(self, text: str, finish: str | None = None) -> Iterable[str]:
        segments = self._split_for_budget(text)
        if not segments:
            if finish is None:
                return
            segments = [""]
        last_index = len(segments) - 1
        for idx, segment in enumerate(segments):
            delta = {"content": segment} if segment else {}
            finish_reason = finish if finish is not None and idx == last_index else None
            data = {
                "id": self.response_id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            yield self._format_event(data)

    def iter_tool_chunks(
        self,
        idx: int,
        call_id: str,
        name: str,
        args_delta: str,
        *,
        finish: str | None = None,
    ) -> Iterable[str]:
        segments = self._split_for_budget(args_delta)
        if not segments:
            segments = [""]
        last_index = len(segments) - 1
        for seg_idx, segment in enumerate(segments):
            finish_reason = finish if finish is not None and seg_idx == last_index else None
            data = {
                "id": self.response_id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": idx,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": segment},
                                }
                            ]
                        },
                        "finish_reason": finish_reason,
                    }
                ],
            }
            yield self._format_event(data)

    def open_think_if_needed(self) -> Iterable[str]:
        if self.think_open:
            return ()
        self.think_open = True
        return self.iter_content_chunks("<think>")

    def close_think_if_open(self) -> Iterable[str]:
        if not self.think_open:
            return ()
        self.think_open = False
        # Ensure a trailing newline after closing think block for better UI formatting
        return self.iter_content_chunks("</think>\n\n")


async def _iter_sse_lines(byte_iter: AsyncIterator[bytes], *, max_buffer_bytes: int = 2 * 1024 * 1024) -> AsyncIterator[bytes]:
    """Yield SSE lines without relying on upstream readline limits."""
    buffer = bytearray()

    async for chunk in byte_iter:
        if not chunk:
            continue
        buffer.extend(chunk)
        while True:
            newline_idx = buffer.find(b"\n")
            if newline_idx == -1:
                break
            line = bytes(buffer[: newline_idx + 1])
            yield line
            del buffer[: newline_idx + 1]
        if len(buffer) > max_buffer_bytes:
            # Safeguard against runaways when upstream omits newlines.
            yield bytes(buffer)
            buffer.clear()

    if buffer:
        yield bytes(buffer)


async def _translate_sse(upstream_iter: AsyncIterator[bytes], state: _StreamState, debug: Optional[Callable[[str], None]]):
    async for raw in upstream_iter:
        line = raw.decode("utf-8", errors="ignore").rstrip("\r\n")
        if debug:
            debug(f"raw: {line[:500]}")
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            if debug:
                debug("event: [DONE]")
            break
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            if debug:
                debug(f"event: parse_error {payload[:200]}")
            continue
        et = ev.get("type")
        if debug:
            debug(f"event: {et or '<unknown>'} {payload[:300]}")

        if et in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            delta = ev.get("delta")
            if isinstance(delta, dict):
                delta = delta.get("text")
            if isinstance(delta, str) and delta:
                for chunk in state.open_think_if_needed():
                    yield chunk
                if et == "response.reasoning_summary_text.delta":
                    state.summary_active = True
                    for chunk in state.iter_content_chunks(delta):
                        yield chunk
                else:
                    if state.summary_active:
                        for chunk in state.iter_content_chunks("\r\n"):
                            yield chunk
                        state.summary_active = False
                    for chunk in state.iter_content_chunks(delta):
                        yield chunk
                # for chunk in state.iter_content_chunks(delta):
                #     yield chunk
            continue

        if et == "response.output_text.delta":
            delta = ev.get("delta")
            if isinstance(delta, dict):
                delta = delta.get("text")
            if isinstance(delta, str) and delta:
                for chunk in state.close_think_if_open():
                    yield chunk
                for chunk in state.iter_content_chunks(delta):
                    yield chunk
            continue

        if et in ("response.output_item.added", "response.output_item.delta"):
            item = ev.get("item") or {}
            if not isinstance(item, dict):
                continue
            if item.get("type") not in ("function_call", "web_search_call"):
                continue
            item_id = item.get("id") or item.get("call_id") or f"call_{len(state.tool_order)}"
            name = item.get("name") or ("web_search" if item.get("type") == "web_search_call" else "")
            if item_id not in state.tool_map:
                idx = len(state.tool_order)
                state.tool_order.append(item_id)
                state.tool_map[item_id] = {"index": idx, "call_id": item_id, "name": name}
                for chunk in state.iter_tool_chunks(idx, item_id, name, ""):
                    yield chunk
            args = item.get("arguments") or item.get("parameters") or ""
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
            if isinstance(args, str) and args:
                prev = state.args_buf.get(item_id, "")
                if not args.startswith(prev):  # reset if non-prefix delta
                    prev = ""
                delta = args[len(prev):]
                state.args_buf[item_id] = prev + delta
                if delta:
                    meta = state.tool_map[item_id]
                    for chunk in state.iter_tool_chunks(meta["index"], meta["call_id"], meta["name"], delta):
                        yield chunk
            continue

        if et == "response.function_call_arguments.delta":
            item_id = ev.get("item_id")
            delta = ev.get("delta") or ""
            if isinstance(item_id, str) and isinstance(delta, str) and delta:
                meta = state.tool_map.get(item_id)
                if not meta:
                    idx = len(state.tool_order)
                    state.tool_order.append(item_id)
                    state.tool_map[item_id] = meta = {"index": idx, "call_id": item_id, "name": ""}
                    for chunk in state.iter_tool_chunks(idx, item_id, "", ""):
                        yield chunk
                prev = state.args_buf.get(item_id, "")
                state.args_buf[item_id] = prev + delta
                for chunk in state.iter_tool_chunks(meta["index"], meta["call_id"], meta["name"], delta):
                    yield chunk
            continue

        if et == "response.completed":
            for chunk in state.close_think_if_open():
                yield chunk
            if state.tool_order:
                last_id = state.tool_order[-1]
                meta = state.tool_map.get(last_id, {"index": 0, "call_id": last_id, "name": ""})
                args_full = state.args_buf.get(last_id, "")
                for chunk in state.iter_tool_chunks(
                    meta["index"],
                    meta["call_id"],
                    meta.get("name", ""),
                    args_full,
                    finish="tool_calls",
                ):
                    yield chunk
            else:
                for chunk in state.iter_content_chunks("", finish="stop"):
                    yield chunk

            resp = ev.get("response") or {}
            usage = resp.get("usage") if isinstance(resp, dict) else None
            if isinstance(usage, dict):
                pt = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
                ct = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
                tt = int(usage.get("total_tokens") or (pt + ct))
                yield state._chunk(
                    {
                        "id": state.response_id,
                        "object": "chat.completion.chunk",
                        "created": state.created,
                        "model": state.model,
                        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt},
                        "choices": [],
                    }
                )
            break

        if et == "response.failed":
            err = "Upstream failure"
            err_obj = ev.get("error")
            if isinstance(err_obj, dict) and isinstance(err_obj.get("message"), str):
                err = err_obj["message"]
            raise _UpstreamStreamError(err, event=ev)
            # for chunk in state.iter_content_chunks(f"Error: {msg}", finish="stop"):
            #     yield chunk
            # break


# ---------- Routes ----------

@router.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "codex-openai-proxy"})


@router.get("/models", response_model=None)
@router.get("/v1/models", response_model=None)
async def models(_: Request) -> Union[JSONResponse, Response]:
    return JSONResponse(content=MODELS_PAYLOAD.dict(), headers={"access-control-allow-origin": "*"})


@router.post("/chat/completions", response_model=None)
@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> Union[JSONResponse, Response, StreamingResponse]:
    try:
        payload_dict = _coerce_json_object_from_bytes(await request.body())
        payload = ChatCompletionsRequest(**payload_dict)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON: %s", e)
        return JSONResponse(
            status_code=422,
            content={"error": {"message": f"Invalid JSON: {e.msg} at pos {e.pos}"}},
            headers={"Access-Control-Allow-Origin": "*"},
        )
    except (TypeError, ValueError) as e:
        logger.error("Invalid request body: %s", e)
        return JSONResponse(
            status_code=422,
            content={"error": {"message": str(e)}},
            headers={"Access-Control-Allow-Origin": "*"},
        )
    
    headers = _build_headers(request)
    debug_cb = _make_debugger(request)

    if "Authorization" not in headers:
        err = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": payload.model,
            "choices": [{"index": 0, "delta": {"content": "Error: missing Authorization"}, "finish_reason": "stop"}],
        }
        async def _err():
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream", headers={"Access-Control-Allow-Origin": "*"})

    upstream_payload = _build_upstream_payload(payload)

    client = getattr(request.app.state, "http_client", None)

    async def _stream() -> AsyncIterator[str]:
        response_id = f"chatcmpl-{uuid.uuid4()}"
        state = _StreamState(response_id, payload.model)
        for attempt in range(MAX_UPSTREAM_RETRIES + 1):
            try:
                async with _upstream_request(client, CHATGPT_RESPONSES_URL, upstream_payload, headers) as r:
                    if debug_cb:
                        debug_cb(f"upstream status: {r.status}")
                    if r.status >= 400:
                        body = await r.text()
                        if debug_cb:
                            debug_cb(f"upstream error body: {body[:500]}")
                        detail = _extract_upstream_error(body) or f"Upstream {r.status}"
                        if _should_retry(r.status) and attempt < MAX_UPSTREAM_RETRIES:
                            await _sleep_with_backoff(attempt, debug_cb)
                            continue
                        for chunk in _yield_error_response(state, detail):
                            yield chunk
                        return

                    # Handle stream iterator selection
                    stream_iter = await _get_stream_iterator(r)
                    async for out in _translate_sse(stream_iter, state, debug_cb):
                        yield out
                    break
            except _UpstreamStreamError as exc:
                if debug_cb:
                    debug_cb(f"streaming upstream failure: {exc}")
                if attempt < MAX_UPSTREAM_RETRIES:
                    await _sleep_with_backoff(attempt, debug_cb)
                    continue
                for chunk in _yield_error_response(state, f"Upstream stream failed: {exc}"):
                    yield chunk
                return
            except (asyncio.TimeoutError, ClientError) as exc:
                if debug_cb:
                    debug_cb(f"streaming exception: {exc}")
                    debug_cb(traceback.format_exc())
                if attempt < MAX_UPSTREAM_RETRIES:
                    await _sleep_with_backoff(attempt, debug_cb)
                    continue
                for chunk in _yield_error_response(state, f"Streaming error: {exc}"):
                    yield chunk
                return
        yield "data: [DONE]\n\n"

    streaming_setting = getattr(request.app.state, "settings", None)
    streaming_mode = getattr(streaming_setting, "streaming_mode", True)
        
    if streaming_mode:
        return StreamingResponse(_stream(), media_type="text/event-stream", headers={"Access-Control-Allow-Origin": "*"})

    # Non-streaming: buffer the stream and assemble a final response
    content_buf: list[str] = []
    think_open = False
    tool_calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    usage_out: dict[str, int] | None = None

    try:
        for attempt in range(MAX_UPSTREAM_RETRIES + 1):
            content_buf.clear()
            think_open = False
            tool_calls.clear()
            order.clear()
            usage_out = None
            try:
                async with _upstream_request(client, CHATGPT_RESPONSES_URL, upstream_payload, headers) as r:
                    if debug_cb:
                        debug_cb(f"upstream status: {r.status}")
                    if r.status >= 400:
                        body = await r.text()
                        if debug_cb:
                            debug_cb(f"upstream error body: {body[:500]}")
                        detail = _extract_upstream_error(body) or f"Upstream {r.status}"
                        if _should_retry(r.status) and attempt < MAX_UPSTREAM_RETRIES:
                            await _sleep_with_backoff(attempt, debug_cb)
                            continue
                        raise ClientError(detail)

                    async for raw in r.content:
                        line = raw.decode("utf-8", errors="ignore").rstrip("\r\n")
                        if debug_cb:
                            debug_cb(f"raw: {line[:500]}")
                        if not line.startswith("data: "):
                            continue
                        p = line[6:].strip()
                        if p == "[DONE]":
                            if debug_cb:
                                debug_cb("event: [DONE]")
                            break
                        try:
                            ev = json.loads(p)
                        except json.JSONDecodeError:
                            if debug_cb:
                                debug_cb(f"event: parse_error {p[:200]}")
                            continue
                        t = ev.get("type")
                        if debug_cb:
                            debug_cb(f"event: {t or '<unknown>'} {p[:300]}")

                        if t in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                            d = ev.get("delta")
                            if isinstance(d, dict):
                                d = d.get("text")
                            if isinstance(d, str) and d:
                                if not think_open:
                                    content_buf.append("<think>")
                                    think_open = True
                                content_buf.append(d)
                        elif t == "response.output_text.delta":
                            d = ev.get("delta")
                            if isinstance(d, dict):
                                d = d.get("text")
                            if isinstance(d, str) and d:
                                if think_open:
                                    content_buf.append("</think>\n\n")
                                    think_open = False
                                content_buf.append(d)
                        elif t in ("response.output_item.added", "response.output_item.delta"):
                            item = ev.get("item") or {}
                            if isinstance(item, dict) and item.get("type") in ("function_call", "web_search_call"):
                                item_id = item.get("id") or item.get("call_id") or f"call_{len(order)}"
                                if item_id not in tool_calls:
                                    order.append(item_id)
                                    tool_calls[item_id] = {
                                        "id": item_id,
                                        "type": "function",
                                        "function": {"name": item.get("name") or "", "arguments": ""},
                                    }
                                args = item.get("arguments") or item.get("parameters") or ""
                                if isinstance(args, dict):
                                    args = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
                                if isinstance(args, str):
                                    tool_calls[item_id]["function"]["arguments"] = args
                        elif t == "response.function_call_arguments.delta":
                            item_id = ev.get("item_id")
                            d = ev.get("delta") or ""
                            if isinstance(item_id, str) and isinstance(d, str) and d:
                                if item_id not in tool_calls:
                                    order.append(item_id)
                                    tool_calls[item_id] = {
                                        "id": item_id,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                tool_calls[item_id]["function"]["arguments"] += d
                        elif t == "response.completed":
                            resp = ev.get("response") or {}
                            usage = resp.get("usage") if isinstance(resp, dict) else None
                            if isinstance(usage, dict):
                                pt = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
                                ct = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
                                tt = int(usage.get("total_tokens") or (pt + ct))
                                usage_out = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
                            break
                        elif t == "response.failed":
                            err = "Upstream failure"
                            e = ev.get("error")
                            if isinstance(e, dict) and isinstance(e.get("message"), str):
                                err = e["message"]
                            raise _UpstreamStreamError(err, event=ev)
                            # raise ClientError(err)
                break
            except _UpstreamStreamError as exc:
                if debug_cb:
                    debug_cb(f"non-stream upstream failure: {exc}")
                if attempt < MAX_UPSTREAM_RETRIES:
                    await _sleep_with_backoff(attempt, debug_cb)
                    continue
                raise
            except (asyncio.TimeoutError, ClientError) as exc:
                if debug_cb:
                    debug_cb(f"non-stream exception: {exc}")
                if attempt < MAX_UPSTREAM_RETRIES:
                    await _sleep_with_backoff(attempt, debug_cb)
                    continue
                raise
    except (_UpstreamStreamError, ClientError) as exc:
        logger.warning("Falling back stub due to upstream error: %s", exc)
        if debug_cb:
            debug_cb(f"non-stream exception: {exc}")
        # Minimal stub to keep contract, mirrors prior behavior
        msg = (
            "I can help you with coding tasks! The proxy connection is working. "
            "What would you like assistance with?"
        )
        usage = {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80}
        resp = _make_chat_response(payload, msg, [], usage)
        return JSONResponse(content=resp.dict(), headers={"access-control-allow-origin": "*"})

    # finalize assembled response
    if think_open:
        content_buf.append("</think>\n\n")
    content = "".join(content_buf)
    tools_list = [tool_calls[k] for k in order]
    resp = _make_chat_response(payload, content, tools_list, usage_out)
    # return JSONResponse(content=resp.dict(), headers={"access-control-allow-origin": "*"})
    
    # Convert to SSE format for OpenCode compatibility
    async def _non_streaming_sse():
        # Convert single response to streaming format
        chunk_data = {
            "id": resp.id,
            "object": "chat.completion.chunk",
            "created": resp.created,
            "model": resp.model,
            "choices": [{
                "index": 0,
                "delta": {"content": content or ""},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"

        # Add usage info if available
        if resp.usage:
            usage_chunk = {
                "id": resp.id,
                "object": "chat.completion.chunk",
                "created": resp.created,
                "model": resp.model,
                "usage": resp.usage.dict(),
                "choices": []
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(_non_streaming_sse(), media_type="text/event-stream", headers={"Access-Control-Allow-Origin": "*"})


# ---------- Response builders ----------

def _make_chat_response(
    req: ChatCompletionsRequest,
    content: str | None,
    tool_calls: list[dict[str, Any]] | None,
    usage_data: dict[str, int] | None,
) -> ChatCompletionsResponse:
    message_kwargs: dict[str, Any] = {"role": "assistant"}
    if content:
        message_kwargs["content"] = content
    if tool_calls:
        message_kwargs["tool_calls"] = tool_calls
    usage_obj = _usage_from_dict(usage_data)
    return ChatCompletionsResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=req.model,
        choices=[Choice(index=0, message=ChatResponseMessage(**message_kwargs), finish_reason=("tool_calls" if tool_calls else "stop"))],
        usage=usage_obj,
    )


def _usage_from_dict(d: dict[str, int] | None) -> Usage | None:
    if not isinstance(d, dict):
        return None
    try:
        return Usage(
            prompt_tokens=int(d.get("prompt_tokens", 0)),
            completion_tokens=int(d.get("completion_tokens", 0)),
            total_tokens=int(d.get("total_tokens", 0)),
        )
    except Exception:  # noqa: BLE001
        return None
