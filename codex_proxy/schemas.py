"""Pydantic models mirroring the Rust proxy structures."""
from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any


class ChatCompletionsRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow", arbitrary_types_allowed=True)
    model: Optional[str] = "gpt-5"  # Default to gpt-5 if not provided
    messages: List[Any]  # Changed to Any to be more flexible
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = Field(default=None, alias="tool_choice")


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ChatResponseMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class Choice(BaseModel):
    index: int
    message: ChatResponseMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionsResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None


class ModelsList(BaseModel):
    object: str = "list"
    data: List[dict]
