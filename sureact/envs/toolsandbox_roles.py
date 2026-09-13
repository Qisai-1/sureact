from __future__ import annotations

import os
import re
from typing import Any

from openai import OpenAI
from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent
from tool_sandbox.roles.openai_api_user import OpenAIAPIUser

DEFAULT_BASE_URL = "http://localhost:8000/v1"


def _client() -> OpenAI:
    client = OpenAI(
        base_url=os.environ.get("SUREACT_LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.environ.get("SUREACT_LLM_API_KEY", "EMPTY"),
    )
    _create = client.chat.completions.create

    def create_with_defaults(**kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("temperature", float(os.environ.get("SUREACT_TEMPERATURE", "0")))
        seed = os.environ.get("SUREACT_SEED")
        if seed is not None:
            kwargs.setdefault("seed", int(seed))
        choice = getattr(client, "_sureact_tool_choice", None)
        if choice and kwargs.get("tools"):
            kwargs.setdefault("tool_choice", choice)
        if os.environ.get("SUREACT_THINK", "0") != "1":
            extra = dict(kwargs.get("extra_body") or {})
            extra.setdefault("chat_template_kwargs", {"enable_thinking": False})
            kwargs["extra_body"] = extra
        return _create(**kwargs)

    client.chat.completions.create = create_with_defaults  # type: ignore[method-assign]
    return client


def _model_name() -> str:
    name = os.environ.get("SUREACT_LLM_MODEL")
    if not name:
        raise RuntimeError(
            "SUREACT_LLM_MODEL is not set"
        )
    return name


_IDENT_SAFE = re.compile(r"[^0-9a-zA-Z_]")


class _NormalizeOpenAIResponse:

    def model_inference(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        response = super().model_inference(*args, **kwargs)  # type: ignore[misc]
        for choice in response.choices:
            tool_calls = choice.message.tool_calls
            if not tool_calls:
                choice.message.tool_calls = None
                continue
            for tc in tool_calls:
                if tc.id:
                    safe = _IDENT_SAFE.sub("_", tc.id)
                    if safe[:1].isdigit():
                        safe = "t_" + safe
                    tc.id = safe
        return response


class LocalAgent(_NormalizeOpenAIResponse, OpenAIAPIAgent):
    """Agent backed by a local OpenAI-compatible server."""

    def __init__(self) -> None:
        self.openai_client = _client()
        self.model_name = _model_name()


class _UserCannotUseAgentTools:

    def respond(self, ending_index: Any = None) -> Any:  # type: ignore[no-untyped-def]
        try:
            return super().respond(ending_index=ending_index)  # type: ignore[misc]
        except KeyError as exc:
            if "is not a known allowed tool" not in str(exc):
                raise
            self._suppress_tool_calls = True
            try:
                return super().respond(ending_index=ending_index)  # type: ignore[misc]
            finally:
                self._suppress_tool_calls = False

    def model_inference(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        response = super().model_inference(*args, **kwargs)  # type: ignore[misc]
        if getattr(self, "_suppress_tool_calls", False):
            for choice in response.choices:
                choice.message.tool_calls = None
                if not (choice.message.content or "").strip():
                    choice.message.content = (
                        "I'm not able to do that myself -- could you take care of it?"
                    )
        return response


class LocalUser(_UserCannotUseAgentTools, _NormalizeOpenAIResponse, OpenAIAPIUser):

    def __init__(self) -> None:
        self.openai_client = _client()
        self.model_name = _model_name()


