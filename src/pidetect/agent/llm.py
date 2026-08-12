"""Phase 6 Tier 1 -- provider abstraction (docs/phase6_tier1_design.md sec 1).

One canonical shape (ToolSchema/Message/ToolCall/LLMTurn) that every provider adapter
translates its own wire format into and out of. Nothing outside this module ever sees a
Gemini- or Ollama-specific request/response object -- src/pidetect/agent/agent.py only
ever talks to an LLMProvider.

Concrete adapters:
    GeminiProvider  -- REST (generativelanguage.googleapis.com), free-tier model, API key
                       from an env var (never hardcoded, never logged).
    OllamaProvider  -- local, OpenAI-compatible /api/chat endpoint with tools=[...].

get_provider(cfg) selects between them from configs/phase6.yaml's `provider` key.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

import requests


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    call_id: str
    name: str
    args: dict
    # Opaque provider-specific round-trip data (e.g. Gemini's thought_signature, required on
    # replay by "thinking" models -- https://ai.google.dev/gemini-api/docs/thought-signatures).
    # agent.py never inspects this; only the adapter that produced it reads it back.
    provider_meta: dict = field(default_factory=dict)


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_call_id: Optional[str] = None   # role="tool": which call this is answering
    name: Optional[str] = None           # role="tool": the tool name
    tool_call: Optional[ToolCall] = None  # role="assistant": set when this turn issued a call


@dataclass
class LLMTurn:
    kind: Literal["tool_call", "final_answer"]
    tool_call: Optional[ToolCall] = None
    text: Optional[str] = None


class LLMProvider(Protocol):
    def step(self, system_prompt: str, messages: list[Message], tools: list[ToolSchema]) -> LLMTurn:
        ...


class ProviderError(RuntimeError):
    """Base class for adapter-level failures (auth, network, malformed tool-call args)."""


class ToolArgsParseError(ProviderError):
    """A provider returned a tool call whose arguments couldn't be parsed as JSON."""


# ---------------------------------------------------------------------------
# Gemini adapter
# ---------------------------------------------------------------------------

def _gemini_safe_schema(schema: dict) -> dict:
    """Gemini's function-calling `parameters` is a restricted OpenAPI 3.0 subset -- it
    rejects standard JSON-Schema keys like "additionalProperties" outright (HTTP 400,
    confirmed against the live API while implementing this). Canonical ToolSchema.parameters
    stays plain JSON Schema (sec 1.2: the one shape tool authors write against); this is the
    provider-specific translation step, kept local to this adapter so nothing upstream needs
    to know Gemini is pickier than the schema format itself allows."""
    if not isinstance(schema, dict):
        return schema
    out = {k: v for k, v in schema.items() if k != "additionalProperties"}
    if "properties" in out:
        out["properties"] = {k: _gemini_safe_schema(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _gemini_safe_schema(out["items"])
    return out


class GeminiProvider:
    """docs/phase6_tier1_design.md sec 1.3.

    Wraps ToolSchema list into one Tool(function_declarations=[...]). Reads
    candidates[0].content.parts: a part with "functionCall" -> tool_call; otherwise the
    concatenated "text" parts -> final_answer. Args arrive as a native dict (Struct via
    the JSON REST binding), no JSON-string parsing needed here (contrast OllamaProvider).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-lite",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_s: float = 60.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout_s = timeout_s

    def _contents(self, messages: list[Message]) -> list[dict]:
        contents = []
        for m in messages:
            if m.role == "system":
                continue  # sent as system_instruction instead
            if m.role == "user":
                contents.append({"role": "user", "parts": [{"text": m.content}]})
            elif m.role == "assistant":
                if m.tool_call is not None:
                    part = {"functionCall": {"name": m.tool_call.name, "args": m.tool_call.args}}
                    # Required on replay for "thinking" models (2026-08 API, confirmed live):
                    # a functionCall part missing thoughtSignature is rejected outright.
                    sig = m.tool_call.provider_meta.get("thought_signature")
                    if sig:
                        part["thoughtSignature"] = sig
                    contents.append({"role": "model", "parts": [part]})
                else:
                    contents.append({"role": "model", "parts": [{"text": m.content}]})
            elif m.role == "tool":
                try:
                    response_obj = json.loads(m.content) if m.content else {}
                except json.JSONDecodeError:
                    response_obj = {"result": m.content}
                if not isinstance(response_obj, dict):
                    response_obj = {"result": response_obj}
                # Confirmed against the live API (2026-08): role "function" is rejected
                # ("Role 'function' is not supported... use SYSTEM/USER/ASSISTANT/MODEL/...");
                # the functionResponse part goes on a "user" turn instead.
                contents.append({"role": "user", "parts": [
                    {"functionResponse": {"name": m.name, "response": response_obj}}
                ]})
        return contents

    def step(self, system_prompt: str, messages: list[Message], tools: list[ToolSchema]) -> LLMTurn:
        url = f"{self.base_url}/models/{self.model}:generateContent"
        body = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": self._contents(messages),
            "tools": [{"functionDeclarations": [
                {"name": t.name, "description": t.description, "parameters": _gemini_safe_schema(t.parameters)}
                for t in tools
            ]}],
        }
        # Free-tier rate limits are tight enough to hit mid-conversation, not just between
        # questions -- retry 429s with backoff (honoring Retry-After when the API sends one)
        # rather than surfacing a transient rate-limit blip as a hard failure.
        max_retries = 4
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    url, params={"key": self.api_key}, json=body, timeout=self.timeout_s,
                )
                if resp.status_code == 429 and attempt < max_retries:
                    delay = float(resp.headers.get("Retry-After", 2 ** attempt * 5))
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                raise ProviderError(f"Gemini request failed: {exc}") from exc

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderError(f"Gemini returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        for i, part in enumerate(parts):
            fc = part.get("functionCall")
            if fc:
                provider_meta = {}
                sig = part.get("thoughtSignature")
                if sig:
                    provider_meta["thought_signature"] = sig
                return LLMTurn(
                    kind="tool_call",
                    tool_call=ToolCall(
                        call_id=fc.get("id") or f"gemini-{i}",
                        name=fc["name"],
                        args=dict(fc.get("args") or {}),
                        provider_meta=provider_meta,
                    ),
                )
        text = "".join(p.get("text", "") for p in parts)
        return LLMTurn(kind="final_answer", text=text)


# ---------------------------------------------------------------------------
# Ollama adapter
# ---------------------------------------------------------------------------

class OllamaProvider:
    """docs/phase6_tier1_design.md sec 1.3.

    Local, OpenAI-compatible /api/chat with tools=[...]. Reads
    message.tool_calls[0].function -- arguments arrives as a JSON STRING here (OpenAI-style
    wire format, unlike Gemini's native dict), so this is the adapter that has to
    json.loads() it; a malformed-JSON response from a small local model is caught and
    raised as ToolArgsParseError (retryable at the agent-loop layer), never crashes silently.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout_s = timeout_s

    def _messages(self, system_prompt: str, messages: list[Message]) -> list[dict]:
        out = [{"role": "system", "content": system_prompt}]
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "assistant" and m.tool_call is not None:
                out.append({
                    "role": "assistant",
                    "content": m.content or "",
                    "tool_calls": [{
                        "id": m.tool_call.call_id,
                        "type": "function",
                        "function": {"name": m.tool_call.name, "arguments": json.dumps(m.tool_call.args)},
                    }],
                })
            elif m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    def step(self, system_prompt: str, messages: list[Message], tools: list[ToolSchema]) -> LLMTurn:
        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model,
            "messages": self._messages(system_prompt, messages),
            "tools": [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ],
            "stream": False,
        }
        try:
            resp = requests.post(url, json=body, timeout=self.timeout_s)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"Ollama request failed (is `ollama serve` running at {self.base_url}?): {exc}") from exc

        data = resp.json()
        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            tc = tool_calls[0]
            fn = tc.get("function", {})
            args_raw = fn.get("arguments")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError as exc:
                    raise ToolArgsParseError(
                        f"Ollama model {self.model!r} returned unparseable tool-call "
                        f"arguments for {fn.get('name')!r}: {args_raw!r}"
                    ) from exc
            else:
                args = dict(args_raw or {})
            call_id = tc.get("id") or f"ollama-{fn.get('name', 'call')}"
            return LLMTurn(kind="tool_call", tool_call=ToolCall(call_id=call_id, name=fn.get("name", ""), args=args))
        return LLMTurn(kind="final_answer", text=msg.get("content", ""))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(cfg: dict) -> LLMProvider:
    """configs/phase6.yaml's `provider` key selects the adapter; no provider-specific
    detail leaks past this function."""
    provider = cfg.get("provider")
    if provider == "gemini":
        key_env = cfg.get("gemini_api_key_env", "GEMINI_API_KEY")
        api_key = os.environ.get(key_env)
        if not api_key:
            raise ProviderError(
                f"provider='gemini' but env var {key_env} is not set. "
                f"Get a free-tier key at https://aistudio.google.com/apikey and export it."
            )
        return GeminiProvider(api_key=api_key, model=cfg.get("gemini_model", "gemini-3.1-flash-lite"))
    if provider == "ollama":
        return OllamaProvider(
            model=cfg.get("ollama_model", "qwen2.5:7b"),
            base_url=cfg.get("ollama_base_url", "http://localhost:11434"),
        )
    raise ValueError(f"Unknown provider in configs/phase6.yaml: {provider!r} (expected 'gemini' or 'ollama')")
