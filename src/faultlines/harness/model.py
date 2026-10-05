"""Model clients.

CompletionEndpointModel renders the Gemma 4 chat template locally (transformers
tokenizer) and calls an OpenAI-compatible /v1/completions endpoint (vLLM) with
special tokens kept. That way the *raw* model text, including broken
<|tool_call> spans, is recorded, which the official harness never shows.

ScriptedModel replays fixed outputs for tests and demos.
"""

from __future__ import annotations

import copy
import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ModelTurn:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatModel(Protocol):
    name: str

    def generate(self, messages: list[dict], tools: list[dict]) -> ModelTurn: ...


class ScriptedModel:
    def __init__(self, outputs: list[str | ModelTurn], name: str = "scripted"):
        self.outputs = list(outputs)
        self.name = name
        self.calls = 0

    def generate(self, messages, tools) -> ModelTurn:
        self.calls += 1
        if not self.outputs:
            return ModelTurn("I have nothing more to do.", "stop", 0, 0)
        o = self.outputs.pop(0)
        return o if isinstance(o, ModelTurn) else ModelTurn(o, "stop", 1000 * self.calls, 50)


def to_gemma_inline(messages: list[dict]) -> list[dict]:
    """Fold role=tool messages into the preceding assistant message as `tool_responses`
    (the shape shown in Google's Gemma 4 function-calling docs)."""
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "tool" and out and out[-1].get("role") == "assistant":
            try:
                resp = json.loads(m["content"])
            except (json.JSONDecodeError, TypeError):
                resp = {"result": m["content"]}
            out[-1].setdefault("tool_responses", []).append({"name": m.get("name", ""), "response": resp})
        else:
            out.append(copy.deepcopy(m))
    return out


class CompletionEndpointModel:
    def __init__(self, base_url: str, model: str, tokenizer_path: str, max_output_tokens: int = 16384,
                 max_model_len: int = 32768, temperature: float = 0.0, top_p: float | None = None,
                 seed: int | None = 0, chat_template_kwargs: dict | None = None,
                 tool_message_style: str = "gemma_inline", timeout: int = 1800):
        from transformers import AutoTokenizer
        self.base_url = base_url.rstrip("/")
        self.name = model
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_output_tokens = max_output_tokens
        self.max_model_len = max_model_len
        self.temperature, self.top_p, self.seed = temperature, top_p, seed
        self.template_kwargs = chat_template_kwargs or {}
        self.style = tool_message_style
        self.timeout = timeout

    def render(self, messages: list[dict], tools: list[dict]) -> str:
        msgs = to_gemma_inline(messages) if self.style == "gemma_inline" else messages
        return self.tok.apply_chat_template(msgs, tools=[{"type": "function", "function": t} for t in tools],
                                            add_generation_prompt=True, tokenize=False, **self.template_kwargs)

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def generate(self, messages: list[dict], tools: list[dict]) -> ModelTurn:
        prompt = self.render(messages, tools)
        n = self.count(prompt)
        max_tokens = min(self.max_output_tokens, self.max_model_len - n - 8)
        if max_tokens < 64:
            return ModelTurn("", "context_overflow", n, 0)
        body: dict[str, Any] = {"model": self.name, "prompt": prompt, "max_tokens": max_tokens,
                                "temperature": self.temperature, "skip_special_tokens": False,
                                "spaces_between_special_tokens": False, "add_special_tokens": False}
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.seed is not None:
            body["seed"] = self.seed
        req = urllib.request.Request(f"{self.base_url}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        ch = data["choices"][0]
        usage = data.get("usage") or {}
        return ModelTurn(ch.get("text", ""), ch.get("finish_reason"), usage.get("prompt_tokens", n),
                         usage.get("completion_tokens"))
