"""LLM client. Which provider and model is set in config.toml [llm].

    llm = get_llm()
    text = llm.chat([{"role": "system", "content": "..."}, {"role": "user", "content": "Hello"}])
    for piece in llm.chat(messages, stream=True): print(piece, end="")

Providers:
  gemini  Google Gemini API (generateContent / streamGenerateContent). Needs an API key.
  ollama  local Ollama (/api/chat).
  openai  any OpenAI-compatible /v1/chat/completions server: llama.cpp server, LM Studio, vLLM.
Uses only the standard library. A server that is not on this machine is refused unless
[llm] allow_remote is true.
"""

from __future__ import annotations

import dataclasses
import json
import time
import urllib.error
import urllib.request
from typing import Iterator
from urllib.parse import quote, urlparse

from .config import LLMSettings, settings

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
# Temporary failures worth retrying: rate limit, server error, overloaded ("high demand").
RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable  # a busy / rate-limited server: another model may work


class _HTTPClient:
    def __init__(self, cfg: LLMSettings):
        host = urlparse(cfg.base_url).hostname or ""
        if host not in LOCAL_HOSTS and not cfg.allow_remote:
            raise LLMError(f"LLM server {cfg.base_url} is not on this machine. Set [llm] allow_remote = true "
                           "in config.toml if you really want to send questions to it.")
        self.cfg = cfg

    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _post(self, path: str, body: dict, stream: bool):
        url = self.cfg.base_url.rstrip("/") + path
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=self._headers())
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = urllib.request.urlopen(req, timeout=self.cfg.timeout_seconds)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                try:  # Gemini and OpenAI-style servers return {"error": {"message": ...}}
                    err = json.loads(detail)["error"]
                    detail = err.get("message", detail) if isinstance(err, dict) else err
                except (ValueError, KeyError, TypeError):
                    pass
                if e.code in RETRY_STATUS and attempt < self.cfg.max_retries:
                    time.sleep(2 ** attempt)  # 1 s, 2 s, 4 s, ...
                    continue
                raise LLMError(f"{self.cfg.provider} returned {e.code} for model {self.cfg.model!r}: "
                               f"{detail[:300]}", retryable=e.code in RETRY_STATUS) from e
            except urllib.error.URLError as e:
                raise LLMError(f"Cannot reach the {self.cfg.provider} server at {self.cfg.base_url} "
                               f"({e.reason}). Is it running / are you online?") from e
        if not stream:
            with resp:
                return json.loads(resp.read().decode("utf-8"))
        return resp

    @staticmethod
    def _sse(resp) -> Iterator[dict]:
        """JSON payloads of a server-sent-events stream ("data: {...}" lines)."""
        with resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                yield json.loads(data)

    def chat(self, messages: list[dict], *, stream: bool = False) -> str | Iterator[str]:
        raise NotImplementedError


class GeminiLLM(_HTTPClient):
    def __init__(self, cfg: LLMSettings):
        super().__init__(cfg)
        self.key = cfg.api_key()
        if not self.key:
            raise LLMError(f"No Gemini API key. Set the {cfg.api_key_env} environment variable, or add a line "
                           f"{cfg.api_key_env}=... to a .env file in 'RAG model' or the repository root (gitignored).")

    def _headers(self) -> dict:
        # The key goes in a header, not the URL, so it never shows up in logs or error text.
        return {"Content-Type": "application/json", "x-goog-api-key": self.key}

    def _body(self, messages: list[dict]) -> dict:
        system = [m["content"] for m in messages if m["role"] == "system"]
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        c = self.cfg
        generation = {"temperature": c.temperature, "topP": c.top_p, "maxOutputTokens": c.max_tokens}
        if c.thinking_budget != -1:
            generation["thinkingConfig"] = {"thinkingBudget": c.thinking_budget}
        body = {"contents": contents, "generationConfig": generation}
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        return body

    @staticmethod
    def _text(event: dict) -> tuple[str, str | None]:
        """(answer text, finish reason) of one response; hidden "thought" parts are skipped."""
        if not event.get("candidates"):
            reason = event.get("promptFeedback", {}).get("blockReason")
            if reason:
                raise LLMError(f"Gemini refused the request ({reason}).")
            return "", None
        cand = event["candidates"][0]
        parts = cand.get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if not p.get("thought")), cand.get("finishReason")

    def _path(self, method: str) -> str:
        return f"/models/{quote(self.cfg.model)}:{method}"

    def chat(self, messages: list[dict], *, stream: bool = False) -> str | Iterator[str]:
        if stream:
            return self._stream(messages)
        text, reason = self._text(self._post(self._path("generateContent"), self._body(messages), False))
        if not text:
            raise LLMError(f"Gemini returned no text (finish reason: {reason}).")
        return text

    def _stream(self, messages: list[dict]) -> Iterator[str]:
        resp = self._post(self._path("streamGenerateContent") + "?alt=sse", self._body(messages), True)
        produced, reason = False, None
        for event in self._sse(resp):
            text, reason = self._text(event)
            if text:
                produced = True
                yield text
        if not produced:
            raise LLMError(f"Gemini returned no text (finish reason: {reason}).")


class OllamaLLM(_HTTPClient):
    def _body(self, messages: list[dict], stream: bool) -> dict:
        c = self.cfg
        return {
            "model": c.model, "messages": messages, "stream": stream, "think": c.thinking_budget != 0,
            "keep_alive": c.keep_alive,
            "options": {"temperature": c.temperature, "top_p": c.top_p,
                        "num_ctx": c.context_window, "num_predict": c.max_tokens},
        }

    def chat(self, messages: list[dict], *, stream: bool = False) -> str | Iterator[str]:
        if not stream:
            return self._post("/api/chat", self._body(messages, False), False)["message"]["content"]
        return self._stream(messages)

    def _stream(self, messages: list[dict]) -> Iterator[str]:
        with self._post("/api/chat", self._body(messages, True), True) as resp:
            for line in resp:
                if not line.strip():
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise LLMError(event["error"])
                piece = event.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if event.get("done"):
                    break


class OpenAICompatibleLLM(_HTTPClient):
    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        key = self.cfg.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _body(self, messages: list[dict], stream: bool) -> dict:
        c = self.cfg
        return {"model": c.model, "messages": messages, "stream": stream,
                "temperature": c.temperature, "top_p": c.top_p, "max_tokens": c.max_tokens}

    def chat(self, messages: list[dict], *, stream: bool = False) -> str | Iterator[str]:
        if not stream:
            return self._post("/v1/chat/completions", self._body(messages, False), False)["choices"][0]["message"]["content"]
        return self._stream(messages)

    def _stream(self, messages: list[dict]) -> Iterator[str]:
        for event in self._sse(self._post("/v1/chat/completions", self._body(messages, True), True)):
            delta = event["choices"][0].get("delta", {}).get("content")
            if delta:
                yield delta


PROVIDERS = {"gemini": GeminiLLM, "ollama": OllamaLLM, "openai": OpenAICompatibleLLM}


class FallbackLLM:
    """Tries [llm] model, then each of fallback_models, when a model stays busy or
    rate-limited after its retries. Other errors (bad key, bad request) are raised at once."""

    def __init__(self, clients: list[_HTTPClient]):
        self.clients = clients
        self.used_model: str | None = None

    def chat(self, messages: list[dict], *, stream: bool = False) -> str | Iterator[str]:
        if stream:
            return self._stream(messages)
        for i, client in enumerate(self.clients):
            try:
                text = client.chat(messages)
                self.used_model = client.cfg.model
                return text
            except LLMError as e:
                if not e.retryable or i == len(self.clients) - 1:
                    raise

    def _stream(self, messages: list[dict]) -> Iterator[str]:
        for i, client in enumerate(self.clients):
            started = False
            try:
                for piece in client.chat(messages, stream=True):
                    if not started:
                        started = True
                        self.used_model = client.cfg.model
                    yield piece
                return
            except LLMError as e:
                # Once text has been streamed, switching model would garble the answer.
                if started or not e.retryable or i == len(self.clients) - 1:
                    raise


def get_llm(cfg: LLMSettings | None = None) -> FallbackLLM:
    cfg = cfg or settings().llm
    try:
        provider = PROVIDERS[cfg.provider]
    except KeyError:
        raise LLMError(f"Unknown [llm] provider {cfg.provider!r}; choose one of: {', '.join(PROVIDERS)}") from None
    models = [cfg.model] + [m for m in cfg.fallback_models if m != cfg.model]
    return FallbackLLM([provider(dataclasses.replace(cfg, model=m)) for m in models])
