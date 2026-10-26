"""Settings, read from config.toml with RAG_<SECTION>_<KEY> environment overrides.

    from rag_model.config import settings
    settings().llm.model
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.toml"
# Secrets such as GEMINI_API_KEY (gitignored): "RAG model/.env", then the repository root's .env.
ENV_FILES = (ROOT / ".env", ROOT.parent / ".env")


@dataclass
class LLMSettings:
    provider: str = "gemini"
    model: str = "gemini-3.5-flash-lite"
    fallback_models: list[str] = field(default_factory=list)
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env: str = "GEMINI_API_KEY"
    allow_remote: bool = True
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    thinking_budget: int = 0
    timeout_seconds: int = 120
    max_retries: int = 1
    context_window: int = 8192
    keep_alive: str = "30m"

    def api_key(self) -> str | None:
        """The key from the environment, or from the gitignored .env file."""
        return os.environ.get(self.api_key_env) or _read_env_file().get(self.api_key_env)


@dataclass
class EmbeddingSettings:
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "auto"
    query_device: str = "auto"
    dtype: str = "float16"
    query_dtype: str = "float32"
    max_seq_length: int = 1024
    query_instruction: str = ("Given a question about legal rights or the law in India, retrieve the "
                              "provisions of the Constitution of India that answer it")


@dataclass
class RetrievalSettings:
    seeds: int = 5
    min_seed_score: float = 0.3
    threshold: float = 0.25
    max_chunks: int = 12
    max_chars: int = 18000


@dataclass
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])
    max_question_chars: int = 2000
    max_history_messages: int = 6
    max_history_chars: int = 4000


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    source: str = ""


def _read_env_file() -> dict[str, str]:
    """KEY=value lines from the .env files (blank lines and # comments ignored, quotes
    stripped). The first file that defines a key wins."""
    values: dict[str, str] = {}
    for path in ENV_FILES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values.setdefault(key.strip(), value.strip().strip("'\""))
    return values


def _coerce(value: str, like):
    if isinstance(like, list):  # "a,b" -> ["a", "b"]
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(like, bool):
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected true/false, got {value!r}")
    return type(like)(value)


def load(path: Path | None = None) -> Settings:
    path = Path(path or os.environ.get("RAG_CONFIG") or DEFAULT_CONFIG)
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    result = Settings(source=str(path) if path.exists() else "(defaults)")
    for section_field in fields(Settings):
        if section_field.name == "source":
            continue
        section = getattr(result, section_field.name)
        given = data.pop(section_field.name, {})
        known = {f.name for f in fields(section)}
        unknown = set(given) - known
        if unknown:
            raise ValueError(f"{path}: unknown setting(s) in [{section_field.name}]: {', '.join(sorted(unknown))}")
        for f in fields(section):
            value = given.get(f.name, getattr(section, f.name))
            env = os.environ.get(f"RAG_{section_field.name.upper()}_{f.name.upper()}")
            if env is not None:
                value = _coerce(env, getattr(section, f.name))
            setattr(section, f.name, value)
    if data:
        raise ValueError(f"{path}: unknown section(s): {', '.join(sorted(data))}")
    return result


@lru_cache(maxsize=1)
def settings() -> Settings:
    return load()
