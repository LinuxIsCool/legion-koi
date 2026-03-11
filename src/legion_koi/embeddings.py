"""Embedding provider infrastructure — pluggable backends.

Mirrors the TypeScript LLM plugin pattern: Protocol + concrete providers + factory.
Each embedding result carries its model name for multi-model storage in PostgreSQL.
"""

import math
import os
from pathlib import Path
from typing import Protocol

import httpx
import structlog

from .constants import MAX_EMBED_CHARS, EMBED_BATCH_SIZE

log = structlog.stdlib.get_logger()


def _load_telus_env() -> dict[str, str]:
    """Load TELUS API secrets from dotenv file."""
    env_path = Path("~/.claude/local/secrets/telus-api.env").expanduser()
    if not env_path.exists():
        return {}
    result = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize an embedding vector."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class Embedder(Protocol):
    """Provider-agnostic embedding interface."""

    def embed(self, text: str, input_type: str = "passage") -> list[float]: ...
    def embed_batch(self, texts: list[str], input_type: str = "passage") -> list[list[float]]: ...
    def get_dimensions(self) -> int: ...
    def get_model(self) -> str: ...
    def is_available(self) -> bool: ...


class TelusEmbedder:
    """TELUS AI Console embeddings — nvidia/nv-embedqa-e5-v5, 1024 dims."""

    def __init__(self, url: str | None = None, key: str | None = None, model: str | None = None):
        env = _load_telus_env()
        self._url = url or os.environ.get("TELUS_EMBED_URL") or env.get("TELUS_EMBED_URL", "")
        self._key = key or os.environ.get("TELUS_EMBED_KEY") or env.get("TELUS_EMBED_KEY", "")
        self._model = model or os.environ.get("TELUS_EMBED_MODEL") or env.get("TELUS_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0),
        )

    def embed(self, text: str, input_type: str = "passage") -> list[float]:
        return self.embed_batch([text], input_type=input_type)[0]

    def embed_batch(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        truncated = [t[:MAX_EMBED_CHARS] for t in texts]
        resp = self._client.post(
            self._url,
            headers={"Authorization": f"Bearer {self._key}"},
            json={
                "input": truncated,
                "model": self._model,
                "encoding_format": "float",
                "input_type": input_type,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        return [_l2_normalize(e) for e in embeddings]

    def get_dimensions(self) -> int:
        return 1024

    def get_model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        if not self._url or not self._key:
            return False
        try:
            result = self.embed("test", input_type="query")
            return len(result) == 1024
        except Exception:
            return False


# Known dimensions for common Ollama embedding models
_OLLAMA_KNOWN_DIMS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
}


class OllamaEmbedder:
    """Ollama local embeddings — sequential, configurable model."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self._base_url = (base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self._model = model or os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self._dimensions: int | None = _OLLAMA_KNOWN_DIMS.get(self._model)
        self._client = httpx.Client(timeout=60.0)

    def embed(self, text: str, input_type: str = "passage") -> list[float]:
        resp = self._client.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text[:MAX_EMBED_CHARS]},
        )
        resp.raise_for_status()
        vec = resp.json()["embedding"]
        if self._dimensions is None:
            self._dimensions = len(vec)
        return _l2_normalize(vec)

    def embed_batch(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        return [self.embed(t, input_type=input_type) for t in texts]

    def get_dimensions(self) -> int:
        if self._dimensions is None:
            # Probe with a test embedding
            vec = self.embed("test")
            self._dimensions = len(vec)
        return self._dimensions

    def get_model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        try:
            self._client.get(f"{self._base_url}/api/version", timeout=5.0).raise_for_status()
            return True
        except Exception:
            return False


# -- Factory --

_default_embedder: Embedder | None = None


def create_embedder(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> Embedder:
    """Create an embedder — auto-detects Telus vs Ollama from env vars.

    Explicit override: provider="telus" or provider="ollama".
    """
    if provider == "ollama":
        return OllamaEmbedder(base_url=base_url, model=model)
    if provider == "telus":
        return TelusEmbedder(url=base_url, model=model)

    # Auto-detect: check for Telus credentials
    env = _load_telus_env()
    telus_url = os.environ.get("TELUS_EMBED_URL") or env.get("TELUS_EMBED_URL")
    telus_key = os.environ.get("TELUS_EMBED_KEY") or env.get("TELUS_EMBED_KEY")
    if telus_url and telus_key:
        return TelusEmbedder(url=base_url or telus_url, key=telus_key, model=model)

    return OllamaEmbedder(base_url=base_url, model=model)


def get_embedder() -> Embedder:
    """Get or create the default embedder (singleton)."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = create_embedder()
    return _default_embedder


def embed_texts(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """Embed texts using the default embedder."""
    return get_embedder().embed_batch(texts, input_type=input_type)


def embed_query(text: str) -> list[float]:
    """Embed a search query using the default embedder with query input_type."""
    return get_embedder().embed(text, input_type="query")
