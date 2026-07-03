from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator

from langfuse import Langfuse


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LangfuseSettings:
    enabled: bool
    public_key: str
    secret_key: str
    base_url: str | None
    app_env: str | None
    app_version: str | None
    capture_full_prompt: bool


class _NoOpObservation:
    id: str | None = None

    def update(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    def score(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    def __enter__(self) -> "_NoOpObservation":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:  # noqa: ARG002
        return False


@lru_cache(maxsize=1)
def get_langfuse_settings() -> LangfuseSettings:
    enabled_raw = os.getenv("LANGFUSE_ENABLED", "true").strip().lower()
    enabled = enabled_raw not in {"0", "false", "no", "off"}

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    base_url = os.getenv("LANGFUSE_BASE_URL", os.getenv("LANGFUSE_HOST", "")).strip() or None
    app_env = os.getenv("APP_ENV", "").strip() or None
    app_version = os.getenv("APP_VERSION", "").strip() or None
    capture_full_prompt = os.getenv("LANGFUSE_CAPTURE_FULL_PROMPT", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    keys_present = bool(public_key and secret_key)
    return LangfuseSettings(
        enabled=enabled and keys_present,
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        app_env=app_env,
        app_version=app_version,
        capture_full_prompt=capture_full_prompt,
    )


@lru_cache(maxsize=1)
def get_langfuse_client() -> Langfuse | None:
    settings = get_langfuse_settings()
    if not settings.enabled:
        return None

    try:
        return Langfuse(
            public_key=settings.public_key,
            secret_key=settings.secret_key,
            base_url=settings.base_url,
            environment=settings.app_env,
            release=settings.app_version,
            tracing_enabled=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse client unavailable; tracing disabled: %s", exc)
        return None


def is_langfuse_enabled() -> bool:
    return get_langfuse_client() is not None


def truncate_value(value: Any, max_chars: int = 1000) -> Any:
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return f"{value[:max_chars]}...<truncated>"
    if isinstance(value, list):
        return [truncate_value(item, max_chars=max_chars) for item in value]
    if isinstance(value, dict):
        return {key: truncate_value(item, max_chars=max_chars) for key, item in value.items()}
    return value


def compact_text(text: str, max_chars: int = 1000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...<truncated>"


@contextmanager
def langfuse_observation(kind: str, **kwargs: Any) -> Iterator[Any]:
    client = get_langfuse_client()
    if client is None:
        yield _NoOpObservation()
        return

    starter = getattr(client, f"start_as_current_{kind}", None)
    if not callable(starter):
        yield _NoOpObservation()
        return

    with starter(**kwargs) as observation:
        yield observation


@contextmanager
def langfuse_span(**kwargs: Any) -> Iterator[Any]:
    with langfuse_observation("span", **kwargs) as observation:
        yield observation


@contextmanager
def langfuse_generation(**kwargs: Any) -> Iterator[Any]:
    with langfuse_observation("generation", **kwargs) as observation:
        yield observation


def safe_update_observation(observation: Any | None, **kwargs: Any) -> None:
    if observation is None:
        return

    try:
        update = getattr(observation, "update", None)
        if callable(update):
            update(**kwargs)
            return
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse observation update failed: %s", exc)
        return

    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.update_current_span(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse current span update failed: %s", exc)


def safe_update_trace(**kwargs: Any) -> None:
    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.update_current_trace(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse trace update failed: %s", exc)


def safe_score_trace(**kwargs: Any) -> None:
    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.score_current_trace(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse trace scoring failed: %s", exc)


def safe_score_observation(observation: Any | None, **kwargs: Any) -> None:
    if observation is None:
        return

    try:
        score = getattr(observation, "score", None)
        if callable(score):
            score(**kwargs)
            return
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse observation scoring failed: %s", exc)
        return

    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.score_current_span(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse current span scoring failed: %s", exc)


def safe_create_score(**kwargs: Any) -> None:
    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.create_score(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse create score failed: %s", exc)


def safe_flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.flush()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse flush failed: %s", exc)


def get_current_trace_id() -> str | None:
    client = get_langfuse_client()
    if client is None:
        return None

    try:
        trace_id = client.get_current_trace_id()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse current trace id unavailable: %s", exc)
        return None

    return str(trace_id) if trace_id else None


def get_current_observation_id() -> str | None:
    client = get_langfuse_client()
    if client is None:
        return None

    try:
        observation_id = client.get_current_observation_id()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse current observation id unavailable: %s", exc)
        return None

    return str(observation_id) if observation_id else None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
