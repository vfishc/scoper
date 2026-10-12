from __future__ import annotations
import time
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable
from scoper.agents.procthor.config import AdapterConfig, RetryConfig
from scoper.agents.procthor.image_encoder import ImageEncoder
from scoper.agents.procthor.message import Message

class VLMAdapterError(RuntimeError):
    pass

class RetryableVLMError(VLMAdapterError):
    pass

class NonRetryableVLMError(VLMAdapterError):
    pass

@dataclass
class CallMetadata:
    elapsed_seconds: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    attempts: list[dict[str, Any]] = field(default_factory=list)

@runtime_checkable
class VLMAdapter(Protocol):

    def call(self, messages: Sequence[Message]) -> str:
        ...

    def supports_images(self) -> bool:
        ...

    def close(self) -> None:
        ...

class BaseAdapter:

    def __init__(self, config: AdapterConfig, retry: RetryConfig, image_encoder: ImageEncoder) -> None:
        self.config = config
        self.retry = retry
        self.image_encoder = image_encoder
        self.last_call_metadata: Optional[CallMetadata] = None

    def call(self, messages: Sequence[Message]) -> str:
        started = time.time()
        self.last_call_metadata = None
        from scoper.benchmark.common.transport_retry import RetryBudget
        self._transport_retry_budget = RetryBudget(getattr(self, 'transport_retry_budget_seconds', None))
        last_exc: Optional[BaseException] = None
        attempts: list[dict[str, Any]] = []
        for attempt in range(self.retry.max_retries + 1):
            attempt_started = time.monotonic()
            started_at_utc = _utc_now()
            try:
                self._transport_retry_budget.remaining()
                text = self._call_once(messages)
                attempts.append(_attempt_record(attempt, started_at_utc, time.monotonic() - attempt_started, status='succeeded'))
                self._record_elapsed(started)
                assert self.last_call_metadata is not None
                self.last_call_metadata.attempts = attempts
                return text
            except NonRetryableVLMError as exc:
                attempts.append(_attempt_record(attempt, started_at_utc, time.monotonic() - attempt_started, status='fatal_error', error=exc))
                self._record_elapsed(started)
                assert self.last_call_metadata is not None
                self.last_call_metadata.attempts = attempts
                raise
            except Exception as exc:
                last_exc = exc
                retryable = self._is_retryable_exception(exc)
                should_retry = retryable and attempt < self.retry.max_retries
                from scoper.benchmark.common.transport_retry import retry_delay
                delay = retry_delay(exc, attempt, self.retry.backoff_base, self.retry.backoff_cap) if should_retry else 0.0
                attempts.append(_attempt_record(attempt, started_at_utc, time.monotonic() - attempt_started, status='retryable_error' if should_retry else 'fatal_error', error=exc, backoff_seconds=delay))
                if not should_retry:
                    self._record_elapsed(started)
                    assert self.last_call_metadata is not None
                    self.last_call_metadata.attempts = attempts
                    raise
                try:
                    self._transport_retry_budget.allow_wait(delay, exc)
                except Exception:
                    self._record_elapsed(started)
                    assert self.last_call_metadata is not None
                    self.last_call_metadata.attempts = attempts
                    raise
                time.sleep(delay)
        self._record_elapsed(started)
        assert self.last_call_metadata is not None
        self.last_call_metadata.attempts = attempts
        assert last_exc is not None
        raise last_exc

    def semantic_request(self, messages: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        return {'messages': copy.deepcopy(list(messages)), 'generation_config': redact_semantic_request({'max_tokens': self.config.max_tokens, 'temperature': self.config.temperature})}

    def supports_images(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def _call_once(self, messages: Sequence[Message]) -> str:
        raise NotImplementedError

    def _record_elapsed(self, started: float) -> None:
        if self.last_call_metadata is None:
            self.last_call_metadata = CallMetadata(elapsed_seconds=time.time() - started)
        else:
            self.last_call_metadata.elapsed_seconds = time.time() - started

    def _is_retryable_exception(self, exc: BaseException) -> bool:
        from scoper.benchmark.common.transport_retry import retryable_exception
        return retryable_exception(exc)

def require_dependency(module_name: str, install_hint: str) -> None:
    raise NonRetryableVLMError("Missing optional dependency '{}'. Install with {}.".format(module_name, install_hint))

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

def _attempt_record(index: int, started_at_utc: str, latency_seconds: float, *, status: str, error: BaseException | None=None, backoff_seconds: float=0.0) -> dict[str, Any]:
    status_code = getattr(error, 'status_code', None) or getattr(error, 'code', None)
    try:
        status_code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status_code = None
    return {'attempt_index': index, 'started_at_utc': started_at_utc, 'latency_seconds': max(0.0, float(latency_seconds)), 'status': status, 'error_type': None if error is None else type(error).__name__, 'status_code': status_code, 'message': None if error is None else _redact_error(str(error)), 'backoff_seconds': max(0.0, float(backoff_seconds))}

def _redact_error(value: str) -> str:
    redacted = re.sub('(?i)bearer\\s+\\S+', 'Bearer <redacted>', value)
    redacted = re.sub('sk-[A-Za-z0-9_-]{8,}', '<redacted>', redacted)
    redacted = re.sub('(?i)(api[_-]?key|authorization|token|password|secret)\\s*[=:]\\s*\\S+', '\\1=<redacted>', redacted)
    return redacted

def redact_semantic_request(value: Any) -> Any:
    credential_keys = {'api_key', 'apikey', 'authorization', 'cookie', 'credentials', 'password', 'secret', 'set_cookie', 'token', 'headers', 'request_headers'}
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            normalized = str(key).casefold().replace('-', '_')
            if normalized in credential_keys or normalized.endswith(('_api_key', '_access_token', '_password', '_secret')):
                continue
            result[str(key)] = redact_semantic_request(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact_semantic_request(item) for item in value]
    return copy.deepcopy(value)
__all__ = ['BaseAdapter', 'CallMetadata', 'NonRetryableVLMError', 'RetryableVLMError', 'VLMAdapter', 'VLMAdapterError', 'redact_semantic_request']
