from __future__ import annotations
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, Sequence, runtime_checkable
from scoper.agents.behavior.config import AdapterConfig, RetryConfig
from scoper.agents.behavior.image_encoder import ImageEncoder
from scoper.agents.behavior.message import Message

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
    attempts: tuple[dict[str, Any], ...] = ()

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
            attempt_started = time.time()
            try:
                self._transport_retry_budget.remaining()
                text = self._call_once(messages)
                attempts.append(_attempt_record(attempt, attempt_started, status='succeeded', error=None, backoff_seconds=0.0))
                self._record_elapsed(started, attempts)
                return text
            except NonRetryableVLMError as exc:
                attempts.append(_attempt_record(attempt, attempt_started, status='fatal_error', error=exc, backoff_seconds=0.0))
                self._record_elapsed(started, attempts)
                raise
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable_exception(exc) or attempt >= self.retry.max_retries:
                    attempts.append(_attempt_record(attempt, attempt_started, status='fatal_error', error=exc, backoff_seconds=0.0))
                    self._record_elapsed(started, attempts)
                    raise
                from scoper.benchmark.common.transport_retry import retry_delay
                delay = retry_delay(exc, attempt, self.retry.backoff_base, self.retry.backoff_cap)
                attempts.append(_attempt_record(attempt, attempt_started, status='retryable_error', error=exc, backoff_seconds=delay))
                try:
                    self._transport_retry_budget.allow_wait(delay, exc)
                except Exception:
                    self._record_elapsed(started, attempts)
                    raise
                time.sleep(delay)
        self._record_elapsed(started, attempts)
        assert last_exc is not None
        raise last_exc

    def supports_images(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def _call_once(self, messages: Sequence[Message]) -> str:
        raise NotImplementedError

    def _record_elapsed(self, started: float, attempts: Sequence[dict[str, Any]]) -> None:
        if self.last_call_metadata is None:
            self.last_call_metadata = CallMetadata(elapsed_seconds=time.time() - started, attempts=tuple(attempts))
        else:
            self.last_call_metadata.elapsed_seconds = time.time() - started
            self.last_call_metadata.attempts = tuple(attempts)

    def _is_retryable_exception(self, exc: BaseException) -> bool:
        from scoper.benchmark.common.transport_retry import retryable_exception
        return retryable_exception(exc)

def require_dependency(module_name: str, install_hint: str) -> None:
    raise NonRetryableVLMError("Missing optional dependency '{}'. Install with {}.".format(module_name, install_hint))

def _attempt_record(attempt_index: int, started: float, *, status: str, error: BaseException | None, backoff_seconds: float) -> dict[str, Any]:
    status_code = getattr(error, 'status_code', None) or getattr(error, 'code', None)
    if isinstance(status_code, bool) or not isinstance(status_code, (int, str)):
        status_code = None
    return {'attempt_index': attempt_index + 1, 'started_at_utc': datetime.fromtimestamp(started, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'latency_seconds': max(0.0, time.time() - started), 'status': status, 'error_type': type(error).__name__ if error is not None else None, 'status_code': status_code, 'message': _redact_error_message(str(error)) if error is not None else None, 'backoff_seconds': float(backoff_seconds)}

def _redact_error_message(message: str) -> str:
    lowered = message.lower()
    for marker in ('api_key=', 'api-key=', 'authorization:', 'bearer ', 'token='):
        start = lowered.find(marker)
        if start >= 0:
            end = message.find(' ', start + len(marker))
            if end < 0:
                end = len(message)
            message = message[:start + len(marker)] + '[REDACTED]' + message[end:]
            lowered = message.lower()
    return message
__all__ = ['BaseAdapter', 'CallMetadata', 'NonRetryableVLMError', 'RetryableVLMError', 'VLMAdapter', 'VLMAdapterError']
