from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import random
import time
import json
import os
from pathlib import Path

def episode_started(task_id):
    path = os.environ.get('EB_EVALUATION_HEARTBEAT_FILE')
    if path:
        p = Path(path)
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps({'task_id': task_id, 'started_monotonic': time.monotonic()}))
        tmp.replace(p)

def episode_deadline_exceeded(path, timeout, process_started):
    started = process_started
    if path and Path(path).exists():
        try:
            started = float(json.loads(Path(path).read_text())['started_monotonic'])
        except (ValueError, KeyError, OSError):
            pass
    return time.monotonic() - started >= timeout

class TransportRetryExhausted(RuntimeError):
    pass

class RetryBudget:

    def __init__(self, seconds=None):
        self.deadline = None if seconds is None else time.monotonic() + float(seconds)

    def remaining(self):
        remaining = float('inf') if self.deadline is None else self.deadline - time.monotonic()
        if remaining <= 0:
            raise TransportRetryExhausted('API retry time budget exhausted')
        return remaining

    def allow_wait(self, delay, cause):
        if delay + 1 >= self.remaining():
            raise TransportRetryExhausted('API retry delay exceeds remaining time budget') from cause

def retryable_exception(exc):
    status = getattr(exc, 'status_code', None)
    try:
        status = int(status)
    except (ValueError, TypeError):
        status = None
    if status in {400, 401, 403, 404, 405, 422}:
        return False
    body = getattr(exc, 'body', {}) or {}
    if isinstance(body, dict):
        error = body.get('error', body)
        code = error.get('code') if isinstance(error, dict) else None
        if code in {'insufficient_quota', 'billing_hard_limit_reached', 'account_deactivated'}:
            return False
    if status in {408, 409, 429} or (status is not None and 500 <= status < 600):
        return True
    return bool({cls.__name__ for cls in type(exc).__mro__} & {'RetryableVLMError', 'RateLimitError', 'APITimeoutError', 'APIConnectionError', 'Timeout', 'TimeoutError', 'ConnectTimeout', 'ReadTimeout', 'ConnectionError', 'ConnectError', 'ReadError', 'RemoteProtocolError'})

def retry_delay(exc, attempt, base, cap):
    delay = min(float(base) ** attempt, float(cap))
    headers = getattr(getattr(exc, 'response', None), 'headers', {}) or {}
    hinted = 0.0
    try:
        if headers.get('retry-after-ms'):
            hinted = float(headers['retry-after-ms']) / 1000
        elif headers.get('retry-after'):
            raw = headers['retry-after']
            try:
                hinted = float(raw)
            except ValueError:
                hinted = (parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds()
    except (ValueError, TypeError, OverflowError):
        hinted = 0.0
    return min(300.0, max(delay, hinted, 0.0)) + random.uniform(0, 0.5)

def is_transport_failure(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        record = getattr(exc, 'evaluation_failure_record', None)
        if isinstance(record, dict) and (record.get('failure') or {}).get('code') == 'model_response_parsing_failed':
            return False
        if type(exc).__name__ == 'DirectResponseParseError':
            return False
        if type(exc).__name__ in {'TransportRetryExhausted', 'VLMBrainInfrastructureError', 'RateLimitError', 'APITimeoutError', 'APIConnectionError', 'InternalServerError'}:
            return True
        exc = getattr(exc, '__cause__', None) or getattr(exc, '__context__', None)
    return False

def transport_error_info(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if getattr(exc, 'status_code', None) is not None:
            return {'type': type(exc).__name__, 'status_code': getattr(exc, 'status_code'), 'retryable': retryable_exception(exc)}
        exc = getattr(exc, '__cause__', None) or getattr(exc, '__context__', None)
    return None

def retryable_record(record):
    if (record.get('failure') or {}).get('code') == 'model_response_parsing_failed':
        return False
    info = (record.get('backend_metrics') or {}).get('transport_failure') or {}
    if info.get('retryable') is False:
        return False
    reason = str((record.get('failure') or {}).get('reason', ''))
    if any((code in reason for code in ('AuthenticationError', 'PermissionDeniedError', 'NotFoundError', 'BadRequestError', 'insufficient_quota', 'billing_hard_limit_reached'))):
        return False
    if record.get('status') == 'infrastructure_failed':
        return True
    failure = record.get('failure') or {}
    return failure.get('code') == 'evaluation_episode_exception'
