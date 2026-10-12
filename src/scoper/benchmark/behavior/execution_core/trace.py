from __future__ import annotations
from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
import time
from typing import Any

@dataclass
class CheckResult:
    allowed: bool
    reasons: list[str]
    metrics: dict[str, Any]

    @classmethod
    def pass_(cls, **metrics: Any) -> 'CheckResult':
        return cls(allowed=True, reasons=[], metrics=metrics)

    @classmethod
    def fail(cls, reasons: list[str] | str, **metrics: Any) -> 'CheckResult':
        if isinstance(reasons, str):
            reasons = [reasons]
        return cls(allowed=False, reasons=reasons, metrics=metrics)

@dataclass
class ActionResult:
    action: str
    args: dict[str, Any]
    success: bool
    precondition: CheckResult
    postcondition: CheckResult | None
    trace_id: str
    frames: int = 0
    start_pose: Any | None = None
    end_pose: Any | None = None
    state_delta: dict[str, Any] | None = None

@dataclass
class TraceEvent:
    trace_id: str
    event: str
    timestamp: float
    payload: dict[str, Any]

class TraceLogger:

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, trace_id: str, event: str, **payload: Any) -> None:
        self.events.append(TraceEvent(trace_id=trace_id, event=event, timestamp=time.time(), payload=payload))

    def as_list(self) -> list[dict[str, Any]]:
        return [to_jsonable(event) for event in self.events]

    def write_json(self, path: str | Path) -> None:
        write_json(path, self.as_list())

def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, 'tolist'):
        return to_jsonable(value.tolist())
    if hasattr(value, 'item'):
        try:
            return value.item()
        except ValueError:
            pass
    return value

def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(to_jsonable(value), f, indent=2, sort_keys=True)
