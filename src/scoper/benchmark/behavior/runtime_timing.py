from __future__ import annotations
import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping
TIMING_SCHEMA_VERSION = 'behavior-runtime-timing.v1'
TIMING_PATH_ENV = 'EB_BEHAVIOR_TIMING_JSONL'
_CURRENT_RECORDER: ContextVar['RuntimeTimingRecorder | None'] = ContextVar('behavior_runtime_timing_recorder', default=None)

@dataclass
class RuntimeTimingRecorder:
    path: Path
    task_id: str
    started_at: float = field(default_factory=time.monotonic)
    counters: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_environment(cls, task_id: str, environment: Mapping[str, str] | None=None) -> 'RuntimeTimingRecorder | None':
        values = os.environ if environment is None else environment
        raw_path = str(values.get(TIMING_PATH_ENV, '') or '').strip()
        if not raw_path:
            return None
        return cls(path=Path(raw_path), task_id=str(task_id))

    def write(self, event: str, **payload: Any) -> None:
        record = {'schema_version': TIMING_SCHEMA_VERSION, 'event': str(event), 'task_id': self.task_id, 'elapsed_seconds': round(time.monotonic() - self.started_at, 6), **payload}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            stream.write('\n')
            stream.flush()

    def count_render(self, purpose: str, *, sim_render_calls: int, rgb: int, segmentation: int) -> None:
        delta = {'sim_render_calls': max(0, int(sim_render_calls)), 'rgb_captures': max(0, int(rgb)), 'segmentation_captures': max(0, int(segmentation))}
        for name, value in delta.items():
            self.counters[name] = self.counters.get(name, 0) + value
        self.write('render', purpose=str(purpose), counts=delta, cumulative_counts=dict(sorted(self.counters.items())))

def current_runtime_timing() -> RuntimeTimingRecorder | None:
    return _CURRENT_RECORDER.get()

def activate_runtime_timing(recorder: RuntimeTimingRecorder) -> Token:
    return _CURRENT_RECORDER.set(recorder)

def deactivate_runtime_timing(token: Token) -> None:
    _CURRENT_RECORDER.reset(token)

@contextmanager
def runtime_phase(phase: str, *, tool_name: str | None=None, status: str='complete') -> Iterator[None]:
    recorder = current_runtime_timing()
    if recorder is None:
        yield
        return
    started = time.monotonic()
    recorder.write('phase_start', phase=str(phase), tool_name_or_null=None if tool_name is None else str(tool_name))
    try:
        yield
    except Exception as exc:
        recorder.write('phase', phase=str(phase), status='error', duration_seconds=round(time.monotonic() - started, 6), tool_name_or_null=None if tool_name is None else str(tool_name), error_type=type(exc).__name__)
        raise
    recorder.write('phase', phase=str(phase), status=str(status), duration_seconds=round(time.monotonic() - started, 6), tool_name_or_null=None if tool_name is None else str(tool_name))

def record_render(purpose: str, *, sim_render_calls: int, rgb: int=0, segmentation: int=0) -> None:
    recorder = current_runtime_timing()
    if recorder is not None:
        recorder.count_render(purpose, sim_render_calls=sim_render_calls, rgb=rgb, segmentation=segmentation)
__all__ = ['TIMING_PATH_ENV', 'TIMING_SCHEMA_VERSION', 'RuntimeTimingRecorder', 'activate_runtime_timing', 'current_runtime_timing', 'deactivate_runtime_timing', 'record_render', 'runtime_phase']
