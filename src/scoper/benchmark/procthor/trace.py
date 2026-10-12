from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping, Optional
from scoper.benchmark.procthor.closed_loop_tool_env import _json_safe
RECORD_TYPE_META = 'meta'
RECORD_TYPE_STEP = 'step'
RECORD_TYPE_SUMMARY = 'summary'

class EpisodeTraceWriter:

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open('w', encoding='utf-8')
        self._closed = False
        self._step_count = 0

    def write_meta(self, record: Mapping[str, Any]) -> None:
        self._write({**record, 'record_type': RECORD_TYPE_META})

    def write_step(self, record: Mapping[str, Any]) -> None:
        self._step_count += 1
        self._write({**record, 'record_type': RECORD_TYPE_STEP})

    def write_summary(self, record: Mapping[str, Any]) -> None:
        self._write({**record, 'record_type': RECORD_TYPE_SUMMARY})

    def close(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True

    @property
    def step_count(self) -> int:
        return self._step_count

    def __enter__(self) -> 'EpisodeTraceWriter':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _write(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError('EpisodeTraceWriter is already closed.')
        line = json.dumps(_json_safe(dict(record)), ensure_ascii=False, sort_keys=True)
        self._fh.write(line)
        self._fh.write('\n')
        self._fh.flush()

def build_meta_record(task_spec: Mapping[str, Any], *, runner_config: Mapping[str, Any], reset: Mapping[str, Any], tool_specs: Optional[Any]=None) -> dict:
    return {'task_id': task_spec['id'], 'category': task_spec['category'], 'house_id': task_spec['source']['house_id'], 'instruction': task_spec['instruction'], 'runner_config': dict(runner_config), 'reset': dict(reset), 'tool_specs': tool_specs}

def build_step_record(*, high_level_step: int, brain_decision: Mapping[str, Any], transition: Mapping[str, Any], hook_events: Optional[list]=None, scheduler_events: Optional[list]=None) -> dict:
    return {'high_level_step': high_level_step, 'brain': dict(brain_decision), 'transition': dict(transition), 'hook_events': list(hook_events or []), 'scheduler_events': list(scheduler_events or [])}

def build_summary_record(*, status: str, final_goal: Mapping[str, Any], metrics: Mapping[str, Any], failure_type: Optional[str]=None, failure_reason: Optional[str]=None, failure_code: Optional[str]=None, failure_details: Optional[Mapping[str, Any]]=None, init_mutations: Optional[list]=None, scene_mutations: Optional[list]=None, hook_events: Optional[list]=None, elapsed_seconds: Optional[float]=None) -> dict:
    return {'status': status, 'final_goal': dict(final_goal), 'metrics': dict(metrics), 'failure_type': failure_type, 'failure_reason': failure_reason, 'failure_code': failure_code, 'failure_details': dict(failure_details or {}), 'init_mutations': list(init_mutations or []), 'scene_mutations': list(scene_mutations or []), 'hook_events': list(hook_events or []), 'elapsed_seconds': elapsed_seconds}

def read_trace(path: Path) -> list:
    records = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
__all__ = ['EpisodeTraceWriter', 'RECORD_TYPE_META', 'RECORD_TYPE_STEP', 'RECORD_TYPE_SUMMARY', 'build_meta_record', 'build_step_record', 'build_summary_record', 'read_trace']
