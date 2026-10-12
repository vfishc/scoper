from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from scoper.benchmark.procthor.online_task_spec import TaskSpecValidationError, validate_online_task_spec, validate_task_bank, validate_task_set

def load_task_bank(path: str, validate: bool=True) -> Dict[str, dict]:
    if Path(path).suffix.lower() == '.json':
        try:
            value = json.loads(_read_text(path))
        except json.JSONDecodeError as exc:
            raise TaskSpecValidationError(f'Invalid JSON task bank: {path}') from exc
        tasks = value.get('tasks') if isinstance(value, dict) else value
        if not isinstance(tasks, list) or any(not isinstance(task, dict) for task in tasks):
            raise TaskSpecValidationError('JSON task bank must be a list or an object containing tasks')
    else:
        tasks = _read_jsonl(path)
    if validate:
        validate_task_bank(tasks)
    return {task['id']: task for task in tasks}

def load_task_set(path: str, task_bank: Optional[Mapping[str, Mapping[str, Any]]]=None, validate: bool=True) -> dict:
    task_set = _read_json(path)
    if validate:
        valid_ids = task_bank.keys() if task_bank is not None else None
        validate_task_set(task_set, task_ids=valid_ids)
    return task_set

def resolve_task_set(task_bank: Mapping[str, Mapping[str, Any]], task_set: Mapping[str, Any], validate: bool=True) -> List[Mapping[str, Any]]:
    if validate:
        validate_task_set(task_set, task_ids=task_bank.keys())
    tasks = []
    for task_id in task_set['task_ids']:
        task = task_bank[task_id]
        if validate:
            validate_online_task_spec(task)
        tasks.append(task)
    return tasks

def load_resolved_task_set(task_bank_path: str, task_set_path: str, validate: bool=True) -> List[Mapping[str, Any]]:
    task_bank = load_task_bank(task_bank_path, validate=validate)
    task_set = load_task_set(task_set_path, task_bank=task_bank, validate=validate)
    return resolve_task_set(task_bank, task_set, validate=validate)

def _read_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskSpecValidationError("Invalid JSON in task bank '{}' at line {}: {}.".format(path, line_number, exc)) from exc
        if not isinstance(record, dict):
            raise TaskSpecValidationError("Task bank '{}' line {} must be a JSON object.".format(path, line_number))
        records.append(record)
    return records

def _read_json(path: str) -> dict:
    try:
        record = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise TaskSpecValidationError("Invalid JSON in task set '{}': {}.".format(path, exc)) from exc
    if not isinstance(record, dict):
        raise TaskSpecValidationError("Task set '{}' must contain one JSON object.".format(path))
    return record

def _read_text(path: str) -> str:
    return Path(path).read_text(encoding='utf-8')
__all__ = ['load_resolved_task_set', 'load_task_bank', 'load_task_set', 'resolve_task_set']
