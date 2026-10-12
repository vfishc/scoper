from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from scoper.benchmark.common import binding_sidecar_manifest, canonical_json_bytes, compile_binding_sidecar

def load_tasks(path: Path) -> list[Mapping[str, Any]]:
    tasks = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f'task bank line {number} is not a JSON object')
        tasks.append(dict(value))
    if not tasks:
        raise ValueError('task bank is empty')
    return tasks

def build_sidecar(task_bank: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = Path(task_bank).read_bytes()
    sidecar = compile_binding_sidecar(load_tasks(task_bank), bank_sha256=hashlib.sha256(raw).hexdigest())
    return (sidecar, binding_sidecar_manifest(sidecar))

def main(argv: Sequence[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args(argv)
    sidecar, manifest = build_sidecar(args.task_bank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(sidecar) + b'\n')
    args.manifest.write_bytes(canonical_json_bytes(manifest) + b'\n')
    print(json.dumps(manifest, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
__all__ = ['build_sidecar', 'load_tasks', 'main']
