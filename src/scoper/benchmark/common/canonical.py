from __future__ import annotations
import hashlib
import json
import math
from typing import Any

def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode('utf-8')

def canonical_json_line(value: Any) -> bytes:
    return canonical_json_bytes(value) + b'\n'

def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def validate_json_value(value: Any, *, json_pointer: str='') -> None:
    pointer = json_pointer or '/'
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'non-finite number at {pointer}')
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_value(item, json_pointer=f"{pointer.rstrip('/')}/{index}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f'non-string mapping key at {pointer}')
            escaped = key.replace('~', '~0').replace('/', '~1')
            validate_json_value(item, json_pointer=f"{pointer.rstrip('/')}/{escaped}")
        return
    raise ValueError(f'non-JSON value at {pointer}: {type(value).__name__}')

def strict_json_loads(value: str | bytes, *, source: str='<memory>') -> Any:

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise ValueError(f'duplicate JSON key {key!r} in {source}')
            output[key] = item
        return output
    try:
        result = json.loads(value, object_pairs_hook=pairs_hook)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f'invalid JSON in {source}: {exc}') from exc
    validate_json_value(result)
    return result
__all__ = ['canonical_json_bytes', 'canonical_json_line', 'sha256_bytes', 'sha256_json', 'strict_json_loads', 'validate_json_value']
