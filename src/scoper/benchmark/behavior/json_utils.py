from __future__ import annotations
from typing import Any

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items() if key != 'head_rgb'}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value
_json_safe = json_safe
__all__ = ['json_safe', '_json_safe']
