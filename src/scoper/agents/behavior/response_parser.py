from __future__ import annotations
import json
import re
from typing import Any, Iterable, Mapping, Optional
from scoper.benchmark.behavior.brains.base import ToolCall
MODEL_VISIBLE_TOOLS = {'navigate_to', 'navigate_to_room', 'scan', 'look_for', 'pick_up', 'put_down', 'drop', 'open', 'close', 'toggle_on', 'toggle_off', 'answer', 'stop', 'wipe', 'cut', 'soak_under', 'fill', 'pour', 'heat', 'cool'}

class ResponseParser:

    def __init__(self, allowed_tools: Optional[Iterable[str]]=None) -> None:
        self.allowed_tools = set(allowed_tools or MODEL_VISIBLE_TOOLS)
        self.last_parse_error: str | None = None

    def set_allowed_tools(self, allowed_tools: Optional[Iterable[str]]) -> None:
        if allowed_tools:
            self.allowed_tools = set(allowed_tools)

    def parse(self, raw_response: Any) -> ToolCall:
        self.last_parse_error = None
        text = self._response_to_text(raw_response)
        candidates = self._json_candidates(text)
        best_error = 'no JSON object found'
        selected = {}
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError as exc:
                best_error = str(exc)
                continue
            from scoper.benchmark.common.direct_response import collect_tool_calls
            calls = collect_tool_calls(data, self._tool_call_from_mapping)
            if calls:
                for tool_call in calls:
                    selected[tool_call.tool_name, json.dumps(tool_call.arguments, sort_keys=True)] = tool_call
                continue
            best_error = 'JSON object did not contain a valid tool call'
        if len(selected) == 1:
            return next(iter(selected.values()))
        if selected:
            best_error = 'multiple conflicting final tool calls'
        self.last_parse_error = best_error
        return ToolCall('stop', {}, reasoning='parse_error: {}; raw={}'.format(best_error, text[:500]))

    def _tool_call_from_mapping(self, data: Any) -> Optional[ToolCall]:
        if not isinstance(data, Mapping):
            return None
        if 'tool_use' in data and isinstance(data['tool_use'], Mapping):
            data = data['tool_use']
        if data.get('type') == 'tool_use':
            name = data.get('name') or data.get('tool_name')
            args = data.get('input') or data.get('arguments') or {}
        else:
            name = data.get('tool_name') or data.get('name') or data.get('tool')
            args = data.get('arguments') or data.get('args') or {}
        if not isinstance(name, str) or name not in self.allowed_tools:
            return None
        if not isinstance(args, Mapping):
            return None
        reasoning = data.get('reasoning') or data.get('rationale') or ''
        return ToolCall(name, dict(args), str(reasoning))

    def _find_tool_call(self, data: Any) -> Optional[ToolCall]:
        direct = self._tool_call_from_mapping(data)
        if direct is not None:
            return direct
        if isinstance(data, Mapping):
            for value in data.values():
                found = self._find_tool_call(value)
                if found is not None:
                    return found
        elif isinstance(data, list):
            for value in data:
                found = self._find_tool_call(value)
                if found is not None:
                    return found
        return None

    def _json_candidates(self, text: str) -> list[str]:
        from scoper.benchmark.common.direct_response import final_answer_text
        text = final_answer_text(text)
        candidates: list[str] = []
        for match in re.finditer('```(?:json)?\\s*(.*?)```', text, re.DOTALL | re.IGNORECASE):
            candidates.extend(_balanced_json_objects(match.group(1)))
        candidates.extend(_balanced_json_objects(_strip_think_blocks(text)))
        return sorted(dict.fromkeys(candidates), key=len, reverse=True)

    def _response_to_text(self, raw_response: Any) -> str:
        if isinstance(raw_response, str):
            return raw_response
        return json.dumps(raw_response, ensure_ascii=False, default=str)

def _strip_think_blocks(text: str) -> str:
    return re.sub('<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

def _balanced_json_objects(text: str) -> list[str]:
    out: list[str] = []
    starts: list[int] = []
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == '{':
            starts.append(index)
        elif char == '}' and starts:
            start = starts.pop()
            if not starts:
                out.append(text[start:index + 1])
    return out
__all__ = ['MODEL_VISIBLE_TOOLS', 'ResponseParser']
