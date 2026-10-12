import re

def collect_tool_calls(data, decode):
    from collections.abc import Mapping
    direct = decode(data)
    if direct is not None:
        return [direct]
    values = data.values() if isinstance(data, Mapping) else data if isinstance(data, list) else ()
    return [call for value in values for call in collect_tool_calls(value, decode)]

def final_answer_text(text):
    parts = re.split('</think>', text, flags=re.I)
    if len(parts) > 1:
        return parts[-1]
    if re.search('<think>', text, flags=re.I):
        return ''
    return text

class DirectResponseParseError(RuntimeError):

    def __init__(self, raw_response):
        super().__init__('model response parsing failed')
        self.evaluation_failure_record = {'model_call_count': 1, 'backend_metrics': {'model_response_parsing_failed': True}, 'failure': {'type': type(self).__name__, 'code': 'model_response_parsing_failed', 'reason': str(self), 'raw_response': str(raw_response)}}

def preserve_partial_episode(error, transitions, metrics, final_goal):
    record = getattr(error, 'evaluation_failure_record', None)
    if not isinstance(record, dict):
        return
    record.update(high_level_steps=len(transitions), environment_tool_calls=len(transitions), model_call_count=len(transitions) + int(record.get('model_call_count', 0)), task_progress=float(final_goal.get('task_progress', final_goal.get('predicate_success', 0.0)) or 0.0), transitions=tuple(transitions), binding_metrics=dict(metrics.get('binding_metrics') or {}), navigation_metrics=dict(metrics.get('navigation_metrics') or {}))
    record['backend_metrics'] = {**metrics, **record.get('backend_metrics', {})}

def record_direct_decision(brain, step, raw, call, parse_error):
    if brain.config.eval_mode != 'standard_vlm':
        return
    records = getattr(brain, 'direct_decision_records', None)
    if records is None:
        records = brain.direct_decision_records = []
    if len(records) >= 256:
        return
    text = str(raw)
    records.append({'step': int(step), 'raw_response': text[:16384], 'raw_response_truncated': len(text) > 16384, 'parse_error': str(parse_error)[:512] if parse_error else None, 'tool_name': None if parse_error else call.tool_name, 'arguments': {} if parse_error else dict(call.arguments), 'reasoning': '' if parse_error else str(call.reasoning)[:4096]})
