from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional
from scoper.benchmark.behavior.brains.base import ToolCall
from scoper.benchmark.behavior.json_utils import _json_safe
from scoper.benchmark.behavior.hooks.base import Hook, HookEvent

class ExecutionTimelineHook(Hook):
    name = 'execution_timeline'

    def __init__(self, *, output_dir: Path, episode_name: str, write_markdown: bool=True, write_json: bool=True) -> None:
        output = Path(output_dir)
        self.timeline_path = output / 'timelines' / '{}.md'.format(episode_name)
        self.json_path = output / 'timelines' / '{}.json'.format(episode_name)
        self.write_markdown = bool(write_markdown)
        self.write_json = bool(write_json)
        self._task_spec: Optional[Mapping[str, Any]] = None
        self._reset_observation: Mapping[str, Any] = {}
        self._init_mutations: List[dict] = []
        self._entries: List[dict] = []
        self._started_at: Optional[float] = None
        self._active_step: Optional[dict] = None
        self._final_goal: Mapping[str, Any] = {}
        self._status: str = 'unknown'
        self._closed = False

    @property
    def manifest(self) -> Optional[dict]:
        if not self._closed:
            return None
        return {'timeline': str(self.timeline_path) if self.write_markdown else None, 'timeline_json': str(self.json_path) if self.write_json else None, 'entry_count': len(self._entries)}

    @property
    def entries(self) -> List[dict]:
        return [dict(entry) for entry in self._entries]

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any], reset_observation: Mapping[str, Any]) -> Optional[List[HookEvent]]:
        self._task_spec = task_spec
        self._reset_observation = dict(reset_observation)
        self._started_at = time.time()
        self._entries = []
        self._closed = False
        init_mutations = list(getattr(env, 'init_mutations', None) or reset_observation.get('init_mutations') or [])
        self._init_mutations = [_json_safe(dict(m)) for m in init_mutations]
        self._add_entry(kind='episode_start', start_seconds=0.0, end_seconds=0.0, title='Episode reset', detail='Task initialized for execution.', payload={'init_mutations': _json_safe(init_mutations)})
        if init_mutations:
            self._add_entry(kind='init_mutation', start_seconds=0.0, end_seconds=0.0, title='Scene initialization adapted for this task', detail=_summarize_init_mutations(init_mutations), payload={'init_mutations': _json_safe(init_mutations)})
        return [HookEvent('timeline_started', 'episode_start', {'timeline': str(self.timeline_path)})]

    def on_scheduler_events(self, env: Any, events: Iterable[Mapping[str, Any]], phase: str, high_level_step: int) -> Optional[List[HookEvent]]:
        emitted = []
        for event in events or []:
            text = _scheduler_event_text(event)
            if not text:
                continue
            bounds = _pop_video_bounds(env, phase='scheduler', high_level_step=high_level_step, event_name=str(event.get('name') or ''))
            now = self._elapsed()
            start = float(bounds.get('start_seconds', now)) if bounds is not None else now
            end = float(bounds.get('end_seconds', start)) if bounds is not None else start
            self._add_entry(kind='scheduler_event', start_seconds=start, end_seconds=end, title=text, detail="Scheduler event '{}' during {}.".format(event.get('name'), phase), high_level_step=high_level_step, payload={'event': _json_safe(dict(event))})
            emitted.append(HookEvent('timeline_scheduler_event', 'scheduler', {'text': text, 'phase': phase, 'high_level_step': high_level_step}))
        return emitted or None

    def before_tool(self, env: Any, tool_call: ToolCall, high_level_step: int) -> Optional[List[HookEvent]]:
        bounds = _pop_video_bounds(env, phase='before_tool', high_level_step=high_level_step, tool_name=tool_call.tool_name)
        start = float(bounds.get('start_seconds', self._elapsed())) if bounds is not None else self._elapsed()
        self._active_step = {'start_seconds': start, 'action_subtitle_end_seconds': float(bounds.get('end_seconds', start)) if bounds is not None else start, 'high_level_step': high_level_step, 'tool_name': tool_call.tool_name, 'arguments': dict(tool_call.arguments), 'reasoning': tool_call.reasoning}
        title = 'Execute {}{}'.format(tool_call.tool_name, _format_arguments(tool_call.arguments))
        if tool_call.tool_name == 'answer':
            title = 'Answer QA {}'.format(tool_call.arguments.get('question_id', 'unknown'))
        return [HookEvent('timeline_before_tool', 'before_tool', {'text': title, 'high_level_step': high_level_step})]

    def after_tool(self, env: Any, tool_call: ToolCall, transition: Mapping[str, Any], high_level_step: int) -> Optional[List[HookEvent]]:
        bounds = _pop_video_bounds(env, phase='after_tool', high_level_step=high_level_step, tool_name=tool_call.tool_name)
        end = float(bounds.get('end_seconds', self._elapsed())) if bounds is not None else self._elapsed()
        active = self._active_step or {}
        start = float(active.get('start_seconds', end))
        title = 'Execute {}{}'.format(tool_call.tool_name, _format_arguments(tool_call.arguments))
        detail = _transition_detail(tool_call, transition)
        if tool_call.tool_name == 'answer':
            title = _answer_title(tool_call, transition)
        self._add_entry(kind='tool_call', start_seconds=start, end_seconds=end, title=title, detail=detail, high_level_step=high_level_step, payload={'tool_call': {'tool_name': tool_call.tool_name, 'arguments': dict(tool_call.arguments), 'reasoning': tool_call.reasoning}, 'transition': _json_safe(dict(transition))})
        self._active_step = None
        return [HookEvent('timeline_after_tool', 'after_tool', {'text': title, 'high_level_step': high_level_step})]

    def on_episode_end(self, env: Any, final_goal: Mapping[str, Any], status: str) -> Optional[List[HookEvent]]:
        self._final_goal = dict(final_goal)
        self._status = status
        bounds = _pop_video_bounds(env, phase='episode_end')
        now = self._elapsed()
        start = float(bounds.get('start_seconds', now)) if bounds is not None else now
        end = float(bounds.get('end_seconds', start)) if bounds is not None else start
        self._add_entry(kind='episode_end', start_seconds=start, end_seconds=end, title='Episode {}'.format(status), detail='Final task_success={}'.format(final_goal.get('task_success')), payload={'final_goal': _json_safe(dict(final_goal))})
        self._write()
        self._closed = True
        return [HookEvent('timeline_finalized', 'episode_end', dict(self.manifest or {}))]

    def _elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, time.time() - self._started_at)

    def _add_entry(self, *, kind: str, start_seconds: float, end_seconds: float, title: str, detail: str, high_level_step: Optional[int]=None, payload: Optional[Mapping[str, Any]]=None) -> None:
        self._entries.append({'kind': kind, 'start_seconds': round(float(start_seconds), 3), 'end_seconds': round(float(end_seconds), 3), 'high_level_step': high_level_step, 'title': str(title), 'detail': str(detail), 'payload': _json_safe(dict(payload or {}))})

    def _write(self) -> None:
        self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
        if self.write_json:
            self.json_path.write_text(json.dumps(self._manifest_record(), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        if self.write_markdown:
            self.timeline_path.write_text(self._markdown(), encoding='utf-8')

    def _manifest_record(self) -> dict:
        task = self._task_spec or {}
        return {'task': _task_overview(task), 'context': _task_context_summary(task), 'init_mutations': _json_safe(self._init_mutations), 'status': self._status, 'final_goal': _json_safe(dict(self._final_goal)), 'entries': list(self._entries)}

    def _markdown(self) -> str:
        task = self._task_spec or {}
        context = _task_context_summary(task)
        lines = ['# Execution Timeline', '', 'Task ID: {}'.format(task.get('id')), '', 'Instruction: {}'.format(task.get('instruction')), '', 'Category: {}'.format(task.get('category')), '', 'Status: {}'.format(self._status), '', '## What The Agent Must Consider']
        for item in context:
            lines.append('- {}'.format(item))
        if not context:
            lines.append('- No additional task constraints were declared.')
        lines.extend(['', '## Scene Initialization'])
        init_mutations = self._init_mutations
        if init_mutations:
            for mutation in init_mutations:
                lines.append('- {}'.format(_mutation_text(mutation)))
        else:
            lines.append('- No scene adaptation was recorded at reset.')
        lines.extend(['', '## Timeline'])
        for entry in self._entries:
            lines.append('- {:.1f}s-{:.1f}s {}'.format(float(entry['start_seconds']), float(entry['end_seconds']), _entry_line(entry)))
        lines.append('')
        return '\n'.join(lines)

def _task_overview(task: Mapping[str, Any]) -> dict:
    return {'task_id': task.get('id'), 'category': task.get('category'), 'house_id': (task.get('source') or {}).get('house_id') if isinstance(task.get('source'), Mapping) else None, 'instruction': task.get('instruction')}

def _task_context_summary(task: Mapping[str, Any]) -> List[str]:
    items: List[str] = []
    goal = task.get('goal') if isinstance(task.get('goal'), Mapping) else {}
    predicates = list(goal.get('predicates') or [])
    if predicates:
        items.append('Goal predicates: {}'.format('; '.join((_predicate_text(p) for p in predicates[:8]))))
    if goal.get('require_model_stop'):
        items.append('The agent must explicitly call stop after satisfying the goal.')
    req = task.get('online_requirements')
    if isinstance(req, Mapping):
        expected_events = list(req.get('expected_events') or [])
        behaviors = list(req.get('expected_brain_behaviors') or [])
        if expected_events:
            items.append('Expected events: {}'.format('; '.join(map(str, expected_events))))
        if behaviors:
            items.append('Expected brain behaviors: {}'.format('; '.join(map(str, behaviors))))
    family = task.get('family_spec')
    if isinstance(family, Mapping):
        if isinstance(family.get('human'), Mapping):
            events = list(family['human'].get('human_events') or [])
            if events:
                items.append('Human interaction events: {}'.format('; '.join((_human_event_text(e) for e in events))))
        if isinstance(family.get('mid_execution'), Mapping):
            items.append('Mid-execution disturbance task is enabled.')
        if isinstance(family.get('recover'), Mapping):
            items.append('Recovery incidents must be handled before final placement.')
        if isinstance(family.get('prepare'), Mapping):
            stages = list(family['prepare'].get('stages') or [])
            if stages:
                items.append('Preparation stages: {}'.format('; '.join((str(s.get('type')) for s in stages if isinstance(s, Mapping)))))
    disturbance = task.get('disturbance_spec')
    if isinstance(disturbance, Mapping):
        disturbances = list(disturbance.get('disturbances') or [])
        failures = list(disturbance.get('failures') or [])
        if disturbances:
            items.append('Runtime disturbances may occur: {}'.format('; '.join((str(d.get('type')) for d in disturbances if isinstance(d, Mapping)))))
        if failures:
            items.append('Injected failure modes may occur: {}'.format('; '.join((str(f.get('mode')) for f in failures if isinstance(f, Mapping)))))
    return items

def _summarize_init_mutations(mutations: Iterable[Mapping[str, Any]]) -> str:
    return '; '.join((_mutation_text(m) for m in list(mutations)[:6]))

def _mutation_text(mutation: Mapping[str, Any]) -> str:
    typ = mutation.get('type') or 'mutation'
    role = mutation.get('object_role') or mutation.get('target_role') or 'object'
    method = mutation.get('method') or mutation.get('mode') or 'applied'
    if typ == 'relocate_object':
        return 'relocated {} to {} using {}'.format(role, mutation.get('to_receptacle_role') or mutation.get('to_receptacle_object_id'), method)
    return '{} on {} via {}'.format(typ, role, method)

def _scheduler_event_text(event: Mapping[str, Any]) -> str:
    name = str(event.get('name') or '')
    payload = event.get('payload') if isinstance(event.get('payload'), Mapping) else {}
    if name == 'human_qa_posted':
        question = payload.get('question')
        if question:
            return 'Human QA posted: {} - {}'.format(payload.get('id'), question)
        return 'Human QA posted: {}'.format(payload.get('id'))
    if name == 'human_insert_posted':
        instruction = payload.get('instruction')
        if instruction:
            return 'Human inserted an extra instruction: {} - {}'.format(payload.get('id'), instruction)
        return 'Human inserted an extra instruction: {}'.format(payload.get('id'))
    if name.startswith('failure_') and name != 'failure_armed':
        return 'Injected failure event: {} on {}'.format(name, payload.get('tool') or payload.get('config_tool') or 'tool')
    if name in {'close_object', 'open_object', 'toggle_on', 'toggle_off', 'dirty_object', 'relocate_object', 'restore_initial_state'}:
        text = 'Scene disturbance: {} on {}'.format(name, payload.get('target_role') or payload.get('object_id') or 'object')
        activation = payload.get('activation')
        if isinstance(activation, Mapping) and activation.get('type') == 'after_tool_result':
            armed_by = activation.get('armed_by') if isinstance(activation.get('armed_by'), Mapping) else {}
            if armed_by:
                text += ' after {}({})'.format(armed_by.get('tool'), armed_by.get('target'))
        return text
    return ''

def _transition_detail(tool_call: ToolCall, transition: Mapping[str, Any]) -> str:
    status = 'success' if transition.get('success') else 'failed'
    feedback = str(transition.get('env_feedback') or '').strip()
    reasoning = str(tool_call.reasoning or '').strip()
    chunks = ['{}.'.format(status)]
    if feedback:
        chunks.append(feedback)
    if reasoning:
        chunks.append('Reasoning: {}'.format(reasoning))
    return ' '.join(chunks)

def _answer_title(tool_call: ToolCall, transition: Mapping[str, Any]) -> str:
    question_id = tool_call.arguments.get('question_id', 'unknown')
    value = tool_call.arguments.get('value')
    raw = transition.get('raw_execution') if isinstance(transition.get('raw_execution'), Mapping) else {}
    question_value = raw.get('question')
    question = None
    if isinstance(question_value, Mapping):
        question = question_value.get('question')
    elif question_value is not None:
        question = question_value
    if question is not None:
        question = str(question)
    if transition.get('success'):
        if question:
            return 'Answer QA {}: {} -> {}'.format(question_id, question, value)
        return 'Answer QA {} with {}'.format(question_id, value)
    return 'Attempt QA answer {}'.format(question_id)

def _format_arguments(arguments: Mapping[str, Any]) -> str:
    if not arguments:
        return '()'
    parts = ['{}={}'.format(k, v) for k, v in sorted(arguments.items())]
    text = ', '.join(parts)
    if len(text) > 80:
        text = text[:77] + '...'
    return '({})'.format(text)

def _predicate_text(predicate: Mapping[str, Any]) -> str:
    if not isinstance(predicate, Mapping):
        return str(predicate)
    typ = predicate.get('type') or 'predicate'
    obj = predicate.get('object_role')
    rec = predicate.get('receptacle_role')
    if obj and rec:
        return '{} {} -> {}'.format(typ, obj, rec)
    if obj:
        return '{} {}'.format(typ, obj)
    return str(typ)

def _human_event_text(event: Mapping[str, Any]) -> str:
    if not isinstance(event, Mapping):
        return str(event)
    if event.get('type') == 'qa':
        return 'QA {}: {}'.format(event.get('id'), event.get('question'))
    if event.get('type') == 'insert':
        return 'insert {}: {}'.format(event.get('id'), event.get('instruction'))
    return '{} {}'.format(event.get('type'), event.get('id'))

def _entry_line(entry: Mapping[str, Any]) -> str:
    title = str(entry.get('title') or '')
    detail = str(entry.get('detail') or '')
    if detail and detail != title:
        return '{} - {}'.format(title, detail)
    return title

def _pop_video_bounds(env: Any, *, phase: str, high_level_step: Optional[int]=None, tool_name: Optional[str]=None, event_name: Optional[str]=None) -> Optional[dict]:
    queue = getattr(env, '_artifact_video_event_bounds', None)
    if not isinstance(queue, list):
        return None
    for index, item in enumerate(queue):
        if not isinstance(item, Mapping):
            continue
        if item.get('phase') != phase:
            continue
        if high_level_step is not None and item.get('high_level_step') != high_level_step:
            continue
        if tool_name is not None and item.get('tool_name') != tool_name:
            continue
        if event_name is not None and item.get('event_name') != event_name:
            continue
        return dict(queue.pop(index))
    return None
__all__ = ['ExecutionTimelineHook']
