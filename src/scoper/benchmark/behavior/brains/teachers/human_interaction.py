from __future__ import annotations
import copy
from typing import Any, Mapping
from scoper.benchmark.behavior.brains.base import BaseBrain, BrainError, BrainInput, ToolCall
from scoper.benchmark.behavior.brains.teachers.common import predicates_to_plan, simple_place_plan, stop

class HumanInteractionTeacher(BaseBrain):

    def __init__(self) -> None:
        self._env: Any = None
        self._task_spec: Mapping[str, Any] = {}
        self._queue: list[ToolCall] = []
        self._answered: set[str] = set()
        self._handled_inserts: set[str] = set()

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        self._env = env
        self._task_spec = task_spec
        self._answered = set()
        self._handled_inserts = set()
        human = (task_spec.get('family_spec') or {}).get('human')
        if not isinstance(human, Mapping):
            raise BrainError('HumanInteractionTeacher requires family_spec.human.')
        primary = human.get('primary_task')
        if not isinstance(primary, Mapping):
            raise BrainError('HumanInteractionTeacher requires human.primary_task.')
        self._queue = _primary_plan(env, task_spec, primary)

    def act(self, brain_input: BrainInput) -> ToolCall:
        for message in (brain_input.prompt_context or {}).get('human_messages') or []:
            if not isinstance(message, Mapping) or message.get('type') != 'qa':
                continue
            question_id = message.get('id')
            if isinstance(question_id, str) and question_id not in self._answered:
                self._answered.add(question_id)
                return ToolCall('answer', {'question_id': question_id, 'value': self._answer_for_question(question_id)}, reasoning='answer pending human QA')
        for message in (brain_input.prompt_context or {}).get('human_messages') or []:
            if not isinstance(message, Mapping) or message.get('type') != 'insert':
                continue
            insert_id = message.get('id')
            if isinstance(insert_id, str) and insert_id not in self._handled_inserts:
                self._handled_inserts.add(insert_id)
                self._queue.extend(predicates_to_plan(self._env, message.get('goal_predicates') or []))
        if self._queue:
            return self._queue.pop(0)
        return stop('human interaction plan complete')

    def _answer_for_question(self, question_id: str) -> str:
        event = _event_by_id(self._task_spec, question_id)
        if not event:
            return 'no'
        if event.get('question_type') == 'post_place_next_action':
            primary = ((self._task_spec.get('family_spec') or {}).get('human') or {}).get('primary_task') or {}
            role = (event.get('activation') or {} if isinstance(event.get('activation'), Mapping) else {}).get('role') or primary.get('receptacle_role')
            if _role_is_currently_open(self._env, str(role)):
                return 'close the destination'
            return 'no destination action needed'
        answer_set = event.get('answer_set')
        if isinstance(answer_set, list) and answer_set:
            return str(answer_set[-1])
        return str(event.get('evaluator_answer') or 'yes')

def _primary_plan(env: Any, task_spec: Mapping[str, Any], primary: Mapping[str, Any]) -> list[ToolCall]:
    ptype = str(primary.get('type') or '')
    if ptype == 'simple_place':
        return simple_place_plan(env, str(primary['object_role']), str(primary['receptacle_role']), close_after=True)
    if ptype == 'put_away_by_category':
        from scoper.benchmark.behavior.brains.teachers.put_away import PutAwayTeacher
        sub_spec = copy.deepcopy(dict(task_spec))
        sub_spec['category'] = 'put_away_by_category'
        sub_spec['family_spec'] = {'family': 'put_away_by_category', 'put_away': copy.deepcopy(primary.get('put_away_ref') or primary.get('put_away') or {}), 'scope': copy.deepcopy((task_spec.get('family_spec') or {}).get('scope') or {})}
        teacher = PutAwayTeacher()
        teacher.on_episode_start(env, sub_spec)
        queue = list(getattr(teacher, '_queue', []))
        if queue and queue[-1].tool_name == 'stop':
            queue.pop()
        return queue
    raise BrainError("Unsupported human primary task type '{}'.".format(ptype))

def _event_by_id(task_spec: Mapping[str, Any], event_id: str) -> Mapping[str, Any] | None:
    human = (task_spec.get('family_spec') or {}).get('human') or {}
    events = human.get('human_events')
    if not isinstance(events, list):
        events = human.get('human_events_template')
    for event in events or []:
        if isinstance(event, Mapping) and event.get('id') == event_id:
            return event
    return None

def _role_is_currently_open(env: Any, role: str) -> bool:
    obj_name = env.role_map.get(role, role)
    obj = env.core.objects.get(obj_name)
    if obj is None:
        return False
    states = getattr(obj, 'states', {}) or {}
    return bool(states.get('open', False))
__all__ = ['HumanInteractionTeacher']
