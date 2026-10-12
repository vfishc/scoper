from __future__ import annotations
import copy
from typing import Any, List, Mapping, Optional
from scoper.benchmark.procthor.brains.base import BaseBrain, BrainError, BrainInput, ToolCall
from scoper.benchmark.procthor.brains.teachers.put_away import PutAwayTeacher

class HumanInteractionTeacher(BaseBrain):

    def __init__(self) -> None:
        self._env: Any = None
        self._task_spec: Optional[Mapping[str, Any]] = None
        self._queue: List[ToolCall] = []
        self._answered_qas: set[str] = set()
        self._handled_inserts: set[str] = set()
        self._stop_emitted = False

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        self._env = env
        self._task_spec = task_spec
        self._answered_qas = set()
        self._handled_inserts = set()
        self._stop_emitted = False
        self._validate_insert_predicates(task_spec)
        primary = task_spec['family_spec']['human']['primary_task']
        primary_type = primary['type']
        if primary_type == 'simple_place':
            self._queue = self._build_simple_place_plan(env, primary)
        elif primary_type == 'put_away_by_category':
            self._queue = self._build_put_away_plan(env, task_spec, primary)
        else:
            raise BrainError("Unsupported human primary task type '{}'.".format(primary_type))

    def act(self, brain_input: BrainInput) -> ToolCall:
        context = brain_input.prompt_context or {}
        messages = list(context.get('human_messages') or [])
        for message in messages:
            if message.get('type') != 'qa':
                continue
            question_id = message.get('id')
            if not isinstance(question_id, str) or question_id in self._answered_qas:
                continue
            self._answered_qas.add(question_id)
            return ToolCall('answer', {'question_id': question_id, 'value': self._evaluate_qa_answer(question_id)}, reasoning='answer human qa {}'.format(question_id))
        for message in messages:
            if message.get('type') != 'insert':
                continue
            insert_id = message.get('id')
            if not isinstance(insert_id, str) or insert_id in self._handled_inserts:
                continue
            self._handled_inserts.add(insert_id)
            self._queue.extend(self._build_insert_plan(insert_id))
        if self._queue:
            return self._queue.pop(0)
        if not self._stop_emitted:
            self._stop_emitted = True
            return ToolCall('stop', {}, reasoning='human interaction plan complete')
        return ToolCall('stop', {}, reasoning='human interaction plan exhausted')

    def _build_simple_place_plan(self, env: Any, primary: Mapping[str, Any]) -> List[ToolCall]:
        object_role = primary['object_role']
        receptacle_role = primary['receptacle_role']
        calls = [_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', receptacle_role)]
        if _role_is_openable(env, receptacle_role) and (not _role_is_open(env, receptacle_role)):
            calls.append(_call('open', receptacle_role))
        calls.append(_call('put_down', receptacle_role))
        if _role_is_openable(env, receptacle_role):
            calls.append(_call('close', receptacle_role))
        return calls

    def _build_put_away_plan(self, env: Any, task_spec: Mapping[str, Any], primary: Mapping[str, Any]) -> List[ToolCall]:
        sub_spec = copy.deepcopy(dict(task_spec))
        sub_spec['category'] = 'put_away_by_category'
        sub_spec['family_spec'] = {'family': 'put_away_by_category', 'scope': copy.deepcopy(task_spec['family_spec']['scope']), 'put_away': copy.deepcopy(primary['put_away_ref'])}
        inner = PutAwayTeacher()
        inner.on_episode_start(env, sub_spec)
        queue = list(getattr(inner, '_queue', []))
        if queue and queue[-1].tool_name == 'stop':
            queue.pop()
        return queue

    def _evaluate_qa_answer(self, question_id: str) -> str:
        event = self._event_by_id(question_id)
        if event is None:
            return 'no'
        question_type = event.get('question_type')
        if question_type == 'scene_yes_no':
            pred = event.get('evaluator_answer_predicate')
            if isinstance(pred, Mapping):
                return _predicate_yes_no(self._env, pred)
            return str(event.get('evaluator_answer', 'no'))
        if question_type == 'progress_status':
            primary = self._task_spec['family_spec']['human']['primary_task']
            object_role = primary.get('object_role')
            object_id = getattr(self._env, 'role_object_ids', {}).get(object_role)
            inventory = _metadata(self._env).get('inventoryObjects') or []
            in_inventory = any((isinstance(item, Mapping) and item.get('objectId') == object_id for item in inventory))
            return 'yes' if in_inventory else 'no'
        if question_type == 'post_place_next_action':
            primary = self._task_spec['family_spec']['human']['primary_task']
            role = _event_activation_role(event) or primary.get('receptacle_role')
            obj = _object_by_role(self._env, role)
            if bool(obj.get('openable', False)) and bool(obj.get('isOpen', False)):
                return 'close the destination'
            return 'no destination action needed'
        return str(event.get('evaluator_answer', 'no'))

    def _build_insert_plan(self, insert_id: str) -> List[ToolCall]:
        event = self._event_by_id(insert_id)
        if event is None or event.get('type') != 'insert':
            return []
        calls: List[ToolCall] = []
        for predicate in event.get('goal_predicates') or []:
            predicate_type = predicate.get('type')
            role = predicate.get('object_role')
            if not isinstance(role, str) or not role:
                continue
            if predicate_type == 'object_toggled_off':
                calls.extend([_call('navigate_to', role), _call('toggle_off', role)])
            elif predicate_type == 'object_toggled_on':
                calls.extend([_call('navigate_to', role), _call('toggle_on', role)])
            elif predicate_type == 'object_closed':
                calls.extend([_call('navigate_to', role), _call('close', role)])
            elif predicate_type == 'object_open':
                calls.extend([_call('navigate_to', role), _call('open', role)])
            elif predicate_type == 'object_clean':
                faucet_role = _find_toggleable_role(self._env, excluded={role})
                sink_role = _find_receptacle_role(self._env, excluded={role, faucet_role})
                if faucet_role is None or sink_role is None:
                    raise BrainError('object_clean insert requires sink/faucet support.')
                calls.extend([_call('navigate_to', role), _call('pick_up', role), _call('navigate_to', sink_role), _call('put_down', sink_role), _call('navigate_to', faucet_role), _call('toggle_on', faucet_role), _call('toggle_off', faucet_role)])
            else:
                raise BrainError("Insert predicate '{}' is not supported in v1.".format(predicate_type))
        return calls

    def _event_by_id(self, event_id: str) -> Optional[Mapping[str, Any]]:
        if self._task_spec is None:
            return None
        events = self._task_spec['family_spec']['human'].get('human_events') or []
        for event in events:
            if isinstance(event, Mapping) and event.get('id') == event_id:
                return event
        return None

    @staticmethod
    def _validate_insert_predicates(task_spec: Mapping[str, Any]) -> None:
        allowed = {'object_toggled_off', 'object_toggled_on', 'object_closed', 'object_open', 'object_clean'}
        for event in task_spec['family_spec']['human'].get('human_events') or []:
            if not isinstance(event, Mapping) or event.get('type') != 'insert':
                continue
            for predicate in event.get('goal_predicates') or []:
                predicate_type = predicate.get('type')
                if predicate_type not in allowed:
                    raise BrainError("Insert predicate '{}' is not supported in v1.".format(predicate_type))

def _call(tool_name: str, target: str) -> ToolCall:
    return ToolCall(tool_name, {'target': target})

def _metadata(env: Any) -> Mapping[str, Any]:
    getter = getattr(env, '_metadata', None)
    if callable(getter):
        return getter()
    controller = getattr(env, 'controller', None)
    event = getattr(controller, 'last_event', None)
    return getattr(event, 'metadata', {}) or {}

def _role_object_id(env: Any, role_name: Optional[str]) -> Optional[str]:
    if not isinstance(role_name, str):
        return None
    return getattr(env, 'role_object_ids', {}).get(role_name)

def _object_by_role(env: Any, role_name: Optional[str]) -> Mapping[str, Any]:
    object_id = _role_object_id(env, role_name)
    if object_id is None:
        return {}
    objects_by_id = getattr(env, '_objects_by_id', None)
    if callable(objects_by_id):
        return objects_by_id().get(object_id, {})
    return {}

def _role_is_openable(env: Any, role_name: str) -> bool:
    return bool(_object_by_role(env, role_name).get('openable', False))

def _role_is_open(env: Any, role_name: str) -> bool:
    return bool(_object_by_role(env, role_name).get('isOpen', False))

def _predicate_yes_no(env: Any, predicate: Mapping[str, Any]) -> str:
    expected = bool(predicate.get('expected', True))
    predicate_type = predicate.get('type')
    obj = _object_by_role(env, predicate.get('object_role'))
    actual = False
    if predicate_type == 'object_open':
        actual = bool(obj.get('isOpen', False))
    elif predicate_type == 'object_closed':
        actual = not bool(obj.get('isOpen', False))
    elif predicate_type == 'object_toggled_on':
        actual = bool(obj.get('isToggled', False))
    elif predicate_type == 'object_toggled_off':
        actual = not bool(obj.get('isToggled', False))
    return 'yes' if actual == expected else 'no'

def _event_activation_role(event: Mapping[str, Any]) -> Optional[str]:
    activation = event.get('activation') or {}
    if isinstance(activation, Mapping):
        role = activation.get('role')
        if isinstance(role, str) and role:
            return role
    return None

def _find_toggleable_role(env: Any, excluded: set) -> Optional[str]:
    for role_name in sorted(getattr(env, 'role_object_ids', {})):
        if role_name in excluded:
            continue
        if bool(_object_by_role(env, role_name).get('toggleable', False)):
            return role_name
    return None

def _find_receptacle_role(env: Any, excluded: set) -> Optional[str]:
    for role_name in sorted(getattr(env, 'role_object_ids', {})):
        if role_name in excluded:
            continue
        if bool(_object_by_role(env, role_name).get('receptacle', False)):
            return role_name
    return None
from scoper.benchmark.procthor.brains.teachers import get_dispatcher
get_dispatcher().register('human_interactive_execution', HumanInteractionTeacher)
__all__ = ['HumanInteractionTeacher']
