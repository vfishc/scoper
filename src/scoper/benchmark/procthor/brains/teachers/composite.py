from __future__ import annotations
import copy
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional
from scoper.benchmark.procthor.brains.base import BaseBrain, BrainError, BrainInput, ToolCall
from scoper.benchmark.procthor.brains.teachers.dispatcher import TeacherDispatcherError
from scoper.benchmark.procthor.online_task_spec import COMPOSITE_SUBTASK_CATEGORIES, FAMILY_CATEGORIES

class CompositeTeacher(BaseBrain):

    def __init__(self) -> None:
        self._env: Any = None
        self._parent_task_spec: Optional[Mapping[str, Any]] = None
        self._subtasks: list[Mapping[str, Any]] = []
        self._subtask_specs: list[dict] = []
        self._current_index = 0
        self._current_teacher: Any = None

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        family_spec = task_spec.get('family_spec')
        if not isinstance(family_spec, Mapping):
            raise BrainError('Composite task is missing family_spec.')
        composite = family_spec.get('composite')
        if not isinstance(composite, Mapping):
            raise BrainError('Composite task is missing family_spec.composite.')
        if composite.get('ordering') != 'sequential':
            raise BrainError('CompositeTeacher only supports sequential ordering.')
        subtasks = list(composite.get('subtasks') or [])
        if len(subtasks) < 2:
            raise BrainError('CompositeTeacher requires at least 2 subtasks, got {}.'.format(len(subtasks)))
        self._env = env
        self._parent_task_spec = task_spec
        self._subtasks = subtasks
        self._subtask_specs = [build_composite_subtask_spec(task_spec, subtask, index) for index, subtask in enumerate(subtasks)]
        self._current_index = 0
        _set_active_subtask(self._env, self._subtask_specs[0])
        self._current_teacher = self._instantiate_current_teacher()

    def act(self, brain_input: BrainInput) -> ToolCall:
        while self._current_teacher is not None:
            subtask_spec = self._current_subtask_spec()
            with _temporarily_bind_task(self._env, subtask_spec):
                call = self._current_teacher.act(brain_input)
            if not isinstance(call, ToolCall):
                raise BrainError('Composite child teacher returned {}, expected ToolCall.'.format(type(call).__name__))
            if call.tool_name != 'stop':
                return _translate_child_tool_call(call, subtask_spec)
            goal = _check_subtask_goal(self._env, subtask_spec)
            if float(goal.get('predicate_success', 0.0)) < 1.0:
                raise BrainError("Composite subtask '{}' stopped before its goal was satisfied.".format(subtask_spec['id']))
            self._current_index += 1
            if self._current_index >= len(self._subtask_specs):
                self._current_teacher = None
                _clear_active_subtask(self._env)
                break
            _set_active_subtask(self._env, self._current_subtask_spec())
            self._current_teacher = self._instantiate_current_teacher()
        return ToolCall('stop', {}, reasoning='composite plan complete')

    def on_episode_end(self, env: Any, final_goal: Mapping[str, Any]) -> None:
        _clear_active_subtask(env)
        if self._current_teacher is not None and hasattr(self._current_teacher, 'on_episode_end'):
            try:
                self._current_teacher.on_episode_end(env, final_goal)
            except Exception:
                pass

    def _current_subtask_spec(self) -> dict:
        if self._current_index >= len(self._subtask_specs):
            raise BrainError('CompositeTeacher current subtask index is out of range.')
        return self._subtask_specs[self._current_index]

    def _instantiate_current_teacher(self) -> Any:
        if self._env is None:
            raise BrainError('CompositeTeacher used before on_episode_start.')
        subtask_spec = self._current_subtask_spec()
        from scoper.benchmark.procthor.brains.teachers import get_dispatcher
        try:
            teacher = get_dispatcher().build(subtask_spec)
        except TeacherDispatcherError as exc:
            raise BrainError("No executable teacher for composite subtask '{}' category '{}': {}".format(subtask_spec['id'], subtask_spec.get('category'), exc)) from exc
        with _temporarily_bind_task(self._env, subtask_spec):
            teacher.on_episode_start(self._env, subtask_spec)
        return teacher

def build_composite_subtask_spec(parent_task_spec: Mapping[str, Any], subtask: Mapping[str, Any], index: int) -> dict:
    if not isinstance(subtask, Mapping):
        raise BrainError('Composite subtask {} must be a mapping.'.format(index))
    category = subtask.get('category') or subtask.get('family')
    if not isinstance(category, str) or not category:
        raise BrainError('Composite subtask {} is missing category/family.'.format(index))
    if category == 'composite':
        raise BrainError('Recursive composite subtasks are not supported.')
    if category not in COMPOSITE_SUBTASK_CATEGORIES:
        raise BrainError("Composite subtask category '{}' is not supported by the sequential teacher.".format(category))
    family_spec = subtask.get('family_spec')
    if category in FAMILY_CATEGORIES:
        if not isinstance(family_spec, Mapping):
            raise BrainError("Composite family subtask '{}' is missing family_spec.".format(category))
        if family_spec.get('family') != category:
            raise BrainError("Composite subtask '{}' family_spec.family '{}' must match category.".format(category, family_spec.get('family')))
    goal_predicates = subtask.get('goal_predicates')
    if not isinstance(goal_predicates, list) or not goal_predicates:
        raise BrainError("Composite subtask '{}' must define non-empty goal_predicates.".format(category))
    subtask_id = subtask.get('subtask_id') or '{}_{}'.format(category, index)
    task = copy.deepcopy(dict(parent_task_spec))
    task['id'] = '{}::{}'.format(parent_task_spec.get('id', 'task'), subtask_id)
    task['subtask_id'] = str(subtask_id)
    task['category'] = category
    role_map = _subtask_role_map(parent_task_spec, index)
    if role_map:
        task['_composite_role_map'] = role_map
        roles = copy.deepcopy(dict(task.get('roles') or {}))
        for local_role, parent_role in role_map.items():
            if parent_role in roles:
                roles.setdefault(local_role, copy.deepcopy(roles[parent_role]))
        task['roles'] = roles
    if isinstance(family_spec, Mapping):
        task['family_spec'] = copy.deepcopy(dict(family_spec))
    else:
        task.pop('family_spec', None)
    task['instruction'] = str(subtask.get('instruction_fragment') or parent_task_spec.get('instruction') or '')
    task['goal'] = {'type': 'predicate', 'predicates': copy.deepcopy(goal_predicates), 'require_model_stop': False}
    disturbance_spec = subtask.get('disturbance_spec')
    if isinstance(disturbance_spec, Mapping):
        task['disturbance_spec'] = copy.deepcopy(dict(disturbance_spec))
    return task

def _check_subtask_goal(env: Any, subtask_spec: Mapping[str, Any]) -> dict:
    checker = getattr(env, 'check_goal', None)
    if not callable(checker):
        raise BrainError('Composite environment does not expose check_goal.')
    with _temporarily_bind_task(env, subtask_spec):
        return dict(checker(model_stopped=False))

def _set_active_subtask(env: Any, subtask_spec: Mapping[str, Any]) -> None:
    if env is not None:
        setattr(env, '_composite_active_task_spec', subtask_spec)

def _clear_active_subtask(env: Any) -> None:
    if env is not None and hasattr(env, '_composite_active_task_spec'):
        setattr(env, '_composite_active_task_spec', None)

def _subtask_role_map(parent_task_spec: Mapping[str, Any], index: int) -> dict[str, str]:
    generation = parent_task_spec.get('metadata', {}).get('composite_generation')
    maps = generation.get('role_namespace') if isinstance(generation, Mapping) else None
    if not isinstance(maps, list) or index >= len(maps) or (not isinstance(maps[index], Mapping)):
        return {}
    return {str(local_role): str(parent_role) for local_role, parent_role in maps[index].items() if isinstance(local_role, str) and isinstance(parent_role, str)}

def _translate_child_tool_call(call: ToolCall, subtask_spec: Mapping[str, Any]) -> ToolCall:
    role_map = subtask_spec.get('_composite_role_map')
    if not isinstance(role_map, Mapping):
        return call
    arguments = copy.deepcopy(dict(call.arguments or {}))
    target = arguments.get('target')
    if isinstance(target, str):
        if target in role_map:
            arguments['target'] = str(role_map[target])
        elif target.startswith('__sliced_role__:'):
            prefix, _, local_role = target.partition(':')
            if local_role in role_map:
                arguments['target'] = '{}:{}'.format(prefix, role_map[local_role])
    return ToolCall(call.tool_name, arguments, reasoning=call.reasoning)

@contextmanager
def _temporarily_bind_task(env: Any, task_spec: Mapping[str, Any]) -> Iterator[None]:
    previous = getattr(env, 'current_task_spec', None)
    had_attr = hasattr(env, 'current_task_spec')
    if had_attr:
        env.current_task_spec = task_spec
    role_object_ids = getattr(env, 'role_object_ids', None)
    previous_aliases: dict[str, tuple[bool, Any]] = {}
    role_map = task_spec.get('_composite_role_map')
    if isinstance(role_object_ids, dict) and isinstance(role_map, Mapping):
        for local_role, parent_role in role_map.items():
            if parent_role not in role_object_ids:
                continue
            previous_aliases[str(local_role)] = (str(local_role) in role_object_ids, role_object_ids.get(str(local_role)))
            role_object_ids[str(local_role)] = role_object_ids[parent_role]
    try:
        yield
    finally:
        for local_role, (existed, value) in previous_aliases.items():
            if existed:
                role_object_ids[local_role] = value
            else:
                role_object_ids.pop(local_role, None)
        if had_attr:
            env.current_task_spec = previous
from scoper.benchmark.procthor.brains.teachers import get_dispatcher
get_dispatcher().register('composite', CompositeTeacher)
__all__ = ['CompositeTeacher', 'build_composite_subtask_spec']
