from __future__ import annotations
import copy
from contextlib import contextmanager
from typing import Any, Iterator, Mapping
from scoper.benchmark.behavior.brains.base import BaseBrain, BrainError, BrainInput, ToolCall
from scoper.benchmark.behavior.brains.teachers.common import goal_cleanup_plan, stop

class CompositeTeacher(BaseBrain):

    def __init__(self) -> None:
        self._env: Any = None
        self._parent_task_spec: Mapping[str, Any] = {}
        self._subtasks: list[Mapping[str, Any]] = []
        self._index = 0
        self._teacher: Any = None
        self._child_spec: dict[str, Any] | None = None
        self._cleanup_queue: list[ToolCall] = []

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        composite = (task_spec.get('family_spec') or {}).get('composite')
        if not isinstance(composite, Mapping):
            raise BrainError('CompositeTeacher requires family_spec.composite.')
        if composite.get('ordering') != 'sequential':
            raise BrainError('Only sequential composite ordering is supported.')
        subtasks = [item for item in composite.get('subtasks') or [] if isinstance(item, Mapping)]
        if len(subtasks) < 2:
            raise BrainError('CompositeTeacher requires at least two subtasks.')
        self._env = env
        self._parent_task_spec = task_spec
        self._subtasks = subtasks
        self._index = 0
        self._cleanup_queue = []
        self._child_spec = None
        self._teacher = self._instantiate_child()

    def act(self, brain_input: BrainInput) -> ToolCall:
        while self._teacher is not None:
            if self._child_spec is None:
                raise BrainError('Composite child teacher is missing its task spec.')
            with _temporarily_bind_task(self._env, self._child_spec):
                call = self._teacher.act(brain_input)
            if call.tool_name != 'stop':
                return call
            with _temporarily_bind_task(self._env, self._child_spec):
                goal = self._env.check_goal(model_stopped=False)
            if float(goal.get('predicate_success', 0.0)) < 1.0:
                raise BrainError("Composite subtask '{}' stopped before its goal was satisfied.".format(self._child_spec['id']))
            self._mark_subtask_complete()
            self._index += 1
            if self._index >= len(self._subtasks):
                self._teacher = None
                self._cleanup_queue = goal_cleanup_plan(self._env, self._parent_task_spec)
                break
            self._teacher = self._instantiate_child()
        if self._cleanup_queue:
            return self._cleanup_queue.pop(0)
        return stop('composite plan complete')

    def _instantiate_child(self) -> Any:
        child_spec = _build_child_task_spec(self._parent_task_spec, self._subtasks[self._index])
        self._child_spec = child_spec
        self._set_current_subtask(self._subtasks[self._index])
        from scoper.benchmark.behavior.brains.teachers.dispatcher import default_dispatcher
        teacher = default_dispatcher().build(child_spec)
        with _temporarily_bind_task(self._env, child_spec):
            teacher.on_episode_start(self._env, child_spec)
        return teacher

    def _set_current_subtask(self, subtask: Mapping[str, Any]) -> None:
        if self._env is None:
            return
        subtask_id = str(subtask.get('subtask_id') or 'subtask_{}'.format(self._index))
        category = str(subtask.get('category') or subtask.get('family') or '')
        setattr(self._env, 'current_composite_subtask_id', subtask_id)
        setattr(self._env, 'current_composite_subtask_index', int(self._index))
        setattr(self._env, 'current_composite_subtask_category', category)
        events = list(getattr(self._env, 'composite_subtask_events', []) or [])
        events.append({'event': 'subtask_start', 'subtask_id': subtask_id, 'subtask_index': int(self._index), 'category': category, 'high_level_step': int(getattr(self._env, 'high_level_step', 0) or 0)})
        setattr(self._env, 'composite_subtask_events', events)

    def _mark_subtask_complete(self) -> None:
        if self._env is None or self._index >= len(self._subtasks):
            return
        subtask = self._subtasks[self._index]
        subtask_id = str(subtask.get('subtask_id') or 'subtask_{}'.format(self._index))
        events = list(getattr(self._env, 'composite_subtask_events', []) or [])
        events.append({'event': 'subtask_complete', 'subtask_id': subtask_id, 'subtask_index': int(self._index), 'category': str(subtask.get('category') or subtask.get('family') or ''), 'high_level_step': int(getattr(self._env, 'high_level_step', 0) or 0)})
        setattr(self._env, 'composite_subtask_events', events)

def _build_child_task_spec(parent: Mapping[str, Any], subtask: Mapping[str, Any]) -> dict[str, Any]:
    category = subtask.get('category') or subtask.get('family')
    if not isinstance(category, str) or not category:
        raise BrainError('Composite subtask is missing category.')
    spec = copy.deepcopy(dict(parent))
    spec['id'] = '{}::{}'.format(parent.get('id', 'task'), subtask.get('subtask_id') or category)
    spec['category'] = category
    family_spec = subtask.get('family_spec')
    if isinstance(family_spec, Mapping):
        spec['family_spec'] = copy.deepcopy(dict(family_spec))
    else:
        spec.pop('family_spec', None)
    spec['instruction'] = str(subtask.get('instruction_fragment') or parent.get('instruction') or '')
    spec['goal'] = {'type': 'predicates', 'predicates': copy.deepcopy(list(subtask.get('goal_predicates') or [])), 'require_model_stop': False, 'bddl_goal': None}
    if isinstance(subtask.get('failure_spec'), list):
        spec['failure_spec'] = copy.deepcopy(list(subtask.get('failure_spec') or []))
    if isinstance(subtask.get('disturbance_spec'), list):
        spec['disturbance_spec'] = copy.deepcopy(list(subtask.get('disturbance_spec') or []))
    return spec

@contextmanager
def _temporarily_bind_task(env: Any, task_spec: Mapping[str, Any]) -> Iterator[None]:
    had_attr = hasattr(env, 'current_task_spec')
    previous = getattr(env, 'current_task_spec', None)
    if had_attr:
        env.current_task_spec = task_spec
    try:
        yield
    finally:
        if had_attr:
            env.current_task_spec = previous
__all__ = ['CompositeTeacher']
