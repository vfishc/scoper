from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence
from scoper.benchmark.common import QAOverlayRuntime, compute_qa_clause_incomplete_state
from scoper.benchmark.behavior.scheduler.base import Scheduler, SchedulerEvent

@dataclass
class HumanEventConfig:
    id: str
    type: str
    activation_type: str
    activation_after_step: int
    activation_milestone: Optional[str] = None
    activation_role: Optional[str] = None
    activation_tool: Optional[str] = None
    activation_subtask_id: Optional[str] = None
    question: Optional[str] = None
    question_type: Optional[str] = None
    answer_set: List[str] = field(default_factory=list)
    evaluator_answer: Optional[str] = None
    evaluator_answer_predicate: Optional[dict] = None
    requires_visible_roles: List[str] = field(default_factory=list)
    instruction: Optional[str] = None
    priority: Optional[str] = None
    goal_predicates: List[dict] = field(default_factory=list)

class HumanInteractionScheduler(Scheduler):
    name = 'human_interaction'

    def __init__(self, events: Sequence[Mapping[str, Any]]=(), *, qa_overlay: Mapping[str, Any] | None=None) -> None:
        self.events: List[HumanEventConfig] = [self._coerce_event(event) for event in events]
        self._overlay_runtime = QAOverlayRuntime(qa_overlay) if qa_overlay else None
        self._fired: set[str] = set()
        self._task_spec: Optional[Mapping[str, Any]] = None
        self._last_active_subtask_id: str | None = None

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]):
        self._task_spec = task_spec
        self._fired = set()
        self._last_active_subtask_id = _active_subtask_id(env)
        qa_state = _ensure_human_qa_state(env, reset=True)
        for event in self.events:
            if event.type == 'qa' and event.evaluator_answer is not None:
                qa_state.setdefault('evaluator_keys', {})[event.id] = str(event.evaluator_answer)
        if self._overlay_runtime is not None:
            self._overlay_runtime.reset(qa_state)
        _sync_public_human_state(env)
        return [SchedulerEvent('human_armed', 'episode_start', {'num_events': len(self.events) + len(self._overlay_runtime.overlay.get('questions') or []) if self._overlay_runtime is not None else len(self.events)})]

    def pre_step(self, env: Any, high_level_step: int, last_transition: Optional[Mapping[str, Any]]):
        events: List[SchedulerEvent] = []
        posted_insert_ids: list[str] = []
        task_action_step = int(getattr(env, 'task_action_step', high_level_step - 1)) + 1
        active_subtask_id = _active_subtask_id(env)
        from scoper.benchmark.common.composite_evaluation import enabled
        if active_subtask_id != self._last_active_subtask_id and (active_subtask_id is not None or enabled(self._task_spec or {})):
            previous_subtask_id = self._last_active_subtask_id
            events.extend(self._overlay_events(env, self._consider_overlay(env, trigger='subtask_handoff', total_step=high_level_step, task_action_step=task_action_step, handoff_subtask_id=active_subtask_id, active_subtask_id_override=previous_subtask_id), phase='pre_step'))
        self._last_active_subtask_id = active_subtask_id
        for event in self.events:
            if event.id in self._fired:
                continue
            if event.activation_type != 'high_level_step':
                continue
            if task_action_step < int(event.activation_after_step):
                continue
            if not _subtask_gate_matches(env, event):
                continue
            if not _visible_roles_satisfied(env, event.requires_visible_roles):
                continue
            posted = self._post_event(env, event, 'pre_step', int(high_level_step))
            if posted is not None:
                events.append(posted)
                if event.type == 'insert':
                    posted_insert_ids.append(event.id)
        for insert_id in posted_insert_ids:
            events.extend(self._overlay_events(env, self._consider_overlay(env, trigger='after_human_insert', total_step=high_level_step, task_action_step=task_action_step, insert_id=insert_id), phase='pre_step'))
        events.extend(self._overlay_events(env, self._consider_overlay(env, trigger='task_step', total_step=high_level_step, task_action_step=task_action_step), phase='pre_step'))
        return events or None

    def intercept_tool_call(self, env, tool_call, high_level_step, default_call):
        if self._overlay_runtime is None or getattr(tool_call, 'tool_name', None) != 'stop':
            return None
        raw_events = self._consider_overlay(env, trigger='before_stop', total_step=high_level_step, task_action_step=int(getattr(env, 'task_action_step', 0)) + 1)
        if not raw_events:
            return None
        if not any((event.get('name') == 'human_qa_posted' for event in raw_events)):
            transition = dict(default_call())
            transition.setdefault('scheduler_intercept', []).extend(self._intercept_event_records(raw_events))
            return transition
        _sync_public_human_state(env)
        from scoper.benchmark.common.evaluation_lifecycle import enabled
        if enabled(env):
            transition = dict(default_call())
            transition.setdefault('scheduler_intercept', []).extend(self._intercept_event_records(raw_events))
            return transition
        reject = getattr(env, '_qa_overlay_rejected_stop_transition', None)
        if not callable(reject):
            raise RuntimeError('QA overlay requires a stop-rejection transition adapter')
        return reject(raw_events)

    @staticmethod
    def _intercept_event_records(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{'name': str(event.get('name') or 'qa_overlay_event'), 'phase': 'intercept', 'payload': dict(event.get('payload') or {})} for event in events]

    def post_step(self, env: Any, high_level_step: int, transition: Mapping[str, Any]):
        events: List[SchedulerEvent] = []
        posted_insert_ids: list[str] = []
        for event in self.events:
            if event.id in self._fired:
                continue
            if event.activation_type != 'milestone':
                continue
            if not _subtask_gate_matches(env, event):
                continue
            if not self._milestone_matches(env, event, transition):
                continue
            qa_state = _ensure_human_qa_state(env)
            qa_state.setdefault('checkpoint_steps', {})[event.id] = int(high_level_step)
            if not _visible_roles_satisfied(env, event.requires_visible_roles):
                continue
            posted = self._post_event(env, event, 'post_step', int(high_level_step))
            if posted is not None:
                events.append(posted)
                if event.type == 'insert':
                    posted_insert_ids.append(event.id)
        for insert_id in posted_insert_ids:
            events.extend(self._overlay_events(env, self._consider_overlay(env, trigger='after_human_insert', total_step=high_level_step, task_action_step=int(getattr(env, 'task_action_step', high_level_step)), insert_id=insert_id, transition=transition), phase='post_step'))
        events.extend(self._overlay_events(env, self._consider_overlay(env, trigger='after_tool_result', total_step=high_level_step, task_action_step=int(getattr(env, 'task_action_step', high_level_step)), transition=transition), phase='post_step'))
        return events or None

    def on_episode_end(self, env: Any, final_goal: Mapping[str, Any]):
        qa_state = _ensure_human_qa_state(env)
        pending = sorted((str(key) for key in (qa_state.get('pending') or {}).keys()))
        if not pending:
            return None
        qa_state['unanswered'] = pending
        return [SchedulerEvent('human_qa_unanswered', 'episode_end', {'ids': pending})]

    @staticmethod
    def _coerce_event(event: Mapping[str, Any]) -> HumanEventConfig:
        activation = event.get('activation') if isinstance(event.get('activation'), Mapping) else {}
        question = event.get('question')
        if question is None:
            question = event.get('question_template')
        instruction = event.get('instruction')
        if instruction is None:
            instruction = event.get('instruction_template')
        return HumanEventConfig(id=str(event['id']), type=str(event['type']), activation_type=str(activation.get('type', 'high_level_step')), activation_after_step=int(activation.get('after_step', 0) or 0), activation_milestone=str(activation['milestone']) if activation.get('milestone') is not None else None, activation_role=str(activation['role']) if activation.get('role') is not None else None, activation_tool=str(activation['tool']) if activation.get('tool') is not None else None, activation_subtask_id=str(activation.get('subtask_id') or activation.get('owner_subtask_id')) if activation.get('subtask_id') is not None or activation.get('owner_subtask_id') is not None else None, question=str(question) if question is not None else None, question_type=event.get('question_type'), answer_set=[str(item) for item in event.get('answer_set') or []], evaluator_answer=str(event.get('evaluator_answer')) if event.get('evaluator_answer') is not None else None, evaluator_answer_predicate=dict(event['evaluator_answer_predicate']) if isinstance(event.get('evaluator_answer_predicate'), Mapping) else None, requires_visible_roles=[str(role) for role in event.get('requires_visible_roles') or []], instruction=str(instruction) if instruction is not None else None, priority=str(event.get('priority')) if event.get('priority') is not None else None, goal_predicates=[dict(predicate) for predicate in event.get('goal_predicates') or [] if isinstance(predicate, Mapping)])

    def _post_event(self, env: Any, event: HumanEventConfig, phase: str, high_level_step: int) -> Optional[SchedulerEvent]:
        qa_state = _ensure_human_qa_state(env)
        if event.type == 'qa' and qa_state.get('pending'):
            return None
        self._fired.add(event.id)
        qa_state.setdefault('posted_steps', {})[event.id] = int(high_level_step)
        activation_payload = _activation_payload(event)
        if event.type == 'qa':
            question = _format_human_text(env, event.question or '')
            answer = self._runtime_answer(env, event)
            if answer is not None:
                qa_state.setdefault('evaluator_keys', {})[event.id] = str(answer)
            pending = {'type': 'qa', 'id': event.id, 'question': question, 'question_type': event.question_type, 'answer_set': list(event.answer_set), 'activation': dict(activation_payload), 'requires_visible_roles': list(event.requires_visible_roles), 'answered': False}
            qa_state.setdefault('pending', {})[event.id] = pending
            _sync_public_human_state(env)
            return SchedulerEvent('human_qa_posted', phase, {'id': event.id, 'after_step': int(high_level_step), 'owner_subtask_id': event.activation_subtask_id, 'question': question, 'question_type': event.question_type, 'answer_set': list(event.answer_set), 'evaluator_answer': qa_state.get('evaluator_keys', {}).get(event.id), 'activation': dict(activation_payload), 'requires_visible_roles': list(event.requires_visible_roles)})
        if event.type == 'insert':
            instruction = _format_human_text(env, event.instruction or '')
            insert = {'type': 'insert', 'id': event.id, 'instruction': instruction, 'priority': event.priority, 'goal_predicates': [dict(predicate) for predicate in event.goal_predicates], 'activation': dict(activation_payload)}
            qa_state.setdefault('inserts', {})[event.id] = insert
            _sync_public_human_state(env)
            return SchedulerEvent('human_insert_posted', phase, {'id': event.id, 'after_step': int(high_level_step), 'owner_subtask_id': event.activation_subtask_id, 'instruction': instruction, 'priority': event.priority, 'goal_predicates': [dict(predicate) for predicate in event.goal_predicates], 'activation': dict(activation_payload)})
        return None

    def _consider_overlay(self, env: Any, **kwargs: Any) -> list[dict[str, Any]]:
        if self._overlay_runtime is None:
            return []
        active_override = kwargs.pop('active_subtask_id_override', None)
        return self._overlay_runtime.consider(_ensure_human_qa_state(env), active_subtask_id=active_override if isinstance(active_override, str) else _active_subtask_id(env), resolve_state=lambda question: self._overlay_resolver_state(env, question, kwargs.get('transition')), target_role_matches=lambda role, transition: _transition_targets_role(env, role, transition), **kwargs)

    @staticmethod
    def _overlay_events(env: Any, events: Sequence[Mapping[str, Any]], *, phase: str) -> list[SchedulerEvent]:
        if not events:
            return []
        _sync_public_human_state(env)
        return [SchedulerEvent(str(event.get('name')), phase, dict(event.get('payload') or {})) for event in events]

    def _overlay_resolver_state(self, env: Any, question: Mapping[str, Any], transition: Mapping[str, Any] | None) -> dict[str, Any]:
        question_type = question.get('question_type')
        if question_type == 'post_action_outcome_choice':
            raw = transition.get('raw_execution') if isinstance(transition, Mapping) else None
            status = raw.get('status') if isinstance(raw, Mapping) else None
            success = transition.get('success') if isinstance(transition, Mapping) else None
            outcome = 'ambiguous' if not isinstance(success, bool) or status in {'ambiguous', 'uncertain', 'verify_required'} else 'success' if isinstance(transition, Mapping) and transition.get('success') is True else 'failure'
            return {'last_tool_outcome': outcome}
        if question_type in {'remaining_requirement_choice', 'target_state_choice', 'target_relation_choice', 'mid_execution_progress_choice'}:
            goal = env.check_goal(model_stopped=False)
            owner = question.get('owner_subtask_id')
            task = getattr(env, 'current_task_spec', None) or self._task_spec or {}
            goal_specs = list((task.get('goal') or {}).get('predicates') or [])
            return {'clause_incomplete': compute_qa_clause_incomplete_state(goal_specs, list(goal.get('predicates') or []), owner_subtask_id=owner if isinstance(owner, str) else None)}
        if question_type == 'active_issue_choice':
            return {'active_issue_ids': self._active_issue_ids(env, question)}
        if question_type == 'composite_progress_choice':
            active = _active_subtask_id(env)
            return {'active_subtask_id': active, 'composite_complete': active is None}
        state = _ensure_human_qa_state(env)
        return {'posted_insert_ids': list((state.get('inserts') or {}).keys())}

    def _active_issue_ids(self, env: Any, question: Mapping[str, Any]) -> list[str]:
        configured = list((question.get('resolver') or {}).get('issue_ids') or [])
        goal = env.check_goal(model_stopped=False)
        predicates = list(goal.get('predicates') or [])
        task = getattr(env, 'current_task_spec', None) or self._task_spec or {}
        family = task.get('family_spec') if isinstance(task, Mapping) else {}
        issues = []
        tidy = family.get('tidy') if isinstance(family, Mapping) and isinstance(family.get('tidy'), Mapping) else {}
        issues.extend(tidy.get('violations') or [])
        recover = family.get('recover') if isinstance(family, Mapping) and isinstance(family.get('recover'), Mapping) else {}
        if isinstance(recover.get('incident'), Mapping):
            issues.append(recover['incident'])
        issues.extend(recover.get('incidents') or [])
        goal_specs = list((task.get('goal') or {}).get('predicates') or []) if isinstance(task, Mapping) else []
        active: list[str] = []
        for issue in issues:
            if not isinstance(issue, Mapping):
                continue
            issue_id = str(issue.get('id') or issue.get('violation_id') or '')
            if issue_id not in configured:
                continue
            role = issue.get('object_role') or issue.get('target_role')
            matched = [index for index, predicate in enumerate(goal_specs) if isinstance(predicate, Mapping) and (role in predicate.values() or (isinstance(predicate.get('args'), Mapping) and role in predicate['args'].values()))]
            if matched and any((index < len(predicates) and (not bool(predicates[index].get('satisfied'))) for index in matched)):
                active.append(issue_id)
        return active

    def _runtime_answer(self, env: Any, event: HumanEventConfig) -> Optional[str]:
        if event.question_type == 'scene_yes_no':
            if event.evaluator_answer_predicate:
                return self._predicate_yes_no(env, event.evaluator_answer_predicate)
            return event.evaluator_answer
        if event.question_type == 'progress_status':
            object_role = self._primary_role('object_role')
            object_name = _object_name_by_role(env, object_role)
            core = getattr(env, 'core', None)
            held_name = getattr(getattr(core, 'ego_state', None), 'held_object_name', None)
            return 'yes' if object_name and object_name == held_name else 'no'
        if event.question_type == 'post_place_next_action':
            receptacle_role = event.activation_role or self._primary_role('receptacle_role')
            obj = _object_by_role(env, receptacle_role)
            states = getattr(obj, 'states', {}) or {}
            abilities = set(getattr(obj, 'abilities', ()) or ())
            openable = 'openable' in abilities or 'open' in states
            if openable and bool(states.get('open', False)):
                return 'close the destination'
            return 'no destination action needed'
        return event.evaluator_answer

    def _predicate_yes_no(self, env: Any, predicate: Mapping[str, Any]) -> str:
        expected = bool(predicate.get('expected', True))
        predicate_type = str(predicate.get('type') or '')
        args = predicate.get('args') if isinstance(predicate.get('args'), Mapping) else {}
        object_role = predicate.get('object_role') or predicate.get('object') or args.get('object') or args.get('object_role')
        obj = _object_by_role(env, str(object_role) if object_role is not None else None)
        states = getattr(obj, 'states', {}) or {}
        actual = False
        if predicate_type == 'object_open':
            actual = bool(states.get('open', False))
        elif predicate_type == 'object_closed':
            actual = not bool(states.get('open', False))
        elif predicate_type == 'object_toggled_on':
            actual = bool(states.get('toggled_on', False))
        elif predicate_type == 'object_toggled_off':
            actual = not bool(states.get('toggled_on', False))
        return 'yes' if actual == expected else 'no'

    def _primary_role(self, field_name: str) -> Optional[str]:
        task_spec = self._task_spec or {}
        human = (task_spec.get('family_spec') or {}).get('human') if isinstance(task_spec.get('family_spec'), Mapping) else {}
        primary = human.get('primary_task') if isinstance(human, Mapping) else {}
        if isinstance(primary, Mapping) and primary.get(field_name) is not None:
            return str(primary.get(field_name))
        return None

    def _milestone_matches(self, env: Any, event: HumanEventConfig, transition: Mapping[str, Any]) -> bool:
        if not bool(transition.get('success', False)):
            return False
        if event.activation_milestone == 'after_successful_navigate_to_role':
            if transition.get('tool_name') != 'navigate_to':
                return False
            return _transition_targets_role(env, event.activation_role, transition)
        if event.activation_milestone == 'after_successful_tool_on_role':
            if event.activation_tool and transition.get('tool_name') != event.activation_tool:
                return False
            raw_execution = transition.get('raw_execution')
            if isinstance(raw_execution, Mapping) and raw_execution.get('status') == 'invalid':
                return False
            return _transition_targets_role(env, event.activation_role, transition)
        return False

def _ensure_human_qa_state(env: Any, *, reset: bool=False) -> dict:
    if reset or not isinstance(getattr(env, '_human_qa_state', None), dict):
        env._human_qa_state = {}
    state = env._human_qa_state
    for key in ('pending', 'answers', 'inserts', 'evaluator_keys', 'posted_steps', 'checkpoint_steps'):
        value = state.get(key)
        if reset or not isinstance(value, dict):
            state[key] = {}
    if reset or not isinstance(state.get('unanswered'), list):
        state['unanswered'] = []
    return state

def _sync_public_human_state(env: Any) -> None:
    state = _ensure_human_qa_state(env)
    posted_steps = state.get('posted_steps') or {}
    messages = []
    for collection_name in ('pending', 'inserts'):
        collection = state.get(collection_name) or {}
        for item_id, item in collection.items():
            entry = dict(item) if isinstance(item, Mapping) else {'id': item_id}
            entry.setdefault('id', item_id)
            entry.setdefault('type', 'qa' if collection_name == 'pending' else 'insert')
            messages.append((int(posted_steps.get(item_id, 10 ** 9)), str(item_id), entry))
    messages.sort(key=lambda row: (row[0], row[1]))
    env.human_messages = [entry for _, _, entry in messages]
    answers = []
    for question_id, answer in (state.get('answers') or {}).items():
        entry = dict(answer) if isinstance(answer, Mapping) else {'value': answer}
        entry.setdefault('question_id', question_id)
        answers.append(entry)
    env.human_answers = answers

def _activation_payload(event: HumanEventConfig) -> dict:
    payload = {'type': event.activation_type, 'after_step': event.activation_after_step, 'milestone': event.activation_milestone, 'role': event.activation_role, 'tool': event.activation_tool, 'subtask_id': event.activation_subtask_id, 'owner_subtask_id': event.activation_subtask_id}
    return {key: value for key, value in payload.items() if value is not None}

def _subtask_gate_matches(env: Any, event: HumanEventConfig) -> bool:
    expected = event.activation_subtask_id
    if expected is None:
        return True
    current = getattr(env, 'current_composite_subtask_id', None)
    return str(current) == str(expected)

def _active_subtask_id(env: Any) -> str | None:
    current = getattr(env, 'current_composite_subtask_id', None)
    return str(current) if current is not None else None

def _format_human_text(env: Any, text: str) -> str:
    formatter = getattr(env, '_format_human_template', None)
    if callable(formatter):
        try:
            return str(formatter(text))
        except Exception:
            return str(text)
    return str(text)

def _visible_roles_satisfied(env: Any, role_names: Sequence[str]) -> bool:
    if not role_names:
        return True
    checker = getattr(env, '_visible_roles_satisfied', None)
    if callable(checker):
        try:
            return bool(checker(role_names))
        except Exception:
            return False
    visible = set()
    visible_fn = getattr(env, '_visible_object_names_at_pose', None)
    core = getattr(env, 'core', None)
    pose = getattr(getattr(core, 'ego_state', None), 'pose', None)
    if callable(visible_fn) and pose is not None:
        try:
            visible = set((str(name) for name in visible_fn(pose)))
        except Exception:
            visible = set()
    for role_name in role_names:
        object_name = _object_name_by_role(env, role_name)
        if object_name not in visible:
            return False
    return True

def _transition_targets_role(env: Any, role_name: Optional[str], transition: Mapping[str, Any]) -> bool:
    if not role_name:
        return False
    expected_name = _object_name_by_role(env, role_name)
    from scoper.benchmark.common.qa_target_matching import transition_targets_role
    return transition_targets_role(role_name, expected_name, transition)

def _object_name_by_role(env: Any, role_name: Optional[str]) -> Optional[str]:
    if not role_name:
        return None
    role_map = getattr(env, 'role_map', {}) or {}
    return role_map.get(str(role_name), str(role_name))

def _object_by_role(env: Any, role_name: Optional[str]) -> Any:
    object_name = _object_name_by_role(env, role_name)
    core = getattr(env, 'core', None)
    objects = getattr(core, 'objects', {}) or {}
    if object_name in objects:
        return objects[object_name]
    return None
__all__ = ['HumanEventConfig', 'HumanInteractionScheduler']
