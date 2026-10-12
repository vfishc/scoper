from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence
from scoper.benchmark.common import QAOverlayRuntime, compute_qa_clause_incomplete_state
from scoper.benchmark.procthor.scheduler.base import Scheduler, SchedulerEvent

@dataclass
class HumanEventConfig:
    id: str
    type: str
    activation_type: str
    activation_after_step: int
    activation_milestone: Optional[str] = None
    activation_role: Optional[str] = None
    activation_tool: Optional[str] = None
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
        self.events: List[HumanEventConfig] = [self._coerce_event(e) for e in events]
        self._overlay_runtime = QAOverlayRuntime(qa_overlay) if qa_overlay else None
        self._fired: set[str] = set()
        self._notice_callback_installed = False
        self._task_spec: Optional[Mapping[str, Any]] = None
        self._last_active_subtask_id: str | None = None

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]):
        self._task_spec = task_spec
        self._fired = set()
        self._last_active_subtask_id = self._active_subtask_id(env)
        qa_state = env._human_qa_state
        qa_state.setdefault('pending', {})
        qa_state.setdefault('answers', {})
        qa_state.setdefault('inserts', {})
        evaluator_keys = qa_state.setdefault('evaluator_keys', {})
        qa_state.setdefault('posted_steps', {})
        qa_state.setdefault('checkpoint_steps', {})
        qa_state.setdefault('answered_steps', {})
        for event in self.events:
            if event.type == 'qa' and event.evaluator_answer is not None:
                evaluator_keys[event.id] = event.evaluator_answer
        if self._overlay_runtime is not None:
            self._overlay_runtime.reset(qa_state)
        self._install_prompt_notice_hook(env)
        return [SchedulerEvent('human_armed', 'episode_start', {'num_events': len(self.events) + len(self._overlay_runtime.overlay.get('questions') or []) if self._overlay_runtime is not None else len(self.events)})]

    def pre_step(self, env: Any, high_level_step: int, last_transition: Optional[Mapping[str, Any]]):
        events: List[SchedulerEvent] = []
        posted_insert_ids: list[str] = []
        task_action_step = int(getattr(env, 'task_action_step', high_level_step - 1)) + 1
        active_subtask_id = self._active_subtask_id(env)
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
            if task_action_step < event.activation_after_step:
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

    def intercept_tool_call(self, env: Any, tool_call: Any, high_level_step: int, default_call: Any):
        if self._overlay_runtime is None or getattr(tool_call, 'tool_name', None) != 'stop':
            return None
        raw_events = self._consider_overlay(env, trigger='before_stop', total_step=high_level_step, task_action_step=int(getattr(env, 'task_action_step', 0)) + 1)
        if not raw_events:
            return None
        if not any((event.get('name') == 'human_qa_posted' for event in raw_events)):
            transition = dict(default_call())
            transition.setdefault('scheduler_intercept', []).extend(self._intercept_event_records(raw_events))
            return transition
        sync = getattr(env, '_sync_public_human_state', None)
        if callable(sync):
            sync()
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
            if not self._milestone_matches(env, event, transition):
                continue
            env._human_qa_state.setdefault('checkpoint_steps', {})[event.id] = int(high_level_step)
            if not self._visible_roles_satisfied(env, event.requires_visible_roles):
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
        pending = sorted((env._human_qa_state.get('pending') or {}).keys())
        if not pending:
            return None
        env._human_qa_state['unanswered'] = pending
        return [SchedulerEvent('human_qa_unanswered', 'episode_end', {'ids': pending})]

    @staticmethod
    def _coerce_event(event: Mapping[str, Any]) -> HumanEventConfig:
        activation = event.get('activation') or {}
        return HumanEventConfig(id=str(event['id']), type=str(event['type']), activation_type=str(activation.get('type', 'high_level_step')), activation_after_step=int(activation.get('after_step', 0)), activation_milestone=str(activation['milestone']) if activation.get('milestone') is not None else None, activation_role=str(activation['role']) if activation.get('role') is not None else None, activation_tool=str(activation['tool']) if activation.get('tool') is not None else None, question=event.get('question'), question_type=event.get('question_type'), answer_set=list(event.get('answer_set') or []), evaluator_answer=event.get('evaluator_answer'), evaluator_answer_predicate=dict(event['evaluator_answer_predicate']) if isinstance(event.get('evaluator_answer_predicate'), Mapping) else None, requires_visible_roles=list(event.get('requires_visible_roles') or []), instruction=event.get('instruction'), priority=event.get('priority'), goal_predicates=[dict(predicate) for predicate in event.get('goal_predicates') or [] if isinstance(predicate, Mapping)])

    def _runtime_answer(self, env: Any, event: HumanEventConfig) -> Optional[str]:
        if event.question_type == 'scene_yes_no':
            pred = event.evaluator_answer_predicate
            if pred:
                return self._predicate_yes_no(env, pred)
            return event.evaluator_answer
        if event.question_type == 'progress_status':
            object_role = self._primary_object_role()
            object_id = getattr(env, 'role_object_ids', {}).get(object_role)
            inventory = env._metadata().get('inventoryObjects') or []
            in_inventory = any((isinstance(item, Mapping) and item.get('objectId') == object_id for item in inventory))
            return 'yes' if in_inventory else 'no'
        if event.question_type == 'post_place_next_action':
            receptacle_role = event.activation_role or self._primary_receptacle_role()
            obj = self._object_by_role(env, receptacle_role)
            if bool(obj.get('openable', False)) and bool(obj.get('isOpen', False)):
                return 'close the destination'
            return 'no destination action needed'
        return event.evaluator_answer

    def _post_event(self, env: Any, event: HumanEventConfig, phase: str, high_level_step: int) -> Optional[SchedulerEvent]:
        qa_state = env._human_qa_state
        if event.type == 'qa' and qa_state.get('pending'):
            return None
        self._fired.add(event.id)
        qa_state.setdefault('posted_steps', {})[event.id] = int(high_level_step)
        activation_payload = {'type': event.activation_type, 'after_step': event.activation_after_step, 'milestone': event.activation_milestone, 'role': event.activation_role, 'tool': event.activation_tool}
        if event.type == 'qa':
            answer = self._runtime_answer(env, event)
            if answer is not None:
                qa_state.setdefault('evaluator_keys', {})[event.id] = answer
            qa_state.setdefault('pending', {})[event.id] = {'id': event.id, 'question': event.question, 'question_type': event.question_type, 'answer_set': list(event.answer_set), 'activation': {key: value for key, value in activation_payload.items() if value is not None}}
            return SchedulerEvent('human_qa_posted', phase, {'id': event.id, 'after_step': high_level_step, 'question': event.question, 'question_type': event.question_type, 'answer_set': list(event.answer_set), 'evaluator_answer': qa_state.get('evaluator_keys', {}).get(event.id), 'activation': {key: value for key, value in activation_payload.items() if value is not None}})
        if event.type == 'insert':
            qa_state.setdefault('inserts', {})[event.id] = {'id': event.id, 'instruction': event.instruction, 'priority': event.priority, 'goal_predicates': [dict(p) for p in event.goal_predicates], 'activation': {key: value for key, value in activation_payload.items() if value is not None}}
            return SchedulerEvent('human_insert_posted', phase, {'id': event.id, 'after_step': high_level_step, 'instruction': event.instruction, 'priority': event.priority, 'goal_predicates': [dict(predicate) for predicate in event.goal_predicates], 'activation': {key: value for key, value in activation_payload.items() if value is not None}})
        return None

    def _consider_overlay(self, env: Any, **kwargs: Any) -> list[dict[str, Any]]:
        if self._overlay_runtime is None:
            return []
        active_override = kwargs.pop('active_subtask_id_override', None)
        return self._overlay_runtime.consider(env._human_qa_state, active_subtask_id=active_override if isinstance(active_override, str) else self._active_subtask_id(env), resolve_state=lambda question: self._overlay_resolver_state(env, question, kwargs.get('transition')), target_role_matches=lambda role, transition: self._transition_targets_role(env, role, transition), **kwargs)

    @staticmethod
    def _overlay_events(env: Any, events: Sequence[Mapping[str, Any]], *, phase: str) -> list[SchedulerEvent]:
        if not events:
            return []
        sync = getattr(env, '_sync_public_human_state', None)
        if callable(sync):
            sync()
        return [SchedulerEvent(str(event.get('name')), phase, dict(event.get('payload') or {})) for event in events]

    @staticmethod
    def _active_subtask_id(env: Any) -> str | None:
        active = getattr(env, '_evaluation_active_subtask_spec', getattr(env, '_composite_active_task_spec', None))
        if not isinstance(active, Mapping):
            return None
        subtask_id = active.get('subtask_id')
        if isinstance(subtask_id, str):
            return subtask_id
        task_id = active.get('id')
        return task_id.rsplit('::', 1)[1] if isinstance(task_id, str) and '::' in task_id else None

    def _overlay_resolver_state(self, env: Any, question: Mapping[str, Any], transition: Mapping[str, Any] | None) -> dict[str, Any]:
        question_type = question.get('question_type')
        if question_type == 'post_action_outcome_choice':
            raw = transition.get('raw_execution') if isinstance(transition, Mapping) else None
            status = raw.get('status') if isinstance(raw, Mapping) else None
            success = transition.get('success') if isinstance(transition, Mapping) else None
            outcome = 'ambiguous' if not isinstance(success, bool) or status in {'ambiguous', 'uncertain', 'verify_required'} else 'success' if isinstance(transition, Mapping) and transition.get('success') is True else 'failure'
            return {'last_tool_outcome': outcome}
        if question_type in {'remaining_requirement_choice', 'target_state_choice', 'target_relation_choice', 'mid_execution_progress_choice'}:
            parent_goal = getattr(env, '_qa_overlay_goal_result', None)
            goal = parent_goal() if callable(parent_goal) else env.check_goal(model_stopped=False)
            owner = question.get('owner_subtask_id')
            task = getattr(env, 'current_task_spec', None) or self._task_spec or {}
            goal_specs = list((task.get('goal') or {}).get('predicates') or [])
            return {'clause_incomplete': compute_qa_clause_incomplete_state(goal_specs, list(goal.get('predicates') or []), owner_subtask_id=owner if isinstance(owner, str) else None)}
        if question_type == 'active_issue_choice':
            return {'active_issue_ids': self._active_issue_ids(env, question)}
        if question_type == 'composite_progress_choice':
            return {'active_subtask_id': self._active_subtask_id(env), 'composite_complete': self._active_subtask_id(env) is None}
        return {'posted_insert_ids': list((env._human_qa_state.get('inserts') or {}).keys())}

    def _active_issue_ids(self, env: Any, question: Mapping[str, Any]) -> list[str]:
        configured = list((question.get('resolver') or {}).get('issue_ids') or [])
        parent_goal = getattr(env, '_qa_overlay_goal_result', None)
        goal = parent_goal() if callable(parent_goal) else env.check_goal(model_stopped=False)
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
        active: list[str] = []
        goal_specs = list((task.get('goal') or {}).get('predicates') or []) if isinstance(task, Mapping) else []
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

    def _milestone_matches(self, env: Any, event: HumanEventConfig, transition: Mapping[str, Any]) -> bool:
        if not bool(transition.get('success', False)):
            return False
        raw_execution = transition.get('raw_execution') or {}
        if event.activation_milestone == 'after_successful_navigate_to_role':
            if transition.get('tool_name') != 'navigate_to':
                return False
            return self._transition_targets_role(env, event.activation_role, transition)
        if event.activation_milestone == 'after_successful_tool_on_role':
            if transition.get('tool_name') != event.activation_tool:
                return False
            if isinstance(raw_execution, Mapping) and raw_execution.get('status') == 'invalid':
                return False
            return self._transition_targets_role(env, event.activation_role, transition)
        return False

    @staticmethod
    def _transition_targets_role(env: Any, role_name: Optional[str], transition: Mapping[str, Any]) -> bool:
        if not role_name:
            return False
        expected_id = getattr(env, 'role_object_ids', {}).get(role_name)
        from scoper.benchmark.common.qa_target_matching import transition_targets_role
        return transition_targets_role(role_name, expected_id, transition)

    def _visible_roles_satisfied(self, env: Any, role_names: Sequence[str]) -> bool:
        for role_name in role_names:
            obj = self._object_by_role(env, role_name)
            if not bool(obj.get('visible', False)):
                return False
        return True

    def _predicate_yes_no(self, env: Any, predicate: Mapping[str, Any]) -> str:
        expected = bool(predicate.get('expected', True))
        predicate_type = predicate.get('type')
        object_role = predicate.get('object_role')
        object_id = getattr(env, 'role_object_ids', {}).get(object_role)
        obj = env._objects_by_id().get(object_id, {}) if object_id else {}
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

    def _primary_object_role(self) -> Optional[str]:
        if not self._task_spec:
            return None
        primary = self._task_spec.get('family_spec', {}).get('human', {}).get('primary_task', {})
        if isinstance(primary, Mapping):
            return primary.get('object_role')
        return None

    def _primary_receptacle_role(self) -> Optional[str]:
        if not self._task_spec:
            return None
        primary = self._task_spec.get('family_spec', {}).get('human', {}).get('primary_task', {})
        if isinstance(primary, Mapping):
            return primary.get('receptacle_role')
        return None

    @staticmethod
    def _object_by_role(env: Any, role_name: Optional[str]) -> Mapping[str, Any]:
        if not isinstance(role_name, str) or not role_name:
            return {}
        object_id = getattr(env, 'role_object_ids', {}).get(role_name)
        objects_by_id = getattr(env, '_objects_by_id', None)
        if object_id is not None and callable(objects_by_id):
            return objects_by_id().get(object_id, {})
        return {}

    def _install_prompt_notice_hook(self, env: Any) -> None:
        if getattr(env, '_human_prompt_context_hook_installed', False):
            self._notice_callback_installed = True
            return
        original_get_prompt_context = env.get_prompt_context

        def _patched_get_prompt_context(*args, **kwargs):
            ctx = dict(original_get_prompt_context(*args, **kwargs))
            qa_state = getattr(env, '_human_qa_state', {}) or {}
            live_messages = []
            for question_id, qa in (qa_state.get('pending') or {}).items():
                entry = dict(qa) if isinstance(qa, Mapping) else {}
                entry.setdefault('id', question_id)
                entry['type'] = 'qa'
                live_messages.append(entry)
            for insert_id, insert in (qa_state.get('inserts') or {}).items():
                entry = dict(insert) if isinstance(insert, Mapping) else {}
                entry.setdefault('id', insert_id)
                entry['type'] = 'insert'
                live_messages.append(entry)
            if live_messages:
                ctx.setdefault('human_messages', []).extend(live_messages)
            return ctx
        env.get_prompt_context = _patched_get_prompt_context
        setattr(env, '_human_prompt_context_hook_installed', True)
        self._notice_callback_installed = True
__all__ = ['HumanEventConfig', 'HumanInteractionScheduler']
