from __future__ import annotations
import copy
from typing import Any, Callable, Mapping
from scoper.benchmark.common.situated_qa import resolve_qa_overlay_answer, validate_qa_overlay

class QAOverlayRuntime:

    def __init__(self, overlay: Mapping[str, Any]) -> None:
        self.overlay = validate_qa_overlay(overlay)
        self._questions = {str(question['id']): copy.deepcopy(question) for question in self.overlay['questions']}
        self._fired: set[str] = set()
        self._queued: list[str] = []
        self._triggered_states: dict[str, tuple[int, dict[str, Any]]] = {}

    def reset(self, qa_state: dict[str, Any]) -> None:
        self._fired.clear()
        self._queued.clear()
        self._triggered_states.clear()
        qa_state.setdefault('pending', {})
        qa_state.setdefault('answers', {})
        qa_state.setdefault('evaluator_keys', {})
        qa_state.setdefault('posted_steps', {})
        qa_state.setdefault('answered_steps', {})
        qa_state.setdefault('qa_overlay_resolver_failures', [])
        qa_state['qa_overlay'] = copy.deepcopy(self.overlay)

    def consider(self, qa_state: dict[str, Any], *, trigger: str, total_step: int, task_action_step: int, active_subtask_id: str | None, resolve_state: Callable[[Mapping[str, Any]], Mapping[str, Any]], transition: Mapping[str, Any] | None=None, insert_id: str | None=None, handoff_subtask_id: str | None=None, target_role_matches: Callable[[str, Mapping[str, Any]], bool] | None=None) -> list[dict[str, Any]]:
        for question_id, question in sorted(self._questions.items(), key=lambda item: (str(item[1].get('owner_subtask_id') or '~'), item[0])):
            if question_id in self._fired or question_id in self._queued:
                continue
            owner = question.get('owner_subtask_id')
            if isinstance(owner, str) and owner != active_subtask_id:
                continue
            if self._activation_matches(question.get('activation') or {}, trigger=trigger, task_action_step=task_action_step, transition=transition, insert_id=insert_id, handoff_subtask_id=handoff_subtask_id, target_role_matches=target_role_matches):
                self._queued.append(question_id)
                if question['question_type'] == 'post_action_outcome_choice':
                    self._triggered_states[question_id] = (int(total_step), copy.deepcopy(dict(resolve_state(question))))
        return self.post_next(qa_state, total_step=total_step, resolve_state=resolve_state)

    def post_next(self, qa_state: dict[str, Any], *, total_step: int, resolve_state: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> list[dict[str, Any]]:
        if qa_state.get('pending') or not self._queued:
            return []
        question_id = self._queued.pop(0)
        question = self._questions[question_id]
        self._fired.add(question_id)
        triggered = self._triggered_states.pop(question_id, None)
        resolver_state = triggered[1] if triggered is not None else resolve_state(question)
        answer = resolve_qa_overlay_answer(question, resolver_state)
        if answer is None:
            failures = qa_state.setdefault('qa_overlay_resolver_failures', [])
            if question_id not in failures:
                failures.append(question_id)
            return [{'name': 'qa_overlay_resolver_failed', 'payload': {'id': question_id, 'question_type': question['question_type']}}]
        qa_state.setdefault('evaluator_keys', {})[question_id] = answer
        qa_state.setdefault('posted_steps', {})[question_id] = int(total_step)
        question_text = question['question']
        if triggered is not None and triggered[0] != int(total_step):
            question_text += ' (Refer to the action at step {}.)'.format(triggered[0])
        if question['question_type'] in {'target_state_choice', 'target_relation_choice', 'mid_execution_progress_choice'}:
            question_text += ' Use the state when this question was first asked, before any subsequent actions.'
        qa_state.setdefault('pending', {})[question_id] = {'answer_set': copy.deepcopy(question['answer_set']), 'id': question_id, 'question': question_text, 'question_type': question['question_type'], 'type': 'qa'}
        return [{'name': 'human_qa_posted', 'payload': {'activation': copy.deepcopy(question['activation']), 'answer_set': copy.deepcopy(question['answer_set']), 'evaluator_answer': answer, 'id': question_id, 'owner_subtask_id': question.get('owner_subtask_id'), 'question': question_text, 'question_type': question['question_type'], 'source': QAOverlayRuntime.__name__}}]

    @staticmethod
    def _activation_matches(activation: Mapping[str, Any], *, trigger: str, task_action_step: int, transition: Mapping[str, Any] | None, insert_id: str | None, handoff_subtask_id: str | None, target_role_matches: Callable[[str, Mapping[str, Any]], bool] | None=None) -> bool:
        activation_type = activation.get('type')
        if activation_type == 'task_step':
            return trigger == 'task_step' and task_action_step >= int(activation['after_step'])
        if activation_type == 'before_stop':
            return trigger in {'before_stop', 'subtask_handoff'}
        if activation_type == 'after_human_insert':
            return trigger == 'after_human_insert' and insert_id == activation.get('insert_id')
        if activation_type == 'subtask_handoff':
            return trigger == 'subtask_handoff' and handoff_subtask_id == activation.get('subtask_id')
        if activation_type != 'after_tool_result' or trigger != 'after_tool_result' or (not isinstance(transition, Mapping)):
            return False
        expected_tool = activation.get('tool_name')
        actual_tool = transition.get('tool_name')
        if expected_tool != actual_tool and (expected_tool, actual_tool) not in {('slice', 'cut'), ('wash', 'soak_under')}:
            return False
        target_role = activation.get('target_role_or_null')
        if (expected_tool, actual_tool) == ('wash', 'soak_under'):
            return True
        if expected_tool in {'cool', 'heat'} and expected_tool == actual_tool:
            return True
        if target_role is None:
            return True
        if target_role_matches is not None:
            return target_role_matches(str(target_role), transition)
        arguments = transition.get('arguments')
        return isinstance(arguments, Mapping) and arguments.get('target') == target_role

    @property
    def complete(self) -> bool:
        return len(self._fired) == len(self._questions) and (not self._queued)
__all__ = ['QAOverlayRuntime']
