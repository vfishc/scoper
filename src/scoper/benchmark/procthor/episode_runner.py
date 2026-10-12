from __future__ import annotations
from scoper.benchmark.common import evaluation_lifecycle
import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence
from scoper.benchmark.procthor.brains.base import ActionResultContext, BaseBrain, Brain, BrainInput, ToolCall
from scoper.benchmark.procthor.closed_loop_tool_env import ProcTHORClosedLoopToolEnv, _agent_pose, _json_safe, get_tool_specs
from scoper.benchmark.procthor.failure_analysis import build_failure_report
from scoper.benchmark.procthor.export.execution_audit import audit_phase
from scoper.benchmark.procthor.export.stream_sampling import logical_action_duration
_STREAM_CALLBACK_NOT_INSTALLED = object()
from scoper.benchmark.procthor.export.ids import action_id, decision_id
from scoper.benchmark.procthor.export.types import EpisodeContext
from scoper.benchmark.procthor.hooks import HookBus
from scoper.benchmark.procthor.metrics import compute_episode_metrics
from scoper.benchmark.procthor.scheduler import NoOpScheduler, Scheduler
from scoper.benchmark.procthor.trace import EpisodeTraceWriter, build_meta_record, build_step_record, build_summary_record
from scoper.benchmark.common import CONFIRMABLE_ACTIONS, RuntimeEvidenceCoordinator, confirmation_intent_for_action

class EpisodeRunnerError(RuntimeError):
    pass

@dataclass
class EpisodeResult:
    task_id: str
    category: str
    house_id: str
    status: str
    final_goal: dict
    metrics: dict
    transitions: List[dict] = field(default_factory=list)
    trace_path: Optional[str] = None
    failure_type: Optional[str] = None
    failure_reason: Optional[str] = None
    failure_code: Optional[str] = None
    failure_details: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    scheduler_events: List[dict] = field(default_factory=list)
    hook_events: List[dict] = field(default_factory=list)
    hook_artifacts: dict = field(default_factory=dict)
    init_mutations: List[dict] = field(default_factory=list)
    scene_mutations: List[dict] = field(default_factory=list)

class EpisodeRunner:

    def __init__(self, env: ProcTHORClosedLoopToolEnv, *, max_high_level_steps: Optional[int]=None, require_passed_status: bool=True, raise_infrastructure_errors: bool=False, runtime_evidence_observers: Sequence[Any]=()):
        self.env = env
        self.max_high_level_steps = max_high_level_steps
        self.require_passed_status = require_passed_status
        self.raise_infrastructure_errors = bool(raise_infrastructure_errors)
        self.runtime_evidence_observers = tuple(runtime_evidence_observers)

    def run(self, task_spec: Mapping[str, Any], *, brain: Brain, hooks: Optional[HookBus]=None, scheduler: Optional[Scheduler]=None, trace_path: Optional[Path]=None, runner_config_extras: Optional[Mapping[str, Any]]=None, episode_context: Optional[EpisodeContext]=None, seed_consumption: Optional[Mapping[str, Mapping[str, Any]]]=None, admission_preflight: Optional[Callable[..., Mapping[str, Any]]]=None) -> EpisodeResult:
        env = self.env
        bus = hooks or HookBus()
        if scheduler is None:
            scheduler = _scheduler_from_task_spec(task_spec)
        scheduler = scheduler or NoOpScheduler()
        start_time = time.time()
        transitions: List[dict] = []
        scheduler_events_all: List[dict] = []
        hook_events_all: List[dict] = []
        init_mutations: List[dict] = []
        scene_mutations: List[dict] = []
        failure_type: Optional[str] = None
        failure_reason: Optional[str] = None
        status = 'failed'
        final_goal: Mapping[str, Any] = {}
        capture_hook = _trajectory_capture_hook(bus)
        runtime_evidence = None if capture_hook is not None else RuntimeEvidenceCoordinator(brain, stream_parent_style='action')
        if runtime_evidence is not None:
            for observer in self.runtime_evidence_observers:
                runtime_evidence.bridge.add_observer(observer)
        original_stream_callback = _STREAM_CALLBACK_NOT_INSTALLED
        runtime_error: Optional[Exception] = None
        budget_exhausted = False
        admission_result: Optional[Mapping[str, Any]] = None
        pre_admission_terminal: Optional[Mapping[str, Any]] = None
        brain_started = False
        writer: Optional[EpisodeTraceWriter] = None
        if trace_path is not None:
            writer = EpisodeTraceWriter(Path(trace_path))
        try:
            if capture_hook is not None:
                if episode_context is None:
                    raise EpisodeRunnerError('episode_context is required for trajectory capture')
                configure = getattr(env, 'configure_trajectory_capture', None)
                if not callable(configure):
                    raise EpisodeRunnerError('env does not support trajectory capture configuration')
                configure(episode_context, capture_hook.audit_sink, episode_context.execution_profile, capture_hook.allowed_agent_fallbacks)
                capture_hook.configure_brain(brain)
            with audit_phase(env, 'reset_init'):
                reset_obs = env.reset_to_task(task_spec)
            if runtime_evidence is not None:
                runtime_evidence.reset(str(task_spec.get('id') or 'runtime-episode'))
            init_mutations = [dict(mutation) for mutation in getattr(env, 'init_mutations', []) or []]
            scene_mutations = [dict(mutation) for mutation in getattr(env, 'scene_mutations', []) or []]
            initial_goal = env.check_goal(model_stopped=False)
            if capture_hook is not None:
                env._trajectory_initial_goal = copy.deepcopy(dict(initial_goal))
            if admission_preflight is not None:
                predicate_success = initial_goal.get('predicate_success')
                if isinstance(predicate_success, bool) or not isinstance(predicate_success, (int, float)):
                    pre_admission_terminal = {'done_reason': 'admission_evaluator_invalid', 'error_class': 'evaluator_invalid', 'error_code': 'ADMISSION_EVALUATOR_DISAGREEMENT'}
                elif float(predicate_success) == 1.0:
                    pre_admission_terminal = {'done_reason': 'admission_initial_goal_true', 'error_class': 'task_invalid', 'error_code': 'ADMISSION_INITIAL_GOAL_TRUE'}
                if pre_admission_terminal is not None:
                    status = 'invalid'
                    failure_type = str(pre_admission_terminal['error_class'])
                    failure_reason = str(pre_admission_terminal['error_code'])
            elif float(initial_goal.get('task_success', 0.0)) != 0.0:
                original_stream_callback = self._install_low_level_hook_bridge(env, bus, runtime_evidence, getattr(brain, 'on_scan_rgb', None))
                hook_events_start = bus.on_episode_start(env, task_spec, reset_obs)
                hook_events_all.extend(hook_events_start)
                if writer is not None:
                    runner_config = self._runner_config(env)
                    if runner_config_extras:
                        runner_config.update(dict(runner_config_extras))
                    meta = build_meta_record(task_spec, runner_config=runner_config, reset={'agent_pose': _agent_pose(env.controller.last_event.metadata), 'initial_goal': initial_goal, 'scene_mutations': scene_mutations, 'init_mutations': init_mutations, 'scheduler_events': [], 'hook_events': hook_events_start}, tool_specs=get_tool_specs(model_visible_only=True))
                    writer.write_meta(meta)
                raise EpisodeRunnerError('Task is already successful at reset.')
            if admission_preflight is None:
                original_stream_callback = self._install_low_level_hook_bridge(env, bus, runtime_evidence, getattr(brain, 'on_scan_rgb', None))
            from scoper.benchmark.common.composite_evaluation import enabled
            if enabled(task_spec):
                env._composite_active_task_spec = None
                if hasattr(env, '_evaluation_active_subtask_spec'):
                    del env._evaluation_active_subtask_spec
            hook_events_start = bus.on_episode_start(env, task_spec, reset_obs)
            hook_events_all.extend(hook_events_start)
            if admission_preflight is None and (isinstance(brain, BaseBrain) or hasattr(brain, 'on_episode_start')):
                brain.on_episode_start(env, task_spec)
                brain_started = True
            from scoper.benchmark.common.composite_evaluation import CompositeEvaluationProgress, enabled
            composite_progress = None
            if enabled(task_spec) and (not isinstance(getattr(env, '_composite_active_task_spec', None), Mapping)):
                from scoper.benchmark.procthor.brains.teachers.composite import build_composite_subtask_spec, _check_subtask_goal, _set_active_subtask
                children = [build_composite_subtask_spec(task_spec, child, i) for i, child in enumerate(task_spec['family_spec']['composite']['subtasks'])]
                composite_progress = CompositeEvaluationProgress(children, lambda child: _check_subtask_goal(env, child), lambda child: setattr(env, '_evaluation_active_subtask_spec', child))
            _mark_scheduler_phase_start(bus, env, 'episode_start', 0)
            with audit_phase(env, 'scheduler', trigger='episode_start'):
                sched_events_start = _materialize(scheduler.on_episode_start(env, task_spec))
            scheduler_events_all.extend(sched_events_start)
            hook_events_all.extend(bus.on_scheduler_events(env, sched_events_start, 'episode_start', 0))
            if writer is not None:
                runner_config = self._runner_config(env)
                if runner_config_extras:
                    runner_config.update(dict(runner_config_extras))
                meta = build_meta_record(task_spec, runner_config=runner_config, reset={'agent_pose': _agent_pose(env.controller.last_event.metadata), 'initial_goal': initial_goal, 'scene_mutations': scene_mutations, 'init_mutations': init_mutations, 'scheduler_events': sched_events_start, 'hook_events': hook_events_start}, tool_specs=get_tool_specs(model_visible_only=True))
                writer.write_meta(meta)
            if admission_preflight is not None and pre_admission_terminal is None:
                if episode_context is None:
                    raise EpisodeRunnerError('episode_context is required for admission preflight')
                with audit_phase(env, 'decision', step_id=0):
                    admission_result = admission_preflight(env, task_spec, episode_context=episode_context)
                from scoper.benchmark.procthor.benchmark.admission_schema import validate_reachability_preflight
                validate_reachability_preflight(admission_result)
                if capture_hook is not None:
                    capture_hook.record_admission_preflight(admission_result)
                if admission_result['status'] != 'pass':
                    status = 'invalid'
                    failure_type = str(admission_result['error_class'])
                    failure_reason = str(admission_result.get('failure_code') or 'admission evidence is invalid')
            if admission_preflight is not None and pre_admission_terminal is None and (admission_result is None or admission_result['status'] == 'pass'):
                original_stream_callback = self._install_low_level_hook_bridge(env, bus, runtime_evidence, getattr(brain, 'on_scan_rgb', None))
                if isinstance(brain, BaseBrain) or hasattr(brain, 'on_episode_start'):
                    brain.on_episode_start(env, task_spec)
                    brain_started = True
            max_steps = self.max_high_level_steps if self.max_high_level_steps is not None else env.max_high_level_steps
            last_transition: Optional[dict] = None
            loop_iterations = 0
            loop_ceiling = max_steps
            while pre_admission_terminal is None and (admission_result is None or admission_result['status'] == 'pass'):
                pending_qa = bool((getattr(env, '_human_qa_state', {}) or {}).get('pending'))
                if loop_iterations >= loop_ceiling:
                    budget_exhausted = True
                    failure_type = 'EpisodeRunnerError'
                    failure_reason = 'Exceeded max high-level steps ({}).'.format(max_steps)
                    break
                if int(getattr(env, 'task_action_step', len(transitions))) >= max_steps and (not pending_qa):
                    budget_exhausted = True
                    failure_type = 'EpisodeRunnerError'
                    failure_reason = 'Exceeded max high-level steps ({}).'.format(max_steps)
                    break
                loop_iterations += 1
                step_id = len(transitions) + 1
                _mark_scheduler_phase_start(bus, env, 'pre_step', step_id)
                with audit_phase(env, 'scheduler', step_id=step_id, trigger='pre_step'):
                    pre_events = _materialize([] if evaluation_lifecycle.finishing(env) else scheduler.pre_step(env, step_id, last_transition))
                scheduler_events_all.extend(pre_events)
                hook_events_all.extend(bus.on_scheduler_events(env, pre_events, 'pre_step', step_id))
                brain_input = BrainInput(observation=env._observation(), prompt_context={**env.get_prompt_context(), 'evaluation_step_budget': max_steps}, last_transition=last_transition, high_level_step=step_id)
                from scoper.benchmark.common.public_binding_events import sync as sync_public_binding_events
                sync_public_binding_events(env, brain_input.prompt_context)
                if runtime_evidence is not None and runtime_evidence.enabled:
                    runtime_evidence.prepare_decision(step=step_id, decision_id=decision_id(step_id), observation=brain_input.observation)
                prepare_capture = getattr(capture_hook, 'prepare_brain_input', None) if capture_hook is not None else None
                if callable(prepare_capture):
                    prepare_capture(env, brain_input, step_id)
                prepare_input = getattr(brain, 'prepare_input', None)
                if callable(prepare_input):
                    brain_input = prepare_input(brain_input)
                    if not isinstance(brain_input, BrainInput):
                        raise EpisodeRunnerError('Brain.prepare_input must return BrainInput')
                decision_context = {'step_id': step_id, 'decision_id': decision_id(step_id), 'input_kind': 'teacher_act' if _is_teacher_brain(brain) else 'brain_act', 'policy_source': _policy_source(brain), 'input_frame_ids': [], 'model_call_id': None}
                set_model_context = getattr(brain, 'set_model_call_context', None)
                if getattr(brain, 'policy_model_io_enabled', True) is not True:
                    set_model_context = None
                model_context = None
                if callable(set_model_context):
                    model_context = capture_hook.model_call_context(step_id) if capture_hook else None
                    set_model_context(model_context)
                    decision_context['model_call_id'] = None if model_context is None else model_context.model_call_id
                with audit_phase(env, 'decision', step_id=step_id):
                    hook_events_all.extend(bus.on_decision_input(env, brain_input, step_id, decision_context))
                try:
                    if runtime_evidence is not None:
                        runtime_evidence.note_policy_call()
                    tool_call = brain.act(brain_input)
                finally:
                    if callable(set_model_context):
                        set_model_context(None)
                if not isinstance(tool_call, ToolCall):
                    raise EpisodeRunnerError('Brain.act must return a ToolCall, got {}.'.format(type(tool_call).__name__))
                public_postcondition_pre = None
                harness_definition = getattr(brain, 'harness_definition', None)
                if harness_definition is not None and harness_definition.legacy_simulator_postconditions_enabled:
                    from scoper.integration.procthor.public_postconditions import capture_public_action_state
                    public_postcondition_pre = capture_public_action_state(env)
                before_events = bus.before_tool(env, tool_call, step_id)
                delegated_error: Optional[Exception] = None

                def _default_call(_tc: ToolCall=tool_call) -> dict:
                    nonlocal delegated_error
                    try:
                        with audit_phase(env, 'agent_action', step_id=step_id, action_id=action_id(step_id)):
                            return env.call_tool(_tc.tool_name, dict(_tc.arguments), reasoning=_tc.reasoning)
                    except Exception as exc:
                        delegated_error = exc
                        raise
                _mark_scheduler_phase_start(bus, env, 'intercept', step_id)
                try:
                    with audit_phase(env, 'scheduler', step_id=step_id, trigger='intercept'):
                        intercepted = None if evaluation_lifecycle.finishing(env) else scheduler.intercept_tool_call(env, tool_call, step_id, _default_call)
                except Exception as exc:
                    if delegated_error is not exc:
                        raise
                    if self.raise_infrastructure_errors:
                        raise
                    intercepted = _synthetic_failed_transition(env, tool_call, step_id, exc)
                intercept_events: List[dict] = []
                if intercepted is not None:
                    transition = dict(intercepted)
                    intercept_payload = transition.pop('scheduler_intercept', None)
                    if intercept_payload is not None:
                        intercept_events = _materialize(intercept_payload)
                        scheduler_events_all.extend(intercept_events)
                        hook_events_all.extend(bus.on_scheduler_events(env, intercept_events, 'intercept', step_id))
                else:
                    try:
                        transition = _default_call()
                    except Exception as exc:
                        if self.raise_infrastructure_errors:
                            raise
                        transition = _synthetic_failed_transition(env, tool_call, step_id, exc)
                immediate_observation = dict(transition.get('observation')) if isinstance(transition.get('observation'), Mapping) else {}
                confirmation_observation: Mapping[str, Any] | None = None
                restored_observation = immediate_observation
                harness_spec = getattr(harness_definition, 'spec', harness_definition)
                confirmation_enabled = 'three_frame_visual_verification' in tuple(getattr(harness_spec, 'capabilities', ()))
                immediate_events: List[dict] = []
                if confirmation_enabled and tool_call.tool_name in CONFIRMABLE_ACTIONS:
                    immediate_events = bus.capture_immediate_post_action(env, tool_call, transition, step_id)
                    private_target_getter = getattr(env, 'last_private_bound_target', None)
                    private_target = private_target_getter() if callable(private_target_getter) else None
                    public_target = (tool_call.arguments or {}).get('target')
                    intent = confirmation_intent_for_action(tool_call.tool_name, target_public_ref_or_null=public_target if isinstance(public_target, str) and public_target else None, has_bound_target=isinstance(private_target, Mapping))
                    try:
                        capture_confirmation = getattr(env, 'capture_confirmation', None)
                        if not callable(capture_confirmation):
                            raise EpisodeRunnerError('v12 backend lacks capture_confirmation')
                        confirmation = capture_confirmation(intent, private_target, video_fps=_video_fps_or_null(bus))
                        candidate = confirmation.get('observation')
                        if isinstance(candidate, Mapping) and candidate.get('head_rgb') is not None:
                            confirmation_observation = {**immediate_observation, **dict(candidate)}
                        restored_candidate = confirmation.get('restored_observation')
                        if isinstance(restored_candidate, Mapping):
                            restored_observation = {**immediate_observation, **dict(restored_candidate)}
                        confirmation_records = confirmation.get('low_level_actions')
                        if isinstance(confirmation_records, list):
                            transition.setdefault('low_level_actions', []).extend(copy.deepcopy(confirmation_records))
                            transition['num_low_level_steps'] = len(transition['low_level_actions'])
                        transition['confirmation'] = copy.deepcopy(confirmation.get('camera'))
                    except Exception as exc:
                        transition['confirmation'] = {'schema_version': 'confirmation-camera.v1', 'status': 'failed', 'error_type': type(exc).__name__, 'error_message': str(exc)}
                    transition['observation'] = confirmation_observation if confirmation_observation is not None else immediate_observation
                    episode_metrics = getattr(env, '_episode_metrics', None)
                    if callable(episode_metrics):
                        transition['episode_metrics'] = episode_metrics()
                if intercepted is not None:
                    record_intercepted = getattr(env, 'record_intercepted_transition', None)
                    if callable(record_intercepted):
                        record_intercepted(transition)
                after_events = bus.after_tool(env, tool_call, transition, step_id)
                if confirmation_enabled and tool_call.tool_name in CONFIRMABLE_ACTIONS:
                    transition['observation'] = restored_observation
                public_postcondition_evidence = None
                if public_postcondition_pre is not None:
                    from scoper.integration.procthor.public_postconditions import build_public_postcondition_evidence, capture_public_action_state
                    public_postcondition_evidence = build_public_postcondition_evidence(public_postcondition_pre, capture_public_action_state(env))
                hook_events_all.extend(before_events)
                hook_events_all.extend(immediate_events)
                hook_events_all.extend(after_events)
                if runtime_evidence is not None and runtime_evidence.enabled:
                    final_observation = transition.get('observation') if isinstance(transition.get('observation'), Mapping) else {}
                    runtime_evidence.record_final_stream(step=step_id, rgb=final_observation.get('head_rgb'), public_observation=final_observation, action_id=action_id(step_id))
                    if confirmation_enabled and tool_call.tool_name in CONFIRMABLE_ACTIONS and (confirmation_observation is not None):
                        runtime_evidence.prepare_action_result(step=step_id, action_id=action_id(step_id), immediate_post_observation=immediate_observation, confirmation_observation=confirmation_observation)
                    else:
                        runtime_evidence.prepare_action_result(step=step_id, action_id=action_id(step_id), post_observation=transition.get('observation') if isinstance(transition.get('observation'), Mapping) else {})
                on_action_result = getattr(brain, 'on_action_result', None)
                if callable(on_action_result):
                    action_result_observation = transition.get('observation') or {}
                    if bool(getattr(harness_definition, 'visual_only_state_enabled', False)):
                        action_result_observation = {'head_rgb': action_result_observation.get('head_rgb')}
                    is_confirmable = confirmation_enabled and tool_call.tool_name in CONFIRMABLE_ACTIONS
                    on_action_result(ActionResultContext(step_id=step_id, decision_id=decision_id(step_id), action_id=action_id(step_id), tool_call=tool_call, public_transition=_public_runner_transition(transition, harness_definition=harness_definition), pre_observation=brain_input.observation, post_observation=action_result_observation, public_postcondition_evidence_or_null=public_postcondition_evidence, immediate_post_observation_or_null=({'head_rgb': immediate_observation.get('head_rgb')} if bool(getattr(harness_definition, 'visual_only_state_enabled', False)) else immediate_observation) if is_confirmable else None, confirmation_observation_or_null=({'head_rgb': confirmation_observation.get('head_rgb')} if bool(getattr(harness_definition, 'visual_only_state_enabled', False)) else confirmation_observation) if is_confirmable and confirmation_observation is not None else None, private_binding_roles_or_null=tuple(env.last_private_binding_roles()) if callable(getattr(env, 'last_private_binding_roles', None)) else None))
                _mark_scheduler_phase_start(bus, env, 'post_step', step_id)
                with audit_phase(env, 'scheduler', step_id=step_id, trigger='post_step'):
                    post_events = _materialize([] if evaluation_lifecycle.finishing(env) else scheduler.post_step(env, step_id, transition))
                scheduler_events_all.extend(post_events)
                hook_events_all.extend(bus.on_scheduler_events(env, post_events, 'post_step', step_id))
                evaluation_lifecycle.after_transition(env, transition)
                _mask_family_feedback_if_needed(env, transition)
                if transition.get('done') and (getattr(env, '_human_qa_state', {}) or {}).get('pending'):
                    transition['done'] = False
                    transition['done_reason'] = None
                if composite_progress is not None:
                    composite_progress.after_step(transition, pending_qa=bool((getattr(env, '_human_qa_state', {}) or {}).get('pending')))
                transitions.append(transition)
                last_transition = transition
                if writer is not None:
                    writer.write_step(build_step_record(high_level_step=step_id, brain_decision={'tool_name': tool_call.tool_name, 'arguments': dict(tool_call.arguments), 'reasoning': tool_call.reasoning}, transition=_json_safe(transition), hook_events=before_events + after_events, scheduler_events=pre_events + intercept_events + post_events))
                if transition.get('done'):
                    break
            final_goal = env.check_goal(model_stopped=env.model_stopped)
            if pre_admission_terminal is not None or (admission_result is not None and admission_result['status'] != 'pass'):
                status = 'invalid'
            elif budget_exhausted:
                status = 'failed'
            elif float(final_goal.get('task_success', 0.0)) == 1.0:
                status = 'passed'
            else:
                failure_type = 'EpisodeRunnerError'
                failure_reason = 'Final task_success is not 1.0.'
        except Exception as exc:
            runtime_error = exc
            failure_type = type(exc).__name__
            failure_reason = str(exc)
            try:
                final_goal = env.check_goal(model_stopped=env.model_stopped)
            except Exception:
                final_goal = {}
            if capture_hook is not None:
                _abort_capture(capture_hook, exc)
        runtime_error_code = getattr(runtime_error, 'code', None)
        if not isinstance(runtime_error_code, str) or not runtime_error_code:
            runtime_error_code = None
        elapsed = time.time() - start_time
        metrics = compute_episode_metrics(transitions=transitions, final_goal=final_goal, done_reason=transitions[-1].get('done_reason') if transitions else None, model_stopped=bool(getattr(env, 'model_stopped', False)), elapsed_seconds=elapsed, scheduler_events=scheduler_events_all, task_spec=task_spec)
        metrics.update(evaluation_lifecycle.diagnostics(env))
        environment_metrics_builder = getattr(env, '_episode_metrics', None)
        environment_metrics = environment_metrics_builder() if callable(environment_metrics_builder) else {}
        for metric_name in ('binding_metrics', 'navigation_metrics'):
            metric_value = environment_metrics.get(metric_name)
            if isinstance(metric_value, Mapping):
                metrics[metric_name] = copy.deepcopy(dict(metric_value))
        teardown_error: Optional[Exception] = runtime_error
        end_events_sched: List[dict] = []
        end_events_hook: List[dict] = []
        try:
            _mark_scheduler_phase_start(bus, env, 'episode_end', getattr(env, 'high_level_step', 0))
            with audit_phase(env, 'scheduler', step_id=getattr(env, 'high_level_step', 0), trigger='episode_end'):
                end_events_sched = _materialize(scheduler.on_episode_end(env, final_goal))
        except Exception as exc:
            if teardown_error is None:
                teardown_error = exc
                failure_type = type(exc).__name__
                failure_reason = str(exc)
            status = 'failed'
            if capture_hook is not None:
                _abort_capture(capture_hook, exc)
        scheduler_events_all.extend(end_events_sched)
        try:
            hook_events_all.extend(bus.on_scheduler_events(env, end_events_sched, 'episode_end', getattr(env, 'high_level_step', 0)))
        except Exception as exc:
            if teardown_error is None:
                teardown_error = exc
                failure_type = type(exc).__name__
                failure_reason = str(exc)
            status = 'failed'
            if capture_hook is not None:
                _abort_capture(capture_hook, exc)
        try:
            end_events_hook = bus.on_episode_end(env, final_goal, status)
            hook_events_all.extend(end_events_hook)
        except Exception as exc:
            if teardown_error is None:
                teardown_error = exc
                failure_type = type(exc).__name__
                failure_reason = str(exc)
            status = 'failed'
            if capture_hook is not None:
                _abort_capture(capture_hook, exc)
        if admission_preflight is None or brain_started:
            try:
                brain.on_episode_end(env, final_goal)
            except Exception as exc:
                if teardown_error is None:
                    teardown_error = exc
                    failure_type = type(exc).__name__
                    failure_reason = str(exc)
                status = 'failed'
                if capture_hook is not None:
                    _abort_capture(capture_hook, exc)
        try:
            self._restore_low_level_hook_bridge(env, original_stream_callback)
        except Exception as exc:
            if teardown_error is None:
                teardown_error = exc
                failure_type = type(exc).__name__
                failure_reason = str(exc)
            status = 'failed'
            if capture_hook is not None:
                _abort_capture(capture_hook, exc)
        if capture_hook is not None:
            try:
                capture_hook.restore_brain(brain)
            except Exception as exc:
                if teardown_error is None:
                    teardown_error = exc
                    failure_type = type(exc).__name__
                    failure_reason = str(exc)
                status = 'failed'
                _abort_capture(capture_hook, exc)
        if capture_hook is not None and teardown_error is None:
            if pre_admission_terminal is not None:
                terminal_error_class = str(pre_admission_terminal['error_class'])
                terminal_error_code = str(pre_admission_terminal['error_code'])
                terminal_done_reason = str(pre_admission_terminal['done_reason'])
            elif admission_result is not None and admission_result['status'] != 'pass':
                terminal_error_class = str(admission_result['error_class'])
                terminal_error_code = str(admission_result['failure_code'])
                terminal_done_reason = _admission_done_reason(admission_result)
            else:
                terminal_error_class = _terminal_error_class(failure_type, status, budget_exhausted=budget_exhausted)
                terminal_error_code = None if status == 'passed' else runtime_error_code or failure_type
                terminal_done_reason = 'max_high_level_steps' if budget_exhausted else transitions[-1].get('done_reason') if transitions else failure_reason
            if any((transition.get('raw_execution', {}).get('synthetic_action_end') is True for transition in transitions)):
                terminal_error_class = 'infra_error'
            if getattr(capture_hook, '_profile_violation', False):
                terminal_error_class = 'evaluator_invalid'
            terminal = {'episode_status': status if status in ('passed', 'failed') else 'invalid', 'done_reason': terminal_done_reason, 'error_class': terminal_error_class, 'error_code': terminal_error_code, 'native_task_success': getattr(capture_hook, '_native_final_success', None), 'compatibility_task_success': getattr(capture_hook, '_effective_final_success', None), 'fallback_summary': copy.deepcopy(getattr(capture_hook, '_fallback_counts', {})), 'seed_consumption': copy.deepcopy(dict(seed_consumption or {})), 'prompt_tokens': None, 'completion_tokens': None}
            try:
                capture_hook.finalize(terminal)
            except Exception as exc:
                if teardown_error is None:
                    teardown_error = exc
                    failure_type = type(exc).__name__
                    failure_reason = str(exc)
                _abort_capture(capture_hook, exc)
                status = 'failed'
        if hasattr(brain, 'direct_decision_records'):
            metrics['direct_decisions'] = list(brain.direct_decision_records)
        failure_report = build_failure_report(transitions=transitions, final_goal=final_goal, failure_type=failure_type, failure_reason=failure_reason, scheduler_events=scheduler_events_all, task_spec=task_spec, model_stopped=bool(getattr(env, 'model_stopped', False)))
        if status != 'passed':
            failure_reason = failure_report.get('failure_reason') or failure_reason
        hook_artifacts = _collect_hook_artifacts(bus)
        if runtime_evidence is not None:
            hook_artifacts['runtime_evidence'] = dict(runtime_evidence.artifacts())
        if writer is not None:
            writer.write_summary(build_summary_record(status=status, final_goal=dict(final_goal), metrics=metrics, failure_type=failure_type, failure_reason=failure_reason, failure_code=failure_report.get('failure_code'), failure_details=failure_report.get('failure_details'), init_mutations=init_mutations, scene_mutations=scene_mutations, hook_events=end_events_hook, elapsed_seconds=elapsed))
            writer.close()
        if self.raise_infrastructure_errors and runtime_error is not None:
            from scoper.benchmark.common.direct_response import preserve_partial_episode
            preserve_partial_episode(runtime_error, transitions, metrics, final_goal)
            raise runtime_error
        return EpisodeResult(task_id=task_spec['id'], category=task_spec['category'], house_id=task_spec['source']['house_id'], status=status, final_goal=dict(final_goal), metrics=metrics, transitions=transitions, trace_path=str(trace_path) if trace_path is not None else None, failure_type=failure_type, failure_reason=failure_reason, failure_code=pre_admission_terminal.get('error_code') if pre_admission_terminal is not None else admission_result.get('failure_code') if admission_result is not None and admission_result['status'] != 'pass' else runtime_error_code or failure_report.get('failure_code'), failure_details={'admission_terminal': copy.deepcopy(dict(pre_admission_terminal))} if pre_admission_terminal is not None else {'admission_preflight': copy.deepcopy(dict(admission_result))} if admission_result is not None and admission_result['status'] != 'pass' else dict(failure_report.get('failure_details') or {}), elapsed_seconds=elapsed, scheduler_events=list(scheduler_events_all), hook_events=list(hook_events_all), hook_artifacts=hook_artifacts, init_mutations=list(init_mutations), scene_mutations=list(scene_mutations))

    @staticmethod
    def _install_low_level_hook_bridge(env: ProcTHORClosedLoopToolEnv, bus: HookBus, runtime_evidence: RuntimeEvidenceCoordinator | None=None, scan_rgb_observer: Callable[..., None] | None=None) -> Any:
        if env.executor is None:
            return None
        original = env.executor.stream_callback
        capture_enabled = _trajectory_capture_hook(bus) is not None
        logical_time_s = 0.0

        def _bridged(action_record, event):
            nonlocal logical_time_s
            record = original(action_record, event) if original else dict(action_record)
            hook_event = event if capture_enabled else getattr(getattr(env, 'controller', None), 'last_event', event)
            try:
                bus.on_low_level(env, record, hook_event)
            except Exception:
                if capture_enabled:
                    raise
            if callable(scan_rgb_observer) and record.get('source') == 'scan' and (record.get('action') == 'RotateRight') and (record.get('execution_substep', 1) == record.get('execution_substeps', 1)) and (not record.get('video_only', False)):
                from scoper.benchmark.common.binding_observation_ledger import register_binding_frame
                from scoper.benchmark.common.evaluation_binding_policy import task_uses_evidence_binding
                scan_event = hook_event
                if task_uses_evidence_binding(getattr(env, 'current_task_spec', None)):
                    scan_event = getattr(getattr(env, 'controller', None), 'last_event', hook_event)
                scan_rgb = getattr(scan_event, 'frame', None)
                objects = (getattr(scan_event, 'metadata', {}) or {}).get('objects', [])
                register_binding_frame(env, scan_rgb, [obj['objectId'] for obj in objects if obj.get('visible')])
                scan_rgb_observer(scan_rgb)
            if runtime_evidence is not None and runtime_evidence.enabled:
                step = int(record.get('high_level_step') or getattr(env, 'high_level_step', 0) or 1)
                metadata = getattr(hook_event, 'metadata', {}) or {}
                timestamp = record.get('timestamp_s')
                if isinstance(timestamp, (int, float)) and (not isinstance(timestamp, bool)):
                    logical_time_s = max(logical_time_s, float(timestamp))
                else:
                    logical_time_s += logical_action_duration(record)
                runtime_evidence.record_stream(step=step, rgb=getattr(hook_event, 'frame', None), public_observation={} if runtime_evidence.visual_only else {'visible_object_catalog': list(getattr(env, '_agent_catalog_view', ()) or ()), 'last_action_success': bool(metadata.get('lastActionSuccess', False))}, logical_time_s=logical_time_s, action_id=action_id(step))
            return record
        env.executor.stream_callback = _bridged
        return original

    @staticmethod
    def _restore_low_level_hook_bridge(env, original) -> None:
        if original is not _STREAM_CALLBACK_NOT_INSTALLED and getattr(env, 'executor', None) is not None:
            env.executor.stream_callback = original

    @staticmethod
    def _runner_config(env: ProcTHORClosedLoopToolEnv) -> dict:
        return {'width': env.width, 'height': env.height, 'cloud_rendering': getattr(env, 'cloud_rendering', None), 'max_high_level_steps': env.max_high_level_steps, 'max_total_low_level_steps': env.max_total_low_level_steps, 'max_invalid_tool_calls': env.max_invalid_tool_calls, 'max_nav_steps_per_tool': env.max_nav_steps_per_tool, 'move_execution_step': env.move_execution_step, 'rotate_execution_degrees': env.rotate_execution_degrees, 'look_execution_degrees': env.look_execution_degrees, 'snap_to_grid': env.snap_to_grid, 'allow_force_navigation_fallback': getattr(env, 'allow_force_navigation_fallback', None), 'max_force_navigation_fallbacks_per_tool': getattr(env, 'max_force_navigation_fallbacks_per_tool', None), 'allow_force_open_container_fallback': getattr(env, 'allow_force_open_container_fallback', None), 'max_force_open_container_fallbacks_per_tool': getattr(env, 'max_force_open_container_fallbacks_per_tool', None), 'allow_force_arm_open_close_collision_fallback': getattr(env, 'allow_force_arm_open_close_collision_fallback', None), 'max_force_arm_open_close_collision_fallbacks_per_tool': getattr(env, 'max_force_arm_open_close_collision_fallbacks_per_tool', None), 'allow_public_placed_open_container_navigation': getattr(env, 'allow_public_placed_open_container_navigation', None)}

def _materialize(events) -> List[dict]:
    if not events:
        return []
    out: List[dict] = []
    for ev in events:
        if hasattr(ev, 'name') and hasattr(ev, 'phase'):
            out.append({'name': ev.name, 'phase': ev.phase, 'payload': dict(getattr(ev, 'payload', {}) or {})})
        elif isinstance(ev, Mapping):
            out.append(dict(ev))
    return out

def _configured_qa_count(task_spec: Mapping[str, Any]) -> int:
    overlay = task_spec.get('qa_overlay')
    count = len(overlay.get('questions') or []) if isinstance(overlay, Mapping) else 0
    family = task_spec.get('family_spec')
    if not isinstance(family, Mapping):
        return count
    human = family.get('human')
    if isinstance(human, Mapping):
        events = human.get('human_events')
        if not isinstance(events, list):
            events = human.get('human_events_template')
        count += sum((1 for event in events or [] if isinstance(event, Mapping) and event.get('type') == 'qa'))
    composite = family.get('composite')
    if isinstance(composite, Mapping):
        for subtask in composite.get('subtasks') or []:
            if isinstance(subtask, Mapping):
                count += _configured_qa_count(subtask)
    return count

def _collect_hook_artifacts(bus: HookBus) -> dict:
    artifacts = {}
    for hook in bus.hooks:
        manifest = getattr(hook, 'manifest', None)
        if manifest is None:
            continue
        if callable(manifest):
            try:
                manifest = manifest()
            except TypeError:
                manifest = None
        if manifest:
            artifacts[getattr(hook, 'name', hook.__class__.__name__)] = dict(manifest)
    return artifacts

def _trajectory_capture_hook(bus: HookBus) -> Any:
    for hook in bus.hooks:
        if getattr(hook, 'name', None) == 'trajectory_capture':
            return hook
    return None

def _abort_capture(capture_hook: Any, error: Exception) -> None:
    try:
        capture_hook.abort({'error_class': 'infra_error', 'error_code': getattr(error, 'code', type(error).__name__), 'message': type(error).__name__})
    except Exception:
        writer = getattr(capture_hook, 'writer', None)
        if writer is not None:
            writer.state = 'ABORTED'

def _mark_scheduler_phase_start(bus: HookBus, env: Any, phase: str, high_level_step: int) -> None:
    for hook in bus.hooks:
        method = getattr(hook, 'mark_scheduler_phase_start', None)
        if callable(method):
            method(env, phase, high_level_step)

def _is_teacher_brain(brain: Any) -> bool:
    module = type(brain).__module__.casefold()
    name = type(brain).__name__.casefold()
    return 'teachers' in module or 'oracle' in name or 'scripted' in name

def _policy_source(brain: Any) -> str:
    name = type(brain).__name__.casefold()
    module = type(brain).__module__.casefold()
    if 'recoveryoracle' in name or 'recovery_oracle' in module:
        return 'recovery_oracle'
    if 'oracle' in name:
        return 'oracle'
    if 'teachers' in module or 'scripted' in name:
        return 'teacher'
    return 'learner'

def _synthetic_failed_transition(env: Any, tool_call: ToolCall, step_id: int, error: Exception) -> dict:
    try:
        observation = env._observation()
    except Exception:
        observation = {}
    try:
        goal_result = env.check_goal(model_stopped=getattr(env, 'model_stopped', False))
    except Exception:
        goal_result = {}
    return {'tool_name': tool_call.tool_name, 'arguments': dict(tool_call.arguments), 'reasoning': tool_call.reasoning, 'high_level_step': step_id, 'success': False, 'env_feedback': 'Executor raised {}.'.format(type(error).__name__), 'done': True, 'done_reason': 'executor_exception', 'observation': observation, 'goal_result': goal_result, 'raw_execution': {'status': 'failed', 'error_type': type(error).__name__, 'error_message': str(error), 'synthetic_action_end': True}, 'low_level_actions': [], 'num_low_level_steps': 0, 'episode_metrics': {}, 'ready_target': None, 'elapsed_seconds': 0.0}

def _public_runner_transition(transition: Mapping[str, Any], *, harness_definition: Any | None=None) -> dict:
    if bool(getattr(harness_definition, 'visual_only_state_enabled', False)):
        return dict(harness_definition.project_completed_tool_record(transition))
    return {key: copy.deepcopy(transition.get(key)) for key in ('tool_name', 'arguments', 'reasoning', 'high_level_step', 'success', 'env_feedback', 'done', 'done_reason', 'num_low_level_steps', 'elapsed_seconds', 'binding', 'bindings') if key in transition}

def _terminal_error_class(failure_type: str | None, status: str, *, budget_exhausted: bool=False) -> str:
    if status == 'passed':
        return 'none'
    if failure_type == 'NativeProfileViolation':
        return 'evaluator_invalid'
    if failure_type in ('VLMBrainInfrastructureError', 'TrajectoryCaptureError'):
        return 'infra_error'
    if budget_exhausted:
        return 'budget_exceeded'
    return 'agent_failure'

def _admission_done_reason(result: Mapping[str, Any]) -> str:
    if result.get('failure_code') == 'ADMISSION_UNREACHABLE':
        return 'admission_unreachable'
    if result.get('error_class') == 'evaluator_invalid':
        return 'admission_evaluator_invalid'
    return 'admission_infra_error'

def _scheduler_from_task_spec(task_spec: Mapping[str, Any]) -> Optional[Scheduler]:
    from scoper.benchmark.procthor.scheduler import ActiveSubtaskScheduler, CompositeScheduler, DisturbanceScheduler, FailureScheduler, HumanInteractionScheduler
    children: list = []
    spec = task_spec.get('disturbance_spec')
    if spec:
        seed = int(spec.get('seed', 0))
        disturbances = spec.get('disturbances')
        if disturbances:
            children.append(DisturbanceScheduler(configs=disturbances, seed=seed))
        failures = spec.get('failures')
        if failures:
            children.append(FailureScheduler(configs=failures, seed=seed))
    family_human = task_spec.get('family_spec', {}).get('human')
    qa_overlay = task_spec.get('qa_overlay')
    if isinstance(family_human, Mapping):
        human_events = list(family_human.get('human_events') or [])
        if human_events or isinstance(qa_overlay, Mapping):
            children.append(HumanInteractionScheduler(human_events, qa_overlay=qa_overlay if isinstance(qa_overlay, Mapping) else None))
    elif isinstance(qa_overlay, Mapping):
        children.append(HumanInteractionScheduler(qa_overlay=qa_overlay))
    composite = task_spec.get('family_spec', {}).get('composite')
    if isinstance(composite, Mapping):
        from scoper.benchmark.procthor.brains.teachers.composite import build_composite_subtask_spec
        for index, subtask in enumerate(composite.get('subtasks') or []):
            if not isinstance(subtask, Mapping):
                continue
            child_family = subtask.get('family_spec')
            child_human = child_family.get('human') if isinstance(child_family, Mapping) else None
            if not isinstance(child_human, Mapping):
                continue
            human_events = list(child_human.get('human_events') or [])
            if not human_events:
                continue
            child_spec = build_composite_subtask_spec(task_spec, subtask, index)
            children.append(ActiveSubtaskScheduler(HumanInteractionScheduler(human_events), subtask_id=str(child_spec['subtask_id']), task_spec=child_spec))
    if not children:
        return None
    if len(children) == 1:
        return children[0]
    return CompositeScheduler(children)

def _mask_family_feedback_if_needed(env: ProcTHORClosedLoopToolEnv, transition: dict) -> None:
    masker = getattr(env, '_mask_verify_feedback_if_needed', None)
    if callable(masker):
        masker(transition)

def _video_fps_or_null(bus: HookBus) -> float | None:
    for hook in bus.hooks:
        recorder = getattr(hook, 'recorder', None)
        fps = getattr(recorder, 'fps', None)
        if isinstance(fps, (int, float)) and (not isinstance(fps, bool)) and (fps > 0):
            return float(fps)
    return None
__all__ = ['EpisodeResult', 'EpisodeRunner', 'EpisodeRunnerError']
