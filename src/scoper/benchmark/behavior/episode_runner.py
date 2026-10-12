from __future__ import annotations
from scoper.benchmark.common import evaluation_lifecycle
import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional
from scoper.benchmark.behavior.behavior_closed_loop_tool_env import BehaviorClosedLoopToolEnv, MODEL_VISIBLE_TOOLS
from scoper.benchmark.behavior.brains.base import ActionResultContext, BaseBrain, Brain, BrainInput, ToolCall
from scoper.benchmark.behavior.export.ids import action_id, decision_id
from scoper.benchmark.behavior.export.types import EpisodeContext, TrajectoryConfigError
from scoper.benchmark.behavior.failure_analysis import build_failure_report
from scoper.benchmark.behavior.hooks import HookBus
from scoper.benchmark.behavior.json_utils import _json_safe
from scoper.benchmark.behavior.metrics import compute_episode_metrics
from scoper.benchmark.behavior.runtime_evidence import BehaviorStreamEvidenceObserver
from scoper.benchmark.behavior.runtime_timing import RuntimeTimingRecorder, activate_runtime_timing, current_runtime_timing, deactivate_runtime_timing, runtime_phase
from scoper.benchmark.behavior.scheduler import NoOpScheduler, Scheduler, SchedulerEvent, build_scheduler_from_task_spec
from scoper.benchmark.behavior.trace import EpisodeTraceWriter
from scoper.benchmark.common import CONFIRMABLE_ACTIONS, RuntimeEvidenceCoordinator, confirmation_intent_for_action
POLICY_SOURCES = frozenset({'teacher', 'oracle', 'recovery_oracle', 'learner', 'pending_interrupt'})

def _materialize(events) -> List[dict]:
    if not events:
        return []
    if isinstance(events, SchedulerEvent) or (hasattr(events, 'name') and hasattr(events, 'phase')):
        events = [events]
    elif isinstance(events, Mapping):
        events = [events]
    out: List[dict] = []
    for ev in events:
        if isinstance(ev, SchedulerEvent) or (hasattr(ev, 'name') and hasattr(ev, 'phase')):
            out.append({'name': ev.name, 'phase': ev.phase, 'payload': dict(getattr(ev, 'payload', {}) or {})})
        elif isinstance(ev, Mapping):
            out.append(dict(ev))
    return out

def _ensure_event_steps(events: List[dict], high_level_step: int) -> None:
    for event in events:
        payload = event.get('payload')
        if not isinstance(payload, dict):
            payload = {}
            event['payload'] = payload
        payload.setdefault('step', int(high_level_step))

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

def _dispatch_low_level_hook_events(bus: HookBus, env: Any, transition: Mapping[str, Any]) -> List[dict]:
    events: List[dict] = []
    for record in transition.get('low_level_actions') or []:
        if not isinstance(record, Mapping):
            continue
        events.extend(bus.on_low_level(env, dict(record), None))
    return events

def _collect_hook_artifacts(bus: HookBus) -> dict:
    artifacts: dict[str, dict] = {}
    for hook in bus.hooks:
        manifest = getattr(hook, 'artifacts', None)
        if not callable(manifest):
            manifest = getattr(hook, 'manifest', None)
        if callable(manifest):
            try:
                manifest = manifest()
            except TypeError:
                manifest = None
        if isinstance(manifest, Mapping):
            artifacts[str(getattr(hook, 'name', hook.__class__.__name__))] = dict(manifest)
    return artifacts

def _capture_hooks(bus: HookBus) -> List[Any]:
    return [hook for hook in bus.hooks if callable(getattr(hook, 'finalize', None))]

def _mark_scheduler_phase(bus: HookBus, env: Any, phase: str, high_level_step: int) -> None:
    for hook in bus.hooks:
        marker = getattr(hook, 'mark_scheduler_phase_start', None)
        if callable(marker):
            marker(env, phase, high_level_step)

def _safe_observation(env: Any) -> Mapping[str, Any]:
    try:
        value = env._observation()
    except Exception:
        return {}
    return value if isinstance(value, Mapping) else {}

def _public_action_transition(transition: Mapping[str, Any], *, harness_definition: Any | None=None) -> Mapping[str, Any]:
    if bool(getattr(harness_definition, 'visual_only_state_enabled', False)):
        return harness_definition.project_completed_tool_record(transition)
    fields = {'tool_name', 'arguments', 'reasoning', 'success', 'feedback', 'frames', 'num_low_level_steps', 'total_low_level_steps', 'env_feedback', 'high_level_step', 'current_room', 'visited_rooms', 'done', 'done_reason', 'status', 'interrupted', 'invalid_tool_call', 'binding', 'bindings'}
    return {key: _json_safe(transition[key]) for key in fields if key in transition}

def _synthetic_failed_transition(env: Any, tool_call: ToolCall, high_level_step: int, exc: BaseException) -> dict:
    return {'tool_name': tool_call.tool_name, 'arguments': dict(tool_call.arguments), 'reasoning': tool_call.reasoning, 'success': False, 'feedback': 'Executor raised {}: {}'.format(type(exc).__name__, exc), 'frames': 0, 'num_low_level_steps': 0, 'total_low_level_steps': int(getattr(env, 'total_low_level_steps', 0)), 'env_feedback': {'error_type': 'executor_exception', 'error_message': '{}: {}'.format(type(exc).__name__, exc)}, 'high_level_step': high_level_step, 'state_delta': None, 'goal_result': {}, 'done': True, 'done_reason': 'executor_exception', 'observation': _safe_observation(env), 'raw_execution': {'status': 'failed', 'error_type': 'executor_exception', 'error_message': '{}: {}'.format(type(exc).__name__, exc)}, 'low_level_actions': []}

def _seed_usage(env: Any, scheduler: Any, brain: Any) -> Mapping[str, Mapping[str, Any]]:
    renderer = getattr(env, 'renderer', None)
    usage = {'environment': {'consumed': True, 'consumer': 'env.reset_to_task', 'reason': None}, 'scheduler': {'consumed': not isinstance(scheduler, NoOpScheduler), 'consumer': type(scheduler).__name__ if not isinstance(scheduler, NoOpScheduler) else None, 'reason': None if not isinstance(scheduler, NoOpScheduler) else 'no_scheduler_randomness'}, 'model': {'consumed': hasattr(brain, 'adapter'), 'consumer': type(brain).__name__ if hasattr(brain, 'adapter') else None, 'reason': None if hasattr(brain, 'adapter') else 'non_model_policy'}, 'data_sampling': {'consumed': False, 'consumer': None, 'reason': 'raw_capture_no_sampling'}, 'renderer': {'consumed': renderer is not None, 'consumer': type(renderer).__name__ if renderer is not None else None, 'reason': None if renderer is not None else 'renderer_absent'}}
    env_usage = getattr(env, '_trajectory_seed_usage', None)
    if isinstance(env_usage, Mapping):
        for namespace in ('environment', 'renderer'):
            if namespace in env_usage:
                usage[namespace] = dict(env_usage[namespace])
    scheduler_usage = getattr(scheduler, '_trajectory_seed_usage', None)
    if isinstance(scheduler_usage, Mapping):
        usage['scheduler'] = dict(scheduler_usage)
    model_usage = getattr(brain, '_trajectory_seed_usage', None)
    if isinstance(model_usage, Mapping):
        usage['model'] = dict(model_usage)
    return usage

def _coerce_pending_tool_call(value: Any) -> Optional[ToolCall]:
    if isinstance(value, ToolCall):
        return value
    if not isinstance(value, Mapping):
        return None
    name = value.get('tool_name') or value.get('name') or value.get('tool')
    if not isinstance(name, str) or not name:
        return None
    args = value.get('arguments') or value.get('args') or {}
    if not isinstance(args, Mapping):
        args = {}
    return ToolCall(name, dict(args), reasoning=str(value.get('reasoning') or 'pending_tool_call'))

def _policy_source(brain: Brain, explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in POLICY_SOURCES - {'pending_interrupt'}:
            raise TrajectoryConfigError('policy_source_invalid', 'policy_source is not a policy enum')
        return explicit
    qualified = '{}.{}'.format(type(brain).__module__, type(brain).__name__).casefold()
    if 'recovery_oracle' in qualified:
        return 'recovery_oracle'
    if '.oracle' in qualified or 'oraclebrain' in qualified:
        return 'oracle'
    if '.teachers.' in qualified or 'teacher' in qualified:
        return 'teacher'
    return 'learner'

class EpisodeRunnerError(RuntimeError):
    pass

@dataclass
class EpisodeResult:
    task_id: str
    category: str
    status: str
    final_goal: dict
    metrics: dict
    transitions: List[dict] = field(default_factory=list)
    trace_path: Optional[str] = None
    failure_type: Optional[str] = None
    failure_reason: Optional[str] = None
    failure_code: Optional[str] = None
    failure_details: dict = field(default_factory=dict)
    scheduler_events: List[dict] = field(default_factory=list)
    hook_events: List[dict] = field(default_factory=list)
    hook_artifacts: dict = field(default_factory=dict)
    init_mutations: List[dict] = field(default_factory=list)
    scene_mutations: List[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0

class EpisodeRunner:

    def __init__(self, env: BehaviorClosedLoopToolEnv, *, max_high_level_steps: Optional[int]=None, raise_infrastructure_errors: bool=False, runtime_evidence_observers: tuple[Any, ...]=()):
        self.env = env
        self.max_high_level_steps = max_high_level_steps
        self.raise_infrastructure_errors = bool(raise_infrastructure_errors)
        self.runtime_evidence_observers = tuple(runtime_evidence_observers)

    def run(self, task_spec: Mapping[str, Any], *, brain: Brain, hooks: Optional[HookBus]=None, scheduler: Optional[Scheduler]=None, trace_path: Optional[Path]=None, rng_seed: int=0, episode_context: EpisodeContext | None=None, admission_preflight: Callable[..., Mapping[str, Any]] | None=None, policy_source: str | None=None) -> EpisodeResult:
        env = self.env
        bus = hooks or HookBus()
        capture_hooks = _capture_hooks(bus)
        runtime_evidence = None if capture_hooks else RuntimeEvidenceCoordinator(brain, stream_parent_style='prefixed')
        stream_evidence_observer: BehaviorStreamEvidenceObserver | None = None
        if capture_hooks and episode_context is None:
            raise TrajectoryConfigError('episode_context_required', 'trajectory capture requires EpisodeContext')
        if episode_context is not None:
            environment_seed = episode_context.seeds.get('environment')
            if isinstance(rng_seed, bool) or not isinstance(rng_seed, int) or rng_seed != environment_seed:
                raise TrajectoryConfigError('seed_mismatch', "rng_seed must exactly equal EpisodeContext.seeds['environment']")
        for capture_hook in capture_hooks:
            configure = getattr(capture_hook, 'configure_model_io', None)
            if callable(configure):
                configure(brain)
        if scheduler is None:
            scheduler = build_scheduler_from_task_spec(task_spec)
            if scheduler is None:
                scheduler = NoOpScheduler()
        start_time = time.time()
        timing_recorder = current_runtime_timing() or RuntimeTimingRecorder.from_environment(str(task_spec.get('id') or 'runtime-episode'))
        timing_token = activate_runtime_timing(timing_recorder) if timing_recorder is not None and current_runtime_timing() is None else None
        if timing_recorder is not None:
            timing_recorder.write('episode', status='start')
        transitions: List[dict] = []
        status = 'failed'
        failure_type: Optional[str] = None
        failure_reason: Optional[str] = None
        final_goal: Mapping[str, Any] = {}
        scheduler_events: List[dict] = []
        hook_events: List[dict] = []
        init_mutations: List[dict] = []
        scene_mutations: List[dict] = []
        reset_pose_diagnostics: dict = {}
        fatal_capture_error: BaseException | None = None
        runtime_error: BaseException | None = None
        teardown_error: BaseException | None = None
        executor_exception: BaseException | None = None
        budget_exhausted = False
        admission_result: Mapping[str, Any] | None = None
        pre_admission_terminal: Mapping[str, Any] | None = None
        brain_terminal_failure: Mapping[str, Any] | None = None
        brain_started = False
        resolved_policy_source = _policy_source(brain, policy_source)
        writer: Optional[EpisodeTraceWriter] = None
        if trace_path is not None:
            writer = EpisodeTraceWriter(Path(trace_path))
        try:
            env.scan_rgb_observer = getattr(brain, 'on_scan_rgb', None)
            env._trajectory_lifecycle_phase = 'reset'
            with runtime_phase('renderer_reset'):
                reset_obs = env.reset_to_task(task_spec, rng_seed=rng_seed)
            if runtime_evidence is not None:
                controller = getattr(env, 'stream_controller', None)
                add_observer = getattr(controller, 'add_observer', None)
                if runtime_evidence.enabled and callable(add_observer):
                    stream_evidence_observer = BehaviorStreamEvidenceObserver(runtime_evidence)
                    add_observer(stream_evidence_observer)
                for observer in self.runtime_evidence_observers:
                    runtime_evidence.bridge.add_observer(observer)
                runtime_evidence.reset(str(task_spec.get('id') or 'runtime-episode'))
            env._trajectory_lifecycle_phase = 'initial_goal'
            initial_goal = env.check_goal(model_stopped=False)
            env._trajectory_lifecycle_phase = 'episode_start_capture'
            from scoper.benchmark.common.composite_evaluation import enabled
            if enabled(task_spec):
                env.current_composite_subtask_id = None
                env._composite_active_task_spec = None
            start_hook_events = bus.on_episode_start(env, task_spec, reset_obs)
            hook_events.extend(start_hook_events)
            if admission_preflight is None and (isinstance(brain, BaseBrain) or hasattr(brain, 'on_episode_start')):
                brain.on_episode_start(env, task_spec)
                brain_started = True
            from scoper.benchmark.common.composite_evaluation import CompositeEvaluationProgress, enabled
            composite_progress = None
            if enabled(task_spec) and getattr(env, 'current_composite_subtask_id', None) is None:
                from scoper.benchmark.behavior.brains.teachers.composite import _build_child_task_spec, _temporarily_bind_task
                children = [dict(_build_child_task_spec(task_spec, child), subtask_id=child['subtask_id']) for child in task_spec['family_spec']['composite']['subtasks']]

                def check_child(child):
                    with _temporarily_bind_task(env, child):
                        return env.check_goal(model_stopped=False)

                def activate_child(child):
                    env.current_composite_subtask_id = None if child is None else child['subtask_id']
                composite_progress = CompositeEvaluationProgress(children, check_child, activate_child)
            _mark_scheduler_phase(bus, env, 'episode_start', 0)
            start_events = _materialize(scheduler.on_episode_start(env, task_spec))
            _ensure_event_steps(start_events, 0)
            scheduler_events.extend(start_events)
            scheduler_hook_events = bus.on_scheduler_events(env, start_events, 'episode_start', 0)
            hook_events.extend(scheduler_hook_events)
            if start_events and (not capture_hooks) and hasattr(env, 'emit_stream_event_keyframe'):
                try:
                    env.emit_stream_event_keyframe(source='scheduler_event', keyframe_reason='episode_start', active_tool_name='episode_start', active_tool_args={'event_count': len(start_events)}, prompt_metadata={'scheduler_events': start_events, 'phase': 'episode_start'}, privileged_metadata={'scheduler_events': start_events})
                except Exception:
                    pass
            init_mutations = _json_safe(list(getattr(env, 'init_mutations', None) or reset_obs.get('init_mutations') or []))
            scene_mutations = _json_safe(list(getattr(env, 'scene_mutations', None) or reset_obs.get('scene_mutations') or []))
            diagnostics_fn = getattr(env, 'reset_pose_diagnostics', None)
            if callable(diagnostics_fn):
                reset_pose_diagnostics = _json_safe(diagnostics_fn())
            if writer is not None:
                writer.write_meta({'task_id': task_spec['id'], 'category': task_spec['category'], 'instruction': task_spec.get('instruction', ''), 'scene_model': task_spec['source'].get('scene_model'), 'runner': {'max_high_level_steps': self.max_high_level_steps or env.max_high_level_steps, 'rng_seed': int(rng_seed)}, 'reset': {'agent_pose': env._agent_pose(), 'pose_diagnostics': reset_pose_diagnostics, 'initial_goal': initial_goal, 'placement_report': _json_safe(getattr(env.scene_result, 'placement_report', {})), 'init_mutations': init_mutations, 'scene_mutations': scene_mutations}, 'scheduler_events': start_events, 'hook_events': start_hook_events + scheduler_hook_events, 'tool_specs': env.get_tool_specs(model_visible_only=True), 'allowed_model_tools': list(MODEL_VISIBLE_TOOLS)})
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
            if admission_preflight is not None and pre_admission_terminal is None:
                if episode_context is None:
                    raise TrajectoryConfigError('episode_context_required', 'admission preflight requires EpisodeContext')
                env._trajectory_lifecycle_phase = 'reachability_preflight'
                admission_result = admission_preflight(env, task_spec, episode_context=episode_context)
                from scoper.benchmark.behavior.benchmark.admission_schema import validate_reachability_preflight
                validate_reachability_preflight(admission_result)
                env._trajectory_admission_preflight = _json_safe(admission_result)
                for capture_hook in capture_hooks:
                    recorder = getattr(capture_hook, 'record_admission_preflight', None)
                    if callable(recorder):
                        recorder(admission_result)
                if admission_result['status'] != 'pass':
                    status = 'invalid'
                    failure_type = str(admission_result.get('error_class') or 'task_invalid')
                    failure_reason = str(admission_result.get('failure_code') or 'admission evidence is invalid')
            if admission_preflight is not None and pre_admission_terminal is None and (admission_result is None or admission_result['status'] == 'pass'):
                env._trajectory_lifecycle_phase = 'rendered_observation_activation'
                activate_rendered_evidence = getattr(env, 'activate_rendered_observation_evidence', None)
                if callable(activate_rendered_evidence):
                    activate_rendered_evidence()
                activate_stream = getattr(env, 'activate_stream_runtime', None)
                if callable(activate_stream):
                    env._trajectory_lifecycle_phase = 'stream_activation'
                    activate_stream(emit_reset_keyframe=False)
                if isinstance(brain, BaseBrain) or hasattr(brain, 'on_episode_start'):
                    brain.on_episode_start(env, task_spec)
                    brain_started = True
            max_steps = self.max_high_level_steps or env.max_high_level_steps
            env.max_high_level_steps = max_steps
            last_transition: Optional[dict] = None
            loop_iterations = 0
            loop_ceiling = max_steps
            pending_tool_queue: List[dict] = []
            while pre_admission_terminal is None and (admission_result is None or admission_result['status'] == 'pass'):
                env._trajectory_lifecycle_phase = 'episode_execution'
                pending_qa = bool((getattr(env, '_human_qa_state', {}) or {}).get('pending'))
                if loop_iterations >= loop_ceiling or (int(getattr(env, 'task_action_step', env.high_level_step)) >= max_steps and (not pending_qa)):
                    budget_exhausted = True
                    failure_type = 'EpisodeRunnerError'
                    failure_reason = 'Exceeded max high-level steps ({}).'.format(max_steps)
                    break
                loop_iterations += 1
                next_step = loop_iterations
                _mark_scheduler_phase(bus, env, 'pre_step', next_step)
                pre_events = _materialize([] if evaluation_lifecycle.finishing(env) else scheduler.pre_step(env, next_step, last_transition))
                _ensure_event_steps(pre_events, next_step)
                scheduler_events.extend(pre_events)
                pre_hook_events = bus.on_scheduler_events(env, pre_events, 'pre_step', next_step)
                hook_events.extend(pre_hook_events)
                if pre_events and (not capture_hooks) and hasattr(env, 'emit_stream_event_keyframe'):
                    try:
                        env.emit_stream_event_keyframe(source='scheduler_event', keyframe_reason='pre_step', active_tool_name='scheduler_pre_step', active_tool_args={'event_count': len(pre_events), 'step': next_step}, prompt_metadata={'scheduler_events': pre_events, 'phase': 'pre_step'}, privileged_metadata={'scheduler_events': pre_events})
                    except Exception:
                        pass
                pending_record = pending_tool_queue.pop(0) if pending_tool_queue else None
                if pending_record is not None:
                    tool_call = _coerce_pending_tool_call(pending_record.get('tool_call') or pending_record)
                    if tool_call is None:
                        raise EpisodeRunnerError('Pending tool call was malformed: {}.'.format(pending_record))
                    brain_source = 'pending_interrupt'
                    brain_input = None
                else:
                    observation = env._observation()
                    raw_brain_input = BrainInput(observation=observation, prompt_context={**env.get_prompt_context(observation), 'evaluation_step_budget': max_steps}, last_transition=last_transition, high_level_step=next_step)
                    from scoper.benchmark.common.public_binding_events import sync as sync_public_binding_events
                    sync_public_binding_events(env, raw_brain_input.prompt_context)
                    if runtime_evidence is not None and runtime_evidence.enabled:
                        runtime_evidence.prepare_decision(step=next_step, decision_id=decision_id(next_step), observation=observation)
                    prepare_input = getattr(brain, 'prepare_input', None)
                    brain_input = prepare_input(raw_brain_input) if callable(prepare_input) else raw_brain_input
                    if not isinstance(brain_input, BrainInput):
                        raise EpisodeRunnerError('Brain.prepare_input must return BrainInput, got {}.'.format(type(brain_input).__name__))
                    brain_source = resolved_policy_source
                decision_identifier = decision_id(next_step)
                action_identifier = action_id(next_step)
                source_stream_obs_id = None
                input_frame_ids: List[str] = []
                if pending_record is not None:
                    source_stream_obs_id = pending_record.get('stream_obs_id')
                    if isinstance(pending_record.get('frame_id'), str):
                        input_frame_ids = [pending_record['frame_id']]
                model_context = None
                if brain_input is not None and hasattr(brain, 'set_model_call_context'):
                    for capture_hook in capture_hooks:
                        reserve = getattr(capture_hook, 'reserve_model_call_context', None)
                        if callable(reserve):
                            model_context = reserve(call_type='step_decision', decision_identifier=decision_identifier, action_identifier=None)
                            break
                decision_context = {'step_id': next_step, 'decision_id': decision_identifier, 'input_kind': 'pending_interrupt' if pending_record is not None else 'brain_act', 'policy_source': brain_source, 'pending_record': _json_safe(pending_record) if pending_record is not None else None, 'source_stream_obs_id': source_stream_obs_id, 'input_frame_ids': input_frame_ids, 'model_call_id': model_context.model_call_id if model_context is not None else None}
                decision_hook_events = bus.on_decision_input(env, brain_input, next_step, decision_context)
                hook_events.extend(decision_hook_events)
                if pending_record is None:
                    if runtime_evidence is not None:
                        runtime_evidence.note_policy_call()
                    if model_context is not None:
                        brain.set_model_call_context(model_context)
                    try:
                        try:
                            with runtime_phase('policy'):
                                tool_call = brain.act(brain_input)
                        except Exception:
                            terminal = getattr(brain, 'terminal_failure', None)
                            if not isinstance(terminal, Mapping) or not terminal:
                                raise
                            brain_terminal_failure = _json_safe(terminal)
                    finally:
                        if model_context is not None:
                            brain.set_model_call_context(None)
                    if brain_terminal_failure is not None:
                        status = 'failed'
                        failure_type = str(brain_terminal_failure.get('error_class') or 'agent_failure')
                        failure_reason = str(brain_terminal_failure.get('error_code') or brain_terminal_failure.get('message') or 'method_terminal_failure')
                        break
                    if not isinstance(tool_call, ToolCall):
                        raise EpisodeRunnerError('Brain.act must return a ToolCall, got {}.'.format(type(tool_call).__name__))
                public_postcondition_pre = None
                harness_definition = getattr(brain, 'harness_definition', None)
                if harness_definition is not None and harness_definition.legacy_simulator_postconditions_enabled:
                    from scoper.integration.behavior.public_postconditions import capture_public_action_state
                    public_postcondition_pre = capture_public_action_state(env)
                before_events = bus.before_tool(env, tool_call, next_step)
                hook_events.extend(before_events)

                def _default_call(_tc: ToolCall=tool_call) -> dict:
                    with runtime_phase('tool_execution', tool_name=_tc.tool_name):
                        return env.call_tool(_tc.tool_name, dict(_tc.arguments), reasoning=_tc.reasoning)
                _mark_scheduler_phase(bus, env, 'intercept', next_step)
                intercept_events: List[dict] = []
                try:
                    intercepted = None if evaluation_lifecycle.finishing(env) else scheduler.intercept_tool_call(env, tool_call, next_step, _default_call)
                    if intercepted is not None:
                        transition = dict(intercepted)
                        intercept_events = _materialize(transition.pop('scheduler_intercept', None))
                    else:
                        transition = _default_call()
                        intercept_events = _materialize(transition.pop('scheduler_intercept', None))
                except Exception as exc:
                    if capture_hooks and type(exc).__name__ == 'TrajectoryCaptureError':
                        raise
                    if self.raise_infrastructure_errors:
                        raise
                    executor_exception = exc
                    transition = _synthetic_failed_transition(env, tool_call, next_step, exc)
                _ensure_event_steps(intercept_events, next_step)
                scheduler_events.extend(intercept_events)
                intercept_hook_events = bus.on_scheduler_events(env, intercept_events, 'intercept', next_step)
                hook_events.extend(intercept_hook_events)
                if intercept_events and (not capture_hooks) and hasattr(env, 'emit_stream_event_keyframe'):
                    try:
                        env.emit_stream_event_keyframe(source='scheduler_event', keyframe_reason='intercept', active_tool_name='scheduler_intercept', active_tool_args={'event_count': len(intercept_events), 'step': next_step}, prompt_metadata={'scheduler_events': intercept_events, 'phase': 'intercept'}, privileged_metadata={'scheduler_events': intercept_events})
                    except Exception:
                        pass
                low_level_hook_events = _dispatch_low_level_hook_events(bus, env, transition)
                hook_events.extend(low_level_hook_events)
                immediate_observation = dict(transition.get('observation')) if isinstance(transition.get('observation'), Mapping) else {}
                confirmation_observation: Mapping[str, Any] | None = None
                restored_observation = immediate_observation
                harness_definition = getattr(brain, 'harness_definition', None)
                harness_spec = getattr(harness_definition, 'spec', harness_definition)
                confirmation_enabled = 'three_frame_visual_verification' in tuple(getattr(harness_spec, 'capabilities', ())) and (not (isinstance(transition.get('raw_execution'), Mapping) and transition['raw_execution'].get('action_executed') is False))
                immediate_events: List[dict] = []
                if confirmation_enabled and tool_call.tool_name in CONFIRMABLE_ACTIONS:
                    immediate_events = bus.capture_immediate_post_action(env, tool_call, transition, next_step)
                    private_getter = getattr(env, 'last_private_bound_target', None)
                    private_target = private_getter() if callable(private_getter) else None
                    public_target = (tool_call.arguments or {}).get('target')
                    intent = confirmation_intent_for_action(tool_call.tool_name, target_public_ref_or_null=public_target if isinstance(public_target, str) and public_target else None, has_bound_target=isinstance(private_target, Mapping))
                    try:
                        capture_confirmation = getattr(env, 'capture_confirmation', None)
                        if not callable(capture_confirmation):
                            raise EpisodeRunnerError('v12 backend lacks capture_confirmation')
                        with runtime_phase('confirmation', tool_name=tool_call.tool_name):
                            confirmation = capture_confirmation(intent, private_target, video_fps=None)
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
                    transition['observation'] = restored_observation
                after_events = bus.after_tool(env, tool_call, transition, next_step)
                public_postcondition_evidence = None
                if public_postcondition_pre is not None:
                    from scoper.integration.behavior.public_postconditions import build_public_postcondition_evidence, capture_public_action_state
                    public_postcondition_evidence = build_public_postcondition_evidence(public_postcondition_pre, capture_public_action_state(env))
                hook_events.extend(immediate_events)
                hook_events.extend(after_events)
                if runtime_evidence is not None and runtime_evidence.enabled:
                    final_observation = transition.get('observation') if isinstance(transition.get('observation'), Mapping) else {}
                    runtime_evidence.record_final_stream(step=next_step, rgb=final_observation.get('head_rgb'), public_observation=final_observation, action_id=action_identifier)
                    if confirmation_enabled and tool_call.tool_name in CONFIRMABLE_ACTIONS and (confirmation_observation is not None):
                        runtime_evidence.prepare_action_result(step=next_step, action_id=action_identifier, immediate_post_observation=immediate_observation, confirmation_observation=confirmation_observation)
                    else:
                        post_observation = transition.get('observation') if isinstance(transition.get('observation'), Mapping) else {}
                        runtime_evidence.prepare_action_result(step=next_step, action_id=action_identifier, post_observation=post_observation)
                on_action_result = getattr(brain, 'on_action_result', None)
                if callable(on_action_result):
                    action_result_observation = transition.get('observation') if isinstance(transition.get('observation'), Mapping) else {}
                    if bool(getattr(harness_definition, 'visual_only_state_enabled', False)):
                        action_result_observation = {'head_rgb': action_result_observation.get('head_rgb')}
                    with runtime_phase('verifier_memory', tool_name=tool_call.tool_name):
                        on_action_result(ActionResultContext(step_id=next_step, decision_id=decision_identifier, action_id=action_identifier, tool_call=tool_call, public_transition=_public_action_transition(transition, harness_definition=harness_definition), pre_public_observation=brain_input.observation if brain_input is not None else {}, post_public_observation=action_result_observation, public_postcondition_evidence_or_null=public_postcondition_evidence, immediate_post_public_observation_or_null=({'head_rgb': immediate_observation.get('head_rgb')} if bool(getattr(harness_definition, 'visual_only_state_enabled', False)) else immediate_observation) if confirmation_enabled and tool_call.tool_name in CONFIRMABLE_ACTIONS else None, confirmation_public_observation_or_null=({'head_rgb': confirmation_observation.get('head_rgb')} if bool(getattr(harness_definition, 'visual_only_state_enabled', False)) else confirmation_observation) if confirmation_enabled and tool_call.tool_name in CONFIRMABLE_ACTIONS and (confirmation_observation is not None) else None, private_binding_roles_or_null=tuple(env.last_private_binding_roles()) if callable(getattr(env, 'last_private_binding_roles', None)) else None))
                _mark_scheduler_phase(bus, env, 'post_step', next_step)
                post_events = _materialize([] if evaluation_lifecycle.finishing(env) else scheduler.post_step(env, next_step, transition))
                _ensure_event_steps(post_events, next_step)
                scheduler_events.extend(post_events)
                post_hook_events = bus.on_scheduler_events(env, post_events, 'post_step', next_step)
                hook_events.extend(post_hook_events)
                if post_events and (not capture_hooks) and hasattr(env, 'emit_stream_event_keyframe'):
                    try:
                        env.emit_stream_event_keyframe(source='scheduler_event', keyframe_reason='post_step', active_tool_name='scheduler_post_step', active_tool_args={'event_count': len(post_events), 'step': next_step}, prompt_metadata={'scheduler_events': post_events, 'phase': 'post_step'}, privileged_metadata={'scheduler_events': post_events})
                    except Exception:
                        pass
                evaluation_lifecycle.after_transition(env, transition)
                if transition.get('done') and (getattr(env, '_human_qa_state', {}) or {}).get('pending'):
                    transition['done'] = False
                    transition['done_reason'] = None
                transitions.append(transition)
                last_transition = transition
                pending_tool_call = _coerce_pending_tool_call(transition.get('pending_tool_call'))
                if pending_tool_call is not None:
                    stream_interrupt = transition.get('stream_interrupt')
                    stream_interrupt = stream_interrupt if isinstance(stream_interrupt, Mapping) else {}
                    pending_tool_queue.append({'source': 'stream_interrupt', 'stream_id': stream_interrupt.get('stream_id'), 'stream_obs_id': stream_interrupt.get('stream_obs_id'), 'frame_id': stream_interrupt.get('frame_id'), 'interrupt_frame_index': transition.get('interrupt_frame_index'), 'interrupt_timestamp_s': transition.get('interrupt_timestamp_s'), 'tool_call': {'tool_name': pending_tool_call.tool_name, 'arguments': dict(pending_tool_call.arguments), 'reasoning': pending_tool_call.reasoning}, 'reasoning': pending_tool_call.reasoning})
                if composite_progress is not None:
                    composite_progress.after_step(transition, pending_qa=bool((getattr(env, '_human_qa_state', {}) or {}).get('pending')))
                step_hook_events = pre_hook_events + decision_hook_events + before_events + intercept_hook_events + low_level_hook_events + after_events + post_hook_events
                step_scheduler_events = pre_events + intercept_events + post_events
                if writer is not None:
                    writer.write_step({'high_level_step': next_step, 'brain': {'tool_name': tool_call.tool_name, 'arguments': dict(tool_call.arguments), 'reasoning': tool_call.reasoning, 'source': brain_source, 'pending_record': _json_safe(pending_record) if pending_record else None}, 'transition': _json_safe(transition), 'hook_events': _json_safe(step_hook_events), 'scheduler_events': _json_safe(step_scheduler_events)})
                if transition.get('done'):
                    break
                if executor_exception is not None:
                    break
            final_goal = env.check_goal(model_stopped=env.model_stopped)
            if pre_admission_terminal is not None:
                pass
            elif admission_result is not None and admission_result['status'] != 'pass':
                pass
            elif executor_exception is not None:
                status = 'failed'
                failure_type = type(executor_exception).__name__
                failure_reason = str(executor_exception)
            elif brain_terminal_failure is not None:
                status = 'failed'
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
            if capture_hooks:
                for capture_hook in capture_hooks:
                    abort = getattr(capture_hook, 'abort', None)
                    if callable(abort):
                        abort({'error_class': 'infra_error', 'error_code': 'runner_capture_abort', 'message': '{}: {}'.format(type(exc).__name__, exc)})
            try:
                final_goal = env.check_goal(model_stopped=env.model_stopped)
            except Exception:
                final_goal = {}
        elapsed = time.time() - start_time
        if fatal_capture_error is None and runtime_error is None:
            _mark_scheduler_phase(bus, env, 'episode_end', getattr(env, 'high_level_step', 0))
        try:
            if fatal_capture_error is not None:
                raise fatal_capture_error
            end_events = [] if runtime_error is not None else _materialize(scheduler.on_episode_end(env, final_goal))
            _ensure_event_steps(end_events, getattr(env, 'high_level_step', 0))
            scheduler_events.extend(end_events)
            end_scheduler_hook_events = bus.on_scheduler_events(env, end_events, 'episode_end', getattr(env, 'high_level_step', 0))
            hook_events.extend(end_scheduler_hook_events)
            if not capture_hooks and hasattr(env, 'emit_stream_event_keyframe'):
                try:
                    env.emit_stream_event_keyframe(source='episode_end', keyframe_reason='episode_end', active_tool_name='episode_end', active_tool_args={'event_count': len(end_events), 'status': status}, prompt_metadata={'scheduler_events': end_events, 'phase': 'episode_end'}, privileged_metadata={'scheduler_events': end_events, 'final_goal': dict(final_goal)})
                except Exception:
                    pass
        except Exception as exc:
            teardown_error = teardown_error or exc
            if capture_hooks:
                for capture_hook in capture_hooks:
                    abort = getattr(capture_hook, 'abort', None)
                    if callable(abort):
                        abort({'error_class': 'infra_error', 'error_code': 'terminal_scheduler_capture_error', 'message': '{}: {}'.format(type(exc).__name__, exc)})
        terminal_done_reason = (str(pre_admission_terminal['done_reason']) if pre_admission_terminal is not None else _admission_done_reason(admission_result)) or (str(brain_terminal_failure.get('error_code')) if brain_terminal_failure is not None else None) or (transitions[-1].get('done_reason') if transitions else None)
        terminal_override = pre_admission_terminal
        if terminal_override is None and brain_terminal_failure is not None:
            terminal_override = {'done_reason': terminal_done_reason, 'error_class': brain_terminal_failure.get('error_class') or 'agent_failure', 'error_code': brain_terminal_failure.get('error_code')}
        env._trajectory_terminal_override = _json_safe(terminal_override or {})
        metrics = compute_episode_metrics(transitions=transitions, final_goal=final_goal, done_reason=terminal_done_reason, model_stopped=bool(getattr(env, 'model_stopped', False)), elapsed_seconds=elapsed, scheduler_events=scheduler_events, task_spec=task_spec)
        metrics.update(evaluation_lifecycle.diagnostics(env))
        environment_metrics_builder = getattr(env, '_episode_metrics', None)
        environment_metrics = environment_metrics_builder() if callable(environment_metrics_builder) else {}
        for metric_name in ('binding_metrics', 'navigation_metrics'):
            metric_value = environment_metrics.get(metric_name)
            if isinstance(metric_value, Mapping):
                metrics[metric_name] = copy.deepcopy(dict(metric_value))
        if hasattr(brain, 'direct_decision_records'):
            metrics['direct_decisions'] = list(brain.direct_decision_records)
        failure_report = build_failure_report(transitions=transitions, final_goal=final_goal, failure_type=failure_type, failure_reason=failure_reason, scheduler_events=scheduler_events, task_spec=task_spec, model_stopped=bool(getattr(env, 'model_stopped', False)))
        try:
            end_hook_events = [] if runtime_error is not None else bus.on_episode_end(env, final_goal, status)
            hook_events.extend(end_hook_events)
        except Exception as exc:
            teardown_error = teardown_error or exc
            end_hook_events = []
        stream_controller = getattr(env, 'stream_controller', None)
        if brain_started and stream_controller is not None and hasattr(stream_controller, 'on_stream_end'):
            try:
                stream_controller.on_stream_end({'final_goal': dict(final_goal), 'status': status})
            except Exception:
                pass
        if brain_started:
            try:
                brain.on_episode_end(env, final_goal)
            except Exception as exc:
                teardown_error = teardown_error or exc
        for capture_hook in capture_hooks:
            if runtime_error is not None:
                continue
            if pre_admission_terminal is not None:
                terminal_error_class = str(pre_admission_terminal['error_class'])
                terminal_error_code = str(pre_admission_terminal['error_code'])
            elif admission_result is not None and admission_result['status'] != 'pass':
                terminal_error_class = str(admission_result['error_class'])
                terminal_error_code = admission_result.get('failure_code')
            elif brain_terminal_failure is not None:
                terminal_error_class = str(brain_terminal_failure.get('error_class') or 'agent_failure')
                terminal_error_code = brain_terminal_failure.get('error_code')
            else:
                terminal_error_class = 'infra_error' if executor_exception is not None else 'none' if status == 'passed' else 'agent_failure'
                terminal_error_code = 'executor_exception' if executor_exception is not None else None
            capture_hook.finalize({'episode_status': status, 'done_reason': terminal_done_reason or failure_reason, 'error_class': terminal_error_class, 'error_code': terminal_error_code, 'seed_usage': _seed_usage(env, scheduler, brain)})
        hook_artifacts = _collect_hook_artifacts(bus)
        if runtime_evidence is not None:
            hook_artifacts['runtime_evidence'] = dict(runtime_evidence.artifacts())
        if writer is not None:
            writer.write_summary({'status': status, 'final_goal': dict(final_goal), 'metrics': metrics, 'reset_pose_diagnostics': reset_pose_diagnostics, 'failure_type': failure_type, 'failure_reason': failure_reason, 'failure_code': failure_report.get('failure_code'), 'failure_details': failure_report.get('failure_details') or {}, 'scheduler_events': scheduler_events, 'hook_events': _json_safe(hook_events), 'hook_artifacts': _json_safe(hook_artifacts), 'init_mutations': _json_safe(init_mutations), 'scene_mutations': _json_safe(scene_mutations), 'elapsed_seconds': elapsed})
            writer.close()
        if stream_evidence_observer is not None:
            controller = getattr(env, 'stream_controller', None)
            remove_observer = getattr(controller, 'remove_observer', None)
            if callable(remove_observer):
                remove_observer(stream_evidence_observer)
        env.scan_rgb_observer = None
        pending_error = runtime_error or teardown_error
        if timing_recorder is not None:
            timing_recorder.write('episode', status='complete' if pending_error is None else 'error', episode_status=status)
        if timing_token is not None:
            deactivate_runtime_timing(timing_token)
        if pending_error is not None and (self.raise_infrastructure_errors or capture_hooks):
            from scoper.benchmark.common.direct_response import preserve_partial_episode
            preserve_partial_episode(pending_error, transitions, metrics, final_goal)
            pending_error._behavior_scene_reuse_receipt_complete = teardown_error is None
            raise pending_error
        return EpisodeResult(task_id=task_spec['id'], category=task_spec['category'], status=status, final_goal=dict(final_goal), metrics=metrics, transitions=transitions, trace_path=str(trace_path) if trace_path is not None else None, failure_type=failure_type, failure_reason=failure_reason, failure_code=pre_admission_terminal.get('error_code') if pre_admission_terminal is not None else admission_result.get('failure_code') if admission_result is not None and admission_result['status'] != 'pass' else brain_terminal_failure.get('error_code') if brain_terminal_failure is not None else failure_report.get('failure_code'), failure_details={'admission_terminal': _json_safe(pre_admission_terminal)} if pre_admission_terminal is not None else {'admission_preflight': _json_safe(admission_result)} if admission_result is not None and admission_result['status'] != 'pass' else {'method_terminal_failure': _json_safe(brain_terminal_failure)} if brain_terminal_failure is not None else dict(failure_report.get('failure_details') or {}), scheduler_events=scheduler_events, hook_events=hook_events, hook_artifacts=hook_artifacts, init_mutations=list(init_mutations or []), scene_mutations=list(scene_mutations or []), elapsed_seconds=elapsed)

def _admission_done_reason(result: Mapping[str, Any] | None) -> str | None:
    if result is None or result.get('status') == 'pass':
        return None
    if result.get('failure_code') == 'ADMISSION_UNREACHABLE':
        return 'admission_unreachable'
    if result.get('error_class') == 'evaluator_invalid':
        return 'admission_evaluator_invalid'
    return 'admission_infra_error'
__all__ = ['EpisodeResult', 'EpisodeRunner', 'EpisodeRunnerError', 'POLICY_SOURCES']
