from __future__ import annotations
from typing import Any, Callable, List, Mapping, Optional
from scoper.benchmark.behavior.scheduler.base import NoOpScheduler, Scheduler, SchedulerEvent
from scoper.benchmark.behavior.scheduler.disturbance import DisturbanceConfig, DisturbanceScheduler, KNOWN_DISTURBANCE_TYPES
from scoper.benchmark.behavior.scheduler.failure_injection import FailureConfig, FailureInjectionScheduler, KNOWN_FAILURE_MODES
from scoper.benchmark.behavior.scheduler.human_interaction import HumanEventConfig, HumanInteractionScheduler

class CompositeScheduler(Scheduler):
    name = 'composite'

    def __init__(self, children: list) -> None:
        self._children = list(children)

    def on_episode_start(self, env, task_spec) -> Optional[List[SchedulerEvent]]:
        events = []
        for child in self._children:
            r = child.on_episode_start(env, task_spec)
            if r:
                events.extend(r)
        return events or None

    def pre_step(self, env, high_level_step, last_transition) -> Optional[List[SchedulerEvent]]:
        events = []
        for child in self._children:
            r = child.pre_step(env, high_level_step, last_transition)
            if r:
                events.extend(r)
        return events or None

    def intercept_tool_call(self, env, tool_call, high_level_step, default_call) -> Optional[dict]:
        for child in self._children:
            result = child.intercept_tool_call(env, tool_call, high_level_step, default_call)
            if result is not None:
                return result
        return None

    def post_step(self, env, high_level_step, transition) -> Optional[List[SchedulerEvent]]:
        events = []
        for child in self._children:
            r = child.post_step(env, high_level_step, transition)
            if r:
                events.extend(r)
        return events or None

    def on_episode_end(self, env, final_goal) -> Optional[List[SchedulerEvent]]:
        events = []
        for child in self._children:
            r = child.on_episode_end(env, final_goal)
            if r:
                events.extend(r)
        return events or None

def build_scheduler_from_task_spec(task_spec: Mapping[str, Any]) -> Scheduler:
    children: list = []
    dist_spec, fail_spec, human_events, seed = normalize_scheduler_specs(task_spec)
    if dist_spec:
        children.append(DisturbanceScheduler(dist_spec, seed=seed))
    if fail_spec:
        children.append(FailureInjectionScheduler(fail_spec, seed=seed))
    qa_overlay = task_spec.get('qa_overlay')
    if human_events or isinstance(qa_overlay, Mapping):
        children.append(HumanInteractionScheduler(human_events, qa_overlay=qa_overlay if isinstance(qa_overlay, Mapping) else None))
    if not children:
        return NoOpScheduler()
    if len(children) == 1:
        return children[0]
    return CompositeScheduler(children)

def normalize_scheduler_specs(task_spec: Mapping[str, Any], *, default_seed: int=0) -> tuple[list[dict], list[dict], list[dict], int]:
    seed = _int_or_default(task_spec.get('scheduler_seed'), default_seed)
    disturbances: list[dict] = []
    failures: list[dict] = []
    human_events: list[dict] = []
    dist_spec = task_spec.get('disturbance_spec')
    if isinstance(dist_spec, Mapping):
        seed = _int_or_default(dist_spec.get('seed'), seed)
        if isinstance(dist_spec.get('disturbances'), list):
            disturbances.extend(_mapping_list(dist_spec.get('disturbances')))
        elif dist_spec.get('type') is not None:
            disturbances.append(dict(dist_spec))
        if isinstance(dist_spec.get('failures'), list):
            failures.extend(_mapping_list(dist_spec.get('failures')))
    elif isinstance(dist_spec, list):
        disturbances.extend(_mapping_list(dist_spec))
    fail_spec = task_spec.get('failure_spec')
    if isinstance(fail_spec, Mapping):
        seed = _int_or_default(fail_spec.get('seed'), seed)
        if isinstance(fail_spec.get('failures'), list):
            failures.extend(_mapping_list(fail_spec.get('failures')))
        elif isinstance(fail_spec.get('configs'), list):
            failures.extend(_mapping_list(fail_spec.get('configs')))
        elif fail_spec.get('mode') is not None:
            failures.append(dict(fail_spec))
    elif isinstance(fail_spec, list):
        failures.extend(_mapping_list(fail_spec))
    family = task_spec.get('family_spec')
    if isinstance(family, Mapping):
        human = family.get('human')
        if isinstance(human, Mapping):
            events = human.get('human_events')
            if not isinstance(events, list):
                events = human.get('human_events_template')
            if isinstance(events, list):
                human_events.extend(_mapping_list(events))
    return (disturbances, failures, human_events, seed)

def _normalized_scheduler_specs(task_spec: Mapping[str, Any]) -> tuple[list[dict], list[dict], list[dict], int]:
    return normalize_scheduler_specs(task_spec)

def _mapping_list(values: Any) -> list[dict]:
    return [dict(item) for item in values or [] if isinstance(item, Mapping)]

def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)
__all__ = ['CompositeScheduler', 'DisturbanceConfig', 'DisturbanceScheduler', 'FailureConfig', 'FailureInjectionScheduler', 'HumanEventConfig', 'HumanInteractionScheduler', 'KNOWN_DISTURBANCE_TYPES', 'KNOWN_FAILURE_MODES', 'NoOpScheduler', 'Scheduler', 'SchedulerEvent', 'build_scheduler_from_task_spec', 'normalize_scheduler_specs']
