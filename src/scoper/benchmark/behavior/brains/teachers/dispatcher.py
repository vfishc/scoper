from __future__ import annotations
from typing import Any, Mapping
from scoper.benchmark.behavior.brains.oracle import can_solve_goal_predicates, unsupported_goal_predicates
LEGACY_TEACHER_CATEGORIES = frozenset({'static_simple', 'static_place', 'hidden_precondition', 'prepare_heat', 'completion_timing', 'dynamic_state_change', 'disturbance_recovery', 'mid_execution_discovery'})
FAMILY_TEACHER_CATEGORIES = frozenset({'prepare_then_place', 'prepare_multi_stage', 'put_away_by_category', 'put_away_visual_judgment', 'put_away_cross_room', 'verify_after_action', 'mid_execution_state_disturbance', 'recover_and_store_incident_object', 'recover_multi_incident', 'tidy_by_policy', 'human_interactive_execution', 'explore_then_act', 'observation_conditioned_room_inspection', 'observation_conditioned_task', 'composite'})

class TeacherDispatcherError(ValueError):
    pass

class TeacherDispatcher:

    def __init__(self) -> None:
        from scoper.benchmark.behavior.brains.oracle import OracleBrain
        self._legacy_cls = OracleBrain
        self._family: dict[str, Any] = {}

    def register(self, category: str, teacher_cls: Any) -> None:
        self._family[category] = teacher_cls

    def select(self, task_spec: Mapping[str, Any]):
        if (task_spec.get('template') or {}).get('template_id') == 'inspect_and_restore_storage_container_001':
            from scoper.benchmark.common.container_inspection.teacher import teacher_class
            return (teacher_class('behavior')(), 'conditional_storage_oracle.v1')
        category = task_spec.get('category')
        if category in self._family:
            return (self._family[category](), 'family:{}'.format(category))
        if category in LEGACY_TEACHER_CATEGORIES:
            return (self._legacy_cls(), 'legacy:{}'.format(category))
        if category in FAMILY_TEACHER_CATEGORIES:
            unsupported = unsupported_goal_predicates(task_spec)
            raise TeacherDispatcherError("No family teacher registered for category '{}'; unsupported goal predicates: {}.".format(category, ', '.join(unsupported) if unsupported else '-'))
        if can_solve_goal_predicates(task_spec):
            return (self._legacy_cls(), 'predicate_fallback')
        unsupported = unsupported_goal_predicates(task_spec)
        raise TeacherDispatcherError("No teacher registered for category '{}'; unsupported goal predicates: {}.".format(category, ', '.join(unsupported) if unsupported else '-'))

    def build(self, task_spec: Mapping[str, Any]):
        brain, _ = self.select(task_spec)
        return brain

def default_dispatcher() -> TeacherDispatcher:
    from scoper.benchmark.behavior.brains.teachers.composite import CompositeTeacher
    from scoper.benchmark.behavior.brains.teachers.explore import ExploreTeacher
    from scoper.benchmark.behavior.brains.teachers.human_interaction import HumanInteractionTeacher
    from scoper.benchmark.behavior.brains.teachers.mid_execution_disturbance import MidExecutionDisturbanceTeacher
    from scoper.benchmark.behavior.brains.teachers.observation_conditioned import ObservationConditionedTeacher
    from scoper.benchmark.behavior.brains.teachers.prepare import PrepareTeacher
    from scoper.benchmark.behavior.brains.teachers.put_away import PutAwayTeacher
    from scoper.benchmark.behavior.brains.teachers.recover import RecoverTeacher
    from scoper.benchmark.behavior.brains.teachers.tidy import TidyTeacher
    from scoper.benchmark.behavior.brains.teachers.verify_after_action import VerifyAfterActionTeacher
    dispatcher = TeacherDispatcher()
    dispatcher.register('prepare_then_place', PrepareTeacher)
    dispatcher.register('prepare_multi_stage', PrepareTeacher)
    dispatcher.register('put_away_by_category', PutAwayTeacher)
    dispatcher.register('put_away_visual_judgment', PutAwayTeacher)
    dispatcher.register('put_away_cross_room', PutAwayTeacher)
    dispatcher.register('verify_after_action', VerifyAfterActionTeacher)
    dispatcher.register('recover_and_store_incident_object', RecoverTeacher)
    dispatcher.register('recover_multi_incident', RecoverTeacher)
    dispatcher.register('tidy_by_policy', TidyTeacher)
    dispatcher.register('human_interactive_execution', HumanInteractionTeacher)
    dispatcher.register('mid_execution_state_disturbance', MidExecutionDisturbanceTeacher)
    dispatcher.register('explore_then_act', ExploreTeacher)
    dispatcher.register('observation_conditioned_room_inspection', ObservationConditionedTeacher)
    dispatcher.register('observation_conditioned_task', ObservationConditionedTeacher)
    dispatcher.register('composite', CompositeTeacher)
    return dispatcher
__all__ = ['TeacherDispatcher', 'TeacherDispatcherError', 'LEGACY_TEACHER_CATEGORIES', 'FAMILY_TEACHER_CATEGORIES', 'default_dispatcher']
