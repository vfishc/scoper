from __future__ import annotations
from typing import Any, Dict, Mapping, Type
from scoper.benchmark.procthor.online_task_spec import V1_FAMILY_CATEGORIES
LEGACY_TEACHER_CATEGORIES = frozenset({'static_simple', 'hidden_precondition', 'mid_execution_discovery', 'dynamic_state_change', 'completion_timing', 'disturbance_recovery', 'failure_recovery'})

class TeacherDispatcherError(ValueError):
    pass

class TeacherDispatcher:

    def __init__(self) -> None:
        from scoper.benchmark.procthor.brains.oracle import OracleBrain
        self._legacy_teacher = OracleBrain
        self._family_teachers: Dict[str, Type[Any]] = {}

    def register(self, category: str, teacher_cls: Type[Any]) -> None:
        if not isinstance(category, str) or not category:
            raise TeacherDispatcherError('Teacher category must be a non-empty string.')
        self._family_teachers[category] = teacher_cls

    def build(self, task_spec: Mapping[str, Any]) -> Any:
        if (task_spec.get('template') or {}).get('template_id') == 'inspect_and_restore_storage_container_001':
            from scoper.benchmark.common.container_inspection.teacher import teacher_class
            return teacher_class('procthor')()
        category = task_spec.get('category')
        if category in self._family_teachers:
            return self._family_teachers[category]()
        if category in LEGACY_TEACHER_CATEGORIES:
            return self._legacy_teacher()
        if category in V1_FAMILY_CATEGORIES:
            raise TeacherDispatcherError("No teacher registered for v1 family category '{}'.".format(category))
        raise TeacherDispatcherError("Unknown task category '{}'.".format(category))
__all__ = ['LEGACY_TEACHER_CATEGORIES', 'TeacherDispatcher', 'TeacherDispatcherError']
