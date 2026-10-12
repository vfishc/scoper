from __future__ import annotations
from scoper.benchmark.procthor.brains.teachers.dispatcher import TeacherDispatcher, TeacherDispatcherError
_DISPATCHER = TeacherDispatcher()

def get_dispatcher() -> TeacherDispatcher:
    return _DISPATCHER
from scoper.benchmark.procthor.brains.teachers import prepare
from scoper.benchmark.procthor.brains.teachers import put_away
from scoper.benchmark.procthor.brains.teachers import recover
from scoper.benchmark.procthor.brains.teachers import tidy
from scoper.benchmark.procthor.brains.teachers import verify_after_action
from scoper.benchmark.procthor.brains.teachers import human_interaction
from scoper.benchmark.procthor.brains.teachers import mid_execution_disturbance
from scoper.benchmark.procthor.brains.teachers import explore
from scoper.benchmark.procthor.brains.teachers import observation_conditioned
from scoper.benchmark.procthor.brains.teachers import composite
__all__ = ['TeacherDispatcher', 'TeacherDispatcherError', 'get_dispatcher']
