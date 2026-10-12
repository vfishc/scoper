from collections.abc import Mapping
from scoper.benchmark.common.evaluation_binding_policy import CONCRETE_BINDING_COMPILER_VERSIONS

class CompositeEvaluationProgress:

    def __init__(self, children, check, activate):
        self.children = tuple(children)
        self.check = check
        self.activate = activate
        self.index = 0
        self.last_step = None
        self.activate(self.children[0])

    def after_step(self, transition, *, pending_qa=False):
        if self.index >= len(self.children) or pending_qa:
            return
        step = transition.get('high_level_step')
        if step is not None and step == self.last_step:
            return
        self.last_step = step
        if transition.get('tool_name') in {'stop', 'scan', 'look_for', 'navigate_to', 'navigate_to_room'}:
            return
        goal = self.check(self.children[self.index])
        if float(goal.get('predicate_success', 0)) < 1:
            return
        self.index += 1
        self.activate(self.children[self.index] if self.index < len(self.children) else None)

def enabled(task):
    return (task.get('task_binding_spec') or {}).get('compiler_version') in CONCRETE_BINDING_COMPILER_VERSIONS and isinstance((task.get('family_spec') or {}).get('composite'), Mapping)
