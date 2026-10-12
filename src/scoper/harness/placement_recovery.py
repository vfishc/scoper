from dataclasses import dataclass, replace
from scoper.harness.recovery import RecoveryPlan

def public_reference_key(value):
    return ''.join(value.casefold().split())

@dataclass
class PlacementRecovery:
    object_ref: str | None
    failed_step: int
    phase: str = 'inspect'
    actions: int = 0
    observations: int = 0
    max_actions: int = 8
    pickup_assumed: bool = False

    def held_state(self, memory):
        if self.object_ref is None:
            return 'unknown'
        observations = []
        for fact in memory.facts:
            if fact.status != 'current' or fact.observed_step < self.failed_step or public_reference_key(fact.subject_public_role) != public_reference_key(self.object_ref):
                continue
            value = fact.object_public_role_or_value
            if fact.relation_or_property == 'held_by_agent' and isinstance(value, bool):
                observations.append((fact.observed_step, 'held' if value else 'released'))
            elif fact.relation_or_property == 'scene_observation' and isinstance(value, str):
                if value.startswith('Visibly held in hand:'):
                    observations.append((fact.observed_step, 'held'))
                elif value.startswith('Visibly outside hand:'):
                    observations.append((fact.observed_step, 'released'))
        if not observations:
            return 'unknown'
        latest = max((step for step, state in observations))
        states = {state for step, state in observations if step == latest}
        return states.pop() if len(states) == 1 else 'unknown'

    def plan(self, memory, context, public_tool_specs):
        tools = {spec['name'] for spec in public_tool_specs}
        if self.phase == 'inspect':
            state = self.held_state(memory)
            if state == 'held':
                self.phase = 'navigate_destination'
            elif state == 'released':
                self.phase = 'navigate_object'
            elif self.observations:
                self.phase = 'exhausted'
        if self.actions >= self.max_actions or context.retry_index >= context.max_retries:
            self.phase = 'exhausted'
        if self.object_ref is None or self.phase in {'exhausted', 'await_pickup_verification'}:
            return RecoveryPlan('agent_failure', 'placement_recovery_exhausted', context.incident_id, None, ())
        destination = context.failed_tool_call['arguments'].get('target')
        calls = {'inspect': {'tool_name': 'scan', 'arguments': {}}, 'navigate_object': {'tool_name': 'navigate_to', 'arguments': {'target': self.object_ref}}, 'pickup': {'tool_name': 'pick_up', 'arguments': {'target': self.object_ref}}, 'navigate_destination': {'tool_name': 'navigate_to', 'arguments': {'target': destination}}, 'retry': context.failed_tool_call}
        call = calls[self.phase]
        if call['tool_name'] not in tools or (self.phase != 'inspect' and (not call['arguments'].get('target'))):
            return RecoveryPlan('agent_failure', 'placement_recovery_tool_unavailable', context.incident_id, None, ())
        return RecoveryPlan('recovery', 'placement_' + self.phase, context.incident_id, None, (call['tool_name'],), required_tool_call_or_null=call)

    def finalize(self, phase, admitted, preparation_result):
        self.actions += 1
        if phase == 'retry' or not admitted:
            self.pickup_assumed = False
        if not admitted:
            self.phase = 'exhausted'
        elif phase == 'inspect':
            self.observations += 1
        elif phase == 'pickup':
            self.phase = 'await_pickup_verification'
        elif phase in {'navigate_object', 'navigate_destination'}:
            ready = bool(preparation_result and preparation_result.get('status') == 'navigation_ready' and preparation_result.get('interaction_ready'))
            self.phase = ('pickup' if phase == 'navigate_object' else 'retry') if ready else 'exhausted'

def placement_object_ref(memory, bindings):
    references = {binding.subject_public_role for binding in bindings if binding.relation_or_property == 'held_by_agent' and binding.object_public_role_or_value is False and (binding.binding_status == 'known') and isinstance(binding.subject_public_role, str)}
    if len(references) == 1:
        return next(iter(references))
    if references:
        return None
    held = {fact.subject_public_role for fact in memory.facts if fact.status == 'current' and fact.relation_or_property == 'held_by_agent' and (fact.object_public_role_or_value is True)}
    return next(iter(held)) if len(held) == 1 else None

def invalidate_old_held_fact(memory, object_ref, step):
    facts = tuple((replace(fact, status='stale') if object_ref is not None and public_reference_key(fact.subject_public_role) == public_reference_key(object_ref) and (fact.relation_or_property == 'held_by_agent') and (fact.observed_step < step) and (fact.status == 'current') else fact for fact in memory.facts))
    return replace(memory, facts=facts, revision=memory.revision + 1).validated() if facts != memory.facts else memory
