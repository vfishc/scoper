from dataclasses import replace
import hashlib
import pytest
from scoper.harness import get_harness_definition
from scoper.harness.controller import VerificationCommitContext
from scoper.harness.effect_binding import EffectBinding
from scoper.harness.evidence_memory import EvidenceFact, MemoryCapacity
from scoper.harness.control_prompts import d6_policy_adapter_descriptor
from scoper.harness.recovery_prompts import D6_POLICY_SYSTEM_BYTES, POLICY_SYSTEM_BYTES
from scoper.harness.recovery_prompts import MEMORY_SYSTEM_BYTES, VERIFIER_SYSTEM_BYTES
from scoper.harness.contracts import EFFECT_VOCABULARY
from scoper.harness.exploration_guard import _required_recovery_call
from scoper.harness.recovery import RecoveryBudgets
from scoper.harness.placement_recovery import PlacementRecovery, placement_object_ref
TOOLS = tuple(({'name': name, 'parameters': {'type': 'object', 'properties': {} if name in {'scan', 'stop'} else {'target': {'type': 'string'}}, 'required': [] if name in {'scan', 'stop'} else ['target'], 'additionalProperties': False}} for name in ('look_for', 'navigate_to', 'open', 'pick_up', 'put_down', 'scan', 'stop')))

def fact(relation, value, step, subject='mug'):
    return EvidenceFact.create(subject_public_role=subject, relation_or_property=relation, object_public_role_or_value=value, confidence=1.0, source_frame_ids=(f'frame-{step}',), source_action_id=f'action-{step}', observed_step=step)

def session(version='19'):
    result = get_harness_definition(harness_version=version).create_session(memory_capacity=MemoryCapacity(), recovery_budgets=RecoveryBudgets(), stream_enabled_tools=('navigate_to', 'scan'))
    result.start('placement-test', instruction='Put the mug on the shelf.')
    result.memory = replace(result.memory, facts=(fact('held_by_agent', True, 1),))
    return result

def plan(current):
    return current.plan_recovery(current.memory, current.recovery_ledger, TOOLS, budgets=RecoveryBudgets(), observation_tool_names=('scan',))

def verdict(current, action, outcome, step, recovery_plan):
    effect, relation, value = {'put_down': ('released', 'held_by_agent', False), 'pick_up': ('picked_up', 'held_by_agent', True), 'open': ('opened', 'open', True)}[action]
    target = 'shelf' if action == 'put_down' else 'mug'
    effects = (effect,) if outcome == 'success' else ()
    context = VerificationCommitContext(action, outcome, effects, 'observable', float(outcome == 'success'), float(outcome == 'failure'), tuple(((name, float(name in effects)) for name in EFFECT_VOCABULARY)), (EffectBinding(effect, 'mug', relation, value, True, 'known'),), ('mug', 'shelf'))
    frames = tuple((f'frame-{step}-{name}' for name in ('pre', 'post', 'confirm')))
    current.event_sink.register_parent(f'action-{step}', frames)
    return current.commit_visual_verification(context, action_id=f'action-{step}', evidence_frame_ids=frames, step=step, recovery_plan=recovery_plan, failed_tool_call={'tool_name': action, 'arguments': {'target': target}})

def stage(current, expected, step, admitted=True):
    recovery_plan = plan(current)
    assert recovery_plan.required_tool_call_or_null == expected
    specs = current.policy_tool_specs(TOOLS, recovery_plan)
    assert [spec['name'] for spec in specs] == [expected['tool_name']]
    public_context = current.public_recovery_context(recovery_plan)
    assert _required_recovery_call({'mode': 'recovery', 'recovery': public_context}) == expected
    current.validate_public_action_preconditions(expected)
    current.recovery_ledger = current.record_recovery_action(current.recovery_ledger, recovery_plan, expected, action_id=f'action-{step}', public_tool_specs=TOOLS, allowed_public_entity_refs=('mug', 'shelf'), public_precondition_satisfied=True, budgets=RecoveryBudgets())
    preparation = None
    if expected['tool_name'] in {'scan', 'navigate_to'}:
        preparation = {'tool_name': expected['tool_name'], 'status': 'scan_completed' if expected['tool_name'] == 'scan' else 'navigation_ready', 'interaction_ready': expected['tool_name'] == 'navigate_to', 'frame_id_or_null': f'frame-{step}'}
    current.finalize_recovery_action(action_id=f'action-{step}', admitted_to_environment=admitted, preparation_result=preparation)
    return recovery_plan

def call(name, target=None):
    return {'tool_name': name, 'arguments': {} if target is None else {'target': target}}

def observe(current, text, step):
    observation = fact('scene_observation', text, step)
    current.memory = replace(current.memory, revision=current.memory.revision + 1, facts=tuple(sorted((item for item in (*current.memory.facts, observation) if item is observation or item.fact_id != observation.fact_id), key=lambda item: item.fact_id)))
