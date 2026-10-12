from __future__ import annotations
import copy
from dataclasses import replace
from typing import Mapping
from scoper.harness.canonical import sha256_json
from scoper.harness.harness import HarnessSession
from scoper.harness.advisory_agenda import validate_advisory_agenda, ADVISORY_AGENDA_SCHEMA_VERSION
from scoper.harness.recovery import RecoveryPlan, plan_recovery_v13

class SCOPERSession(HarnessSession):

    def start(self, episode_id, *, instruction=''):
        super().start(episode_id, instruction=instruction)
        self.agenda = {'items': [], 'active_item_id_or_null': None}
        self._agenda_signature = None
        self._task_messages = []
        self._message_ids = set()
        self._deferred_incidents = set()
        self._verified_fact_ids = set()
        self._verified_sources = {}
        self._confirmed_failures = set()

    @property
    def task_context(self):
        return self.task_instruction + ''.join(('\nAdditional requirement: ' + v for v in self._task_messages))

    def add_task_message(self, message_id: str, text: str):
        if not self.active or not message_id or (not isinstance(text, str)) or (not text.strip()):
            raise ValueError('invalid public task message')
        if message_id not in self._message_ids:
            self._message_ids.add(message_id)
            self._task_messages.append(text.strip())

    @property
    def agenda_signature(self):
        return sha256_json({'instruction': self.task_context, 'memory': self.memory.to_mapping()})

    def agenda_request(self, *, initial=False):
        if initial:
            return {'instruction': self.task_context}
        self.reopen_unsupported()
        return {'instruction': self.task_context, 'previous_agenda': copy.deepcopy(self.agenda), 'committed_memory': self.memory.to_mapping(), 'verified_effect_fact_ids': sorted(self._verified_fact_ids)}

    def _eligible(self, item):
        facts = {f.fact_id: f for f in self.memory.facts if f.status == 'current'}
        refs = item['evidence_refs']
        if not refs or any((ref not in facts for ref in refs)):
            return False
        if item['criterion'] == 'execution':
            return all((self._verified_sources.get(ref) == (facts[ref].source_action_id, facts[ref].source_frame_ids) for ref in refs))
        return True

    def accept_agenda(self, proposal: Mapping, *, initial=False):
        if not isinstance(proposal, Mapping) or set(proposal) != {'items', 'active_item_id_or_null'}:
            raise ValueError('invalid agenda fields')
        raw = copy.deepcopy(dict(proposal))
        criteria = {}
        for item in raw['items']:
            criterion = item.pop('criterion', None)
            if criterion not in {'inspection', 'execution', 'constraint'}:
                raise ValueError('invalid agenda criterion')
            criteria[item['id']] = criterion
        parsed = dict(validate_advisory_agenda({**raw, 'schema_version': ADVISORY_AGENDA_SCHEMA_VERSION}))
        parsed.pop('schema_version')
        known = {f.fact_id for f in self.memory.facts} | {i.incident_id for i in self.memory.incidents} | {o.obligation_id for o in self.memory.pending_obligations}
        for item in parsed['items']:
            item['criterion'] = criteria[item['id']]
            if any((ref not in known for ref in item['evidence_refs'])):
                raise ValueError('unknown agenda evidence reference')
            if initial and (item['evidence_refs'] or item['status'] == 'supported'):
                raise ValueError('initial agenda cannot assert environmental support')
            if item['status'] == 'supported' and (not self._eligible(item)):
                raise ValueError('ineligible agenda support')
        self.agenda = parsed
        self._agenda_signature = self.agenda_signature
        return copy.deepcopy(parsed)

    def reopen_unsupported(self):
        for item in self.agenda['items']:
            if item['status'] == 'supported' and (not self._eligible(item)):
                item['status'] = 'watching' if item['mode'] in {'conditional', 'ongoing'} else 'pending'

    def commit_visual_verification(self, verification, **kwargs):
        result = super().commit_visual_verification(verification, **kwargs)
        if verification.effective_outcome == 'success':
            self._verified_fact_ids.update((f.fact_id for f in self.memory.facts if f.status == 'current' and f.source_action_id == kwargs['action_id'] and (f.relation_or_property != 'scene_observation')))
        for fact in self.memory.facts:
            if fact.fact_id in self._verified_fact_ids and fact.source_action_id == kwargs['action_id']:
                self._verified_sources[fact.fact_id] = (fact.source_action_id, fact.source_frame_ids)
        if verification.effective_outcome == 'failure':
            source = getattr(kwargs['recovery_plan'], 'incident_id_or_null', None)
            self._confirmed_failures.update((i.incident_id for i in self.memory.incidents if i.status == 'open' and (i.action_id == kwargs['action_id'] or i.incident_id == source)))
        live = {i.incident_id for i in self.memory.incidents if i.status == 'open'}
        self._deferred_incidents.intersection_update(live)
        self._placement_recoveries = {k: v for k, v in self._placement_recoveries.items() if k in live}
        self.reopen_unsupported()
        return result

    def plan_recovery(self, memory, ledger, public_tool_specs, *, budgets, observation_tool_names):
        tools = tuple(sorted((spec['name'] for spec in public_tool_specs)))
        live = [i for i in self.memory.incidents if i.status == 'open' and i.incident_id not in self._deferred_incidents]
        live.sort(key=lambda i: (i.created_step, i.incident_id))
        if live:
            incident = live[0]
            context = self._recovery_contexts.get(incident.incident_id)
            placement = self._placement_recoveries.get(incident.incident_id)
            if placement is not None:
                return self._public_plan(placement.plan(self.memory, self._target_context(context), public_tool_specs), public_tool_specs)
            if context is not None and (context.retry_index >= context.max_retries or context.phase == 'exhausted'):
                if context.failure_kind != 'ambiguous' or incident.incident_id in self._confirmed_failures:
                    return RecoveryPlan('agent_failure', 'confirmed_failure_recovery_exhausted', incident.incident_id, None, ())
                self._deferred_incidents.add(incident.incident_id)
                self._assumed_completed_actions.append({'tool_call': copy.deepcopy(dict(context.failed_tool_call)), 'required_effects': list(context.required_effects), 'source_incident_id': incident.incident_id, 'evidence_frame_ids': list(context.evidence_frame_ids), 'retry_count': context.retry_index, 'verification_status': 'uncertain', 'created_step': max((f.observed_step for f in self.memory.facts), default=-1), 'assumed_effects': self._assumed_effects_for_context(context)})
                self._last_recovery_notice = {**context.to_mapping(), 'resolution': 'unresolved_continue'}
                return RecoveryPlan('step', 'recovery_assumed_continue', None, None, tools)
        projected = replace(self.memory, incidents=tuple((i for i in self.memory.incidents if i.incident_id not in self._deferred_incidents)), pending_obligations=tuple((o for o in self.memory.pending_obligations if o.source_incident_id_or_null not in self._deferred_incidents)))
        return self._public_plan(plan_recovery_v13(projected, ledger, public_tool_specs, contexts={k: self._target_context(v) for k, v in self._recovery_contexts.items()}, preparation_tool_names=('look_for', 'scan', 'navigate_to', 'navigate_to_room'), max_retries=2), public_tool_specs)

    def reconcile_planning_assumptions(self):
        kept = []
        for assumption in self._assumed_completed_actions:
            disputed = False
            for effect in assumption.get('assumed_effects', ()):
                for fact in self.memory.facts:
                    if fact.status == 'current' and fact.observed_step > assumption.get('created_step', -1) and (fact.subject_public_role == effect.get('subject_public_role')) and (fact.relation_or_property == effect.get('relation_or_property')) and (fact.object_public_role_or_value != effect.get('object_public_role_or_value')):
                        disputed = True
            if disputed:
                incident_id = assumption['source_incident_id']
                self._deferred_incidents.discard(incident_id)
                self._confirmed_failures.add(incident_id)
            else:
                kept.append(assumption)
        self._assumed_completed_actions = kept

    @staticmethod
    def _target_context(context):
        call = copy.deepcopy(dict(context.failed_tool_call))
        if 'object_ref' in call['arguments']:
            call['arguments']['target'] = call['arguments'].pop('object_ref')
        return replace(context, failed_tool_call=call)

    @staticmethod
    def _public_plan(plan, specs):
        if plan.required_tool_call_or_null is None:
            return plan
        call = copy.deepcopy(dict(plan.required_tool_call_or_null))
        spec = next((t for t in specs if t['name'] == call['tool_name']))
        schema = spec.get('parameters', spec.get('arguments_schema', {}))
        props = schema['properties']
        if 'target' in call['arguments'] and 'target' not in props and ('object_ref' in props):
            call['arguments']['object_ref'] = call['arguments'].pop('target')
        if call['tool_name'] == 'scan' and 'num_views' in schema.get('required', ()):
            call['arguments']['num_views'] = props['num_views'].get('minimum', 1)
        return replace(plan, required_tool_call_or_null=call)

    def finalize_recovery_action(self, *, action_id, admitted_to_environment, preparation_completed=None, preparation_result=None):
        pending = self._pending_recovery_actions.pop(action_id, None)
        if pending is None:
            return self._recovery_ledger
        incident_id = pending['incident_id']
        context = self._recovery_contexts.get(incident_id)
        if context is None:
            return self._recovery_ledger
        attempts = dict(self._recovery_ledger.incident_attempts)
        placement = self._placement_recoveries.get(incident_id)
        if placement is not None:
            phase = pending.get('placement_phase')
            placement.actions += 1
            if not admitted_to_environment:
                placement.phase = 'exhausted'
                placement.pickup_assumed = False
            elif phase == 'inspect':
                placement.observations += 1
            elif phase == 'pickup':
                placement.phase = 'await_pickup_verification'
            elif phase == 'navigate_object':
                placement.phase = 'pickup'
            elif phase == 'navigate_destination':
                placement.phase = 'retry'
        if pending['is_original_retry']:
            if admitted_to_environment:
                count = min(context.max_retries, context.retry_index + 1)
                context = replace(context, retry_index=count, dispatched_attempts=count)
                attempts[incident_id] = attempts.get(incident_id, 0) + 1
            else:
                context = replace(context, phase='exhausted')
        else:
            tool = pending['tool_call']['tool_name']
            phase = 'navigate' if tool in {'scan', 'look_for'} else 'retry'
            if not admitted_to_environment:
                phase = 'exhausted'
            context = replace(context, phase=phase, preparation_attempts=context.preparation_attempts + 1)
        self._recovery_contexts[incident_id] = context
        self._recovery_ledger = replace(self._recovery_ledger, total_recovery_actions=self._recovery_ledger.total_recovery_actions + int(admitted_to_environment), incident_attempts=tuple(sorted(attempts.items())))
        return self._recovery_ledger

    def _reject_executor_memory_update(self, *args, **kwargs):
        raise ValueError('SCOPER v21 requires visual evidence; executor labels cannot update memory')
    record_public_tool_failure = _reject_executor_memory_update
    record_public_preparation_failure = _reject_executor_memory_update
    record_public_recovery_success = _reject_executor_memory_update
    commit_public_postcondition = _reject_executor_memory_update

    def finish(self, *, assert_closed=True):
        try:
            super().finish(assert_closed=assert_closed)
        finally:
            self.agenda = {'items': [], 'active_item_id_or_null': None}
            self._agenda_signature = None
            self._task_messages = []
            self._message_ids = set()
            self._deferred_incidents = set()
            self._verified_fact_ids = set()
            self._verified_sources = {}
            self._confirmed_failures = set()
