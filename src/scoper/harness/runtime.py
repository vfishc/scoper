from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Mapping, Protocol
import copy
import re
from scoper.harness.canonical import sha256_bytes, strict_json_loads
from scoper.harness.harness import get_harness_definition
from scoper.harness.prompts import AGENDA_SYSTEM_BYTES, POLICY_SYSTEM_BYTES, MEMORY_SYSTEM_BYTES, VERIFIER_SYSTEM_BYTES, ANSWER_SYSTEM_BYTES, VERIFICATION_THRESHOLDS
from scoper.harness.evidence_memory import MemoryCapacity
from scoper.harness.memory_router import route_memory, build_active_roles
from scoper.harness.memory_proposer import build_memory_proposer_request
from scoper.harness.controller import ProposalContext, VerificationCommitContext, evaluate_memory_proposal
from scoper.harness.recovery import RecoveryBudgets
from scoper.harness.exploration_guard import ExplorationLoopGuard
from scoper.harness.verifier import build_verification_action, build_verifier_request, validate_verifier_output, gate_online_visual_selection
from scoper.harness.contracts import EFFECT_VOCABULARY
from scoper.harness.validation import public_json

@dataclass(frozen=True)
class RGBFrame:
    frame_id: str
    data: bytes
    mime_type: str = 'image/png'
    width: int = 512
    height: int = 512

    def __post_init__(self):
        if not self.frame_id or not isinstance(self.data, bytes) or (not self.data):
            raise ValueError('RGB frame needs an identity and image bytes')
        if self.mime_type not in {'image/png', 'image/jpeg'} or self.width <= 0 or self.height <= 0:
            raise ValueError('invalid RGB frame metadata')

    def metadata(self):
        return {'media_sha256': sha256_bytes(self.data), 'mime_type': self.mime_type, 'width': self.width, 'height': self.height}

@dataclass(frozen=True)
class ExpectedEffect:
    canonical_action: str
    target: str | None
    relation: str | None
    bindings: tuple
    public_roles: tuple[str, ...]

@dataclass(frozen=True)
class ExecutionReceipt:
    post: RGBFrame
    process_frames: tuple[RGBFrame, ...] = ()
    admitted: bool = True
    preparation_result: Mapping | None = None

class Backbone(Protocol):
    checkpoint_sha256: str

    def complete(self, role: str, system: bytes, request: Mapping, frames: tuple[RGBFrame, ...]) -> Mapping:
        ...

class EnvironmentAdapter(Protocol):
    backend_name: str

    def execute(self, call: Mapping) -> ExecutionReceipt:
        ...

    def expected_effect(self, call: Mapping, public_memory: Mapping) -> ExpectedEffect | None:
        pass

    def stop(self) -> None:
        ...

    def submit_answer(self, answer: Mapping) -> None:
        ...

    def poll_public_messages(self) -> tuple[Mapping, ...]:
        pass

class JSONBackbone:

    def __init__(self, generate, checkpoint_sha256):
        if len(checkpoint_sha256) != 64 or any((c not in '0123456789abcdef' for c in checkpoint_sha256)):
            raise ValueError('checkpoint identity must be a SHA256')
        self.generate = generate
        self.checkpoint_sha256 = checkpoint_sha256

    def complete(self, role, system, request, frames):
        result = self.generate(role, system, copy.deepcopy(request), frames)
        return strict_json_loads(result) if isinstance(result, (str, bytes)) else result

class SCOPERRuntime:

    def __init__(self, backbone: Backbone, tools, public_vocabulary, *, capture_sink=None):
        self.backbone = backbone
        self._checkpoint = backbone.checkpoint_sha256
        self.tools = tuple(sorted(copy.deepcopy(tools), key=lambda t: t['name']))
        self.vocabulary = tuple(sorted(set(public_vocabulary)))
        public_json({'tools': list(self.tools), 'names': list(self.vocabulary)}, pointer='/scoper_public_interface')
        if len({t['name'] for t in self.tools}) != len(self.tools):
            raise ValueError('duplicate tools')
        self.s = get_harness_definition(harness_version='scoper-1').create_session(memory_capacity=MemoryCapacity(), recovery_budgets=RecoveryBudgets(), stream_enabled_tools=('look_for', 'navigate_to', 'navigate_to_room', 'scan'))
        self.guard = ExplorationLoopGuard()
        self.capture_sink = capture_sink
        self.frames = deque(maxlen=32)
        self.history = deque(maxlen=48)
        self._processed = set()
        self._frame_hashes = {}
        self._questions = deque()
        self._question_ids = set()
        self.steps = 0
        self.model_calls = 0
        self.model_calls_by_role = {}
        self.stopped = False
        self.terminated = False
        self.termination_reason = None
        self.budget = 0
        self._feedback = None
        self._pending = None
        self._plan = None
        self._context = None
        self._serial = 0

    def _call(self, role, system, request, frames=()):
        if self.backbone.checkpoint_sha256 != self._checkpoint:
            raise ValueError('base model changed during episode')
        request = copy.deepcopy(request)
        if role in {'memory', 'verifier'}:
            request['checkpoint_sha256'] = self._checkpoint
        public_json(request, pointer='/scoper_model_request', include_future_action=True)
        result = self.backbone.complete(role, system, copy.deepcopy(request), tuple(frames))
        self.model_calls += 1
        self.model_calls_by_role[role] = self.model_calls_by_role.get(role, 0) + 1
        if not isinstance(result, Mapping):
            raise ValueError('model output must be an object')
        if self.capture_sink is not None:
            self.capture_sink(role, copy.deepcopy(request), copy.deepcopy(result))
        return result

    def _append(self, frames):
        for frame in frames:
            digest = sha256_bytes(frame.data)
            if frame.frame_id in self._frame_hashes:
                if self._frame_hashes[frame.frame_id] != digest:
                    raise ValueError('frame ID reused for different image')
                continue
            self._frame_hashes[frame.frame_id] = digest
            self.frames.append(frame)

    @staticmethod
    def sample(frames, limit=8):
        frames = tuple(frames)
        if len(frames) <= limit:
            return frames
        return tuple((frames[round(i * (len(frames) - 1) / (limit - 1))] for i in range(limit)))

    def start(self, episode_id, instruction, initial_frame, *, budget=24):
        if self.s.active:
            raise RuntimeError('finish the existing episode first')
        if budget not in {24, 36, 48}:
            raise ValueError('episode budget must be 24, 36 or 48')
        self.frames.clear()
        self.history.clear()
        self._processed.clear()
        self._frame_hashes.clear()
        self._questions.clear()
        self._question_ids.clear()
        self.guard.start()
        self.steps = 0
        self.model_calls = 0
        self.model_calls_by_role = {}
        self.budget = budget
        self._serial = 0
        self.stopped = False
        self.terminated = False
        self.termination_reason = None
        self._feedback = None
        self._pending = None
        self.s.start(episode_id, instruction=instruction)
        self._append((initial_frame,))
        self.s.accept_agenda(self._call('agenda', AGENDA_SYSTEM_BYTES, self.s.agenda_request(initial=True)), initial=True)

    def ingest_public_messages(self, messages):
        for message in messages:
            kind = message.get('kind')
            if kind == 'task_update' and set(message) == {'kind', 'id', 'text'}:
                self.s.add_task_message(message['id'], message['text'])
            elif kind == 'question' and set(message) == {'kind', 'question_id', 'text', 'choices'}:
                if not isinstance(message['choices'], list) or not 2 <= len(message['choices']) <= 4:
                    raise ValueError('invalid QA choices')
                if message['question_id'] not in self._question_ids:
                    self._question_ids.add(message['question_id'])
                    self._questions.append(copy.deepcopy(message))
            else:
                raise ValueError('invalid public message fields')

    def _maintain(self):
        frames = self.sample([f for f in self.frames if f.frame_id not in self._processed])
        if frames:
            ids = tuple((f.frame_id for f in frames))
            images = {f'window_rgb_{i}': f.metadata() for i, f in enumerate(frames)}
            for mode in ('observe', 'reconcile'):
                if mode == 'reconcile' and (not self.s.memory.facts):
                    continue
                self._serial += 1
                parent = f'memory-{self.steps}-{self._serial}'
                candidates = tuple(sorted((f.fact_id for f in self.s.memory.facts if not (f.fact_id in self.s._verified_fact_ids and set(f.source_frame_ids) & set(ids))))) if mode == 'reconcile' else ()
                if mode == 'reconcile' and (not candidates):
                    continue
                request = build_memory_proposer_request(proposal_id=parent, mode=mode, base_revision=self.s.memory.revision, evidence_frame_ids=ids, images=images, previous_evidence_memory=self.s.memory, public_observation_or_null=None, observed_action_or_null=None, public_transition_or_null=None, public_entity_refs=() if mode == 'observe' else self.vocabulary, candidate_memory_ids=candidates, public_instruction=self.s.task_context if mode == 'observe' else None, public_task_context={} if mode == 'observe' else None, observation_tool_call_or_null=None)
                if mode == 'reconcile':
                    request['public_instruction'] = self.s.task_context
                proposal = self._call('memory', MEMORY_SYSTEM_BYTES, request, frames)
                if proposal.get('mode') != mode or proposal.get('proposal_id') != parent or proposal.get('checkpoint_sha256') != self._checkpoint:
                    raise ValueError('memory proposal correlation mismatch')
                decision = evaluate_memory_proposal(self.s.memory, proposal, ProposalContext(parent, self.steps, ids, self.vocabulary, candidate_fact_ids=candidates), capacity=self.s.memory_capacity, harness_version='scoper-1')
                if decision.accepted_by_harness:
                    self.s.memory = decision.memory
            self._processed.update(ids)
        self.s.reconcile_planning_assumptions()
        self.s.reopen_unsupported()
        if self.s.agenda_signature != self.s._agenda_signature:
            self.s.accept_agenda(self._call('agenda', AGENDA_SYSTEM_BYTES, self.s.agenda_request()))

    def _routed(self):
        obligations = [o for o in self.s.memory.pending_obligations if o.status == 'open']
        roles = tuple((r for o in obligations if o.source_incident_id_or_null for r in o.public_roles))
        last = self.history[-1]['arguments'] if self.history else {}
        target = next((v for v in last.values() if isinstance(v, str)), None)
        active = build_active_roles(unresolved_incident_roles=roles, first_pending_obligation_roles=obligations[0].public_roles if obligations else (), last_public_transition_target_or_null=target)
        needs = ' '.join((item['requirement'] + ' ' + (item['next_intent_or_null'] or '') for item in self.s.agenda['items'] if item['status'] != 'supported'))
        agenda_roles = tuple((name for name in self.vocabulary if re.search('(?<!\\w)' + re.escape(name) + '(?!\\w)', needs, re.I)))
        active = tuple(dict.fromkeys((*active, *agenda_roles)))
        return route_memory(self.s.memory, active, public_instruction=self.s.task_context).to_mapping()

    def _validate_call(self, call, tools):
        if set(call) != {'tool_name', 'arguments'} or not isinstance(call['arguments'], dict):
            raise ValueError('invalid ToolCall')
        spec = next((t for t in tools if t['name'] == call['tool_name']), None)
        if spec is None:
            raise ValueError('tool not available')
        schema = spec.get('parameters', spec.get('arguments_schema', {}))
        args = call['arguments']
        props = schema.get('properties', {})
        if not set(schema.get('required', ())) <= set(args) or (schema.get('additionalProperties') is False and set(args) - set(props)):
            raise ValueError('tool arguments do not match schema')
        from scoper.harness.recovery import _validate_arguments
        _validate_arguments(args, schema)
        for key, value in args.items():
            rule = props.get(key, {})
            if rule.get('type') == 'string' and (not isinstance(value, str) or not value):
                raise ValueError('invalid argument type')
            if 'enum' in rule and value not in rule['enum']:
                raise ValueError('invalid argument enum')
            if key in {'target', 'object_ref', 'object', 'receptacle', 'room', 'source'} and value not in self.vocabulary:
                raise ValueError('argument is outside public vocabulary')
        public_json(call, pointer='/tool_call', include_future_action=True)

    def decide(self):
        if not self.s.active or self.terminated:
            raise RuntimeError('episode not active')
        if self._pending is not None:
            raise RuntimeError('pending action must be completed')
        if self.steps >= self.budget:
            self.terminated = True
            self.termination_reason = 'budget_exhausted'
            return None
        self._maintain()
        routed = self._routed()
        images = tuple(self.frames)[-4:]
        if self._questions:
            q = self._questions[0]
            answer = self._call('answer', ANSWER_SYSTEM_BYTES, {'question': q, 'instruction': self.s.task_context, 'history': list(self.history), 'evidence_memory': routed}, images)
            if set(answer) != {'question_id', 'value'} or answer['question_id'] != q['question_id'] or answer['value'] not in q['choices']:
                raise ValueError('invalid QA answer')
            self._pending = {'tool_name': 'answer', 'arguments': dict(answer)}
            return copy.deepcopy(self._pending)
        if self.stopped:
            self.terminated = True
            self.termination_reason = 'stopped'
            return None
        self._plan = self.s.plan_recovery(self.s.memory, self.s.recovery_ledger, self.tools, budgets=self.s.recovery_budgets, observation_tool_names=('scan', 'look_for'))
        if self._plan.mode == 'agent_failure':
            self.terminated = True
            self.termination_reason = self._plan.reason
            return None
        tools = self.s.policy_tool_specs(self.tools, self._plan)
        recovery = self.s.public_recovery_context(self._plan)
        if recovery:
            recovery = {k: v for k, v in recovery.items() if k not in {'last_preparation_result_or_null', 'binding_failure_reason_or_null', 'public_failure_reason_or_null', 'uniquely_resolved_public_target'}}
        context = {'instruction': self.s.task_context, 'mode': self._plan.mode, 'step': self.steps, 'remaining_budget': self.budget - self.steps, 'tool_history': list(self.history), 'last_public_transition': self.history[-1] if self.history else None, 'allowed_tool_specs': list(tools), 'public_vocabulary': list(self.vocabulary), 'evidence_memory': routed, 'agenda': self.s.agenda, 'recovery': recovery, 'planning_assumptions': list(self.s.public_assumed_effects())}
        required = self._plan.required_tool_call_or_null
        for attempt in range(3):
            call = copy.deepcopy(required) if required else dict(self._call('policy', POLICY_SYSTEM_BYTES, context, images))
            try:
                self._validate_call(call, tools)
            except ValueError:
                context['reselection_reason'] = 'Return one valid call using the allowed tool schema and public vocabulary.'
                if required:
                    break
                continue
            rejection = self.guard.rejection(context, call, reask_index=attempt, public_transition_feedback_or_null=self._feedback)
            if rejection is None:
                self.guard.accept(context, call, public_transition_feedback_or_null=self._feedback)
                self._pending = call
                self._context = context
                return copy.deepcopy(call)
            context = self.guard.reask_context(context, rejection)
            tools = context['allowed_tool_specs']
            context['reselection_reason'] = {k: v for k, v in rejection.items() if k in {'reason', 'message'}}
        self.terminated = True
        self.termination_reason = 'control_exhausted'
        return None

    def execute_pending(self, adapter: EnvironmentAdapter):
        if self._pending is None:
            raise RuntimeError('no pending action')
        call = self._pending
        aid = f'action-{self.steps}'
        pre = self.frames[-1]
        if call['tool_name'] == 'answer':
            adapter.submit_answer(call['arguments'])
            self._questions.popleft()
        elif call['tool_name'] == 'stop':
            adapter.stop()
            self.stopped = True
        else:
            expected = adapter.expected_effect(copy.deepcopy(call), self._routed())
            if expected:
                for role in expected.public_roles:
                    if role not in self.vocabulary:
                        raise ValueError('non-public effect reference')
                for binding in expected.bindings:
                    public_json(binding.to_mapping(), pointer='/effect_binding')
                    if binding.binding_status == 'known':
                        if binding.subject_public_role not in self.vocabulary:
                            raise ValueError('non-public effect subject')
                        if binding.relation_or_property in {'inside', 'on_top'} and binding.object_public_role_or_value not in self.vocabulary:
                            raise ValueError('non-public effect destination')
                action = build_verification_action(canonical_action=expected.canonical_action, target_public_ref_or_null=expected.target, intended_relation_or_null=expected.relation, effect_bindings=expected.bindings)
            if self._plan.mode in {'recovery', 'observe'}:
                self.s.recovery_ledger = self.s.record_recovery_action(self.s.recovery_ledger, self._plan, call, action_id=aid, public_tool_specs=self.tools, allowed_public_entity_refs=self.vocabulary, public_precondition_satisfied=True, budgets=self.s.recovery_budgets)
            receipt = adapter.execute(copy.deepcopy(call))
            if not isinstance(receipt, ExecutionReceipt):
                raise TypeError('adapter must return ExecutionReceipt')
            self.s.finalize_recovery_action(action_id=aid, admitted_to_environment=receipt.admitted, preparation_result=receipt.preparation_result)
            process = self.sample((*receipt.process_frames, receipt.post))
            self._append(process)
            self._feedback = {'tool_name': call['tool_name'], 'arguments': copy.deepcopy(call['arguments']), 'binding_admission_or_null': {'status': 'success' if receipt.admitted else 'rejected'}}
            if receipt.admitted and expected:
                ids = (pre.frame_id, receipt.post.frame_id)
                request = build_verifier_request(verifier_call_id=aid, action=action, evidence_frame_ids=ids, pre_rgb=pre.metadata(), post_rgb=receipt.post.metadata())
                request['thresholds'] = {**VERIFICATION_THRESHOLDS, 'effect_thresholds': {e: 0.5 for e in EFFECT_VOCABULARY}}
                output = self._call('verifier', VERIFIER_SYSTEM_BYTES, request, (pre, receipt.post))
                if output.get('verifier_call_id') != aid or tuple(output.get('evidence_frame_ids', ())) != ids or output.get('checkpoint_sha256') != self._checkpoint:
                    raise ValueError('verifier correlation mismatch')
                selection = gate_online_visual_selection(validate_verifier_output(output, request['thresholds'], required_effects=action['required_effects']))
                v = VerificationCommitContext(expected.canonical_action, selection.selected_outcome, selection.selected_effects, selection.selected_observability, output['outcome_probabilities']['success'], output['outcome_probabilities']['failure'], tuple(((e, output['effect_probabilities'][e]) for e in EFFECT_VOCABULARY)), expected.bindings, expected.public_roles)
                self.s.event_sink.register_parent(aid, ids)
                self.s.commit_visual_verification(v, action_id=aid, evidence_frame_ids=ids, step=self.steps, recovery_plan=self._plan, failed_tool_call=call)
        self.history.append(copy.deepcopy(call))
        self.steps += 1
        self._pending = None

    def run(self, adapter):
        if adapter.backend_name not in {'procthor', 'behavior'}:
            raise ValueError('unsupported backend')
        while not self.terminated:
            self.ingest_public_messages(adapter.poll_public_messages())
            call = self.decide()
            if call is None:
                break
            self.execute_pending(adapter)
        return {'harness_version': 'scoper-1', 'harness_spec_sha256': self.s.definition.harness_spec_sha256, 'high_level_steps': self.steps, 'model_calls': self.model_calls, 'model_calls_by_role': dict(self.model_calls_by_role), 'termination': self.termination_reason}

    def finish(self):
        self.s.finish()
        self.guard.finish()
        self.frames.clear()
        self.history.clear()
        self._questions.clear()
        self._processed.clear()
        self._pending = None
        self.terminated = True
