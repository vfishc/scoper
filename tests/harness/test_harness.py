import copy
import json
from dataclasses import replace
import pytest
from scoper.harness import get_harness_definition
from scoper.harness.runtime import SCOPERRuntime, JSONBackbone, RGBFrame, ExecutionReceipt, ExpectedEffect
from scoper.harness.evidence_memory import EvidenceFact
from scoper.harness.effect_binding import EffectBinding
from scoper.harness.contracts import EFFECT_VOCABULARY
from scoper.harness.verifier import select_verifier_result
CHECKPOINT = 'a' * 64
TOOLS = [{'name': name, 'parameters': {'type': 'object', 'properties': {} if name in {'scan', 'stop'} else {'object_ref': {'type': 'string'}}, 'required': [] if name in {'scan', 'stop'} else ['object_ref'], 'additionalProperties': False}} for name in ('scan', 'navigate_to', 'pick_up', 'open', 'close', 'stop')]

def frame(i):
    return RGBFrame(f'f{i}', f'image-{i}'.encode())

def agenda(status='pending', criterion='execution', refs=()):
    return {'items': [{'id': 'a1', 'requirement': 'Open the cabinet', 'mode': 'required', 'status': status, 'evidence_refs': list(refs), 'next_intent_or_null': None, 'criterion': criterion}], 'active_item_id_or_null': None}

def proposal(request):
    return {'schema_version': 'memory-proposer-io.v1', 'proposal_id': request['proposal_id'], 'mode': request['mode'], 'base_revision': request['base_revision'], 'evidence_frame_ids': request['evidence_frame_ids'], 'fact_mutations': [], 'incident_mutations': [], 'obligation_mutations': [], 'per_fact_status': [], 'relevant_memory_ids': [], 'abstain_reason_or_null': 'No new evidence', 'confidence': 1.0, 'checkpoint_sha256': CHECKPOINT}

class Model:

    def __init__(self, calls=('stop',), outcome='success'):
        self.calls = iter(calls)
        self.requests = []
        self.outcome = outcome

    def __call__(self, role, system, request, frames):
        self.requests.append((role, copy.deepcopy(request), tuple((f.frame_id for f in frames))))
        if role == 'agenda':
            return agenda()
        if role == 'memory':
            return proposal(request)
        if role == 'answer':
            return {'question_id': request['question']['question_id'], 'value': request['question']['choices'][0]}
        if role == 'policy':
            name = next(self.calls)
            return {'tool_name': name, 'arguments': {} if name in {'stop', 'scan'} else {'object_ref': 'cabinet'}}
        if role == 'verifier':
            effects = {e: float(e in request['action']['required_effects'] and self.outcome == 'success') for e in EFFECT_VOCABULARY}
            probs = {'outcome': {o: float(o == self.outcome) for o in ('success', 'failure', 'ambiguous')}, 'effects': effects, 'observability': {'observable': 1.0, 'partially_observable': 0.0, 'unobservable': 0.0}}
            thresholds = {'commit_threshold': 0.5, 'failure_threshold': 0.5, 'effect_thresholds': {e: 0.5 for e in EFFECT_VOCABULARY}}
            sel = select_verifier_result(probs, thresholds, required_effects=request['action']['required_effects'])
            return {'schema_version': 'transition-verifier-io.v1', 'verifier_call_id': request['verifier_call_id'], 'evidence_frame_ids': request['evidence_frame_ids'], 'checkpoint_sha256': CHECKPOINT, 'outcome_temperature': 1.0, 'outcome_probabilities': probs['outcome'], 'effect_probabilities': effects, 'observability_probabilities': probs['observability'], 'selected_outcome': sel.selected_outcome, 'selected_effects': list(sel.selected_effects), 'selected_observability': sel.selected_observability}
        raise AssertionError(role)

class Adapter:

    def __init__(self, backend='procthor', post_question=False):
        self.backend_name = backend
        self.actions = []
        self.answers = []
        self.stopped = False
        self.post_question = post_question

    def expected_effect(self, call, memory):
        if call['tool_name'] == 'open':
            return ExpectedEffect('open', 'cabinet', None, (EffectBinding('opened', 'cabinet', 'open', True, True, 'known'),), ('cabinet',))

    def execute(self, call):
        self.actions.append(call)
        return ExecutionReceipt(frame(len(self.actions) * 20), tuple((frame(i) for i in range((len(self.actions) - 1) * 20 + 1, len(self.actions) * 20))), True)

    def stop(self):
        self.stopped = True

    def submit_answer(self, answer):
        self.answers.append(answer)

    def poll_public_messages(self):
        if self.stopped and self.post_question:
            self.post_question = False
            return ({'kind': 'question', 'question_id': 'q1', 'text': 'Is it open?', 'choices': ['yes', 'no']},)
        return ()

def runtime(model):
    r = SCOPERRuntime(JSONBackbone(model, CHECKPOINT), TOOLS, ['cabinet'])
    r.start('episode', 'Open the cabinet', frame(0))
    return r

@pytest.mark.parametrize('backend', ['procthor', 'behavior'])
@pytest.mark.parametrize('json_transport', [False, True])
def test_loop_and_stop_qa(backend, json_transport):
    model = Model(('scan', 'open', 'stop'))
    generate = (lambda *args: json.dumps(model(*args))) if json_transport else model
    r = runtime(generate)
    a = Adapter(backend, True)
    result = r.run(a)
    assert result['high_level_steps'] == 4
    assert result['termination'] == 'stopped' and len(a.actions) == 2 and (len(a.answers) == 1)
    assert model.requests[0][0] == 'agenda' and model.requests[0][2] == ()
    assert set(model.requests[0][1]) == {'instruction'}
    assert all((len(frames) <= 4 for role, _, frames in model.requests if role in {'policy', 'answer'}))
    assert all((len(frames) <= 8 for role, _, frames in model.requests if role == 'memory'))
    assert [frames for role, _, frames in model.requests if role == 'verifier'] == [('f20', 'f40')]
    assert r.s._verified_fact_ids
    assert 'binding_admission_or_null' not in json.dumps(model.requests)
    r.finish()

def test_agenda_support_and_contradiction():
    r = runtime(Model())
    s = r.s
    with pytest.raises(ValueError):
        s.accept_agenda(agenda('supported'), initial=True)
    fact = EvidenceFact.create(subject_public_role='cabinet', relation_or_property='open', object_public_role_or_value=True, confidence=1.0, source_frame_ids=('f0',), source_action_id='observe0', observed_step=0)
    s.memory = replace(s.memory, facts=(fact,))
    s.accept_agenda(agenda('supported', 'inspection', (fact.fact_id,)))
    with pytest.raises(ValueError):
        s.accept_agenda(agenda('supported', 'execution', (fact.fact_id,)))
    s.memory = replace(s.memory, facts=(replace(fact, status='contradicted'),))
    s.reopen_unsupported()
    assert s.agenda['items'][0]['status'] == 'pending'
    with pytest.raises(ValueError):
        s.accept_agenda(agenda('pending', refs=('missing',)))

def test_task_updates_and_model_identity():
    m = Model()
    r = runtime(m)
    before = r.s.agenda_signature
    r.ingest_public_messages([{'kind': 'task_update', 'id': 'u1', 'text': 'Also close the cabinet.'}])
    assert before != r.s.agenda_signature
    r.decide()
    assert 'Also close' in [req['instruction'] for role, req, _ in m.requests if role == 'agenda'][-1]
    r.backbone.checkpoint_sha256 = 'b' * 64
    with pytest.raises(ValueError, match='base model changed'):
        r._call('agenda', b'', {})

def test_frame_sampling_and_budget():
    frames = [frame(i) for i in range(100)]
    selected = SCOPERRuntime.sample(frames)
    assert len(selected) == 8 and selected[0] == frames[0] and (selected[-1] == frames[-1])
    r = runtime(Model())
    r.steps = 24
    assert r.decide() is None and r.termination_reason == 'budget_exhausted'
    r2 = runtime(Model())
    with pytest.raises(ValueError):
        r2._append((RGBFrame('f0', b'changed'),))

def test_capture_independence():
    records = []
    m1 = Model()
    m2 = Model()
    r1 = runtime(m1)
    r2 = runtime(m2)
    r2.capture_sink = lambda *args: records.append(args)
    assert r1.run(Adapter()) == r2.run(Adapter())
    assert m1.requests == m2.requests and records

@pytest.mark.parametrize('outcome,termination', [('failure', 'confirmed_failure_recovery_exhausted'), ('ambiguous', 'stopped')])
def test_bounded_recovery_preserves_incident_and_no_failure_assumptions(outcome, termination):
    model = Model(('open', 'stop'), outcome=outcome)
    r = runtime(model)
    a = Adapter()
    result = r.run(a)
    assert result['termination'] == termination
    assert [call['tool_name'] for call in a.actions] == ['open', 'scan', 'navigate_to', 'open', 'scan', 'navigate_to', 'open']
    assert r.s.memory.incidents and all((i.status == 'open' for i in r.s.memory.incidents))
    assert bool(r.s.public_assumed_effects()) == (outcome == 'ambiguous')
    assert not r.s._verified_fact_ids
    assert max((c.retry_index for c in r.s._recovery_contexts.values())) == 2

def test_callback_boundary_prevents_physical_actions_after_stop():
    from scoper.harness.adapters import CallbackEnvironmentAdapter
    a = Adapter()
    adapter = CallbackEnvironmentAdapter('behavior', execute=a.execute, expected_effect=a.expected_effect, stop_and_freeze_score=a.stop, submit_answer=a.submit_answer, poll_public_messages=a.poll_public_messages)
    runtime(Model()).run(adapter)
    with pytest.raises(RuntimeError):
        adapter.execute({'tool_name': 'open', 'arguments': {'object_ref': 'cabinet'}})
    adapter.submit_answer({'question_id': 'q', 'value': 'yes'})
    assert a.answers

def test_observation_discovery_commits_without_creating_failure():
    base = Model()

    def generate(role, system, request, frames):
        if role == 'memory' and request['mode'] == 'observe':
            out = proposal(request)
            out['abstain_reason_or_null'] = None
            out['fact_mutations'] = [{'operation': 'add', 'public_entity_ref': 'cabinet', 'relation_or_property': 'scene_observation', 'object_public_role_or_value': 'The cabinet is visibly open.', 'confidence': 1.0}]
            return out
        return base(role, system, request, frames)
    r = runtime(generate)
    r.run(Adapter())
    assert len(r.s.memory.facts) == 1
    assert not r.s.memory.incidents and (not r.s.memory.pending_obligations)
    assert len([q for q in base.requests if q[0] == 'agenda']) == 2

def test_invalid_call_is_reselected_within_bound():
    base = Model()

    def generate(role, system, request, frames):
        if role == 'policy':
            return {'tool_name': 'private_magic', 'arguments': {}}
        return base(role, system, request, frames)
    r = runtime(generate)
    a = Adapter()
    result = r.run(a)
    assert result['termination'] == 'control_exhausted' and result['high_level_steps'] == 0
    assert result['model_calls_by_role']['policy'] == 3 and (not a.actions)

def test_explicit_failure_followed_by_uncertainty_still_cannot_be_assumed_success():
    base = Model(('open', 'stop'), 'failure')
    count = 0

    def generate(role, system, request, frames):
        nonlocal count
        if role == 'verifier':
            base.outcome = 'failure' if count == 0 else 'ambiguous'
            count += 1
        return base(role, system, request, frames)
    r = runtime(generate)
    result = r.run(Adapter())
    assert result['termination'] == 'confirmed_failure_recovery_exhausted'
    assert not r.s.public_assumed_effects()

def test_contradictory_later_evidence_revokes_uncertain_planning():
    r = runtime(Model(('open', 'stop'), 'ambiguous'))
    r.run(Adapter())
    assert r.s.public_assumed_effects()
    f = EvidenceFact.create(subject_public_role='cabinet', relation_or_property='open', object_public_role_or_value=False, confidence=1.0, source_frame_ids=('later',), source_action_id='later', observed_step=20)
    r.s.memory = replace(r.s.memory, facts=(f,))
    r.s.reconcile_planning_assumptions()
    assert not r.s.public_assumed_effects() and (not r.s._deferred_incidents)

@pytest.mark.parametrize('pickup_outcome', ['success', 'failure', 'ambiguous'])
def test_placement_recovery_reacquires_released_object(pickup_outcome):
    from tests.harness.recovery_fixtures import session, verdict, plan, stage, call, observe
    s = session('scoper-1')
    verdict(s, 'put_down', 'failure', 2, plan(s))
    stage(s, call('scan'), 3)
    observe(s, 'Visibly outside hand: resting on the floor.', 3)
    stage(s, call('navigate_to', 'mug'), 4)
    pickup_plan = stage(s, call('pick_up', 'mug'), 5)
    verdict(s, 'pick_up', pickup_outcome, 5, pickup_plan)
    if pickup_outcome == 'failure':
        assert plan(s).mode == 'agent_failure'
        assert not s.public_assumed_effects()
    else:
        assert bool(s.public_assumed_effects()) == (pickup_outcome == 'ambiguous')
        stage(s, call('navigate_to', 'shelf'), 6)
        retry_plan = stage(s, call('put_down', 'shelf'), 7)
        verdict(s, 'put_down', 'success', 7, retry_plan)
        assert all((i.status == 'resolved' for i in s.memory.incidents))
        assert not s.public_assumed_effects()

def test_rejected_physical_call_does_not_buy_fresh_exploration_budget():
    from scoper.harness.exploration_guard import ExplorationLoopGuard
    guard = ExplorationLoopGuard()
    guard.start()
    context = {'mode': 'step', 'instruction': 'Open the cabinet', 'allowed_tool_specs': TOOLS, 'tool_history': [], 'evidence_memory': {'facts': [], 'incidents': [], 'pending_obligations': []}}
    scan = {'tool_name': 'scan', 'arguments': {}}
    for i in range(2):
        context['step'] = i
        assert guard.rejection(context, scan, reask_index=0) is None
        guard.accept(context, scan)
    rejected_before = guard.rejection(context, scan, reask_index=0)
    assert rejected_before is not None
    physical = {'tool_name': 'open', 'arguments': {'object_ref': 'cabinet'}}
    guard.accept(context, physical)
    context['step'] = 99
    feedback = {**physical, 'binding_admission_or_null': {'status': 'rejected'}}
    assert guard.rejection(context, scan, reask_index=0, public_transition_feedback_or_null=feedback) is not None

@pytest.mark.parametrize('method', ['record_public_tool_failure', 'record_public_preparation_failure', 'record_public_recovery_success', 'commit_public_postcondition'])
def test_legacy_executor_labels_cannot_update_v21_memory(method):
    r = runtime(Model())
    before = r.s.memory
    with pytest.raises(ValueError, match='requires visual evidence'):
        getattr(r.s, method)(success=True)
    assert r.s.memory == before
