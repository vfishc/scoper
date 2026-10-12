import json
import struct
import zlib
from scoper.harness import JSONBackbone, RGBFrame, ExecutionReceipt
from scoper.harness.backends import ProcTHORAdapter, BEHAVIORAdapter
from scoper.harness.evaluation import evaluate_episode
CHECKPOINT = 'a' * 64
TOOLS = [dict(name=n, parameters=dict(type='object', properties={}, required=[], additionalProperties=False)) for n in ['scan', 'stop']]

def png():

    def chunk(tag, data):
        return struct.pack('!I', len(data)) + tag + data + struct.pack('!I', zlib.crc32(tag + data) & 4294967295)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 512, 512, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress((b'\x00' + bytes([128, 128, 128]) * 512) * 512)) + chunk(b'IEND', b'')

def run(backend, harness=True, json_transport=False):
    state = dict(scans=0, stopped=False, posted=False, answered=False, correct=False)
    frames = lambda n: RGBFrame(f'frame-{n}', png())
    calls = iter(['scan', 'stop'])
    requests = []

    def generate(role, system, request, images):
        requests.append((role, request, images))
        if role == 'agenda':
            result = dict(items=[dict(id='a1', requirement='Observe the scene', mode='required', status='pending', evidence_refs=[], next_intent_or_null=None, criterion='inspection')], active_item_id_or_null=None)
        elif role == 'memory':
            result = dict(schema_version='memory-proposer-io.v1', proposal_id=request['proposal_id'], mode=request['mode'], base_revision=request['base_revision'], evidence_frame_ids=request['evidence_frame_ids'], fact_mutations=[], incident_mutations=[], obligation_mutations=[], per_fact_status=[], relevant_memory_ids=[], abstain_reason_or_null='Fixture has no semantic updates', confidence=1.0, checkpoint_sha256=CHECKPOINT)
        elif role == 'policy':
            result = dict(tool_name=next(calls), arguments={})
        elif role == 'answer':
            result = dict(question_id=request['question']['question_id'], value='yes')
        else:
            raise AssertionError(role)
        return json.dumps(result) if json_transport else result

    def execute(call):
        assert call['tool_name'] == 'scan'
        state['scans'] += 1
        return ExecutionReceipt(frames(state['scans']))

    def stop():
        state['stopped'] = True

    def poll():
        if state['stopped'] and (not state['posted']):
            state['posted'] = True
            return (dict(kind='question', question_id='question-1', text='Was an observation requested?', choices=['yes', 'no']),)
        return ()

    def answer(a):
        state['answered'] = True
        state['correct'] = a['value'] == 'yes'

    def counts():
        return dict(coverage_matched=int(state['scans'] > 0), coverage_total=1, qa_configured=1, qa_triggered=int(state['posted']), qa_correct=int(state['correct']))
    adapter = (ProcTHORAdapter if backend == 'procthor' else BEHAVIORAdapter)(execute=execute, expected_effect=lambda c, m: None, goal_predicates=lambda: [state['scans'] > 0], submit_answer=answer, poll_public_messages=poll, scoring_counts=counts, on_stop=stop)
    result = evaluate_episode(dict(id='fixture', instruction='Observe, then stop.'), adapter, JSONBackbone(generate, CHECKPOINT), TOOLS, [], frames(0), harness=harness)
    assert result['high_level_steps'] == 3 and result['task_success'] == 1 and (result['qa_accuracy'] == 1)
    assert not any(('goal_satisfied' in r or 'evaluator_answer' in r for _, r, _ in requests))
    return result
if __name__ == '__main__':
    print(json.dumps({'run_type': 'scripted_fixture', 'results': [run(b, h) for b in ['procthor', 'behavior'] for h in [False, True]]}, indent=2))
