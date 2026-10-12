from collections import deque
from copy import deepcopy
from math import isfinite
from scoper.harness.runtime import SCOPERRuntime, ExecutionReceipt
from scoper.harness.evaluation_policy import step_budget, standard_system_rules
from scoper.harness.prompts import ANSWER_SYSTEM_BYTES

def score_episode(*, stopped, goal_satisfied, coverage_matched, coverage_total, qa_configured, qa_triggered, qa_correct):
    values = (coverage_matched, coverage_total, qa_configured, qa_triggered, qa_correct)
    if any((type(v) is not int or v < 0 for v in values)):
        raise ValueError('counts must be nonnegative integers')
    if coverage_matched > coverage_total or not qa_correct <= qa_triggered <= qa_configured:
        raise ValueError('inconsistent denominators')
    goals = tuple(goal_satisfied)
    if not goals or any((type(v) is not bool for v in goals)):
        raise ValueError('nonempty Boolean goal results required')
    return dict(task_success=float(stopped and all(goals)), task_progress=sum(goals) / len(goals), coverage=coverage_matched / coverage_total if coverage_total else None, coverage_matched=coverage_matched, coverage_total=coverage_total, qa_configured=qa_configured, qa_triggered=qa_triggered, qa_correct=qa_correct, qa_trigger_rate=qa_triggered / qa_configured if qa_configured else None, qa_accuracy=qa_correct / qa_triggered if qa_triggered else None)

def aggregate(records):
    if not records:
        raise ValueError('no scored episodes')

    def mean(key):
        values = [r[key] for r in records if r[key] is not None]
        return sum(values) / len(values) if values else None
    n, t, c = (sum((r[k] for r in records)) for k in ('qa_configured', 'qa_triggered', 'qa_correct'))
    return dict(episodes=len(records), task_success_rate=mean('task_success'), task_progress=mean('task_progress'), coverage=mean('coverage'), qa_configured=n, qa_triggered=t, qa_correct=c, qa_trigger_rate=t / n if n else None, qa_accuracy=c / t if t else None)

class DirectRuntime:

    def __init__(self, backbone, tools, vocabulary):
        self.backbone = backbone
        self.checkpoint = backbone.checkpoint_sha256
        self.tools = deepcopy(tools)
        self.vocabulary = sorted(set(vocabulary))
        self.frames = deque(maxlen=32)
        self.history = deque(maxlen=48)
        self.questions = deque()
        self.question_ids = set()

    def start(self, episode_id, instruction, initial_frame, *, budget=24):
        if budget not in {24, 36, 48}:
            raise ValueError('invalid episode budget')
        self.instruction, self.budget = (instruction, budget)
        self.frames.clear()
        self.frames.append(initial_frame)
        self.history.clear()
        self.questions.clear()
        self.question_ids.clear()
        self.steps = self.model_calls = 0
        self.stopped = False

    def run(self, adapter):
        while self.steps < self.budget:
            for event in adapter.poll_public_messages():
                if event['kind'] == 'task_update':
                    self.instruction += '\nAdditional requirement: ' + event['text']
                elif event['kind'] == 'question' and event['question_id'] not in self.question_ids:
                    self.questions.append(deepcopy(event))
                    self.question_ids.add(event['question_id'])
                elif event['kind'] != 'question':
                    raise ValueError('unsupported public event')
            if self.stopped and (not self.questions):
                break
            request = dict(instruction=self.instruction, tool_history=list(self.history), remaining_budget=self.budget - self.steps, allowed_tool_specs=self.tools, public_vocabulary=self.vocabulary)
            question = self.questions.popleft() if self.questions else None
            if question:
                request['question'] = question
            if self.backbone.checkpoint_sha256 != self.checkpoint:
                raise ValueError('backbone changed during episode')
            result = self.backbone.complete('answer' if question else 'policy', ANSWER_SYSTEM_BYTES if question else standard_system_rules().encode(), request, tuple(self.frames))
            self.model_calls += 1
            if question:
                if set(result) != {'question_id', 'value'} or result['question_id'] != question['question_id']:
                    raise ValueError('invalid answer response')
                adapter.submit_answer(result)
                call = dict(tool_name='answer', arguments=deepcopy(result))
            else:
                call = deepcopy(dict(result))
                if set(call) != {'tool_name', 'arguments'} or not isinstance(call['arguments'], dict):
                    raise ValueError('expected one structured tool call')
                specs = {s['name']: s['parameters'] for s in self.tools}
                if call['tool_name'] not in specs or call['tool_name'] in {'answer', 'handoff'}:
                    raise ValueError('unknown or evaluator-owned action')
                schema = specs[call['tool_name']]
                if not set(schema.get('required', ())).issubset(call['arguments']):
                    raise ValueError('missing tool arguments')
                if schema.get('additionalProperties') is False and (not set(call['arguments']).issubset(schema.get('properties', {}))):
                    raise ValueError('unexpected tool arguments')
                if call['tool_name'] == 'stop':
                    adapter.stop()
                    self.stopped = True
                else:
                    receipt = adapter.execute(call)
                    if not isinstance(receipt, ExecutionReceipt):
                        raise TypeError('explicit public RGB receipt required')
                    frames = SCOPERRuntime.sample((*receipt.process_frames, receipt.post))
                    self.frames.extend(frames)
            self.history.append(call)
            self.steps += 1
        return dict(high_level_steps=self.steps, model_calls=self.model_calls, termination='stopped' if self.stopped else 'budget_exhausted')

    def finish(self):
        self.frames.clear()
        self.history.clear()
        self.questions.clear()

def evaluate_episode(task, adapter, backbone, tools, vocabulary, initial_frame, *, harness=True):
    if any((t['name'] == 'handoff' for t in tools)):
        raise ValueError('composite handoff belongs to the evaluator, not the model')
    runtime = (SCOPERRuntime if harness else DirectRuntime)(backbone, tools, vocabulary)
    try:
        runtime.start(task['id'], task['instruction'], initial_frame, budget=step_budget(task))
        record = runtime.run(adapter)
        return dict(task_id=task['id'], backend=adapter.backend_name, mode='harness' if harness else 'direct', **record, **adapter.finalize())
    finally:
        runtime.finish()
