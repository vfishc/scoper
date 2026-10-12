import hashlib
import json
import re
from pathlib import Path
SCHEMA = 'necessary-action-coverage.v2'
TOOLS = frozenset({'navigate_to', 'navigate_to_room', 'pick_up', 'put_down', 'open', 'close', 'toggle_on', 'toggle_off', 'slice', 'heat', 'cool', 'soak_under', 'drop'})

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def execution_targets(task):
    roles = task.get('roles') or {}
    targets = set(roles)
    for role in roles.values():
        if isinstance(role, dict):
            targets.update((str(role[key]) for key in ('object_id', 'bddl_inst') if role.get(key)))
    return targets

def transition_signature(transition, held_before='', known_roles=()):
    tool = {'cut': 'slice'}.get(transition.get('tool_name'), transition.get('tool_name'))
    if tool not in TOOLS or transition.get('success') is not True:
        return None
    raw = transition.get('raw_execution') or {}
    if tool == 'navigate_to' and raw.get('room_id') and (not raw.get('object_id')):
        tool = 'navigate_to_room'
    if raw.get('success') is False or raw.get('status') in {'failed', 'failure', 'error'}:
        return None
    target = raw.get('object_id') or raw.get('target_object_id') or raw.get('room_id')
    if not target and re.search('\\.[a-z]\\.\\d+_\\d+$', str(raw.get('target', ''))):
        target = raw['target']
    if not target and raw.get('target') in known_roles:
        target = raw['target']
    if tool == 'navigate_to' and raw.get('interaction_ready') is False:
        return None
    subject = ''
    if tool in {'put_down', 'drop'}:
        subject = raw.get('held_object_id') or raw.get('placed_object_id') or held_before
        if not subject:
            return None
        if tool == 'drop':
            target = subject
    if not target:
        return None
    return (str(tool), str(target), str(subject))

def held_after(transition):
    held = (transition.get('observation') or {}).get('held_objects') or []
    if len(held) != 1:
        return ''
    return str(held[0].get('objectId') or held[0].get('object_id') or '') if isinstance(held[0], dict) else str(held[0])

def stage_after(task, transition, previous):
    children = ((task.get('family_spec') or {}).get('composite') or {}).get('subtasks') or []
    if not children:
        return 0
    predicates = (transition.get('parent_goal_result') or transition.get('goal_result') or {}).get('predicates') or []
    offset = 0
    for index, child in enumerate(children):
        n = len(child.get('goal_predicates') or [])
        if index >= previous and (not n or len(predicates) < offset + n or (not all((row.get('satisfied') is True for row in predicates[offset:offset + n])))):
            return max(previous, index)
        offset += n
    return len(children)

def compile_reference(task, records, *, trace_sha256):
    nodes, seen, unresolved = ([], {}, [])
    stage = 0
    held = ''
    for row in records:
        t = row.get('transition', row)
        signature = transition_signature(t, held, execution_targets(task))
        if t.get('tool_name') in TOOLS and t.get('success') is True and (signature is None):
            unresolved.append(t.get('high_level_step'))
        if signature:
            children = ((task.get('family_spec') or {}).get('composite') or {}).get('subtasks') or []
            current = min(stage, max(0, len(children) - 1))
            key = (current, *signature)
            if key not in seen:
                tool, target, subject = signature
                dependencies = []
                prior = [n for n in nodes if n['stage'] == current and n['target'] == target]
                if prior and tool != 'navigate_to':
                    dependencies.append(prior[-1]['id'])
                if subject:
                    pickups = [n for n in nodes if n['stage'] == current and n['tool'] == 'pick_up' and (n['target'] == subject)]
                    if pickups:
                        dependencies.append(pickups[-1]['id'])
                node = {'id': len(nodes), 'stage': current, 'tool': tool, 'target': target, 'subject': subject, 'requires': sorted(set(dependencies))}
                nodes.append(node)
                seen[key] = node['id']
        stage = stage_after(task, t, stage)
        held = held_after(t)
    payload = {'schema': SCHEMA, 'task_id': task['id'], 'trace_sha256': trace_sha256, 'nodes': nodes, 'unresolved_steps': unresolved, 'status': 'ready' if nodes and (not unresolved) else 'needs_review'}
    payload['reference_sha256'] = digest(payload)
    return payload

class NecessaryActionCoverage:
    name = 'necessary_action_coverage'

    def __init__(self, references):
        self.references = references

    def reset(self, context):
        self.task = context['task']
        self.reference = self.references[str(self.task['id'])]
        if self.reference['task_id'] != self.task['id']:
            raise ValueError('oracle reference task mismatch')
        payload = {k: v for k, v in self.reference.items() if k != 'reference_sha256'}
        if digest(payload) != self.reference['reference_sha256']:
            raise ValueError('oracle reference hash mismatch')
        if self.reference['status'] != 'ready':
            raise ValueError('oracle reference needs review')
        self.covered = set()
        self.stage = 0
        self.held = ''

    def on_transition(self, transition):
        signature = transition_signature(transition, self.held, execution_targets(self.task))
        if signature:
            for node in self.reference['nodes']:
                if node['id'] not in self.covered and node['stage'] <= self.stage and ((node['tool'], node['target'], node['subject']) == signature) and (set(node['requires']) <= self.covered):
                    self.covered.add(node['id'])
                    break
        self.stage = stage_after(self.task, transition, self.stage)
        self.held = held_after(transition)

    def finalize(self, result):
        total = len(self.reference['nodes'])
        return {'schema': SCHEMA, 'matched': len(self.covered), 'total': total, 'coverage': len(self.covered) / total, 'reference_sha256': self.reference['reference_sha256']}

def load_references(path, bank_sha256):
    value = json.loads(Path(path).read_text())
    if value['schema'] != SCHEMA or value['bank_sha256'] != bank_sha256:
        raise ValueError('oracle reference bank mismatch')
    return value['tasks']
