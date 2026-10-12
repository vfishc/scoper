import copy
import hashlib
import random
from pathlib import Path
from scoper.benchmark.common.container_inspection.definition import *

def build(source, cabinet, content, branch, index):
    task = copy.deepcopy(source)
    backend = task['backend']
    lineage = task['id']
    for key in ('family_spec', 'failure_spec', 'disturbance_spec', 'qa_overlay', 'generation', 'reset_plan', 'init_source_predicates', 'oracle_teacher_hint', 'online_requirements', 'validation', 'task_binding_spec', 'evaluation_binding', 'binding_policy', 'renderer_asset_provenance'):
        task.pop(key, None)
    roles = {'cabinet': copy.deepcopy(source['roles'][cabinet])}
    if content:
        roles['contents'] = copy.deepcopy(source['roles'][content])
    task['roles'] = roles
    task['category'] = CATEGORY
    task['template'] = {'template_id': TEMPLATE, 'template_version': '1.0'}
    if 'template_id' in task['source']:
        task['source']['template_id'] = TEMPLATE
        task['source']['template_version'] = '1.0'
    task['instruction'] = INSTRUCTION
    token = hashlib.sha256(f'v2:{lineage}:{cabinet}:{content}:{branch}:{index}'.encode()).hexdigest()[:16]
    task['id'] = f'{TEMPLATE}__{backend}__{token}'
    task['tags'] = [TEMPLATE, CATEGORY]
    predicate = 'object_closed' if branch == 'dishware' else 'object_open'
    task['goal'] = {'type': 'predicates' if backend == 'behavior' else 'predicate', 'predicates': [{'type': predicate, **({'args': {'object': 'cabinet'}} if backend == 'behavior' else {'object_role': 'cabinet'})}], 'require_model_stop': True}
    task['conditional_storage_policy'] = {'schema_version': 'conditional-storage-policy.v1', 'cabinet_role': 'cabinet', 'dishware_types': sorted(DISHWARE), 'expected_branch': branch, 'mandatory_observation_gate': False, 'initial_state': 'closed'}
    task['provenance'] = {'source_task_id': lineage, 'revision': TEMPLATE, 'changes': ['conditional_goal', 'initial_contents', 'initial_closed_state']}
    task['online_requirements'] = {'expected_events': [], 'expected_brain_behaviors': []}
    task['validation'] = {'status': 'candidate', 'checks': {}, 'failure_reason': None}
    if backend == 'behavior':
        roles['cabinet'].setdefault('init_states', {})['open'] = False
        task['init'] = []
        if content:
            obj = roles['contents']
            obj.pop('spawn_placement', None)
            obj['room_id'] = roles['cabinet'].get('room_id')
            obj['center'] = list(roles['cabinet']['center'])
            task['init'].append({'predicate': 'inside', 'args': ['contents', 'cabinet'], 'interior_floor_clearance': 0.04})
        task['init'].append({'predicate': 'closed', 'args': ['cabinet']})
    else:
        task['init'] = {'agent_pose': copy.deepcopy(source['init']['agent_pose']), 'object_pose_overrides': [], 'object_state_overrides': [{'type': 'set_open_state', 'object_role': 'cabinet', 'is_open': False}]}
        if content:
            task['init']['object_pose_overrides'].append({'type': 'relocate_object', 'object_role': 'contents', 'mode': 'same_room_as', 'to_receptacle_role': 'cabinet', 'open_target_during_relocate': True, 'keep_object_kinematic': True, 'prefer_receptacle_center': True, 'collision_checked_relocation': True, 'open_parent_during_relocate': True, 'restore_parent_state_after_relocate': True})
    return task
