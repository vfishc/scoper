import hashlib
PROTOCOL_VERSION = 'direct-vlm-evaluation.v3'
SCAN_DESCRIPTION = 'Look around for additional RGB views; usable at different task stages and during recovery.'

def standard_system_rules(direct_multistep_v2=False):
    rules = 'Standard visual-evaluation rules:\n- Infer world state and action outcomes from the supplied RGB observations. Previous requested actions are history, not proof of success.\n- Use short natural-language references to exactly one object or receptacle. Never invent or use simulator IDs, task role handles, or private state.\n- You may navigate toward a natural object name explicitly present in the instruction even when it is not visible yet.\n- Navigate to the relevant object before physical interaction. If navigation did not reach the intended target, observe or navigate again before interacting.\n- Pick_up targets the movable object; put_down targets its destination receptacle. Open a closed container before using its interior.\n- Complete every clause, including composite subtasks and required device cleanup. Use the available tool definitions and visual evidence to choose the next action.'
    if direct_multistep_v2:
        rules += '\n- Maintain a checklist for every clause in the full instruction, including all composite subtasks.'
    return rules

def child_count(task):
    children = ((task.get('family_spec') or {}).get('composite') or {}).get('subtasks') or []
    if children and len(children) not in (2, 3):
        raise ValueError('evaluation supports only two or three composite children')
    return len(children) or 1

def step_budget(task, override=None):
    if override is not None:
        if isinstance(override, bool) or int(override) <= 0:
            raise ValueError('step budget must be positive')
        return int(override)
    return {1: 24, 2: 36, 3: 48}[child_count(task)]

def episode_seed(task_id, seed):
    if not task_id:
        raise ValueError('task id required for stable seed')
    return int.from_bytes(hashlib.sha256(f'{seed}:{task_id}'.encode()).digest()[:4], 'big')

def layout_key(task):
    source = task.get('source') or {}
    metadata = task.get('metadata') or {}
    if source.get('scene_model'):
        return str(source['scene_model']) + ':' + str(source.get('layout_variant_id') or source.get('layout_id') or '')
    return str(source.get('layout_id') or source.get('house_id') or source.get('scene_model') or metadata.get('layout_id') or task.get('house_id') or '')

def standard_turn_rules():
    return 'Visual closed-loop policy:\n- Navigate to the relevant object before physical interaction.\n- Infer action outcomes from RGB; a requested action does not prove success.\n- Use scan whenever another view is needed, including during recovery or another subtask; each scan supplies at most eight frames and consumes one step. Avoid repeating unchanged views.\n- If placement released the object without placing it correctly, navigate to the object and pick it up again before another placement attempt.\n- The green/cyan sphere attached to the gripper is a simulator marker, not a held object. Identify a separate object; do not ignore real green objects.\n- Call stop when current visual evidence supports all clauses of the instruction, including all composite subtasks. No explicit instruction to say stop is required.'
