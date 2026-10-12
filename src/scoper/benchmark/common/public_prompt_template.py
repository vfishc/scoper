from string import Formatter
FIELD_EXPOSURE = {'object_catalog_summary': 'visible_catalog', 'held_objects': 'held_objects', 'agent_pose': 'agent_pose', 'current_room': 'current_room', 'target_handles': 'target_handles', 'room_handles': 'room_handles', 'human_messages': 'human_messages', 'visible_issues': 'visible_issues', 'scheduler_notices': 'scheduler_notices', 'prompt_context_json': 'structured_memory', 'goal': 'goal_progress', 'goal_progress': 'goal_progress', 'tool_progress': 'goal_progress', 'task_progress': 'goal_progress', 'task_success': 'goal_progress'}

def prepare_public_template(template, exposure, *, hide_empty_human_messages=False):
    denied = {name for name, flag in FIELD_EXPOSURE.items() if not getattr(exposure, flag, False)}
    if hide_empty_human_messages:
        denied.add('human_messages')
    private_guidance = ['last result failed', 'goal is complete', 'task is complete']
    if not getattr(exposure, 'target_handles', False):
        private_guidance += ['role handles', 'target handles', 'target_object', 'target_receptacle', 'stateful_object']
    if not getattr(exposure, 'room_handles', False):
        private_guidance += ['room ids', 'room_id', 'room_target_handles']
    out = []
    for line in template.splitlines():
        fields = {field.split('.')[0].split('[')[0] for _, field, _, _ in Formatter().parse(line) if field}
        literals = ''.join((literal for literal, _, _, _ in Formatter().parse(line))).lower()
        if fields & denied:
            if out and out[-1].strip().endswith(':'):
                out.pop()
            continue
        if any((phrase in literals for phrase in private_guidance)):
            continue
        out.append(line)
    return '\n'.join(out)

def require_complete_tool_schemas(rendered, serialized_schemas):
    if serialized_schemas not in rendered:
        raise ValueError('Public tool schemas missing or modified during prompt assembly')
