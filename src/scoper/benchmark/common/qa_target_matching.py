from collections.abc import Mapping

def transition_targets_role(role_name, expected_id, transition):
    if not role_name:
        return False
    raw = transition.get('raw_execution') or {}
    ready = transition.get('ready_target') or {}
    raw = raw if isinstance(raw, Mapping) else {}
    ready = ready if isinstance(ready, Mapping) else {}
    raw_id = raw.get('object_id') or raw.get('target_object_id') or raw.get('target')
    ready_id = ready.get('object_id')
    if expected_id and (raw_id == expected_id or (not raw_id and ready_id == expected_id)):
        return True
    if raw_id and raw_id == ready_id and (ready.get('binding_role') == role_name):
        return True
    args = transition.get('arguments') or {}
    target = args.get('target') if isinstance(args, Mapping) else None
    return target == role_name or bool(expected_id and target == expected_id)
