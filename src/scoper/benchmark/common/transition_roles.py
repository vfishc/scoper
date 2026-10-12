def target_roles(env, transition):
    roles = getattr(env, 'role_object_ids', None) or getattr(env, 'role_map', None) or {}
    arguments = transition.get('arguments') or {}
    target = arguments.get('target')
    result = {str(target)} if target is not None and (not roles or target in roles) else set()
    raw = transition.get('raw_execution') or {}
    object_id = raw.get('object_id') or raw.get('target_object_id') or raw.get('target')
    if object_id:
        result.update((str(role) for role, value in roles.items() if str(value) == str(object_id)))
    return result
