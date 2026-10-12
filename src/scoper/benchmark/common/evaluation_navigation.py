from copy import deepcopy
VERSION = 'evaluation-navigation.v2'

def enabled(env):
    return getattr(env, 'eval_mode', None) == 'standard_vlm'

def reset(env):
    env._relocation_navigation_receipts = {}

def relocated(env, object_id, previous_position):
    if not enabled(env) or not object_id:
        return
    if not hasattr(env, '_relocation_navigation_receipts'):
        reset(env)
    env._relocation_navigation_receipts[str(object_id)] = deepcopy(previous_position)

def consume_previous_position(env, object_id):
    if not enabled(env) or not object_id:
        return None
    return getattr(env, '_relocation_navigation_receipts', {}).pop(str(object_id), None)

def has_previous_position(env, object_id):
    return enabled(env) and str(object_id) in getattr(env, '_relocation_navigation_receipts', {})
