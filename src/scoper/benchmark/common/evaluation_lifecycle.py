from copy import deepcopy
from scoper.benchmark.common.evaluation_binding_policy import task_uses_refined_binding

def enabled(env):
    return getattr(env, 'eval_mode', None) == 'standard_vlm' and task_uses_refined_binding(getattr(env, 'current_task_spec', None))

def reset(env):
    env._evaluation_stop_goal = None
    env._evaluation_stop_step = None
    env._evaluation_finish_blocked = 0

def finishing(env):
    return enabled(env) and getattr(env, '_evaluation_stop_goal', None) is not None

def frozen_goal(env):
    return deepcopy(env._evaluation_stop_goal) if finishing(env) else None

def guard_tool(env, tool_name, arguments):
    if not finishing(env) or tool_name == 'answer':
        return None
    env.high_level_step += 1
    env._evaluation_finish_blocked += 1
    transition = {'tool_name': tool_name, 'arguments': dict(arguments), 'high_level_step': env.high_level_step, 'task_action_step': getattr(env, 'task_action_step', 0), 'success': False, 'done': False, 'done_reason': None, 'env_feedback': 'Task execution has ended. Only answers to pending questions are allowed.', 'observation': env._observation(), 'goal_result': frozen_goal(env), 'raw_execution': {'action': tool_name, 'status': 'qa_finish_only', 'action_executed': False}, 'low_level_actions': [], 'num_low_level_steps': 0, 'episode_metrics': env._episode_metrics()}
    env.tool_history.append(transition)
    return transition

def after_transition(env, transition):
    if not enabled(env):
        return
    if not finishing(env) and transition.get('tool_name') == 'stop' and getattr(env, 'model_stopped', False):
        env._evaluation_stop_goal = deepcopy(transition.get('parent_goal_result') or env.check_goal(model_stopped=True))
        env._evaluation_stop_step = env.high_level_step
    if finishing(env):
        pending = bool((getattr(env, '_human_qa_state', {}) or {}).get('pending'))
        transition['goal_result'] = frozen_goal(env)
        transition['done'] = not pending
        transition['done_reason'] = None if pending else 'task_stopped_qa_finished'

def diagnostics(env):
    if not enabled(env):
        return {}
    pending = len((getattr(env, '_human_qa_state', {}) or {}).get('pending') or {})
    frozen = frozen_goal(env) or {}
    return {'evaluation_lifecycle': 'direct-vlm-stop-qa.v1', 'task_success_at_stop': frozen.get('task_success') if finishing(env) else None, 'qa_finish_budget_exhausted': bool(finishing(env) and pending and (getattr(env, 'high_level_step', 0) >= getattr(env, 'max_high_level_steps', float('inf')))), 'task_stop_step': getattr(env, '_evaluation_stop_step', None), 'task_score_frozen': finishing(env), 'qa_finish_blocked_calls': getattr(env, '_evaluation_finish_blocked', 0), 'qa_pending_at_end': pending}
