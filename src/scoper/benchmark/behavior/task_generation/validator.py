from __future__ import annotations
from typing import Any
from scoper.benchmark.behavior.behavior_closed_loop_tool_env import BehaviorClosedLoopToolEnv
from scoper.benchmark.behavior.brains.oracle import OracleBrain
from scoper.benchmark.behavior.episode_runner import EpisodeRunner
from scoper.benchmark.behavior.task_spec import validate_behavior_task_spec

def validate_one_task(task_spec: dict, *, check_oracle: bool=True) -> dict:
    result: dict[str, Any] = {'task_id': task_spec.get('id'), 'structure_ok': False, 'oracle_ok': None, 'failure_reason': None}
    try:
        validate_behavior_task_spec(task_spec)
        result['structure_ok'] = True
    except Exception as exc:
        result['failure_reason'] = 'structure: {}'.format(exc)
        return result
    if not check_oracle:
        return result
    try:
        env = BehaviorClosedLoopToolEnv()
        runner = EpisodeRunner(env, max_high_level_steps=50)
        ep = runner.run(task_spec, brain=OracleBrain(), rng_seed=0)
        result['oracle_ok'] = ep.final_goal.get('task_success') == 1.0
        if not result['oracle_ok']:
            result['failure_reason'] = 'oracle: task_success={} predicates={}'.format(ep.final_goal.get('task_success'), ep.final_goal.get('predicates'))
    except Exception as exc:
        result['oracle_ok'] = False
        result['failure_reason'] = 'oracle_error: {}'.format(exc)
    return result

def validate_task_bank(tasks: list[dict], *, check_oracle: bool=True) -> dict:
    results = [validate_one_task(t, check_oracle=check_oracle) for t in tasks]
    passed = [r for r in results if r['structure_ok']]
    solvable = [r for r in results if r.get('oracle_ok') is True]
    return {'total': len(tasks), 'passed': len(passed), 'failed': len(tasks) - len(passed), 'solvable': len(solvable), 'unsolvable': len(passed) - len(solvable), 'report': results}
__all__ = ['validate_one_task', 'validate_task_bank']
