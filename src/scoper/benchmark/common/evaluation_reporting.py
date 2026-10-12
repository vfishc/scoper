from collections import defaultdict

def native_qa_questions(task):
    found = {}

    def visit(spec):
        family = spec.get('family_spec') or {}
        for event in (family.get('human') or {}).get('human_events') or []:
            if event.get('type') == 'qa':
                key = str(event['id'])
                if key in found and found[key] != event:
                    raise ValueError('Conflicting native QA ID: ' + key)
                found[key] = event
        for child in (family.get('composite') or {}).get('subtasks') or []:
            visit(child)
    visit(task)
    return list(found.values())

def template_key(task):
    children = ((task.get('family_spec') or {}).get('composite') or {}).get('subtasks') or []
    if children:
        return 'composite_' + str(len(children))
    template = (task.get('template') or {}).get('template_id') or (task.get('generation') or {}).get('template_id') or task.get('template_id')
    if template:
        return str(template)
    parts = str(task['id']).split('__')
    if len(parts) >= 3:
        return parts[1]
    raise ValueError('Missing template identity: ' + str(task['id']))

def summarize_evaluation(results, tasks):
    tasks = {t['id']: t for t in tasks}
    records = {r['task_id']: r for r in results}
    if len(records) != len(results) or not records.keys() <= tasks.keys():
        raise ValueError('Duplicate or unexpected evaluation task IDs')

    def aggregate(ids):
        present = [records[i] for i in ids if i in records]
        scored = [r for r in present if r.get('status') != 'infrastructure_failed' and (r.get('failure') or {}).get('code') != 'evaluation_episode_exception']

        def mean(values):
            return sum(values) / len(values) if values else None

        def bm(r):
            return r.get('backend_metrics') or {}
        coverage = [r['metrics']['necessary_action_coverage']['coverage'] for r in scored if 'necessary_action_coverage' in (r.get('metrics') or {})]
        qa_records = [r for r in scored if (tasks[r['task_id']].get('qa_overlay') or {}).get('questions')]
        total = sum((len(tasks[r['task_id']]['qa_overlay']['questions']) for r in qa_records))
        answered = sum((bm(r).get('num_qa_answered', 0) for r in qa_records))
        posted = sum((bm(r).get('num_qa_posted', 0) for r in qa_records))
        correct = sum((bm(r).get('num_qa_correct', 0) for r in qa_records))
        native_records = [r for r in scored if native_qa_questions(tasks[r['task_id']])]
        native_total = sum((len(native_qa_questions(tasks[r['task_id']])) for r in native_records))
        native_correct = sum(((bm(r).get('native_qa') or {}).get('num_qa_correct', 0) for r in native_records))
        native_posted = sum(((bm(r).get('native_qa') or {}).get('num_qa_posted', 0) for r in native_records))
        native_answered = sum(((bm(r).get('native_qa') or {}).get('num_qa_answered', 0) for r in native_records))
        ratio = lambda value, denominator: value / denominator if denominator else None
        children = [child for r in scored for child in bm(r).get('subtask_records', [])]
        return {'runner_passed_count': sum((r.get('status') == 'passed' for r in scored)), 'task_success_with_runner_failure_count': sum((bool(r.get('task_success')) and r.get('status') != 'passed' for r in scored)), 'qa_finish_pending_count': sum((bool(bm(r).get('qa_pending_at_end')) for r in scored)), 'task_score_frozen_count': sum((bool(bm(r).get('task_score_frozen')) for r in scored)), 'selected': len(ids), 'recorded': len(present), 'pending': len(ids) - len(present), 'scored': len(scored), 'infrastructure_failed': len(present) - len(scored), 'task_success_count': sum((bool(r.get('task_success')) for r in scored)), 'task_success_rate': mean([float(bool(r.get('task_success'))) for r in scored]), 'mean_task_progress': mean([r.get('task_progress', 0) for r in scored]), 'mean_necessary_action_coverage': mean(coverage), 'coverage_scored_count': len(coverage), 'mean_high_level_steps': mean([r.get('high_level_steps', 0) for r in scored]), 'mean_elapsed_seconds': mean([bm(r)['elapsed_seconds'] for r in scored if 'elapsed_seconds' in bm(r)]), 'step_budget_violation_count': sum((r.get('high_level_steps', 0) > bm(r).get('max_high_level_steps', float('inf')) for r in scored)), 'qa_episode_count': len(qa_records), 'num_qa_total': total, 'native_qa_total': native_total, 'native_qa_correct': native_correct, 'native_qa_accuracy': ratio(native_correct, native_total), 'native_qa_posted': native_posted, 'native_qa_answered': native_answered, 'native_qa_trigger_rate': ratio(native_posted, native_total), 'native_qa_answer_rate': ratio(native_answered, native_total), 'native_qa_metrics_missing_count': sum(('native_qa' not in bm(r) for r in native_records)), 'all_qa_total': total + native_total, 'all_qa_correct': correct + native_correct, 'all_qa_accuracy': ratio(correct + native_correct, total + native_total), 'num_qa_posted': posted, 'num_qa_unposted': total - posted, 'num_qa_answered': answered, 'num_qa_unanswered': total - answered, 'num_qa_correct': correct, 'qa_metrics_missing_count': sum(('num_qa_total' not in bm(r) for r in qa_records)), 'num_qa_resolver_failures': sum((bm(r).get('num_qa_resolver_failures', 0) for r in qa_records)), 'qa_trigger_rate': ratio(posted, total), 'qa_answer_rate': ratio(answered, total), 'qa_accuracy': ratio(correct, total), 'qa_accuracy_answered': ratio(correct, answered), 'qa_accuracy_triggered': ratio(correct, posted), 'composite_children_recorded': len(children), 'composite_children_success_count': sum((bool(c.get('subtask_success')) for c in children)), 'composite_child_success_rate': mean([float(bool(c.get('subtask_success'))) for c in children]), 'qa_episode_mean_accuracy': mean([ratio(bm(r).get('num_qa_correct', 0), len(tasks[r['task_id']]['qa_overlay']['questions'])) for r in qa_records]), 'qa_success_rate': mean([float(bm(r).get('qa_success', 0)) for r in qa_records]), 'combined_success_rate': mean([float(bool(r.get('task_success')) and bm(r).get('qa_success', 0) == 1) for r in qa_records])}
    grouped = defaultdict(list)
    for task_id, task in tasks.items():
        grouped[template_key(task)].append(task_id)
    by_template = {key: aggregate(ids) for key, ids in sorted(grouped.items())}
    categories = defaultdict(list)
    families = defaultdict(list)
    for task_id, task in tasks.items():
        category = task.get('category', 'unknown')
        categories[category].append(task_id)
        family = 'composite' if category == 'composite' else 'mid_execution' if category == 'mid_execution_state_disturbance' else 'failure' if category == 'verify_after_action' else 'recovery' if category.startswith('recover_') else 'observation' if category in {'observation_conditioned_room_inspection', 'explore_then_act', 'put_away_visual_judgment', 'tidy_by_policy', 'put_away_by_category', 'put_away_cross_room'} else 'human_interaction' if category == 'human_interactive_execution' else 'regular'
        families[family].append(task_id)
    macro = {}
    for key in ('task_success_rate', 'mean_task_progress', 'mean_necessary_action_coverage', 'qa_accuracy', 'qa_trigger_rate', 'qa_answer_rate', 'combined_success_rate'):
        values = [group[key] for group in by_template.values() if group[key] is not None]
        macro[key] = sum(values) / len(values) if values else None
        macro[key + '_template_count'] = len(values)
    return {**aggregate(list(tasks)), 'schema_version': 'evaluation-summary.v2', 'by_template': by_template, 'template_macro': macro, 'by_category': {key: aggregate(ids) for key, ids in sorted(categories.items())}, 'by_family': {key: aggregate(ids) for key, ids in sorted(families.items())}, 'qa_denominator': 'all configured overlay questions on scored episodes; unposted/unanswered count as incorrect'}
