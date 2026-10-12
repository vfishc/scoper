from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence
from scoper.benchmark.behavior.task_generation.template_activation import TEMPLATE_DISABLED_FAILURE_CODE
VISUAL_DEFERRED_CODES = frozenset({'visual_discoverability', 'visibility_query_or_controlled_spawn_visibility', 'VISUAL_STREAM_UNAVAILABLE', 'RENDERER_UNAVAILABLE'})
UNSUPPORTED_CODE_PREFIXES = ('UNSUPPORTED', 'scheduler:unsupported:', 'family_policy:', 'family_scheduler:', 'family_teacher:')
PROCTHOR_NON_GOAL_CODES = frozenset({'TEMPLATE_NOT_IN_STAGE4_SCOPE', 'TEMPLATE_NOT_IN_STAGE4_5_GOLD_SCOPE', TEMPLATE_DISABLED_FAILURE_CODE})

def stage4_validation_records(generation: Mapping[str, Any], structure_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_task = {str(item.get('task_id')): item for item in structure_results}
    rows: list[dict[str, Any]] = []
    for task in generation.get('tasks') or []:
        result = by_task.get(str(task.get('id'))) or {}
        accepted = bool(result.get('structure_ok'))
        rows.append(_record(stage='stage4_task_generation', status='accepted' if accepted else 'rejected', task=task, result=result, failure_code=result.get('failure_code')))
    for rejected in generation.get('rejected') or []:
        rows.append(_record(stage='stage4_task_generation', status='rejected', rejected=rejected, failure_code=rejected.get('failure_code')))
    return rows

def stage5_validation_records(generation: Mapping[str, Any], validation_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tasks_by_id = {str(task.get('id')): task for task in generation.get('tasks') or []}
    rows = [_record(stage='stage5_reset_mutation', status='accepted' if result.get('stage5_reset_mutation_ok') else 'rejected', task=tasks_by_id.get(str(result.get('task_id'))), result=result, failure_code=result.get('failure_code'), deferred_codes=result.get('deferred_codes') or result.get('mutation_deferred_codes') or []) for result in validation_results]
    for rejected in generation.get('rejected') or []:
        rows.append(_record(stage='stage5_reset_mutation', status='rejected', rejected=rejected, failure_code=rejected.get('failure_code')))
    return rows

def stage6_validation_records(generation: Mapping[str, Any], stage6_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tasks_by_id = {str(task.get('id')): task for task in generation.get('tasks') or []}
    rows = [_record(stage='stage6_execution', status='accepted' if result.get('execution_ok') else 'rejected', task=tasks_by_id.get(str(result.get('task_id'))), result=result, failure_code=result.get('failure_code'), deferred_codes=result.get('deferred_codes') or []) for result in stage6_results]
    for rejected in generation.get('rejected') or []:
        rows.append(_record(stage='stage6_execution', status='rejected', rejected=rejected, failure_code=rejected.get('failure_code')))
    return rows

def summarize_validation_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, Any]] = {}
    by_family: dict[str, dict[str, Any]] = {}
    for field, output in (('category', by_category), ('family', by_family)):
        values = sorted({str(record.get(field) or 'unknown') for record in records})
        for value in values:
            items = [record for record in records if str(record.get(field) or 'unknown') == value]
            output[value] = _summary_item(items)
    return {'total': len(records), 'accepted_total': sum((1 for record in records if record.get('status') == 'accepted')), 'rejected_total': sum((1 for record in records if record.get('status') == 'rejected')), 'buckets': dict(sorted(Counter((str(record.get('bucket') or 'unknown') for record in records)).items())), 'by_category': by_category, 'by_family': by_family}

def family_coverage(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_family[_family_from_task(task)].append(task)
    families = {}
    for family in sorted(by_family):
        items = by_family[family]
        families[family] = {'generated_count': len(items), 'has_profile_backed_example': any((_is_profile_backed(task) for task in items)), 'template_ids': sorted({str((task.get('template') or {}).get('template_id') or (task.get('source') or {}).get('template_id')) for task in items if (task.get('template') or {}).get('template_id') or (task.get('source') or {}).get('template_id')}), 'categories': sorted({str(task.get('category') or 'unknown') for task in items})}
    return {'family_count': len(families), 'families': families, 'all_profile_backed_families': sorted((family for family, item in families.items() if item['has_profile_backed_example']))}

def split_records(records: Sequence[Mapping[str, Any]], status: str) -> list[dict[str, Any]]:
    return [dict(record) for record in records if record.get('status') == status]

def _record(*, stage: str, status: str, task: Mapping[str, Any] | None=None, result: Mapping[str, Any] | None=None, rejected: Mapping[str, Any] | None=None, failure_code: Any=None, deferred_codes: Iterable[Any]=()) -> dict[str, Any]:
    source = result or rejected or task or {}
    metadata = task or source
    template_id = metadata.get('template', {}).get('template_id') if isinstance(metadata.get('template'), Mapping) else None
    template_id = template_id or source.get('template_id')
    category = metadata.get('category') or source.get('category')
    failure_code = failure_code or source.get('failure_code')
    deferred = sorted((str(item) for item in deferred_codes or source.get('deferred_codes') or [] if item))
    row = {'stage': stage, 'status': status, 'bucket': _bucket(status=status, failure_code=failure_code, deferred_codes=deferred, source=source), 'task_id': metadata.get('id') or source.get('task_id'), 'template_id': template_id, 'category': category, 'family': _family_from_task(task) if task else str(source.get('family') or category or 'unknown'), 'scene_model': (metadata.get('source') or {}).get('scene_model') if isinstance(metadata.get('source'), Mapping) else source.get('scene_model'), 'failure_code': failure_code, 'failure_reason': source.get('failure_reason'), 'deferred_codes': deferred}
    if rejected:
        row['stage3_status'] = rejected.get('stage3_status')
    return row

def _bucket(*, status: str, failure_code: Any, deferred_codes: Sequence[str], source: Mapping[str, Any]) -> str:
    if status == 'accepted':
        return 'accepted'
    code = str(failure_code or '')
    deferred = {str(item) for item in deferred_codes}
    if code == 'TASK_SPEC_VALIDATION_FAILED' or source.get('structure_ok') is False:
        return 'invalid_task_spec'
    if code in PROCTHOR_NON_GOAL_CODES:
        return 'procthor_specific_non_goal'
    if deferred & VISUAL_DEFERRED_CODES:
        return 'renderer_or_visual_stream_unavailable'
    if code.startswith(UNSUPPORTED_CODE_PREFIXES) or any((item.startswith(UNSUPPORTED_CODE_PREFIXES) for item in deferred)):
        return 'unsupported_symbolic_capability'
    if deferred:
        return 'unsupported_symbolic_capability'
    return 'execution_or_validation_failed'

def _summary_item(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {'total': len(items), 'accepted': sum((1 for item in items if item.get('status') == 'accepted')), 'rejected': sum((1 for item in items if item.get('status') == 'rejected')), 'buckets': dict(sorted(Counter((str(item.get('bucket') or 'unknown') for item in items)).items())), 'template_ids': sorted({str(item.get('template_id')) for item in items if item.get('template_id')})}

def _family_from_task(task: Mapping[str, Any] | None) -> str:
    if not isinstance(task, Mapping):
        return 'unknown'
    family_spec = task.get('family_spec')
    if isinstance(family_spec, Mapping) and family_spec.get('family'):
        return str(family_spec.get('family'))
    return str(task.get('category') or 'unknown')

def _is_profile_backed(task: Mapping[str, Any]) -> bool:
    source = task.get('source') if isinstance(task.get('source'), Mapping) else {}
    return bool(source.get('stage3_match_status') or source.get('scene_model'))
__all__ = ['family_coverage', 'split_records', 'stage4_validation_records', 'stage5_validation_records', 'stage6_validation_records', 'summarize_validation_records']
