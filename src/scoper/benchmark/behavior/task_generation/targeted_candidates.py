from __future__ import annotations
import hashlib, itertools, random
from collections import Counter
from pathlib import Path
from scoper.benchmark.common.canonical import canonical_json_bytes
from scoper.benchmark.common.task_identity_policy import conflicts
from scoper.benchmark.behavior.task_generation.scene_matcher import load_procthor_templates, match_scene_to_templates
from scoper.benchmark.behavior.task_generation.profile_task_generator import generate_stage4_tasks
from scoper.benchmark.behavior.task_generation.composite import build_composite_task_spec
from scoper.benchmark.behavior.task_generation.static_scene_admission import evaluate_static_scene_admission

def generate_targeted(profile, *, template_ids, job_type, seed, template_dir, assets_root, composite_id, composite_size=None, draws=8, group_limit=2048):
    templates = {t['template_id']: t for t in load_procthor_templates(template_dir)}
    requested = list(template_ids)
    if len(set(requested)) != len(requested):
        raise ValueError('Repeated template positions require explicit slot handling')
    pools = {name: [] for name in requested}
    counts = Counter()
    reports = []
    for name in requested:
        seen = set()
        for draw in range(draws):
            matches = match_scene_to_templates(profile, [templates[name]], max_bindings_per_template=4, candidate_seed=int(seed) + draw)
            if not matches[0].get('bindings'):
                counts['NO_BINDING:' + name] += 1
                continue
            generated = generate_stage4_tasks(profiles=[profile], templates=[templates[name]], match_records=matches, template_ids=[name], max_tasks_per_template_scene=4, seed=int(seed) + draw)
            for task in generated['tasks']:
                if job_type == 'composite' and (task.get('disturbance_spec') or task.get('failure_spec')):
                    counts['UNSUPPORTED_CHILD_SCHEDULER'] += 1
                    continue
                if conflicts(task):
                    counts['CHILD_IDENTITY_CONFLICT'] += 1
                    continue
                digest = hashlib.sha256(canonical_json_bytes({k: v for k, v in task.items() if k != 'id'})).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                pools[name].append(task)
        random.Random(str(seed) + name).shuffle(pools[name])
    cache = {}
    root = Path(assets_root)
    root = root if (root / 'datasets/behavior-1k-assets').is_dir() else root.parent.parent

    def admitted(task):
        key = hashlib.sha256(canonical_json_bytes(task)).hexdigest()
        if key not in cache:
            report = evaluate_static_scene_admission(task, rng_seed=int(seed), max_high_level_steps=180, assets_root=root)
            reports.append({'task_id': task.get('id'), 'report': report})
            cache[key] = report.get('status') == 'admitted'
        return cache[key]

    def proof(reason, selected=False):
        return {'reason': reason, 'selected_template_ids': requested if selected else [], 'candidate_counts': {k: len(v) for k, v in pools.items()}, 'rejection_code_counts': dict(counts), 'reports': reports, 'search_version': 'targeted-composite.v1', 'start_pose_policy': 'parent adopts first child pose; full parent symbolic validation is mandatory'}
    if any((not p for p in pools.values())):
        return (None, proof('REQUIRED_TEMPLATE_HAS_NO_CANDIDATES'))
    if job_type == 'atomic':
        for task in pools[requested[0]]:
            if admitted(task):
                return (task, proof('SELECTED', True))
        return (None, proof('ATOMIC_STATIC_REJECTED'))
    if len(requested) != int(composite_size or 0):
        raise ValueError('Composite size mismatch')
    indices = list(itertools.product(*(range(len(pools[n])) for n in requested)))
    random.Random(seed).shuffle(indices)
    for positions in indices[:group_limit]:
        children = [pools[name][index] for name, index in zip(requested, positions)]
        result = build_composite_task_spec(children, composite_id=composite_id, max_subtasks=len(requested), require_independent_goals=True, reject_scheduler_specs=True)
        if result.task is None:
            for row in result.rejected:
                counts[row.get('code', 'COMPOSITE_BUILD_REJECTED')] += 1
            continue
        if conflicts(result.task):
            counts['PARENT_IDENTITY_CONFLICT'] += 1
            continue
        if not all((admitted(t) for t in children)):
            counts['CHILD_STATIC_REJECTED'] += 1
            continue
        if admitted(result.task):
            return (result.task, proof('SELECTED', True))
        counts['PARENT_STATIC_REJECTED'] += 1
    return (None, proof('TARGETED_GROUPS_EXHAUSTED'))
