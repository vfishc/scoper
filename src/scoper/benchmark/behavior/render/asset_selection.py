import copy
from pathlib import Path
from scoper.benchmark.common.asset_diversity import POLICY_VERSION, choose_asset
from scoper.benchmark.common.task_identity_policy import physical_identity

def pin_spawn_models(task, *, assets_root, usage=None, seed=None):
    resolved = copy.deepcopy(dict(task))
    usage = usage if usage is not None else {}
    identities = {}
    assignments = []
    for name, role in resolved.get('roles', {}).items():
        if not role.get('requires_spawn'):
            continue
        identity = physical_identity(role, name)
        if role.get('model') not in (None, '', 'unknown'):
            pin = (role.get('category'), role['model'])
            if identity in identities and identities[identity] != pin:
                raise ValueError('CONFLICTING_FIXED_ASSET_ALIASES:' + str(identity))
            identities[identity] = pin
    for name, role in sorted(resolved.get('roles', {}).items()):
        if not role.get('requires_spawn'):
            continue
        identity = physical_identity(role, name)
        category = str(role.get('category') or '')
        if not category or Path(category).name != category or category in ('.', '..'):
            raise ValueError('INVALID_SPAWN_CATEGORY:' + name)
        if identity in identities:
            pinned_category, model = identities[identity]
            if pinned_category != category:
                raise ValueError('CONFLICTING_FIXED_ASSET_ALIASES:' + str(identity))
        else:
            paths = (Path(assets_root) / 'objects' / category).glob('*/usd/*.encrypted.usd')
            candidates = [dict(object_type=role.get('object_type') or category, category=category, model=p.parent.parent.name) for p in paths if p.name == p.parent.parent.name + '.encrypted.usd']
            model = choose_asset(candidates, seed=(seed, task.get('id'), name), usage=usage)['model']
            identities[identity] = (category, model)
            assignments.append({'role': name, 'category': category, 'model': model})
        role['model'] = model
    provenance = {'schema_version': POLICY_VERSION, 'status': 'skipped_by_user', 'audit_performed': False, 'failures': [], 'assignments': assignments, 'requires_symbolic_and_oracle_admission': True}
    resolved['renderer_asset_provenance'] = provenance
    return (resolved, provenance)
