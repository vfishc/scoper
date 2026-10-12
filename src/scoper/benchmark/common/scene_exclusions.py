from copy import deepcopy

def prune_procthor_house(house, task):
    excluded = set((task.get('source') or {}).get('excluded_object_ids') or [])
    if not excluded:
        return house

    def role_ids(view):
        ids = {r.get('object_id') for r in (view.get('roles') or {}).values()}
        for child in ((view.get('family_spec') or {}).get('composite') or {}).get('subtasks', []):
            ids.update(role_ids(child))
        return ids
    protected = role_ids(task)
    found = set()

    def visit(objects, parent_removed=False):
        kept = []
        for obj in objects:
            oid = obj.get('id')
            if oid in excluded:
                found.add(oid)
            removed = parent_removed or oid in excluded
            if removed and oid in protected:
                raise ValueError('Scene exclusion would remove a task role: ' + str(oid))
            children = visit(obj.get('children', []), removed)
            if not removed:
                item = deepcopy(obj)
                if 'children' in item:
                    item['children'] = children
                kept.append(item)
        return kept
    result = deepcopy(house)
    result['objects'] = visit(house.get('objects', []))
    if found != excluded:
        raise ValueError('Scene exclusions missing from source house: ' + str(sorted(excluded - found)))
    return result
