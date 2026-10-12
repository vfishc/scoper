import hashlib
POLICY_VERSION = 'balanced-concrete-assets.v1'

def choose_asset(candidates, *, seed, usage, forbidden_types=()):
    legal = {(str(c['object_type']), str(c['category']), str(c['model'])) for c in candidates if c.get('object_type') and c.get('category') and c.get('model') and (c['object_type'] not in forbidden_types)}
    if not legal:
        raise ValueError('NO_ELIGIBLE_DISTINCT_ASSET')

    def keys(asset):
        kind, category, model = asset
        return ('type:' + kind, 'model:' + category + '/' + model)

    def rank(asset):
        tk, mk = keys(asset)
        tie = hashlib.sha256((str(seed) + '\x00' + '\x00'.join(asset)).encode()).hexdigest()
        return (usage.get(tk, 0), usage.get(mk, 0), tie)
    chosen = min(legal, key=rank)
    for key in keys(chosen):
        usage[key] = usage.get(key, 0) + 1
    return dict(zip(('object_type', 'category', 'model'), chosen))
