import re
ASSET_DISPLAY_NAMES = {'bottom cabinet no top': 'base cabinet', 'bottom cabinet': 'base cabinet', 'top cabinet': 'wall cabinet', 'furniture sink': 'sink', 'derived faucet': 'faucet', 'pewter teapot': 'teapot'}

def public_object_text(text):
    value = str(text or '')
    for old, new in sorted(ASSET_DISPLAY_NAMES.items(), key=lambda pair: -len(pair[0])):
        pattern = '(?<!\\w)' + '[ _]+'.join(map(re.escape, old.split())) + '(?!\\w)'
        value = re.sub(pattern, new, value, flags=re.I)
    return value

def public_object_type(name):
    return ASSET_DISPLAY_NAMES.get(name, name)
