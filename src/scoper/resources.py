from pathlib import Path
import os
from importlib.util import find_spec

def package_root():
    return Path(__file__).resolve().parent

def project_root():
    return Path(os.environ.get('SCOPER_RESOURCE_ROOT', str(Path(__file__).resolve().parents[2])))

def behavior_assets_root():
    return Path(os.environ.get('SCOPER_BEHAVIOR_ASSETS', str(project_root() / 'assets/behavior-1k-assets')))

def resolve_resource(value):
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    text = path.as_posix()
    prefixes = {'embodiedbench/envs/eb_procthor/': package_root() / 'benchmark/procthor', 'eb_behavior/': package_root() / 'benchmark/behavior', 'scene_augmentation/': package_root() / 'benchmark/behavior/scene_geometry', 'datasets/behavior-1k-assets/': behavior_assets_root()}
    if text == 'datasets/behavior-1k-assets':
        return behavior_assets_root()
    for prefix, root in prefixes.items():
        if text.startswith(prefix):
            return root / text[len(prefix):]
    for prefix, module in [('bddl3/bddl/', 'bddl'), ('OmniGibson/omnigibson/', 'omnigibson')]:
        if text.startswith(prefix):
            spec = find_spec(module)
            if spec is not None and spec.origin:
                return Path(spec.origin).parent / text[len(prefix):]
    return project_root() / path
