from pathlib import Path
from scoper.resources import project_root, resolve_resource, behavior_assets_root

def package_root():
    return Path(__file__).resolve().parent

def repo_root():
    return project_root()

def data_dir():
    return package_root() / 'data'

def assets_root():
    return behavior_assets_root()

def resolve_repo_path(path):
    return resolve_resource(path)

def default_scene_json_path(scene_model, *, assets=None):
    return (Path(assets) if assets is not None else assets_root()) / 'scenes' / str(scene_model) / 'json' / f'{scene_model}_best.json'
