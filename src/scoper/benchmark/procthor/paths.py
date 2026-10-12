from pathlib import Path
from scoper.resources import project_root, resolve_resource

def package_root():
    return Path(__file__).resolve().parent

def repo_root():
    return project_root()

def data_dir():
    return package_root() / 'data'

def prompt_templates_dir():
    return data_dir() / 'prompt_templates'

def resolve_repo_path(path):
    return resolve_resource(path)
