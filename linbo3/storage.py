"""One current COMSOL dataset, isolated overwrite-only notebook analyses."""
from pathlib import Path
import shutil
_profile='main'
PROFILES=('main','comparison','single_pulse','unclassified')
def set_profile(name):
    global _profile
    if name not in PROFILES:raise ValueError('Неизвестный профиль ноутбука')
    _profile=name
def analysis_root(run_dir):
    return Path(run_dir)/'analysis'/_profile
def remove_owned(path,owner):
    path,owner=Path(path),Path(owner).resolve()
    target=path.resolve()
    if target==owner or not target.is_relative_to(owner) or path.is_symlink():
        raise ValueError(f'Небезопасный путь очистки: {path}')
    if path.is_dir():shutil.rmtree(path)
    elif path.exists():path.unlink()
def prune_cache(cache,expected):
    cache=Path(cache);expected={Path(p).name for p in expected}
    for path in cache.glob('*.npz'):
        if path.name not in expected:remove_owned(path,cache)
