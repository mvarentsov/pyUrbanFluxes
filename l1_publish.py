"""Stage and publish the complete L1 dataset and its diagnostic reports."""

import hashlib
import json
from pathlib import Path
import shutil
from tempfile import mkdtemp

from l0_data import PROCESSED_ROOT
from l1_data import build_l1_products, source_configs, source_state


PRODUCTS = ('processed/L1', 'reports/L1')


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _files(folder):
    return {p.relative_to(folder).as_posix(): _sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def _targets(root):
    paths = {'processed/L1': root/'L1', 'reports/L1': root.parent/'reports/L1'}
    for p in paths.values():
        if p.resolve() != p or not p.is_relative_to(root.parent) or (p.exists() and not p.is_dir()):
            raise ValueError(f'Unexpected L1 target: {p}')
    return paths


def stage_l1(run_dir, processed_root=PROCESSED_ROOT, *, verify_radiation=True):
    """Make a separate candidate and a verified snapshot of the previous L1."""
    root = Path(processed_root).resolve(strict=True)
    run = Path(run_dir).resolve()
    if run.is_relative_to(root) or run.is_relative_to(root.parent/'reports'):
        raise ValueError('Use a working directory outside processed and reports')
    run.mkdir(parents=True)
    previous = {}
    for key, target in _targets(root).items():
        previous[key] = {'present': target.is_dir(), 'files': _files(target)}
        if target.is_dir():
            backup = run/'backup'/key
            shutil.copytree(target, backup, copy_function=shutil.copy2)
            if _files(backup) != previous[key]['files']:
                raise RuntimeError(f'L1 backup differs: {target}')

    candidate = run/'candidate'
    summary, inputs = build_l1_products(root, candidate/'processed', verify_radiation=verify_radiation)
    products = {key: _files(candidate/key) for key in PRODUCTS}
    if set(products['processed/L1']) != {Path(p).relative_to('L1').as_posix() for p in summary['file']}:
        raise RuntimeError('L1 summary does not describe the staged CSVs')
    manifest = {'root': str(root), 'inputs': inputs, 'previous': previous, 'products': products}
    (run/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    return summary


def publish_l1(run_dir, processed_root=PROCESSED_ROOT):
    """Replace the two L1 directories on their own volume and roll back on failure."""
    root = Path(processed_root).resolve(strict=True)
    run = Path(run_dir).resolve(strict=True)
    if (run/'published.json').exists():
        raise ValueError(f'This staged run has already been published: {run}')
    candidate = run/'candidate'
    plan = json.loads((run/'manifest.json').read_text(encoding='utf8'))
    if plan['root'] != str(root) or source_state(source_configs(root)) != plan['inputs']:
        raise RuntimeError('The current L0 differs from the staged sources')
    targets = _targets(root)
    for key, target in targets.items():
        before = plan['previous'][key]
        if target.is_dir() != before['present'] or _files(target) != before['files']:
            raise RuntimeError(f'Published L1 changed after staging: {target}')
        if before['present'] and _files(run/'backup'/key) != before['files']:
            raise RuntimeError(f'L1 backup differs: {target}')
        if _files(candidate/key) != plan['products'][key]:
            raise RuntimeError(f'Staged L1 changed: {candidate/key}')

    scratch_root = root.parent/'_temp'
    if scratch_root.resolve() != scratch_root or not scratch_root.is_relative_to(root.parent):
        raise ValueError(f'Unexpected temporary directory: {scratch_root}')
    scratch_root.mkdir(exist_ok=True)
    lock = scratch_root/'l1_publish.lock'
    with lock.open('x'):
        pass
    try:
        scratch = Path(mkdtemp(prefix='l1_publish_', dir=scratch_root)).resolve()
        if scratch.parent != scratch_root:
            raise ValueError(f'Unexpected temporary workspace: {scratch}')
        _replace_l1(scratch, candidate, targets, plan, root.parent)
    finally:
        lock.unlink()
    result = {'files': len(plan['products']['processed/L1']), 'backup': str(run/'backup')}
    (run/'published.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    return result


def _rename_tree(source, destination, data_root):
    for path in (source, destination):
        if path.resolve() != path or path == data_root or not path.is_relative_to(data_root):
            raise ValueError(f'Unexpected L1 move: {path}')
    if destination.exists():
        raise FileExistsError(destination)
    source.rename(destination)


def _replace_l1(scratch, candidate, targets, plan, data_root):
    installed, moved = [], []
    safe_to_discard = True
    try:
        for key in targets:
            copy = scratch/'new'/key
            shutil.copytree(candidate/key, copy, copy_function=shutil.copy2)
            if _files(copy) != plan['products'][key]:
                raise RuntimeError(f'Copy differs: {copy}')
        for key, target in targets.items():
            before = plan['previous'][key]
            if target.is_dir() != before['present'] or _files(target) != before['files']:
                raise RuntimeError(f'Published L1 changed while copying: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                previous = scratch/'old'/key
                previous.parent.mkdir(parents=True, exist_ok=True)
                _rename_tree(target, previous, data_root)
                moved.append((previous, target))
            replacement = scratch/'new'/key
            _rename_tree(replacement, target, data_root)
            installed.append((target, replacement))
        for key, target in targets.items():
            if _files(target) != plan['products'][key]:
                raise RuntimeError(f'Published L1 differs: {target}')
    except BaseException:
        safe_to_discard = False
        for new, original in reversed(installed):
            _rename_tree(new, original, data_root)
        for previous, original in reversed(moved):
            _rename_tree(previous, original, data_root)
        safe_to_discard = True
        raise
    finally:
        if safe_to_discard:
            if scratch.resolve() != scratch or scratch.parent != data_root/'_temp':
                raise ValueError(f'Unexpected temporary directory: {scratch}')
            shutil.rmtree(scratch)
