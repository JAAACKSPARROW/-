#!/usr/bin/env python3
"""Prepare a fresh run of the existing skin-aging harness; never train or overwrite."""
import argparse
import datetime as dt
import difflib
import hashlib
import json
import re
import shlex
import shutil
import sys
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(workspace, run_id):
    workspace = workspace.resolve(strict=True)
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', run_id):
        raise ValueError('run-id must be a simple name, without slashes or dots')
    source = workspace / 'reproduction'
    author = workspace / 'skinaging_predictor_ZJU'
    required = ['src/common.py', 'run_reproduction.py', 'requirements.txt', 'predict.py']
    for relative in required:
        if not (source / relative).is_file():
            raise ValueError(f'Missing existing project file: reproduction/{relative}')
    if not (source / 'sources').is_dir() or not author.is_dir():
        raise ValueError('The existing sources/ and original author repository are required')
    destination = source / 'runs' / run_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Run already exists; evidence will not be overwritten: {destination}')
    if not destination.resolve().is_relative_to(workspace):
        raise ValueError('Run destination must stay within the workspace')

    old = (source / 'src/common.py').read_text(encoding='utf-8')
    replacements = {
        'ROOT = Path(__file__).resolve().parents[2]':
            "ROOT = Path(os.environ['REPRO_WORKSPACE_ROOT'])",
        "REP = ROOT / 'reproduction'":
            "REP = Path(os.environ['REPRO_OUTPUT_ROOT'])",
    }
    new = old
    for before, after in replacements.items():
        if old.count(before) != 1:
            raise ValueError('common.py path setup differs from the supported harness; inspect it before adapting')
        new = new.replace(before, after)
    author_files = [p for p in sorted(author.rglob('*'))
                    if p.is_file() and '.git' not in p.relative_to(author).parts]
    if not author_files:
        raise ValueError('Author directory has no source/data files; initialize the submodule first')
    author_hashes = [{'path': str(p.relative_to(author)), 'sha256': sha(p)} for p in author_files]

    destination.mkdir(parents=True, exist_ok=False)
    for folder in ['results', 'models', 'data_audit', 'logs', 'executed_notebooks', 'cache/ipython']:
        (destination / folder).mkdir(parents=True)
    for folder in ['src', 'sources']:
        shutil.copytree(source / folder, destination / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
    for relative in required[1:]:
        shutil.copy2(source / relative, destination / relative)
    (destination / 'src/common.py').write_text(new, encoding='utf-8')
    patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                       fromfile='source/src/common.py', tofile='run/src/common.py'))
    (destination / 'logs/path_isolation_only.diff').write_text(patch, encoding='utf-8')
    copied = []
    for folder in ['src', 'sources']:
        for path in sorted((destination / folder).rglob('*')):
            if path.is_file():
                original = source / path.relative_to(destination)
                copied.append({'path': str(path.relative_to(destination)),
                               'source_sha256': sha(original), 'run_sha256': sha(path),
                               'change': 'path routing only' if path == destination / 'src/common.py' else 'none'})
    for relative in required[1:]:
        copied.append({'path': relative, 'source_sha256': sha(source / relative),
                       'run_sha256': sha(destination / relative), 'change': 'none'})

    # Keep the harness's checkpoint-hash lookup and Torch's cache pointed at the same files.
    shared_cache = source / 'cache/torch'
    shared_cache.mkdir(parents=True, exist_ok=True)
    (destination / 'cache/torch').symlink_to(shared_cache, target_is_directory=True)
    python = workspace / '.venv/bin/python'
    if not python.is_file():
        python = Path(sys.executable)
    environment = {
        'REPRO_WORKSPACE_ROOT': str(workspace), 'REPRO_OUTPUT_ROOT': str(destination),
        'TORCH_HOME': str(destination / 'cache/torch'),
        'HF_HOME': str(destination / 'cache/huggingface'),
        'MPLCONFIGDIR': str(destination / 'cache/matplotlib'),
        'KERAS_HOME': str(destination / 'cache/keras'),
        'PYTHONDONTWRITEBYTECODE': '1',
    }
    command = ['env', *[f'{k}={v}' for k, v in environment.items()],
               str(python), str(destination / 'run_reproduction.py')]
    plan = {
        'status': 'PREPARED_NOT_EXECUTED', 'run_id': run_id,
        'prepared_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'workspace': str(workspace), 'run_directory': str(destination),
        'cwd': str(workspace), 'command_argv': command,
        'shell_command': shlex.join(command),
        'reuse_pretrained_weights_only': True,
        'rerun_embeddings_cleaning_and_training': True,
        'scientific_protocol_changed': False,
        'log_file': str(destination / 'logs/full_pipeline.log'),
        'execution_note': 'Run the command from cwd, capture stdout/stderr and exit code; record actual start/end times separately.',
        'publication_exclusions': ['cache/', 'logs/jupyter_runtime/', 'credentials', 'Git history bundles'],
    }
    for relative, value in [('run_plan.json', plan), ('logs/source_equivalence.json', copied),
                            ('logs/author_input_hashes_before.json', author_hashes)]:
        (destination / relative).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        plan = prepare(args.workspace, args.run_id)
    except (ValueError, OSError) as error:
        parser.exit(2, f'Preparation stopped: {error}\n')
    print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
