# Explicit data adapters. Downloading is opt-in and records immutable file hashes.
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

import numpy as np

SMD_SOURCE = 'https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/'


def _positive_integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value > 0


def make_windows(series, window_size=1, stride=1):
    x = np.asarray(series, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('series must be a nonempty finite (time,channels) array')
    if not _positive_integer(window_size) or window_size > len(x) or not _positive_integer(stride):
        raise ValueError('invalid window_size or stride')
    ends = np.arange(window_size - 1, len(x), stride)
    view = np.lib.stride_tricks.sliding_window_view(x, window_size, axis=0)
    return view[::stride].transpose(0, 2, 1), ends


def split_train(values):
    x = np.asarray(values)
    if x.ndim < 1 or len(x) < 10:
        raise ValueError('at least ten ordered training rows are required')
    a, b = int(.6 * len(x)), int(.8 * len(x))
    return x[:a], x[a:b], x[b:]


def _machine_name(machine):
    match = re.fullmatch(r'machine-([123])-(\d+)', machine)
    if match is None or not 1 <= int(match[2]) <= (8, 9, 11)[int(match[1]) - 1]:
        raise ValueError('invalid SMD machine name')
    return machine


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_smd(directory, machine):
    machine = _machine_name(machine)
    root = Path(directory)
    manifest_path = root / 'download_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('source') != SMD_SOURCE:
            raise ValueError('cached data source differs from expected SMD source')
        files = manifest.get('files', {})
        for group in ('train', 'test', 'test_label'):
            name = f'{group}/{machine}.txt'
            if name not in files or file_sha256(root / name) != files[name].get('sha256'):
                raise ValueError('unverified or modified cached SMD file: ' + name)
    train = np.loadtxt(root / 'train' / f'{machine}.txt', delimiter=',', ndmin=2)
    test = np.loadtxt(root / 'test' / f'{machine}.txt', delimiter=',', ndmin=2)
    labels = np.loadtxt(root / 'test_label' / f'{machine}.txt', ndmin=1)
    if train.shape[1] != 38 or test.shape[1] != 38 or min(len(train), len(test)) < 1:
        raise ValueError('SMD train/test must contain 38 channels')
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError('SMD contains nonfinite values')
    if labels.shape != (len(test),) or not np.isin(labels, (0, 1)).all():
        raise ValueError('SMD labels must be binary and aligned to test rows')
    return train, test, labels.astype(bool)


def download_smd(directory, machines=('machine-1-1', 'machine-2-1', 'machine-3-1')):
    machines = sorted(set(_machine_name(m) for m in machines))
    if not machines:
        raise ValueError('at least one SMD machine is required')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'download_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {
        'source': SMD_SOURCE, 'files': {}}
    if manifest['source'] != SMD_SOURCE:
        raise ValueError('cached data source differs from expected SMD source')
    names = {'README.md': 'README.md', 'LICENSE': 'LICENSE'}
    for machine in machines:
        for group in ('train', 'test', 'test_label'):
            name = f'{group}/{machine}.txt'
            names[name] = 'ServerMachineDataset/' + name
    for name, remote in names.items():
        target = root / name
        if target.exists():
            if name not in manifest['files'] or file_sha256(target) != manifest['files'][name]['sha256']:
                raise ValueError('unverified or modified cached SMD file: ' + name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(SMD_SOURCE + remote, timeout=60) as response:
            content = response.read()
        with target.open('xb') as stream:
            stream.write(content)
        manifest['files'][name] = {'url': SMD_SOURCE + remote,
                                    'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
        # Persist after every successful file so interrupted downloads are resumable.
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for machine in machines:
        load_smd(root, machine)
    return manifest
