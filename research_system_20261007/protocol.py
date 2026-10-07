"""Explicit data roles, causal alarm rules, immutable artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np

PACKS = (6, 8, 9, 10)


def split_roles(ids, held_pack):
    ids = np.asarray(ids)
    if ids.ndim != 1 or set(np.unique(ids)) != set(PACKS) or held_pack not in PACKS:
        raise ValueError('expected exactly packs 6/8/9/10')
    fit, cal, dev = [], [], []
    for pack in PACKS:
        idx = np.flatnonzero(ids == pack)
        if len(idx) < 10:
            raise ValueError('too few rows for disjoint fit/calibration/development')
        if pack != held_pack:
            a, b = int(.6*len(idx)), int(.8*len(idx))
            fit.extend(idx[:a]); cal.extend(idx[a:b]); dev.extend(idx[b:])
    return tuple(np.asarray(part, dtype=np.int64) for part in (fit, cal, dev, np.flatnonzero(ids==held_pack)))


def confirm(exceeded, ids, persistence=5, initial_run=0):
    exceeded, ids = np.asarray(exceeded, dtype=bool), np.asarray(ids)
    if exceeded.ndim != 1 or ids.shape != exceeded.shape or persistence < 1:
        raise ValueError('invalid alarm inputs')
    out = np.zeros(len(ids), dtype=bool)
    run = int(initial_run)
    for i, flag in enumerate(exceeded):
        if i and ids[i] != ids[i-1]: run = 0
        run = run+1 if flag else 0
        out[i] = run >= persistence
    return out


def summarize(scores, threshold, ids, persistence=5):
    scores = np.asarray(scores, dtype=float)
    ids = np.asarray(ids)
    if scores.ndim != 1 or not len(scores) or scores.shape != ids.shape or not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError('invalid finite scores / threshold / ids')
    exceeded = scores > threshold
    confirmed = confirm(exceeded, ids, persistence)
    starts = confirmed & ~np.r_[False, confirmed[:-1]]
    starts[1:] |= confirmed[1:] & (ids[1:] != ids[:-1])
    first = np.flatnonzero(confirmed)
    return dict(n_windows=len(scores), threshold=float(threshold), n_exceeded=int(exceeded.sum()),
                exceedance_rate=float(exceeded.mean()), n_confirmed=int(confirmed.sum()),
                confirmation_rate=float(confirmed.mean()), alarm_segments=int(starts.sum()),
                first_confirmed=int(first[0]) if len(first) else None)


def select_candidate(rows):
    if not rows: raise ValueError('no development candidates')
    # Exploratory surrogate utility, NOT estimated true-fault utility.
    ranked = [dict(row, utility=.55*row['detection']+.25*row['worst_detection']+
                   .20*row['localization']-.30*row['ref_alarm']) for row in rows]
    return max(ranked, key=lambda row: (row['utility'], -row.get('complexity', 0)))


def file_hash(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write_json(file, content):
    Path(file).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def seal(directory, content):
    directory = Path(directory)
    content = dict(content, files={p.relative_to(directory).as_posix():file_hash(p)
                    for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'manifest.json'})
    target = directory/'manifest.json'
    if target.exists(): raise FileExistsError(target)
    write_json(target, content)


def verify_files(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete': raise ValueError('run incomplete')
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name!='manifest.json'}
    if actual != set(manifest['files']): raise ValueError('artifact inventory changed')
    for name, digest in manifest['files'].items():
        resolved=(directory/name).resolve()
        if not resolved.is_relative_to(directory.resolve()) or file_hash(resolved)!=digest:
            raise ValueError('artifact changed: '+name)
    return manifest
