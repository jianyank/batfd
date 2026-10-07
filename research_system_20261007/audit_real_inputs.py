"""固定四缓存的只读证据清单；不推理、不生成标签、不批准部署。"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

RESEARCH = Path(__file__).resolve().parent
REPO = RESEARCH.parent
NAMES = ('StandTrainData', 'StandTestData1', 'StandTestData2', 'StandTestData3')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def time_summary(times, ids):
    result = {'available': times is not None, 'window_time_mapping_proven': False,
              'exposure_computed': False, 'packs': {}}
    if times is None:
        return result
    times, ids = np.asarray(times), np.asarray(ids)
    if times.shape != ids.shape or times.ndim != 1 or times.dtype.kind != 'M':
        raise ValueError('time must be a 1D datetime array matching ids')
    for pack in np.unique(ids):
        t = times[ids == pack].astype('datetime64[ns]')
        valid = ~np.isnat(t)
        adjacent = valid[1:] & valid[:-1]
        ns = t.astype(np.int64)
        dt = (ns[1:][adjacent] - ns[:-1][adjacent]) / 1e9
        positive = dt[dt > 0]
        result['packs'][str(int(pack))] = {
            'n_windows': len(t), 'nat_count': int((~valid).sum()),
            'first_valid_timestamp': str(t[valid][0]) if valid.any() else None,
            'last_valid_timestamp': str(t[valid][-1]) if valid.any() else None,
            'min_timestamp': str(t[valid].min()) if valid.any() else None,
            'max_timestamp': str(t[valid].max()) if valid.any() else None,
            'adjacent_valid_pairs': len(dt), 'negative_deltas': int((dt < 0).sum()),
            'zero_deltas': int((dt == 0).sum()),
            'median_positive_delta_seconds': float(np.median(positive)) if len(positive) else None,
            'max_positive_delta_seconds': float(positive.max()) if len(positive) else None}
    return result


def check_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    files = []
    for group, records in manifest.items():
        if isinstance(records, dict):
            records = [{'path': str(path.parent / name), 'sha256': digest}
                       for name, digest in records.items() if isinstance(digest, str) and len(digest) == 64]
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict) or not {'path', 'sha256'} <= rec.keys():
                continue
            source = Path(rec['path'])
            if not source.is_absolute():
                source = REPO / source
            current = sha256(source) if source.is_file() else None
            status = 'missing' if current is None else 'match' if current == rec['sha256'] else 'hash_mismatch'
            files.append({'group': group, 'path': str(source), 'expected_sha256': rec['sha256'],
                          'current_sha256': current, 'status': status})
    return {'manifest': file_record(path), 'files': files,
            'all_recorded_files_match': bool(files) and all(x['status'] == 'match' for x in files),
            'scope': '文件版本复查，不等同语义或真实效能验证'}


def inventory():
    result = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'BLOCKED', 'real_evaluation_executed': False,
              'training_or_inference_executed': False, 'deployment_approved': False,
              'datasets': [], 'prior_evidence_checks': []}
    seen = {}
    for name in NAMES:
        cache = REPO / 'outputs' / 'cache' / name
        ids = np.load(cache / 'ids.npy', mmap_mode='r', allow_pickle=False)
        signal = np.load(cache / 'signal.npy', mmap_mode='r', allow_pickle=False)
        meta = json.loads((cache / 'meta.json').read_text(encoding='utf-8'))
        if signal.shape != (len(ids), 256, 20) or ids.ndim != 1:
            raise ValueError('unexpected cache shape: ' + name)
        packs, counts = np.unique(ids, return_counts=True)
        record = {'name': name, 'raw_mat': file_record(REPO / 'data' / (name + '.mat')),
                  'cache_files': [file_record(cache / 'meta.json')], 'meta': meta,
                  'id_counts': {str(int(p)): int(n) for p, n in zip(packs, counts)},
                  'id_contiguous_blocks': int(1 + (ids[1:] != ids[:-1]).sum()),
                  'arrays': {}, 'duplicate_cached_windows': 0,
                  'windows_matching_earlier_datasets': 0, 'overlap_examples': []}
        for filename in ('signal.npy', 'ids.npy', 'cond.npy', 'hidden.npy', 'time.npy'):
            path = cache / filename
            if not path.exists():
                record['arrays'][filename] = {'available': False}
                continue
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            if a.shape[0] != len(ids):
                raise ValueError('cache length mismatch: ' + str(path))
            nonfinite = 0
            if a.dtype.kind != 'M':
                for start in range(0, len(a), 256):
                    nonfinite += int((~np.isfinite(a[start:start + 256])).sum())
            record['arrays'][filename] = {'available': True, 'shape': list(a.shape),
                                         'dtype': str(a.dtype), 'nonfinite_count': nonfinite if a.dtype.kind != 'M' else None}
            record['cache_files'].append(file_record(path))
        time_path = cache / 'time.npy'
        times = np.load(time_path, mmap_mode='r', allow_pickle=False) if time_path.exists() else None
        record['time'] = time_summary(times, ids)
        sidecar = REPO / 'outputs/cache/matlab_meta' / ('time_' + name + '.mat')
        record['time_sidecar'] = file_record(sidecar) if sidecar.is_file() else None
        for index, window in enumerate(signal):
            key = hashlib.sha256(window.tobytes(order='C')).digest()
            if key in seen:
                previous_name, previous_index = seen[key]
                record['duplicate_cached_windows'] += 1
                if previous_name != name:
                    record['windows_matching_earlier_datasets'] += 1
                    if len(record['overlap_examples']) < 5:
                        record['overlap_examples'].append({'window_0based': index, 'earlier_dataset': previous_name,
                                                           'earlier_window_0based': previous_index})
            else:
                seen[key] = (name, index)
        result['datasets'].append(record)
    for rel in ('local_source_trace_20261004T041842Z/manifest.json',
                'mat_cross_file_audit_20261003T120500Z/manifest.json',
                'mcmaster_event_binding_followup_20261004/manifest.json',
                'mcmaster_event_binding_followup_20261004/protected_baseline.json'):
        result['prior_evidence_checks'].append(check_manifest(REPO / 'outputs/diagnostics' / rel))
    result['limitations'] = [
        '有时间数组不证明窗口起止/时区/采样周期/有效暴露时长；不输出每包天指标。',
        '缓存窗口按float32逻辑C序逐字节哈希，仅检测完全重复；无重复不证明实体独立或无采样重叠。',
        '当前原MAT与缓存指纹不证明缓存生成链路；meta.source_path仅是历史记录。',
        '旧manifest哈希一致只证明记录文件不变；漂移不通过且不修改旧证据。',
        '未取得独立健康/故障标签及业务门槛，不输出真实性能、不据此迭代算法。']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if (RESEARCH / 'runs').resolve() not in out.parents:
        raise ValueError('output must be a new directory under research/runs')
    out.mkdir(parents=True, exist_ok=False)
    report = inventory()
    (out / 'inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'datasets': len(report['datasets']),
                      'prior_evidence_all_match': [x['all_recorded_files_match'] for x in report['prior_evidence_checks']],
                      'real_evaluation_executed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
