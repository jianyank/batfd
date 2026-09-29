"""故障指标缓存：绑定重建内容与重建代次，强制重建会级联失效。"""
from pathlib import Path

import numpy as np

from .. import provenance
from ..models import inference
from . import fault_metric

FILES = ("feat.npy", "sigma.npy", "cos.npy", "q3.npy")
MANIFEST = "metrics.json"


def dependencies(directory: Path, v_meas, v_rec) -> dict:
    recon = provenance.read_manifest(directory / inference.RECON_MANIFEST)
    return {"schema_version": 1,
            "reconstruction_generation": recon.get("generation") if recon else None,
            "v_meas": provenance.array_digest(v_meas),
            "v_rec": provenance.array_digest(v_rec),
            "implementation": provenance.source_digest([
                "batfd/features/cache.py", "batfd/features/fault_metric.py", "batfd/provenance.py"]),
            "numpy": np.__version__}


def get_metrics(cfg, tag, dataset_name, v_meas, v_rec, *, force=False):
    d = inference.reconstruction_dir(cfg, tag) / dataset_name
    deps = dependencies(d, v_meas, v_rec)
    manifest = provenance.read_manifest(d / MANIFEST)
    if force or not provenance.cache_valid(d, manifest, deps, FILES):
        m = fault_metric.paper_metrics(np.asarray(v_meas), np.asarray(v_rec))
        arrays = (m.oriented(), m.sigma_v, m.cos_sim, m.q3_err)
        d.mkdir(parents=True, exist_ok=True)
        (d / MANIFEST).unlink(missing_ok=True)
        for name, a in zip(FILES, arrays):
            np.save(d / name, a.astype(np.float64))
        provenance.seal_cache(d, MANIFEST, deps, FILES)
        print(f"[metric] {dataset_name}: 指标 {arrays[0].shape} 已缓存")
    return tuple(np.load(d / name, mmap_mode="r") for name in FILES)


def load_metrics(cfg, cache, tag, dataset_name):
    """仅验证并读取；下游不得跳过重建依赖校验直接 np.load。"""
    meas, rec, _ = inference.load_reconstruction(cfg, cache, tag, dataset_name)
    d = inference.reconstruction_dir(cfg, tag) / dataset_name
    deps = dependencies(d, meas, rec)
    manifest = provenance.read_manifest(d / MANIFEST)
    if not provenance.cache_valid(d, manifest, deps, FILES):
        raise ValueError(f"指标缓存缺失、过期或损坏：{d}；请先重跑 scripts/05_detect.py")
    return tuple(np.load(d / name, mmap_mode="r") for name in FILES)
