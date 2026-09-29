"""内容指纹和缓存清单；不以文件名、大小或修改时间代替来源校验。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(array: np.ndarray) -> str:
    a = np.asarray(array)
    h = hashlib.sha256(f"{a.dtype.str}:{a.shape}".encode())
    # 按第一维分块，避免把整个 memmap 复制到内存。
    a = a.reshape((1,)) if a.ndim == 0 else a
    row_bytes = max(1, int(np.prod(a.shape[1:])) * a.dtype.itemsize)
    step = max(1, (1024 * 1024) // row_bytes)
    for start in range(0, len(a), step):
        h.update(np.ascontiguousarray(a[start:start + step]).tobytes())
    return h.hexdigest()


def source_digest(paths) -> str:
    # 统一换行符，避免 Windows checkout 单独改变实现指纹。
    return digest({p: (ROOT / p).read_text(encoding="utf-8") for p in sorted(paths)})


def read_manifest(path: Path) -> dict | None:
    try:
        value = path.read_text(encoding="utf-8")
        result = json.loads(value)
        return result if isinstance(result, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def cache_valid(directory: Path, manifest: dict | None, dependencies: dict, names) -> bool:
    if manifest is None or manifest.get("dependencies") != dependencies:
        return False
    files = manifest.get("files", {})
    return isinstance(files, dict) and all(
        (directory / name).is_file() and files.get(name) == file_digest(directory / name)
        for name in names
    )


def seal_cache(directory: Path, filename: str, dependencies: dict, names) -> dict:
    manifest = {"schema_version": 1, "generation": uuid.uuid4().hex,
                "dependencies": dependencies,
                "files": {n: file_digest(directory / n) for n in names}}
    write_manifest(directory / filename, manifest)
    return manifest
