# Build the delivery artifacts: wheel, source archive, checksums and delivery record.
# Read-only with respect to the framework tree; writes only under dist/.
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
DIST = ROOT / "dist"

SOURCE_FILES = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "src/chronoguard/__init__.py",
    "src/chronoguard/data.py",
    "src/chronoguard/detector.py",
    "src/chronoguard/evaluation.py",
    "src/chronoguard/features.py",
    "examples/demo.py",
    "examples/benchmark.py",
    "examples/benchmark_smd_windows.py",
    "tests/test_core.py",
    "tests/test_data_evaluation.py",
]

DOC_FILES = {
    "docs/ALGORITHM_LIBRARY.md": ROOT / "docs/ALGORITHM_LIBRARY.md",
    "docs/DATASETS.md": ROOT / "docs/DATASETS.md",
    "docs/THIRD_PARTY_NOTICES.md": ROOT / "docs/THIRD_PARTY_NOTICES.md",
    "benchmarks/20261008/REPORT.md": ROOT / "benchmarks/20261008/REPORT.md",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version():
    for line in (FRAMEWORK / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("version not found in pyproject.toml")


def build_wheel(workdir):
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(FRAMEWORK), "--no-deps", "-w", str(workdir), "-q"],
        check=True)
    wheels = sorted(workdir.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel, found {len(wheels)}")
    return wheels[0]


def build_source_archive(version, target):
    prefix = f"chronoguard_ts-{version}"
    entries = {}
    for rel in SOURCE_FILES:
        source = FRAMEWORK / rel
        if not source.is_file():
            raise SystemExit(f"missing source file: {rel}")
        entries[f"{prefix}/{rel}"] = source
    for rel, source in DOC_FILES.items():
        if not source.is_file():
            raise SystemExit(f"missing document: {rel}")
        entries[f"{prefix}/{rel}"] = source
    manifest = {
        "package": "chronoguard-ts",
        "version": version,
        "created": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "note": ("Source archive. Core flow needs no PyTorch, CUDA, MATLAB or Office. "
                 "The peer-equivalence test loads the original research features.py, which ships "
                 "only with the source repository; outside it that single test skips by design."),
        "files": {name: {"bytes": src.stat().st_size, "sha256": sha256(src)}
                  for name, src in sorted(entries.items())},
    }
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, src in sorted(entries.items()):
            archive.write(src, name)
        archive.writestr(f"{prefix}/DELIVERY_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2))
    manifest["files"][f"{prefix}/DELIVERY_MANIFEST.json"] = {
        "bytes": None, "sha256": None, "note": "manifest itself; contains no self hash"}
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=DIST)
    args = parser.parse_args()
    args.dist.mkdir(parents=True, exist_ok=True)
    ver = version()

    with tempfile.TemporaryDirectory() as tmp:
        wheel = build_wheel(Path(tmp))
        wheel_target = args.dist / wheel.name
        shutil.copyfile(wheel, wheel_target)

    source_target = args.dist / f"chronoguard_ts-{ver}-source.zip"
    source_manifest = build_source_archive(ver, source_target)

    previous = sorted(p.name for p in args.dist.iterdir()
                      if p.name not in (wheel_target.name, source_target.name, "CHECKSUMS.sha256",
                                        "DELIVERY.json"))
    artifacts = [wheel_target, source_target]
    checksum_lines = [f"{sha256(p)}  {p.name}" for p in artifacts]
    (args.dist / "CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    record = {
        "project": "chronoguard-ts",
        "version": ver,
        "date": datetime.now().astimezone().date().isoformat(),
        "status": "built",
        "built_from": str(FRAMEWORK.relative_to(ROOT)),
        "files": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in artifacts],
        "source_archive_files": len(source_manifest["files"]) - 1,
        "exclusions": ["private_data", "external_datasets", "models", "per_sample_predictions",
                       "git", "caches", "logs"],
        "stale_from_previous_build": previous,
    }
    (args.dist / "DELIVERY.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
