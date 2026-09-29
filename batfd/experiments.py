"""轻量实验身份：新结果与历史目录隔离，来源/参数/内容均可核验。"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import platform
import re
import subprocess

import numpy as np
import sklearn
import torch

from . import provenance

DATASETS = ("StandTrainData", "StandTestData1", "StandTestData2", "StandTestData3")


def experiment_inputs(cfg, model, tag) -> dict:
    out = Path(cfg["paths"]["outputs_dir"])
    checkpoint = out / "runs" / tag / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint}")
    paths = {f"{ds}/{name}.npy": out / "cache" / ds / f"{name}.npy"
             for ds in DATASETS for name in ("signal", "ids", "cond", "time", "hidden")}
    paths["labels.csv"] = out / "tables" / "labels.csv"
    paths["fault_table.yaml"] = provenance.ROOT / "configs" / "fault_table.yaml"
    source_paths = [str(p.relative_to(provenance.ROOT)).replace("\\", "/")
                    for folder in ("batfd", "scripts")
                    for p in (provenance.ROOT / folder).rglob("*.py")]
    return {"schema_version": 1, "model": model, "tag": tag,
            "checkpoint_sha256": provenance.file_digest(checkpoint),
            "config": {k: v for k, v in cfg.items() if k != "paths" and not k.startswith("_")},
            "sources": {k: provenance.file_digest(p) if p.is_file() else None for k, p in paths.items()},
            "implementation_sha256": provenance.source_digest(source_paths),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "torch": str(torch.__version__), "sklearn": sklearn.__version__}}


class Experiment:
    def __init__(self, directory: Path, manifest: dict):
        self.directory = directory
        self.manifest = manifest
        self.id = manifest["experiment_id"]

    @classmethod
    def create(cls, cfg, model, tag):
        # JSON 往返冻结配置快照，避免调用者随后修改字典污染已记录的身份。
        inputs = json.loads(json.dumps(experiment_inputs(cfg, model, tag), default=str))
        eid = provenance.digest(inputs)[:20]
        directory = Path(cfg["paths"]["outputs_dir"]) / "experiments" / eid
        manifest = {"experiment_id": eid, "inputs": inputs}
        path = directory / "experiment.json"
        if path.exists():
            existing = provenance.read_manifest(path)
            if existing is None or existing.get("inputs") != inputs:
                raise ValueError(f"实验清单损坏或身份冲突：{path}")
        else:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=provenance.ROOT,
                                      capture_output=True, text=True, check=False)
            manifest["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
            provenance.write_manifest(path, manifest)
        print(f"[experiment] {eid} -> {directory}")
        return cls(directory, manifest)

    @classmethod
    def open(cls, output_root: Path, eid: str):
        if not re.fullmatch(r"[0-9a-f]{20}", eid):
            raise ValueError(f"实验 ID 格式无效：{eid}")
        directory = output_root / "experiments" / eid
        manifest = provenance.read_manifest(directory / "experiment.json")
        if (manifest is None or manifest.get("experiment_id") != eid
                or provenance.digest(manifest.get("inputs"))[:20] != eid):
            raise ValueError(f"实验清单缺失或校验失败：{directory}")
        return cls(directory, manifest)

    def write_table(self, stage: str, rows: list[dict], parameters: dict) -> Path:
        if not rows:
            raise ValueError(f"{stage} 没有结果，拒绝写空表")
        output_root = self.directory.parents[1]
        recon = output_root / "runs" / self.manifest["inputs"]["tag"] / "recon"
        cache_refs = {str(p.relative_to(recon)): provenance.file_digest(p)
                      for p in sorted(recon.glob("*/*.json"))}
        artifact = {"schema_version": 1, "experiment_id": self.id, "stage": stage,
                    "parameters": parameters, "cache_manifests": cache_refs,
                    "result_sha256": provenance.digest(rows)}
        aid = provenance.digest(artifact)[:20]
        artifact["artifact_id"] = aid
        data = [{**r, "experiment_id": self.id, "artifact_id": aid, "schema_version": 1} for r in rows]
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
        payload = text.getvalue().encode("utf-8-sig")
        destination = self.directory / "tables" / f"{stage}_{aid}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        # 内容参与文件名。同一身份只接受字节相同的重放，不覆盖异常旧文件。
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError(f"产物内容冲突：{destination}")
        if not destination.exists():
            with destination.open("xb") as fh:
                fh.write(payload)
        artifact["table_sha256"] = provenance.file_digest(destination)
        provenance.write_manifest(destination.with_suffix(".json"), artifact)
        return destination


def read_table(path: Path, *, allow_legacy=False) -> list[dict]:
    sidecar = path.with_suffix(".json")
    meta = provenance.read_manifest(sidecar)
    if meta is None:
        if not allow_legacy or sidecar.exists():
            raise ValueError(f"表格清单缺失或损坏：{path}；历史结果需显式 --allow-legacy")
    else:
        identity = {k: v for k, v in meta.items() if k not in ("artifact_id", "table_sha256")}
        if provenance.digest(identity)[:20] != meta.get("artifact_id"):
            raise ValueError(f"产物身份校验失败：{path}")
        if meta.get("table_sha256") != provenance.file_digest(path):
            raise ValueError(f"表格内容校验失败：{path}")
        exp = Experiment.open(path.parents[3], meta.get("experiment_id", ""))
        if path.parent != exp.directory / "tables":
            raise ValueError(f"表格与实验目录不匹配：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if meta is not None and any(r.get("experiment_id") != meta["experiment_id"]
                                or r.get("artifact_id") != meta.get("artifact_id") for r in rows):
        raise ValueError(f"表格行身份校验失败：{path}")
    return rows
