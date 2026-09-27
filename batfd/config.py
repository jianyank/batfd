"""配置加载与路径解析。

配置只有一个来源（configs/base.yaml），代码里不出现魔法数字。
路径基准用 ``Path(__file__).resolve().parent.parent`` 显式上溯到 pypack/ 根，
不靠相对推断（MATLAB 版 fileparts 式写法在脚本移位后会指错目录）。

用法: ``config.load()`` 读默认的 configs/base.yaml 并返回 ``Config``；
``cfg.paths.data_dir`` 已是绝对路径，``cfg["train"]`` 与 ``cfg.train`` 均可取值。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# batfd/config.py -> batfd/ -> pypack/
PYPACK_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG: Path = PYPACK_ROOT / "configs" / "base.yaml"

# 这些键的值是相对 pypack/ 的路径，load() 会把它们解析成绝对路径
_PATH_KEYS = ("data_dir", "outputs_dir")


class Config(dict):
    """dict 的薄包装，额外提供属性式访问与路径解析。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 配置写错时的显式报错
            raise AttributeError(
                f"配置里没有 '{name}'；已有的顶层键：{sorted(self.keys())}"
            ) from exc

    @property
    def root(self) -> Path:
        return PYPACK_ROOT

    def path(self, *parts: str) -> Path:
        """拼一个相对 pypack/ 的路径。"""
        return PYPACK_ROOT.joinpath(*parts)


def load(path: str | Path | None = None) -> Config:
    """读取 YAML 配置并把 ``paths`` 里的相对路径解析为绝对路径。"""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"配置文件顶层应是映射，实际是 {type(raw).__name__}")

    raw = Config(raw)
    raw["_config_path"] = str(cfg_path)

    paths = raw.setdefault("paths", {})
    for key in _PATH_KEYS:
        if key in paths:
            p = Path(paths[key])
            paths[key] = p if p.is_absolute() else (PYPACK_ROOT / p).resolve()

    # 产物目录当场建好，脚本里就不必各自 ensure
    out = Path(paths["outputs_dir"])
    for sub in ("tables", "figures", "logs", "cache", "runs"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    return raw


def dataset_path(cfg: Config, which: str) -> Path:
    """取某个数据集 .mat 的绝对路径。``which`` ∈ {train, test1, test2, test3}。"""
    name = cfg["data"]["files"][which]
    return Path(cfg["paths"]["data_dir"]) / name


def as_1based(cols: list[int]) -> list[int]:
    """校验列号是 1-based（防止有人误填 0-based）。"""
    for c in cols:
        if c < 1:
            raise ValueError(f"列号应为 1-based 正整数，收到 {c}")
    return cols
