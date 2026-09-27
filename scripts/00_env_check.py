"""P0：环境自检 + 数据读取通路验证。

打印环境、以 ``scipy.io.whosmat`` 只读四个 .mat 变量清单（不加载数据）、深读一个文件校验
信号转置 / ID 归一化 / 时间解码 / Hiddall；并把窗口数与 ID 分布对照 MATLAB 侧独立诊断报告
``outputs/diag/diag_report2.txt`` 的实测值 —— 期望值与 Python 读取路径无共享，对不上即
Python 侧有 bug（该诊断件不在本仓库内）。

用法：``conda run -n batfd python scripts/00_env_check.py [--probe FILE]``
"""

from __future__ import annotations

import argparse
import csv
import gc
import platform
import sys
from pathlib import Path

import numpy as np

# 允许从 pypack/ 直接跑，无需安装
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batfd import config, console  # noqa: E402
from batfd.data import mat_io  # noqa: E402

console.setup()  # 本机控制台默认 GBK，中文与 .mat 文件头会乱码甚至抛异常

# MATLAB 侧独立诊断脚本 diag_data2.m 的实测输出，用作交叉校验；
# 该诊断件不在本仓库内，故把这些实测值硬编码在此以便逐项对照
EXPECTED = {
    "StandTrainData": {"total": 26508, "counts": {6: 9127, 8: 11147, 9: 5093, 10: 1141}},
    "StandTestData1": {"total": 8110, "counts": {6: 1881, 8: 4533, 9: 599, 10: 1097}},
    "StandTestData2": {"total": 21013, "counts": {1: 1830, 2: 305, 3: 565, 4: 15487, 5: 2826}},
    "StandTestData3": {
        "total": 40093,
        "counts": {17: 2618, 18: 10996, 19: 4276, 20: 3870, 21: 18333},
    },
}


def print_env() -> list[dict]:
    """打印环境信息，返回可写进 CSV 的行。"""
    print("=" * 72)
    print("1. 运行环境")
    print("=" * 72)
    print(f"  python      {sys.version.split()[0]}  ({platform.machine()})")
    print(f"  executable  {sys.executable}")

    rows = [{"item": "python", "value": sys.version.split()[0]}]

    for mod in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "torch", "pyarrow"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  {mod:12s}{ver}")
            rows.append({"item": mod, "value": ver})
        except Exception as exc:  # noqa: BLE001
            print(f"  {mod:12s}缺失 ({type(exc).__name__})")
            rows.append({"item": mod, "value": f"MISSING ({type(exc).__name__})"})

    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"  CUDA 可用    {cuda}")
        rows.append({"item": "cuda_available", "value": str(cuda)})
        if cuda:
            print(f"  GPU         {torch.cuda.get_device_name(0)}")
            print(f"  VRAM        {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
            rows.append({"item": "gpu", "value": torch.cuda.get_device_name(0)})
    except Exception as exc:  # noqa: BLE001
        print(f"  CUDA 检查失败：{type(exc).__name__}: {exc}")
        rows.append({"item": "cuda_available", "value": f"check-failed {type(exc).__name__}"})

    try:
        import psutil

        vm = psutil.virtual_memory()
        print(f"  物理内存    共 {vm.total / 2**30:.1f} GiB，可用 {vm.available / 2**30:.1f} GiB")
        rows.append({"item": "ram_total_gib", "value": f"{vm.total / 2**30:.1f}"})
        rows.append({"item": "ram_available_gib", "value": f"{vm.available / 2**30:.1f}"})
    except Exception as exc:  # noqa: BLE001
        print(f"  psutil 缺失：{exc}")

    return rows


def print_var_lists(cfg) -> list[dict]:
    """只用 whosmat 读变量清单，不加载数据。"""
    print()
    print("=" * 72)
    print("2. 四个 .mat 的变量清单（whosmat，不加载数据）")
    print("=" * 72)
    rows = []
    # whosmat 在 test1 上会失败；回退 loadmat 需把整文件读进内存，
    # 故只对小于该阈值的文件启用回退（避免为一份变量清单去载入 364 MB）
    fallback_limit_mb = 150.0

    for which in ("train", "test1", "test2", "test3"):
        p = config.dataset_path(cfg, which)
        ver = mat_io.mat_header_version(p)
        size_mb = p.stat().st_size / 2**20
        print(f"\n  [{which}] {p.name}  {size_mb:.1f} MB  MAT={ver}")
        print(f"      header: {mat_io.mat_header_text(p)}")
        try:
            listing = mat_io.whosmat(p, allow_fallback=size_mb < fallback_limit_mb)
        except Exception as exc:  # noqa: BLE001
            print(f"      !! whosmat 失败且未回退（{size_mb:.0f} MB 超过回退阈值 "
                  f"{fallback_limit_mb:.0f} MB）：{type(exc).__name__}: {exc}")
            rows.append(
                {
                    "file": p.name,
                    "mat_version": ver,
                    "size_mb": f"{size_mb:.1f}",
                    "variable": "(whosmat 失败未回退)",
                    "class": type(exc).__name__,
                    "shape": str(exc)[:80],
                }
            )
            continue
        print(f"      变量清单来源: {mat_io.last_whosmat_method()}")
        for name, shape, cls in listing:
            print(f"      {name:16s} {cls:10s} {shape}")
            rows.append(
                {
                    "file": p.name,
                    "mat_version": ver,
                    "size_mb": f"{size_mb:.1f}",
                    "variable": name,
                    "class": cls,
                    "shape": str(shape),
                }
            )
        if ver != "v5":
            print(f"      !! 非 v5 格式，当前读取路径不适用")
    return rows


def probe(cfg, fname: str) -> tuple[list[dict], bool]:
    """深读一个文件，逐项对照 MATLAB 侧实测值。"""
    print()
    print("=" * 72)
    print(f"3. 深读验证：{fname}")
    print("=" * 72)

    p = Path(cfg["paths"]["data_dir"]) / fname
    ds = mat_io.load_raw(p, cfg)

    exp = EXPECTED.get(ds.name)
    checks: list[dict] = []
    ok_all = True

    def check(item: str, got, want) -> None:
        nonlocal ok_all
        good = str(got) == str(want)
        ok_all = ok_all and good
        print(f"    {'OK  ' if good else 'FAIL'} {item:34s} got={got}  want={want}")
        checks.append({"item": item, "got": str(got), "want": str(want), "ok": str(good)})

    print(f"  信号 signal  shape={ds.signal.shape} dtype={ds.signal.dtype}")
    print(f"  ID           dtype={ds.ids.dtype} 取值={sorted(ds.id_counts().keys())}")
    print(f"  ID 成块数    {ds.id_blocks()}（等于 ID 个数说明每个 ID 严格成块）")
    print(f"  时间         状态={ds.time_status}")
    if ds.time is not None:
        print(f"               最早={ds.time.min()}  最晚={ds.time.max()}")
    print(f"  Hiddall      {'无' if ds.hidden is None else ds.hidden.shape}")

    print("\n  --- 与 MATLAB 侧 diag_report2.txt 逐项对照 ---")
    if exp is None:
        print("    （该文件没有预置期望值，跳过）")
    else:
        check("窗口总数", ds.n_windows, exp["total"])
        got_counts = ds.id_counts()
        check("ID 集合", sorted(got_counts.keys()), sorted(exp["counts"].keys()))
        for i, c in sorted(exp["counts"].items()):
            check(f"ID={i} 窗口数", got_counts.get(i, "缺失"), c)

    if ds.hidden is not None:
        print("\n  --- Hiddall 各通道统计（对照 diag_report2.txt）---")
        for j in range(ds.hidden.shape[1]):
            col = ds.hidden[:, j]
            print(
                f"    c{j + 1}: mean={col.mean():.5f} std={col.std():.5f} "
                f"min={col.min():.5f} max={col.max():.5f} 唯一值={len(np.unique(col))}"
            )

    print("\n  --- 通道统计（前 4 列 + 最后 2 列抽样）---")
    sig = ds.signal
    for j in [0, 1, 2, 3, 16, 17, 18, 19]:
        col = sig[:, :, j].reshape(-1)
        print(
            f"    col{j + 1:2d}: mean={col.mean():+.5f} std={col.std():.5f} "
            f"min={col.min():+.5f} max={col.max():+.5f}"
        )

    # 偶数列是否逐位相同（确认「8 路电流是同一个总电流的复制」）
    print("\n  --- 校验：偶数列(电流)是否逐位相同 ---")
    base = sig[:, :, 1]
    for j in [3, 5, 7, 9, 11, 13, 15]:
        same = bool(np.array_equal(base, sig[:, :, j]))
        print(f"    col2 vs col{j + 1}: 逐位相同 = {same}")

    del ds
    gc.collect()
    return checks, ok_all


def write_csv(cfg, env_rows: list[dict], var_rows: list[dict], check_rows: list[dict]) -> Path:
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "p0_data_audit.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key1", "key2", "key3", "key4"])
        for r in env_rows:
            w.writerow(["env", r["item"], r["value"], "", ""])
        for r in var_rows:
            w.writerow(["variables", r["file"], r["variable"], r["class"], r["shape"]])
        for r in check_rows:
            w.writerow(["check", r["item"], r["got"], r["want"], r["ok"]])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="P0 环境自检与数据读取通路验证")
    ap.add_argument(
        "--probe",
        default=None,
        help="深读验证用的文件名（默认取 configs/base.yaml 的 audit.probe_file）",
    )
    args = ap.parse_args()

    cfg = config.load()
    env_rows = print_env()
    var_rows = print_var_lists(cfg)

    fname = args.probe or cfg["audit"]["probe_file"]
    check_rows, ok = probe(cfg, fname)

    out = write_csv(cfg, env_rows, var_rows, check_rows)
    print()
    print("=" * 72)
    print(f"审计表已写出：{out}")
    print(f"对照结果：{'全部通过' if ok else '存在不一致 —— 需先排查数据层'}")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
