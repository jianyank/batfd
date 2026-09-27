"""P1：把四个 .mat 转成缓存，并回读校验。

把关：回读缓存后统计窗口总数与逐 ID 窗口数，对照 MATLAB 侧 ``outputs/diag/diag_report2.txt``
实测值；对不上即数据层有 bug，后续实验都不可信。内存：一次只 load 一个 .mat，落盘后立即
释放；最大的 test3（364 MB）v5+double 载入约 1.6 GB、转 float32 后约 0.8 GB，峰值约 2.5 GB。

用法：``conda run -n batfd python scripts/01_preprocess.py``
"""

from __future__ import annotations

import csv
import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batfd import config, console, progress  # noqa: E402
from batfd.data import cache, mat_io  # noqa: E402

console.setup()  # 本机控制台默认 GBK，中文与 .mat 文件头会乱码甚至抛异常

# 与 00_env_check.py 同源：MATLAB 侧 diag_report2.txt 的实测值
EXPECTED = {
    "StandTrainData": {"total": 26508, "counts": {6: 9127, 8: 11147, 9: 5093, 10: 1141}},
    "StandTestData1": {"total": 8110, "counts": {6: 1881, 8: 4533, 9: 599, 10: 1097}},
    "StandTestData2": {"total": 21013, "counts": {1: 1830, 2: 305, 3: 565, 4: 15487, 5: 2826}},
    "StandTestData3": {
        "total": 40093,
        "counts": {17: 2618, 18: 10996, 19: 4276, 20: 3870, 21: 18333},
    },
}


def main() -> int:
    cfg = config.load()
    rows: list[dict] = []
    all_ok = True

    for which in ("train", "test1", "test2", "test3"):
        src = config.dataset_path(cfg, which)
        print()
        print("=" * 72)
        print(f"处理 [{which}] {src.name}")
        print("=" * 72)

        # 先定下 signal.npy 的落点，让 load_raw 分块写盘，而不是在内存里再放一份
        # (N,T,C) float32：本机内存放不下 float64 输入 + float32 转置副本（会被 OOM kill）
        sig_path = cache.dataset_cache_dir(cfg, src.stem) / "signal.npy"
        ds = mat_io.load_raw(src, cfg, signal_out=sig_path)
        print(
            f"  载入完成：signal{tuple(ds.signal.shape)} {ds.signal.dtype}  "
            f"ID {sorted(ds.id_counts().keys())}  "
            f"时间={ds.time_status}  "
            f"Hiddall={'无' if ds.hidden is None else ds.hidden.shape}"
        )

        meta = cache.save_cache(cfg, ds, signal_already_written=True)
        print(f"  已写入缓存：{cache.dataset_cache_dir(cfg, ds.name)}")

        # 回读磁盘上的缓存做校验（不要信任刚写进去的内存状态）
        del ds
        gc.collect()

        ds_name = meta["name"]
        back = cache.load_cache(cfg, ds_name)
        got_total = int(back["signal"].shape[0])
        got_counts = np.bincount(back["ids"]) if back["ids"].size else np.array([])
        got_ids = sorted(int(i) for i in np.unique(back["ids"]))

        exp = EXPECTED.get(ds_name)
        checks = [("窗口总数", got_total, exp["total"] if exp else "n/a")]
        checks.append(("ID 集合", got_ids, sorted(exp["counts"].keys()) if exp else "n/a"))
        if exp:
            for i, c in sorted(exp["counts"].items()):
                got = int(got_counts[i]) if i < len(got_counts) else 0
                checks.append((f"ID={i} 窗口数", got, c))

        print("\n  --- 回读校验（对照 diag_report2.txt）---")
        for item, got, want in checks:
            ok = str(got) == str(want)
            all_ok = all_ok and ok
            print(f"    {'OK  ' if ok else 'FAIL'} {item:20s} got={got}  want={want}")
            rows.append(
                {
                    "dataset": ds_name,
                    "which": which,
                    "item": item,
                    "got": got,
                    "want": want,
                    "ok": ok,
                }
            )

        rows.append(
            {
                "dataset": ds_name,
                "which": which,
                "item": "signal_shape",
                "got": str(tuple(back["signal"].shape)),
                "want": "n/a",
                "ok": True,
            }
        )
        rows.append(
            {
                "dataset": ds_name,
                "which": which,
                "item": "time_status",
                "got": back["meta"]["time_status"],
                "want": "n/a",
                "ok": True,
            }
        )
        rows.append(
            {
                "dataset": ds_name,
                "which": which,
                "item": "has_hidden",
                "got": back["meta"]["has_hidden"],
                "want": "n/a",
                "ok": True,
            }
        )

        # 抽检一个窗口的首末若干值，确认转置没有把维度搞混
        sig = np.asarray(back["signal"][0])
        print(
            f"\n  抽检窗口 0：col1 首6步={np.round(sig[:6, 0], 4).tolist()}"
            f"  col2 首6步={np.round(sig[:6, 1], 4).tolist()}"
        )

        del back
        gc.collect()

    out = Path(cfg["paths"]["outputs_dir"]) / "tables" / "p1_cache_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "which", "item", "got", "want", "ok"])
        w.writeheader()
        w.writerows(rows)

    print()
    print("=" * 72)
    print(f"缓存清单：{cache.cached_names(cfg)}")
    print(f"审计表：{out}")
    print(f"校验结果：{'全部通过' if all_ok else '存在不一致 —— 必须先排查数据层'}")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
