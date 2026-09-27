"""P2：验证 Hiddall 的通道↔单体映射，并生成故障起点标签。

1. 通道↔单体映射：``Hiddall`` 第 2..9 通道依次对应单体 1..8 是**未验证的假设**，用从 PDF
   原文核实的 Table 5（``configs/fault_table.yaml``）对照 —— pack 6→Cell 8（应第 9 通道）、
   pack 9→Cell 2（第 3 通道）、pack 10→Cell 8/7（第 9/8 通道）、pack 8→全部单体无定位信息；
   做法是比较各通道故障段相对正常段的偏移，看发散最大者是否落在预期位置。
2. 拼接连续性：训练集无时间戳，只能假设「train 段在前、test1 段在后」，检查内阻在拼接处
   的跳变，跳变过大的包其标签不可信。
3. 故障起点标签 + 敏感性网格，见 ``batfd/eval/onset.py``。

产出 ``outputs/tables/labels.csv``、``onset_sensitivity.csv``、``p2_channel_mapping.csv``。

用法：``python scripts/02_verify_labels.py``
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batfd import config, console  # noqa: E402
from batfd.data import cache  # noqa: E402
from batfd.eval import onset as onset_mod  # noqa: E402

console.setup()


def load_fault_table() -> dict:
    p = Path(__file__).resolve().parent.parent / "configs" / "fault_table.yaml"
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def channel_divergence(res: np.ndarray, n_train: int) -> dict[str, np.ndarray]:
    """逐单体算故障段相对正常段的稳健偏移（单位：正常段的稳健尺度）。

    返回中位数 / 90 分位 / 最大值：单体内阻发散往往到故障段**末期**才显现，只看中位数会
    被前段稀释，三者指向同一单体结论才稳。``res`` 的列 = 单体 1..8（对应 Hiddall 第 2..9
    通道，见 configs/base.yaml）。
    """
    normal = res[:n_train]
    fault = res[n_train:]
    # 减同包跨单体中位数，吸收整包随温度/老化的共同漂移，只留「该单体相对同包伙伴跑了多少」
    dev_n = normal - np.median(normal, axis=1, keepdims=True)
    dev_f = fault - np.median(fault, axis=1, keepdims=True)
    mu, sig = onset_mod.robust_scale(dev_n)
    z = (dev_f - mu) / np.maximum(sig, 1e-12)
    return {
        "median": np.median(z, axis=0),
        "q90": np.percentile(z, 90, axis=0),
        "max": z.max(axis=0),
    }


def main() -> int:
    cfg = config.load()
    ft = load_fault_table()

    caches = {n: cache.load_cache(cfg, n) for n in ("StandTrainData", "StandTestData1")}
    series = onset_mod.build_pack_series(cfg, caches)

    print("=" * 78)
    print("1. 拼接连续性检查（train 段 -> test1 段）")
    print("=" * 78)
    print("  判据：拼接处前后各 5 窗口的中位数之差 ÷ 训练段标准差。")
    print("  超过阈值说明两段不连续，该包的时序拼接不可信。\n")

    cont_rows = []
    cont_ok: dict[int, bool] = {}
    for pid, s in sorted(series.items()):
        c = onset_mod.continuity_check(s, threshold=3.0)
        cont_ok[pid] = c["ok"]
        print(
            f"  pack {pid:>2}: n_train={c['n_train']:>5}  max jump={c['jump_max']:.3f}  "
            f"{'OK' if c['ok'] else 'SUSPECT 单体 ' + str(c['suspect_cells'])}"
        )
        print(f"            逐单体 jump = {c['jump_cells']}")
        cont_rows.append({"pack_id": pid, **{k: v for k, v in c.items() if k != "pack_id"}})

    print()
    print("=" * 78)
    print("2. Hiddall 通道 <-> 单体 映射验证")
    print("=" * 78)
    print("  把故障段的通道偏移按大小排序，看发散最大的通道是否与 Table 5 的故障单体一致。\n")

    map_rows = []
    for pid, s in sorted(series.items()):
        div = channel_divergence(s.res, s.n_train)
        expected = [int(c) for c in ft["packs"][pid]["cells"]]
        desc = ft["packs"][pid]["description"]

        print(f"  pack {pid:>2}  Table5: {desc}")
        # div 的列索引 0..7 就是单体 1..8；对应的 Hiddall 通道号 = 单体号 + 1
        per_stat_hit: dict[str, bool | None] = {}
        for stat in ("median", "q90", "max"):
            v = div[stat]
            order = np.argsort(-v)
            top1_cell = int(order[0]) + 1
            rank = {int(c) + 1: int(np.flatnonzero(order == c)[0]) + 1 for c in range(len(v))}
            hit = (top1_cell in expected) if expected else None
            per_stat_hit[stat] = hit
            print(
                f"      {stat:>6}: 排序 " + ", ".join(f"C{int(c) + 1}={v[c]:+.2f}" for c in order)
            )
            if expected:
                print(
                    f"              预期单体 {expected} 的名次 "
                    f"{ {c: rank[c] for c in expected} }  -> top1 = C{top1_cell} 命中={hit}"
                )
        if not expected:
            print(f"      Table5 未指明单体，无法做映射验证")
        print()

        map_rows.append(
            {
                "pack_id": pid,
                "description": desc,
                "expected_cells": expected,
                "top1_by_stat": {
                    stat: {"cell": int(np.argmax(div[stat])) + 1, "hit": per_stat_hit[stat]}
                    for stat in ("median", "q90", "max")
                },
                "divergence": {k: [float(x) for x in v] for k, v in div.items()},
            }
        )

    print("=" * 78)
    print("3. 故障起点标签（Tier A）与敏感性网格")
    print("=" * 78)
    print("  起点 = 该单体『自身内阻 − 同包内其他单体中位数』首次连续 P 个窗口")
    print("         超过 median + k·MAD（基线只用正常期前段）。\n")

    onset_rows: list[dict] = []
    label_rows: list[dict] = []
    for pid, s in sorted(series.items()):
        rows = onset_mod.onset_grid(s, cfg)
        summ = onset_mod.onset_summary(rows, pid)
        usable = cont_ok.get(pid, False)
        print(
            f"  pack {pid:>2}: n_total={s.n:>6} (train {s.n_train})  "
            f"包起点 点估计={summ['onset_point']}  "
            f"区间=[{summ['onset_early']}, {summ['onset_late']}]  "
            f"有起点单体 {summ['n_cells_with_onset']}/{s.res.shape[1]}  "
            f"稳定单体 {summ['stable_cells']}  不稳定 {summ['unstable_cells']}"
            f"{'' if summ['stable_only'] else '  !! 无稳定单体，点估计不牢靠'}"
            f"{'' if usable else '   <-- 拼接可疑，标签不可信'}"
        )

        # 逐单体：点估计 + 区间，用来看「哪个单体先坏」以及是否符合 Table 5
        expected = [int(c) for c in ft["packs"][pid]["cells"]]
        for c, v in summ["per_cell"].items():
            if v["median"] is None:
                continue
            mark = "  <-- Table5" if c in expected else ""
            print(
                f"            单体 {c}: 点估计 {v['median']:>6}  区间 [{v['min']}, {v['max']}]  "
                f"有效 {v['n_valid']}/{v['n_total']}{mark}"
            )
        print()

        for r in rows:
            onset_rows.append(
                {"pack_id": pid, "usable": usable, **{k: v for k, v in r.items() if k != "pack_id"}}
            )
        label_rows.append(
            {
                "pack_id": pid,
                "description": ft["packs"][pid]["description"],
                "cells_expected": expected,
                "n_total": s.n,
                "n_train": s.n_train,
                "boundary_ok": usable,
                "onset_point": summ["onset_point"],
                "onset_early": summ["onset_early"],
                "onset_late": summ["onset_late"],
                "stable_cells": summ["stable_cells"],
                "unstable_cells": summ["unstable_cells"],
                "onset_in_test1": (
                    summ["onset_point"] is not None and summ["onset_point"] >= s.n_train
                ),
                "per_cell": {c: v for c, v in summ["per_cell"].items() if v["median"] is not None},
            }
        )

    out_dir = Path(cfg["paths"]["outputs_dir"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "p2_channel_mapping.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["pack_id", "description", "expected_cells", "stat",
             "top1_cell", "top1_hit", "divergence_C1..C8 (已减跨单体中位数)"]
        )
        for r in map_rows:
            for stat in ("median", "q90", "max"):
                t = r["top1_by_stat"][stat]
                w.writerow([
                    r["pack_id"], r["description"], r["expected_cells"], stat,
                    t["cell"], t["hit"],
                    " ".join(f"{x:+.3f}" for x in r["divergence"][stat]),
                ])

    with (out_dir / "onset_sensitivity.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["pack_id", "usable", "k", "persistence", "baseline_windows",
                        "sigma_floor_abs", "onset_cells_pack_min", "onset_cells"],
        )
        w.writeheader()
        for r in onset_rows:
            w.writerow(r)

    with (out_dir / "labels.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["pack_id", "description", "cells_expected", "n_total", "n_train",
                    "boundary_ok", "onset_point", "onset_early", "onset_late",
                    "stable_cells", "unstable_cells", "onset_in_test1",
                    "per_cell_point_median"])
        for r in label_rows:
            w.writerow([
                r["pack_id"], r["description"], r["cells_expected"], r["n_total"], r["n_train"],
                r["boundary_ok"], r["onset_point"], r["onset_early"], r["onset_late"],
                r["stable_cells"], r["unstable_cells"], r["onset_in_test1"],
                {c: v["median"] for c, v in r["per_cell"].items()},
            ])

    print("=" * 78)
    print(f"产出：{out_dir / 'p2_channel_mapping.csv'}")
    print(f"      {out_dir / 'onset_sensitivity.csv'}")
    print(f"      {out_dir / 'labels.csv'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
