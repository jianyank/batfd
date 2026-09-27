"""P3：训练论文基线（LFAAE）。

1. 逐层尺寸核对：假前向一次，对照论文 Table 1 逐行比对每层实际输出形状（该表 stride 有
   排印错误，不实测无法确认其余部分是否真对上）。
2. 单批过拟合自检：固定小批次反复训练，重建 MSE 应趋近 0；压不下去说明模型/损失/梯度回传
   有结构性错误，此时跑全量纯属浪费时间。
3. 全量训练：只用 pack 6/8/9/10 的正常期数据（与论文训练集一致），标准化统计量也只在该
   训练集上拟合。

用法：``python scripts/03_train_baseline.py [--epochs N] [--skip-overfit]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batfd import config, console, progress  # noqa: E402
from batfd.baselines import lfaae  # noqa: E402
from batfd.data import cache, dataset as ds_mod  # noqa: E402

console.setup()


def overfit_single_batch(cfg: dict, cache_dict: dict, *, steps: int = 400, bs: int = 8) -> float:
    """单批过拟合自检，返回末次重建 MSE。"""
    split = ds_mod.chronological_split(
        np.asarray(cache_dict["ids"]), cache_dict.get("time"),
        float(cfg["split"]["val_ratio"]),
    )
    ds = ds_mod.WindowDataset(cfg, cache_dict, split.train_idx[:bs])
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lfaae.build(cfg).to(device)
    model.fit_normalizers(batch["x"], batch["v"])
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    x = batch["x"].to(device)
    v = batch["v"].to(device)
    target = model.cell_norm(v)

    first = last = None
    for s in range(steps):
        loss = torch.nn.functional.mse_loss(model(x), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s == 0:
            first = float(loss.detach())
        last = float(loss.detach())

    print(f"[overfit] 重建 MSE {first:.6f} -> {last:.8f}"
          f"（降至 {100 * last / max(first, 1e-12):.4f}%）")
    if last > 0.05 * first:
        print("[overfit] !! 未显著下降，训练通路可能有问题，建议先排查再跑全量")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--skip-overfit", action="store_true")
    ap.add_argument("--tag", default="lfaae")
    ap.add_argument("--no-progress", action="store_true",
                    help="关闭进度条（输出重定向到文件时会自动关闭，无需手动指定）")
    args = ap.parse_args()

    if args.no_progress:
        progress.set_enabled(False)

    cfg = config.load()
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "runs"

    print("=" * 78)
    print("1. 与论文 Table 1 的逐层尺寸核对")
    print("=" * 78)
    ok = lfaae.verify_shapes(cfg)
    print(f"  结论：{'全部吻合' if ok else '存在不一致 —— 实现与论文不是同一个网络，先排查'}")
    if not ok:
        return 1

    print()
    print("=" * 78)
    print("2. 训练集")
    print("=" * 78)
    tr_cache = cache.load_cache(cfg, "StandTrainData")
    print(f"  缓存 {tr_cache['meta']['source_file']}："
          f"{tr_cache['signal'].shape}，ID {sorted(tr_cache['meta']['id_counts'].keys())}")
    print("  注：与论文一致，只用 pack 6/8/9/10 的正常期数据；"
          "标准化统计量也只在该训练集上拟合。")

    if not args.skip_overfit:
        print()
        print("=" * 78)
        print("3. 单批过拟合自检")
        print("=" * 78)
        overfit_single_batch(cfg, tr_cache)
    else:
        print()
        print("3. 单批过拟合自检：已跳过（--skip-overfit）")

    print()
    print("=" * 78)
    print("4. 全量训练")
    print("=" * 78)
    res = lfaae.train(cfg, tr_cache, out_dir=out_dir, tag=args.tag, epochs=args.epochs)

    print()
    print("=" * 78)
    print(f"checkpoint: {res['out_dir']}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
