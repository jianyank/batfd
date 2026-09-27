"""P4：训练单体分辨非对称自编码器，并跑消融阶梯与潜变量诊断。

消融阶梯（cell_resolved / λ2 / λ3；直接回应论文「无消融实验」的缺口）::

    no_late_no_kl        cell_resolved=是   λ2=0    λ3=0
    no_kl                cell_resolved=是   λ2=0.1  λ3=0
    no_late              cell_resolved=是   λ2=0    λ3=1e-3
    full                 cell_resolved=是   λ2=0.1  λ3=1e-3
    shared_latent        cell_resolved=否   λ2=0    λ3=0
    shared_latent_late   cell_resolved=否   λ2=0.1  λ3=0

``shared_latent`` 即论文那种「未指明哪个潜变量对哪个单体」的形态，与 ``full`` 对比才能把
「单体分辨结构」与「物理对齐」的贡献分开（论文 Table 3 只报最终数字，未做隔离）。潜变量
诊断跑在 pack 6/8/9/10 的**完整生命周期**（train 正常期 + test1 故障期）上：单体间分化
恰恰发生在故障期，只用正常期窗口诊断会因无分化而失去意义。

用法：``python scripts/04_train_ours.py [--only NAME ...] [--epochs N] [--smoke]``
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batfd import config, console, progress  # noqa: E402
from batfd.data import cache as cache_mod  # noqa: E402
from batfd.eval import latent_diag  # noqa: E402
from batfd.models import train as train_mod  # noqa: E402

console.setup()

ABLATIONS: dict[str, dict] = {
    "no_late_no_kl":      dict(cell_resolved=True,  lambda2=0.0, lambda3=0.0),
    "no_kl":              dict(cell_resolved=True,  lambda2=0.1, lambda3=0.0),
    "no_late":            dict(cell_resolved=True,  lambda2=0.0, lambda3=1e-3),
    "full":               dict(cell_resolved=True,  lambda2=0.1, lambda3=1e-3),
    "shared_latent":      dict(cell_resolved=False, lambda2=0.0, lambda3=0.0),
    "shared_latent_late": dict(cell_resolved=False, lambda2=0.1, lambda3=0.0),
}
DEFAULT_ORDER = ["no_late_no_kl", "no_kl", "no_late", "full", "shared_latent", "shared_latent_late"]


def apply_overrides(cfg: dict, name: str) -> dict:
    """把消融配置写进 cfg 的副本（不改原对象，避免配置串味）。"""
    import copy

    c = copy.deepcopy(dict(cfg))
    spec = ABLATIONS[name]
    c["model"]["cell_resolved"] = spec["cell_resolved"]
    c["train"]["lambda2"] = spec["lambda2"]
    c["train"]["lambda3"] = spec["lambda3"]
    return config.Config(c)


def concat_lifecycle(caches: dict[str, dict]) -> dict:
    """把 train 与 test1 拼成「完整生命周期」缓存，供潜变量诊断使用。两文件覆盖的都是
    pack 6/8/9/10，拼起来才含故障期；时间轴不参与诊断，置 None。
    """
    tr, te = caches["StandTrainData"], caches["StandTestData1"]
    if tr.get("hidden") is None or te.get("hidden") is None:
        raise ValueError("两个数据集都必须有 Hiddall")
    return {
        "signal": np.concatenate([np.asarray(tr["signal"]), np.asarray(te["signal"])], axis=0),
        "ids": np.concatenate([np.asarray(tr["ids"]), np.asarray(te["ids"])], axis=0),
        "hidden": np.concatenate([np.asarray(tr["hidden"]), np.asarray(te["hidden"])], axis=0),
        "cond": np.concatenate([np.asarray(tr["cond"]), np.asarray(te["cond"])], axis=0),
        "time": None,
        "name": "lifecycle_train+test1",
        "meta": {"note": "train 正常期 + test1 故障期，用于潜变量诊断"},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定的消融配置")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="先跑单批过拟合自检")
    ap.add_argument("--no-progress", action="store_true",
                    help="关闭进度条（输出重定向到文件时会自动关闭，无需手动指定）")
    args = ap.parse_args()

    if args.no_progress:
        progress.set_enabled(False)

    cfg = config.load()
    runs_dir = Path(cfg["paths"]["outputs_dir"]) / "runs"
    train_cache = cache_mod.load_cache(cfg, "StandTrainData")
    caches = {
        "StandTrainData": train_cache,
        "StandTestData1": cache_mod.load_cache(cfg, "StandTestData1"),
    }
    lifecycle = concat_lifecycle(caches)
    print(f"[data] 潜变量诊断用数据集：{lifecycle['signal'].shape}（train + test1）")

    names = args.only or DEFAULT_ORDER
    bad = [n for n in names if n not in ABLATIONS]
    if bad:
        raise SystemExit(f"未知配置 {bad}，可选 {list(ABLATIONS)}")

    diag_rows: list[dict] = []
    for name in names:
        print()
        print("=" * 78)
        print(f"配置：{name}   {ABLATIONS[name]}")
        print("=" * 78)
        c = apply_overrides(cfg, name)
        tag = f"ours_{name}"

        if args.smoke:
            print("--- 单批过拟合自检（用独立模型，不污染正式训练）---")
            from batfd.data import dataset as ds_mod

            smoke_model = train_mod.build_model(c).to(train_mod.pick_device(c))
            smoke_ds = ds_mod.WindowDataset(c, train_cache, np.arange(16))
            smoke_bundle = train_mod.TrainBundle(
                cfg=c,
                model=smoke_model,
                train_ds=smoke_ds,
                val_ds=smoke_ds,
                split=ds_mod.SplitIndex(  # type: ignore[arg-type]
                    train_idx=np.arange(16), val_idx=np.arange(16), per_id={}
                ),
                target_scaler=train_mod.fit_target_scaler(train_cache, np.arange(16), c),
                device=train_mod.pick_device(c),
                out_dir=runs_dir / f"{tag}_smoke",
            )
            train_mod.overfit_single_batch(c, smoke_bundle, steps=200, batch_size=8)
            del smoke_model, smoke_bundle
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        bundle = train_mod.prepare(c, train_cache, out_dir=runs_dir, tag=tag)
        res = train_mod.train(bundle, epochs=args.epochs)

        # 潜变量诊断：跑在完整生命周期上（含故障期）
        probe = latent_diag.probe_latent(
            bundle.model, c, lifecycle, device=bundle.device
        )
        rep = latent_diag.report(probe, label=f"{name}（train+test1 完整生命周期）")
        diag_rows.append(
            {
                "config": name,
                "cell_resolved": ABLATIONS[name]["cell_resolved"],
                "lambda2": ABLATIONS[name]["lambda2"],
                "lambda3": ABLATIONS[name]["lambda3"],
                "best_val_loss": res["best_val_loss"],
                "best_epoch": res["best_epoch"],
                "diag_mean": rep["diagonal_advantage"]["diag_mean"],
                "offdiag_mean": rep["diagonal_advantage"]["offdiag_mean"],
                "diagonal_advantage": rep["diagonal_advantage"]["advantage"],
                "discrepancy_capture_mean": rep["discrepancy_capture"]["mean"],
                "soc_alignment": rep["soc_alignment"]["corr"],
                "corr_matrix": rep["diagonal_advantage"]["matrix"].round(4).tolist(),
                "discrepancy_per_cell": [
                    None if not np.isfinite(x) else round(float(x), 4)
                    for x in rep["discrepancy_capture"]["per_cell"]
                ],
            }
        )
        del probe, bundle
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(cfg["paths"]["outputs_dir"]) / "tables" / "ablation_latent_diag.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(diag_rows[0].keys()))
        w.writeheader()
        for r in diag_rows:
            w.writerow({**r, "corr_matrix": json.dumps(r["corr_matrix"])})

    print()
    print("=" * 78)
    print("消融汇总（潜变量诊断在 train+test1 完整生命周期上）")
    print("=" * 78)
    print(f"{'配置':>22} {'单体分辨':>8} {'λ2':>6} {'λ3':>8} {'对角优势':>10} {'差异捕获':>10} {'SOC对齐':>9}")
    for r in diag_rows:
        print(
            f"{r['config']:>22} {str(r['cell_resolved']):>8} {r['lambda2']:>6} {r['lambda3']:>8} "
            f"{r['diagonal_advantage']:>+10.4f} {r['discrepancy_capture_mean']:>+10.4f} "
            f"{r['soc_alignment']:>+9.4f}"
        )
    print(f"\n表：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
