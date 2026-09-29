"""P10：把 outputs/tables 的结果画成图（中英双版本）。只读表、**不重算任何指标**，数字都
来自 ``outputs/tables/*.csv`` —— 唯一例外是图 5 需要 pack 5 的**逐窗口完整序列**
（``pack5_case.csv`` 只有分数前 15 名、不含最早越限窗口），该序列由 ``scripts/09_pack5_case.py``
导出为 ``pack5_series.csv``，本脚本只读它，以免在绘图脚本里重写一遍 LOF 计算、两份实现漂移。

输出 ``outputs/figures/<fig>_<lang>.png``（300dpi，看）与 ``.pdf``（矢量，投稿），``lang`` ∈
{zh, en}；数据缺失的图**跳过并打印原因**，不画空图。

用法：``python scripts/10_figures.py [--only 1 3] [--no-pdf] [--langs zh en]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batfd import config, console, experiments, provenance  # noqa: E402
from batfd.viz import plots, style  # noqa: E402

console.setup()

FIG_NAMES = {
    1: "fig1_tradeoff",
    2: "fig2_latent_diag",
    3: "fig3_leadtime",
    4: "fig4_localize",
    5: "fig5_pack5_case",
    6: "fig6_dualtrack",
}


def read_table(path: Path, *, allow_legacy=False) -> list[dict]:
    return experiments.read_table(path, allow_legacy=allow_legacy)


def table_paths(tables, pattern, artifact_ids=None):
    roots = [tables] if isinstance(tables, Path) else tables
    paths = sorted(p for root in roots for p in root.glob(pattern))
    if artifact_ids:
        paths = [p for p in paths if p.stem.rsplit("_", 1)[-1] in artifact_ids]
    return paths


def single_table(tables, pattern, *, artifact_ids=None):
    paths = table_paths(tables, pattern, artifact_ids)
    if len(paths) > 1:
        raise ValueError(f"{pattern} 存在多个产物；请用 --artifact-ids 明确选择：{paths}")
    return paths[0] if paths else None


def num(v, default=None):
    s = (v or "").strip()
    if s in ("", "nan", "None"):
        return default
    try:
        return float(s)
    except ValueError:
        return default


def load_tradeoff(tables, *, allow_legacy=False, artifact_ids=None) -> dict[str, dict]:
    """tag -> {lof_mode -> {(q, m) -> row}}"""
    out: dict[str, dict] = {}
    for p in table_paths(tables, "tradeoff_*.csv", artifact_ids):
        for r in read_table(p, allow_legacy=allow_legacy):
            q, m = num(r["q"]), int(num(r["persistence"], 0))
            if q is None:
                continue
            points = out.setdefault(r["tag"], {}).setdefault(r["lof_mode"], {})
            if (q, m) in points and not allow_legacy:
                raise ValueError(f"权衡表重复口径：{r['tag']} / {q} / {m}；请用 --artifact-ids 选择")
            points[(q, m)] = {
                "trigger_rate_before_onset": num(r["trigger_rate_before_onset"], float("nan")),
                "early_detection_rate": num(
                    r["detection_rate"] if allow_legacy and "early_detection_rate" not in r
                    else r["early_detection_rate"], float("nan")),
            }
    return out


def load_latent(tables: Path, runs: Path) -> list[dict]:
    """``ablation_latent_diag.csv`` 的 6 行，另挂一列 ``val_recon_best_epoch``。

    ⚠ 表里的 ``best_val_loss`` **不能**当重建能力用：``val_loss`` 含 ``λ2·val_late +
    λ3·val_reg``，各配置 λ 不同（no_kl 的 λ2=0.1、no_late_no_kl 的 λ2=0），排出来是
    「谁的加权项多谁大」而非「谁重建得准」（原稿图 2 就把它贴上了 val_recon 的标签）。

    真正可跨配置比较的是 ``val_recon``：取 ``best_epoch`` 那一行，与
    ``_save_checkpoint`` 选中的权重严格对应（``best_epoch`` 由 ``models/train.py`` 按
    argmin val_loss 记为 1 起始的 epoch，与 ``history.csv`` 的 ``epoch`` 列同源）。
    """
    p = tables / "ablation_latent_diag.csv"
    rows = read_table(p, allow_legacy=True) if p.exists() else []
    order = {c: i for i, c in enumerate(style.ABLATION_CONFIGS)}
    rows = sorted(rows, key=lambda r: order.get(r["config"], 99))

    for r in rows:
        r["val_recon_best_epoch"] = None
        ep = num(r.get("best_epoch"))
        hist = runs / f"ours_{r['config']}" / "history.csv"
        if ep is None or not hist.exists():
            continue
        for h in read_table(hist, allow_legacy=True):
            if int(num(h["epoch"], -1)) == int(ep):
                r["val_recon_best_epoch"] = num(h.get("val_recon"))
                break
    return rows


def load_detect(tables, *, allow_legacy=False, artifact_ids=None) -> dict[str, dict]:
    """tag -> {pack_id -> row}，只取 fixed 口径 + test1（即下面两个 continue 条件）。

    ⚠ 必须把**所有**匹配文件的行走合并：子集文件（如 ``..._pack_baseline.csv``）的
    ``tag`` 与完整表相同，按 tag 直接赋值会被后读的覆盖掉。
    """
    out: dict[str, dict] = {}
    for p in table_paths(tables, "detection_*.csv", artifact_ids):
        for r in read_table(p, allow_legacy=allow_legacy):
            if (r["lof_mode"] != "train_novelty" or r["threshold_method"] != "fixed"
                    or r["dataset"] != "StandTestData1"):
                continue
            try:
                pid = int(r["pack_id"])
            except ValueError:
                continue
            packs = out.setdefault(r["tag"], {})
            if pid in packs and not allow_legacy:
                raise ValueError(f"检测表重复口径：{r['tag']} / pack {pid}；请用 --artifact-ids 选择")
            packs[pid] = r
    return out


def load_localize(tables, *, allow_legacy=False, artifact_ids=None) -> dict[str, list[dict]]:
    """tag -> 9 行（固定聚合口径 fault_period × raw，四个组合结论一致）。"""
    out: dict[str, list[dict]] = {}
    for p in table_paths(tables, "localization_*.csv", artifact_ids):
        rows = [r for r in read_table(p, allow_legacy=allow_legacy)
                if r["aggregation"] == "fault_period" and r["variant"] == "raw"]
        if rows:
            key = rows[0]["tag"]
            if key in out and not allow_legacy:
                raise ValueError(f"定位表存在多个产物：{key}；请用 --artifact-ids 选择")
            out[key] = sorted(rows, key=lambda r: int(r["pack_id"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="只画指定编号的图，如 --only 1 3")
    ap.add_argument("--langs", nargs="*", default=list(style.LANGS),
                    choices=list(style.LANGS))
    ap.add_argument("--no-pdf", action="store_true", help="只出 PNG")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment-ids", nargs="+", help="显式选择实验 ID，可跨模型比较")
    source.add_argument("--allow-legacy", action="store_true", help="只读历史 tables，明确不保证来源可核验")
    ap.add_argument("--artifact-ids", nargs="+", help="同一实验有多份参数变体时明确选择")
    args = ap.parse_args()

    style.setup()
    cfg = config.load()
    out_root = Path(cfg["paths"]["outputs_dir"])
    runs = out_root / "runs"
    if args.allow_legacy:
        tables = out_root / "tables"
        print("[warn] 历史模式：旧表来源不可验证，不与新实验混用。")
    else:
        selected = [experiments.Experiment.open(out_root, eid) for eid in args.experiment_ids]
        tables = [e.directory / "tables" for e in selected]
    inputs = table_paths(tables, "*.csv", args.artifact_ids)
    if args.artifact_ids:
        found = {p.stem.rsplit("_", 1)[-1] for p in inputs}
        if found != set(args.artifact_ids):
            ap.error("部分 --artifact-ids 不属于所选实验或不存在")
    for p in inputs:
        read_table(p, allow_legacy=args.allow_legacy)
    source_files = inputs + (sorted(runs.glob("*/history.csv")) if args.allow_legacy else [])
    selection = {"experiment_ids": args.experiment_ids, "legacy_unverified": args.allow_legacy,
                 "tables": {str(p): provenance.file_digest(p) for p in source_files},
                 "parameters": vars(args),
                 "implementation": provenance.source_digest([
                     "scripts/10_figures.py", "batfd/viz/plots.py", "batfd/viz/style.py"])}
    outdir = out_root / "figures" / "selections" / provenance.digest(selection)[:20]
    want = set(args.only) if args.only else set(FIG_NAMES)
    pdf = not args.no_pdf

    print("=" * 78)
    print(f"P10 出图：{sorted(want)}  语言 {args.langs}  输出 {outdir}")
    print("=" * 78)

    written: list[str] = []
    skipped: list[str] = []

    def emit(num_: int, builder):
        """builder(lang) -> fig；对每种语言各画一次。"""
        for lang in args.langs:
            try:
                fig = builder(lang)
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"图 {num_} [{lang}]：{type(exc).__name__}: {exc}")
                print(f"  [skip] 图 {num_} [{lang}] {type(exc).__name__}: {exc}")
                continue
            paths = style.save(fig, outdir, FIG_NAMES[num_], lang, pdf=pdf)
            written.extend(paths)
            print(f"  [ok]   图 {num_} [{lang}] -> " + ", ".join(Path(x).name for x in paths))

    if 1 in want:
        trade = load_tradeoff(tables, allow_legacy=args.allow_legacy, artifact_ids=args.artifact_ids)
        if not trade:
            skipped.append("图 1：没有 tradeoff_*.csv")
            print("  [skip] 图 1：没有 tradeoff_*.csv")
        else:
            emit(1, lambda lang: plots.fig_tradeoff(trade, lang))

    if 2 in want:
        # P4 历史消融表尚无实验清单，新模式不得混入旧数字。
        diag = load_latent(tables, runs) if args.allow_legacy else []
        if not diag:
            skipped.append("图 2：没有 ablation_latent_diag.csv")
            print("  [skip] 图 2：没有 ablation_latent_diag.csv")
        else:
            miss = [r["config"] for r in diag if r.get("val_recon_best_epoch") is None]
            if miss:
                print(f"  [warn] 图 2：{miss} 取不到 val_recon，该面板会显示 n/a")
            emit(2, lambda lang: plots.fig_latent(diag, lang))

    if 3 in want:
        det = load_detect(tables, allow_legacy=args.allow_legacy, artifact_ids=args.artifact_ids)
        if not det:
            skipped.append("图 3：没有 detection_*_train_novelty_*.csv")
            print("  [skip] 图 3：没有 detection_*_train_novelty_*.csv")
        else:
            emit(3, lambda lang: plots.fig_leadtime(det, lang))

    if 4 in want:
        loc = load_localize(tables, allow_legacy=args.allow_legacy, artifact_ids=args.artifact_ids)
        if not loc:
            skipped.append("图 4：没有 localization_*.csv")
            print("  [skip] 图 4：没有 localization_*.csv")
        else:
            emit(4, lambda lang: plots.fig_localize(loc, lang))

    if 5 in want:
        sp = single_table(tables, "pack5_series.csv" if args.allow_legacy else "pack5_series_*.csv",
                          artifact_ids=args.artifact_ids)
        if sp is None:
            skipped.append("图 5：缺 pack5_series（跑 scripts/09_pack5_case.py 生成）")
            print("  [skip] 图 5：缺 pack5_series")
        else:
            series = read_table(sp, allow_legacy=args.allow_legacy)
            # 相关系数由序列自身的两列现算 —— 与 09 打印的是同一对数组，值必然一致
            sc = np.array([num(x["lof_score"], float("nan")) for x in series])
            cur = np.array([num(x["mean_abs_i"], float("nan")) for x in series])
            ok = np.isfinite(sc) & np.isfinite(cur)
            r = float(np.corrcoef(sc[ok], cur[ok])[0, 1]) if ok.sum() > 2 else None
            emit(5, lambda lang: plots.fig_pack5(series, lang, corr_pack=r))

    if 6 in want:
        fp = single_table(tables, "far_dualtrack.csv" if args.allow_legacy else "far_dualtrack_*.csv",
                          artifact_ids=args.artifact_ids)
        if fp is None:
            skipped.append("图 6：没有 far_dualtrack.csv")
            print("  [skip] 图 6：没有 far_dualtrack.csv")
        else:
            emit(6, lambda lang: plots.fig_dualtrack(read_table(fp, allow_legacy=args.allow_legacy), lang))

    print()
    print("=" * 78)
    print(f"写出 {len(written)} 个文件；跳过 {len(skipped)} 项")
    for s in skipped:
        print("  - " + s)
    print("=" * 78)
    provenance.write_manifest(outdir / "selection.json", {**selection, "written": written, "skipped": skipped})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
