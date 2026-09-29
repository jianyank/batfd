"""六张结果图。每个函数只负责画，数据由 ``scripts/10_figures.py`` 读表后传入。

1. **不在这里重算任何指标。** 数字一律来自显式选择的实验表（或明确允许的历史表），图与表不可能对不上。
2. **不确定的要画成"不确定"，不能画成 0**：图 2 里 nan（0/0，结构上无定义）显示成
   显式的 n/a，画成 0 会被读成"测出来是 0"。
3. 负结果如实呈现：第 3、4 张图刻意让"完全重合"和"同一处失手"可见，不做视觉修饰。
"""

from __future__ import annotations

import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from . import style
from .style import L, model_name

def fig_tradeoff(tradeoff: dict[str, dict], lang: str, *, per_pack_tag: str | None = "lfaae"):
    """``tradeoff``: tag -> {mode -> {(q, m) -> row dict}}

    ⚠ 只标注 ``ANNOT_Q`` 里的两个分位：曲线右侧 100% 平台上有 5–9 个点挤在同一 y，
    全标出来会糊成一团黑字。少标不丢信息，``f1_note`` 已说明"沿曲线 q 递增"。
    """
    ANNOT_Q = {0.99, 0.999}
    ms = [1, 5, 10]
    models = [t for t in ("lfaae", "ours_full", "ours_shared_latent") if t in tradeoff]
    fig, axes = plt.subplots(1, len(ms), figsize=(4.0 * len(ms), 3.6), sharey=True)

    for ax, m in zip(np.atleast_1d(axes), ms):
        # per_pack 作对照（仅基线有数据）
        if per_pack_tag and per_pack_tag in tradeoff and "per_pack" in tradeoff[per_pack_tag]:
            pk = tradeoff[per_pack_tag]["per_pack"]
            pts = sorted((r["trigger_rate_before_onset"] * 100, r["early_detection_rate"] * 100)
                         for (q, mm), r in pk.items() if mm == m)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls="--",
                        color=style.GREY, marker="x", ms=5, lw=1.1,
                        label=L(lang, "f1_perpack"))

        for t in models:
            rows = tradeoff[t]["train_novelty"]
            pts = sorted(((r["trigger_rate_before_onset"] * 100, r["early_detection_rate"] * 100, q)
                          for (q, mm), r in rows.items() if mm == m), key=lambda z: z[0])
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=style.MODEL_COLOR.get(t, "#333"),
                    marker=style.MODEL_MARKER.get(t, "o"), ms=5.5, lw=1.6,
                    label=model_name(lang, t))
            for x, y, q in pts:
                if q not in ANNOT_Q:
                    continue
                ax.annotate(f"{q:g}", (x, y), textcoords="offset points",
                            xytext=(3, 4), fontsize=6.5, color="#555")

        ax.axvline(1.0, color="#999", lw=0.9, ls=":")
        ax.set_title(L(lang, "f1_panel_m", m=m))
        ax.set_xlabel(L(lang, "f1_x"))
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylim(-8, 108)
        ax.set_yticks([0, 25, 50, 75, 100])

    ax0 = np.atleast_1d(axes)[0]
    ax0.set_ylabel(L(lang, "f1_y"))
    # 图例与图注都占坐标区下方空间：先留 30% 底带，再按画布坐标分层放（图例 0.20、脚注 0.02）
    fig.subplots_adjust(bottom=0.30)
    handles, labs = ax0.get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, 0.20),
               ncol=4, fontsize=8.5)
    # 名义虚警率贴着坐标区底边标；不要用 y=-8（在 ylim 之外，会被读成图例的一项）
    ax0.annotate(L(lang, "f1_nominal"), xy=(1.0, 0.025), xycoords=("data", "axes fraction"),
                 xytext=(4, 0), textcoords="offset points", fontsize=6.5, color="#777")
    fig.suptitle(L(lang, "f1_title"), y=1.02, fontsize=12)
    fig.text(0.5, 0.02, L(lang, "f1_note") + "　" + L(lang, "f1_note_perpack")
             + "\n" + L(lang, "f1_caveat"), ha="center", va="bottom",
             fontsize=7.5, color="#555")
    return fig


def fig_latent(diag: list[dict], lang: str):
    """``diag``: ablation_latent_diag.csv 的 6 行。"""
    configs = [r["config"] for r in diag]
    x = np.arange(len(configs))
    # 有单体分辨结构 = 深色；共享潜向量 = 浅色 + 斜纹
    colors = ["#c00000" if r["cell_resolved"] == "True" else "#d9d9d9" for r in diag]
    hatch = ["" if r["cell_resolved"] == "True" else "///" for r in diag]

    # ⚠ 最后一格用 val_recon，**不能**用 best_val_loss：后者含 λ2·val_late + λ3·val_reg，
    # 各配置的 λ 不同，跨配置排出来的是「谁的加权项多」而不是重建能力。
    panels = [
        ("diagonal_advantage", "f2_diag", "diverging"),
        ("discrepancy_capture_mean", "f2_disc", "diverging"),
        ("soc_alignment", "f2_soc", "plain"),
        ("val_recon_best_epoch", "f2_recon", "plain"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    for ax, (col, key, kind) in zip(axes.ravel(), panels):
        vals, undefined = [], []
        for r in diag:
            v = r.get(col)
            # 值可能来自 CSV（字符串）也可能由 10_figures.py 挂上（float/None）
            raw = "" if v is None else (v if isinstance(v, str) else f"{v:g}")
            if raw.strip().lower() in ("nan", "", "none"):
                vals.append(np.nan)
                undefined.append(True)
            else:
                vals.append(float(raw))
                undefined.append(False)
        vals = np.asarray(vals, dtype=float)

        if kind == "diverging":
            ax.axhline(0.0, color="#333", lw=0.9)
            if np.isfinite(vals).any():
                lim = max(0.05, np.nanmax(np.abs(vals)) * 1.35)
                ax.set_ylim(-lim, lim)
        else:
            fin = vals[np.isfinite(vals)]
            if fin.size:
                # ⚠ y 下限必须含负数：soc_alignment 里 no_late = -0.047，
                # 若从 0 起，那根负条被裁得只剩一条线，等于把数据藏了。
                lo, hi = min(0.0, float(fin.min())), max(0.0, float(fin.max()))
                # 留白按量程定比例（不能用固定值：val_recon 量程仅 0.037，固定垫会挤到上半格）
                span = (hi - lo) if hi > lo else max(abs(hi), 1e-6)
                pad = 0.22 * span
                ax.set_ylim(lo - pad, hi + pad)
                if lo < 0:
                    # 有配置是负相关：零线要画出来，否则负条看着像一根短正条
                    ax.axhline(0.0, color="#333", lw=0.9)

        drawn = np.where(np.isfinite(vals), vals, 0.0)
        bars = ax.bar(x, drawn, color=colors, edgecolor="#333", lw=0.7,
                      hatch=hatch, width=0.66)
        for b, v, und in zip(bars, vals, undefined):
            if und:
                # 结构上无定义：不画成 0，画成显式的 n/a
                b.set_visible(False)
                ax.annotate(L(lang, "f2_undefined"), (b.get_x() + b.get_width() / 2, 0),
                            ha="center", va="center", fontsize=6.6, color="#a00000",
                            fontweight="bold")
            else:
                off = 3 if v >= 0 else -9
                ax.annotate(f"{v:.4g}" if abs(v) < 1 else f"{v:.3g}",
                            (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, off),
                            ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([style.config_name(lang, c) for c in configs], fontsize=7.5)
        ax.set_title(L(lang, key), fontsize=10)

    handles = [
        Patch(facecolor="#c00000", edgecolor="#333", label=f"{L(lang,'f2_legend_res')}: {L(lang,'f2_yes')}"),
        Patch(facecolor="#d9d9d9", edgecolor="#333", hatch="///",
              label=f"{L(lang,'f2_legend_res')}: {L(lang,'f2_no')}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(L(lang, "f2_title"), fontsize=12)
    fig.text(0.5, -0.055, L(lang, "f2_note"), ha="center", fontsize=7.5,
             color="#555", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


def fig_leadtime(detect: dict[str, dict], lang: str):
    """``detect``: tag -> {pack_id -> row dict}，已筛 threshold_method=fixed、test1。"""
    packs = [6, 8, 9, 10]
    tags = [t for t in style.DETECT_MODELS if t in detect]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))

    # 每个包内把 7 个模型横向铺开，让"完全重合"可见（重合就是画的同一个 y）
    span = 0.72
    offs = np.linspace(-span / 2, span / 2, len(tags))
    for i, p in enumerate(packs):
        ax.axvspan(i - 0.5, i + 0.5, color="#f4f4f4" if i % 2 else "#fafafa", zorder=0)
        for t, off in zip(tags, offs):
            r = detect[t].get(p)
            if r is None:
                continue
            lead = (r.get("lead_days") or "").strip()
            if lead in ("", "nan", "None"):
                continue
            v = float(lead)
            colour = style.MODEL_COLOR.get(t, "#333")
            if t == "lfaae":
                ax.plot([i + off], [v], marker="*", ms=13, color=colour,
                        mec="white", mew=0.7, zorder=4, label=model_name(lang, t))
            else:
                ax.plot([i + off], [v], marker=style.MODEL_MARKER.get(t, "D"), ms=6.5,
                        color=colour, mec="white", mew=0.5, alpha=0.95, zorder=3,
                        label=model_name(lang, t) if i == 0 else None)

    ax.axhline(0.0, color="#333", lw=1.1)
    ax.annotate(L(lang, "f3_zero"), (-0.44, 0), xytext=(0, 4),
                textcoords="offset points", fontsize=7, color="#333")

    win = {}
    for t in tags:
        for p, r in detect[t].items():
            w = (r.get("windows_before_onset") or "").strip()
            if w not in ("", "nan", "None"):
                win[p] = int(w)
    ax.set_xticks(range(len(packs)))
    ax.set_xticklabels(
        [f"pack {p}\n({L(lang,'f3_windwos', n=win.get(p,'?'))})" for p in packs]
    )
    ax.set_xlabel(L(lang, "f3_x"))
    ax.set_ylabel(L(lang, "f3_y"))
    ax.set_xlim(-0.5, len(packs) - 0.5)

    seen, handles = set(), []
    for h, lab in zip(*ax.get_legend_handles_labels()):
        if lab and lab not in seen:
            seen.add(lab)
            handles.append(h)
    # ⚠ 图例必须放右上：lower left 的图例框会精确盖住 pack 6 那一列的 7 个点
    # （都在 +3.8 附近）与「起点」标注 —— 数据和标注一起被藏了。
    # 右上（pack 9/10 上方）是空的：那些值都在 ±16 以内。
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=7.5)

    fig.suptitle(L(lang, "f3_title"), fontsize=12)
    fig.text(0.5, -0.06, L(lang, "f3_note"), ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig


def fig_localize(loc: dict[str, list[dict]], lang: str):
    """``loc``: tag -> 该模型 fault_period/raw 的 9 行。"""
    tags = [t for t in ("lfaae", "ours_full", "ours_shared_latent") if t in loc]
    ref = loc[tags[0]]
    packed = [(r["pack_id"], r["dataset"], r["expected_cells"]) for r in ref]
    x = np.arange(len(packed))

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    span = 0.5
    offs = np.linspace(-span / 2, span / 2, len(tags))
    for i, (pid, _ds, _exp) in enumerate(packed):
        for t, off in zip(tags, offs):
            row = next((r for r in loc[t] if r["pack_id"] == pid), None)
            if row is None:
                continue
            rank = int(row["best_rank_of_expected"])
            hit = rank <= 3
            colour = style.MODEL_COLOR.get(t, "#333") if hit else "#d62728"
            ax.plot([i + off], [rank], marker="o" if hit else "X", ms=8 if hit else 9,
                    color=colour, mec="white", mew=0.6, zorder=3,
                    label=model_name(lang, t) if i == 0 else None)

    # 名次是序数量，用线性轴；set_ylim(大, 小) 即让第 1 名在顶部
    maxr = max(int(r["best_rank_of_expected"]) for t in tags for r in loc[t])
    ax.set_ylim(max(maxr + 1, 4), 0.4)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.axhline(3.5, color=style.OK_COLOR, lw=1.0, ls="--")
    ax.annotate("top-3", (len(packed) - 0.45, 3.5), xytext=(0, -9),
                textcoords="offset points", fontsize=7.5, color=style.OK_COLOR, ha="right")

    # 标出共同失手的 pack 5
    miss = [i for i, (pid, _d, _e) in enumerate(packed)
            if all(int(next(r for r in loc[t] if r["pack_id"] == pid)["best_rank_of_expected"]) > 3
                   for t in tags)]
    for i in miss:
        ax.axvspan(i - 0.45, i + 0.45, color="#ffecec", zorder=0)
        ax.annotate(L(lang, "f4_miss"), (i, 1.02), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=style.BAD_COLOR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"pack {p}" for p, _d, _e in packed], rotation=0, fontsize=8)
    ax.set_ylabel(L(lang, "f4_y"))
    ax.legend(loc="lower right", ncol=1)
    fig.suptitle(L(lang, "f4_title"), fontsize=12)
    fig.text(0.5, -0.05, L(lang, "f4_note"), ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig


def fig_pack5(series: list[dict], lang: str, *, corr_pack: float | None = None):
    """``series``: pack5_series.csv，逐窗口。

    ⚠ 报警时刻用 ``confirmed_*``（持久性规则**确认**后的报警），不能用 ``flagged_*``
    （单窗越限，报警时刻会更早）。
    """
    day = np.array([float(r["day_from_first_record"]) for r in series])
    sc = np.array([float(r["lof_score"]) for r in series])

    def first(col: str):
        return next((i for i, r in enumerate(series) if r.get(col) == "True"), None)

    i_fix = first("confirmed_fixed")
    i_base = first("confirmed_pack_baseline")
    i_max = int(np.argmax(sc))

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.plot(day, sc, lw=0.7, color="#1f4e79", alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel(L(lang, "f5_y"))
    ax.set_xlabel(L(lang, "f5_x"))

    if i_fix is not None:
        ax.axvline(day[i_fix], color="#c00000", lw=1.4, ls="--")
        ax.annotate(L(lang, "f5_alarm_fixed", d=day[i_fix]),
                    (day[i_fix], sc.max()), xytext=(6, -6),
                    textcoords="offset points", fontsize=7.5, color="#c00000")
    if i_base is not None:
        ax.axvline(day[i_base], color="#2e7d32", lw=1.4, ls=":")
        ax.annotate(L(lang, "f5_alarm_base", d=day[i_base]),
                    (day[i_base], sc.max() * 0.30), xytext=(6, -6),
                    textcoords="offset points", fontsize=7.5, color="#2e7d32")

    ax.plot([day[i_max]], [sc[i_max]], marker="v", ms=10, color="#e08214",
            mec="white", mew=0.7, zorder=5)
    ax.annotate(L(lang, "f5_outlier", d=day[i_max], s=sc[i_max]),
                (day[i_max], sc[i_max]), xytext=(10, 14),
                textcoords="offset points", fontsize=7.5, color="#8a4b08",
                arrowprops=dict(arrowstyle="->", color="#8a4b08", lw=0.8))

    txt = [L(lang, "f5_note")]
    if corr_pack is not None:
        txt.append(L(lang, "f5_corr", r=corr_pack))
    fig.suptitle(L(lang, "f5_title"), fontsize=12)
    fig.text(0.5, -0.07, "\n".join(txt), ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig


def fig_dualtrack(rows: list[dict], lang: str):
    """``rows``: far_dualtrack.csv 的 16 行（4 包 × 4 口径）。"""
    methods = [m for m in style.METHODS if any(r["threshold_method"] == m for r in rows)]
    packs = sorted({int(r["pack_id"]) for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    # (a) 汇总：每口径的轨 A / 轨 B
    ax = axes[0]
    x = np.arange(len(methods))
    a_tot = []
    for m in methods:
        sub = [r for r in rows if r["threshold_method"] == m]
        w = sum(float(r["A_windows"]) for r in sub if (r["A_windows"] or "").strip())
        c = sum(float(r["A_confirmed"]) for r in sub if (r["A_confirmed"] or "").strip())
        a_tot.append(100.0 * c / w if w else np.nan)
    # 轨 B 的汇总用同一份加权口径（分母 = 各包起点前窗口数之和）
    b_tot = []
    for m in methods:
        sub = [r for r in rows if r["threshold_method"] == m]
        w = sum(float(r["B_windows"]) for r in sub if (r["B_windows"] or "").strip())
        b_tot.append(100.0 * np.average(
            [float(r["B_trigger_rate"]) for r in sub if (r["B_trigger_rate"] or "").strip()],
            weights=[float(r["B_windows"]) for r in sub if (r["B_trigger_rate"] or "").strip()]
        ) if w else np.nan)

    wd = 0.36
    b1 = ax.bar(x - wd / 2, a_tot, wd, color="#1f4e79", label=L(lang, "f6_a"))
    b2 = ax.bar(x + wd / 2, b_tot, wd, color="#e08214", label=L(lang, "f6_b"))
    for bars, vals in ((b1, a_tot), (b2, b_tot)):
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}%", (bar.get_x() + bar.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 2),
                            ha="center", fontsize=7)
            else:
                ax.annotate(L(lang, "f6_na"), (bar.get_x() + bar.get_width() / 2, 0.4),
                            ha="center", fontsize=7, color="#a00000", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([L(lang, f"m_{m}") for m in methods], fontsize=7.5)
    ax.set_ylabel(L(lang, "f6_y_ab"))
    ax.set_title("(a) " + L(lang, "f6_x"), fontsize=10)
    ax.legend(loc="upper right")

    # (b) 逐包轨 A —— 展示"包间离散很大"
    ax = axes[1]
    for p in packs:
        vals = [100.0 * float(r["A_far"]) for r in rows
                if int(r["pack_id"]) == p and (r["A_far"] or "").strip()]
        ax.plot([p] * len(vals), vals, marker="o", ms=6, ls="none",
                color="#1f4e79", alpha=0.8)
    ax.set_xticks(packs)
    ax.set_xticklabels([L(lang, "f6_pack", p=p) for p in packs])
    ax.set_ylabel(L(lang, "f6_y_a"))
    ax.set_title("(b) " + L(lang, "f6_a"), fontsize=10)

    fig.suptitle(L(lang, "f6_title"), fontsize=12)
    fig.text(0.5, -0.10,
             "　".join([L(lang, "f6_note"), L(lang, "f6_note_a"),
                        L(lang, "f6_note_na"), L(lang, "f6_note_disp"),
                        L(lang, "f6_only_lfaae")]),
             ha="center", fontsize=7, color="#555", wrap=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig
