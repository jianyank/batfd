"""绘图的字体、配色与中英标签。

所有图都出**中英两版**（中文供审阅、英文对齐 Elsevier 手稿），标签集中在 ``LABELS``、
绘图函数只认 ``L(zh|en, key)``，避免"改了中文忘了英文"的漂移。

字体：Windows 的 matplotlib 默认字体没有汉字，中文会渲染成方框（豆腐块）。
``setup()`` 把 ``Microsoft YaHei``／``SimHei`` 排到 sans-serif 最前，并关掉
``axes.unicode_minus`` —— 否则负号也会变方框。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境（脚本/CI）下也能出图

import matplotlib.pyplot as plt  # noqa: E402

LANGS = ("zh", "en")

# 打印与文件名用的 ASCII 短名，避免中文文件名在各平台上的编码问题
LANG_SLUG = {"zh": "zh", "en": "en"}

_CJK_FONTS = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]


def setup() -> None:
    """全局样式。绘图脚本第一件事调用。"""
    plt.rcParams.update(
        {
            "font.sans-serif": _CJK_FONTS,
            "font.family": "sans-serif",
            "axes.unicode_minus": False,  # 关键：否则负号是方框
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,  # 投稿要求：字体嵌入为 TrueType，不要转曲
        }
    )


# 配色：同一模型在所有图里必须同色，读者才能跨图对照
MODEL_COLOR = {
    "lfaae": "#1f4e79",                        # 深蓝 = 论文基线
    "ours_full": "#c00000",                    # 红 = 本方法完整配置
    "ours_shared_latent": "#e08214",           # 橙 = 无单体分辨的对照组
    "ours_no_late_no_kl": "#8c8c8c",
    "ours_no_kl": "#7f7f7f",
    "ours_no_late": "#a6a6a6",
    "ours_shared_latent_late": "#bf812d",
}
MODEL_MARKER = {
    "lfaae": "o",
    "ours_full": "^",
    "ours_shared_latent": "s",
}

# 图 3 的七个模型（顺序固定，避免每次跑图例顺序不一样）
DETECT_MODELS = [
    "lfaae",
    "ours_full",
    "ours_no_late_no_kl",
    "ours_no_kl",
    "ours_no_late",
    "ours_shared_latent_late",
    "ours_shared_latent",
]

# 图 2 的六个消融配置（顺序与 ablation_latent_diag.csv 一致）
ABLATION_CONFIGS = [
    "no_late_no_kl",
    "no_kl",
    "no_late",
    "full",
    "shared_latent",
    "shared_latent_late",
]

METHODS = ["fixed", "quantile_regression", "binned_quantile", "pack_baseline"]

OK_COLOR = "#2e7d32"    # 绿系：正向/命中
BAD_COLOR = "#c62828"   # 红系：负向/失手
GREY = "#8c8c8c"        # 对照/退化曲线


# ⚠ 标签是**纯文本**，matplotlib 不渲染 Markdown。
# 写 ``**加粗**`` 只会把星号原样画在图上（已踩过，图 6 脚注一行四个星号）。
# 要强调就用「」或把话说透，不要用记号。
LABELS: dict[str, dict[str, str]] = {
    "zh": {
        # 通用
        "div_before": "早于起点",
        "div_after": "晚于起点",
        # 图 1 权衡曲线
        "f1_title": "图 1　权衡曲线：检出率 vs 起点前触发率",
        "f1_x": "起点前触发率（%，越低越好）",
        "f1_y": "检出率（%）",
        "f1_panel_m": "持久性 m = {m}",
        "f1_note": "所有曲线共用同一批窗口；每个点是一个分位 q（0.9 → 0.999），"
                   "仅标注 q = 0.99 与 0.999，其余点不标以免糊成一团。",
        "f1_note_perpack": "灰色虚线 = per_pack 模式（仅基线有数据），在 m≥5 上塌到检出率 0。",
        "f1_perpack": "per_pack（自参照）",
        "f1_nominal": "名义虚警率 1%",
        "f1_caveat": "注意：检出率分母只有 4 个包（分辨率 25%），"
                     "「同检出率」为粗匹配",
        # 图 2 潜变量诊断
        "f2_title": "图 2　潜变量诊断消融（6 配置）",
        "f2_diag": "对角优势",
        "f2_disc": "单体间差异捕获度",
        "f2_soc": "SOC 对齐相关系数",
        "f2_recon": "验证重建误差 val_recon（最优 epoch）",
        "f2_undefined": "结构上无定义\n(0/0)",
        "f2_legend_res": "单体分辨结构",
        "f2_legend_align": "对齐损失 λ2",
        "f2_note": "共享潜向量配置的对角优势恒为 0、差异捕获度恒为 nan —— "
                   "不是训练不足，是那种参数化下该量恒为 0。",
        "f2_yes": "有",
        "f2_no": "无",
        "f2_with": "有 (λ2=0.1)",
        "f2_without": "无 (λ2=0)",
        # 图 3 检测提前量
        "f3_title": "图 3　检测提前量：7 个模型 × 4 个有标签包",
        "f3_x": "包（括号内为起点前窗口数）",
        "f3_y": "提前量（天，正 = 报警早于起点）",
        "f3_zero": "起点",
        "f3_note": "同色重叠 = 该值完全相同。pack 6 上 6 个 ours 配置完全重合于 +54.0。",
        "f3_windwos": "{n} 窗",
        # 图 4 单体定位
        "f4_title": "图 4　单体定位：真值单体的排名（9 个可标注包）",
        "f4_y": "真值单体中的最优排名（1 = 第 1 名）",
        "f4_hit": "命中",
        "f4_miss": "失手",
        "f4_note": "三模型一律 8/9 命中，且都在 pack 5 失手"
                   "（pack 5 标的是 Voltage drop，属口径外）。",
        "f4_second": "第二真值单体名次",
        # 图 5 pack 5
        "f5_title": "图 5　pack 5 误报案例：LOF 分数与两个阈值的报警时刻",
        "f5_x": "距本包首条记录的天数",
        "f5_y": "LOF 分数（对数轴）",
        "f5_alarm_fixed": "固定阈值报警\n第 {d:.2f} 天",
        "f5_alarm_base": "逐包基线报警\n第 {d:.2f} 天",
        "f5_outlier": "分数最高窗口\n第 {d:.2f} 天（LOF={s:.1f}）",
        "f5_note": "天数的零点是「本包首条记录」，与论文的投运日不同，"
                   "因此只复现机制、不声称复现论文的第 94 天。",
        "f5_corr": "corr(LOF 分数, 平均电流) = {r:+.3f}",
        # 图 6 双轨
        "f6_title": "图 6　双轨虚警报告",
        "f6_a": "轨 A　真虚警率",
        "f6_b": "轨 B　起点前触发率",
        "f6_y_a": "轨 A 真虚警率（%）",
        "f6_y_b": "轨 B 起点前触发率（%）",
        "f6_x": "阈值口径",
        "f6_y_ab": "比例（%）",
        "f6_na": "不适用",
        "f6_note": "两轨分母不同、场景不同，「绝不可相减或合并」，只能各自横向比较。",
        "f6_note_a": "轨 A 已逐包留一（in-sample 会退化到 0）。",
        "f6_note_na": "pack_baseline 以每包自身前 10% 为参照系，无「留一」可言。",
        "f6_note_disp": "轨 A 的包间离散很大，原因尚未归因。",
        "f6_pack": "包 {p}",
        "f6_only_lfaae": "注意：本图仅含基线模型。",
        # 方法名
        "m_fixed": "固定阈值\n(训练集 99 分位)",
        "m_quantile_regression": "工况条件分位数\n(梯度提升)",
        "m_binned_quantile": "工况分位分箱\n+单调平滑",
        "m_pack_baseline": "逐包自身基线\n(前 10%)",
    },
    "en": {
        "div_before": "earlier than onset",
        "div_after": "later than onset",
        "f1_title": "Fig. 1  Trade-off curve: detection rate vs pre-onset trigger rate",
        "f1_x": "Pre-onset trigger rate (%; lower is better)",
        "f1_y": "Detection rate (%)",
        "f1_panel_m": "persistence m = {m}",
        "f1_note": "All curves share the same windows; each point is one quantile q "
                   "(0.9 to 0.999). Only q = 0.99 and 0.999 are labelled, "
                   "to keep the plateau readable.",
        "f1_note_perpack": "Grey dashed = per_pack mode (baseline only); "
                           "collapses to 0% detection for m≥5.",
        "f1_perpack": "per_pack (self-referential)",
        "f1_nominal": "nominal FAR 1%",
        "f1_caveat": "Note: detection denominator is only 4 packs (25% resolution); "
                     "“same detection rate” is a coarse match",
        "f2_title": "Fig. 2  Latent diagnostics ablation (6 configurations)",
        "f2_diag": "Diagonal advantage",
        "f2_disc": "Inter-cell discrepancy capture",
        "f2_soc": "SOC alignment (correlation)",
        "f2_recon": "Validation reconstruction error val_recon (best epoch)",
        "f2_undefined": "structurally\nundefined (0/0)",
        "f2_legend_res": "Cell-resolved structure",
        "f2_legend_align": "Alignment loss λ2",
        "f2_note": "For shared-latent configurations the diagonal advantage is exactly 0 "
                   "and the discrepancy capture is nan — not under-training, but "
                   "identically 0 under that parameterisation.",
        "f2_yes": "yes",
        "f2_no": "no",
        "f2_with": "yes (λ2=0.1)",
        "f2_without": "no (λ2=0)",
        "f3_title": "Fig. 3  Detection lead time: 7 models × 4 labelled packs",
        "f3_x": "Pack (pre-onset window count in brackets)",
        "f3_y": "Lead time (days; positive = alarm earlier than onset)",
        "f3_zero": "onset",
        "f3_note": "Overlapping same-colour markers = identical values. "
                   "On pack 6 all six ours configurations coincide at +54.0.",
        "f3_windwos": "{n} win.",
        "f4_title": "Fig. 4  Cell localisation: rank of the true cell(s), 9 labelled packs",
        "f4_y": "Best rank among the true cells (1 = top)",
        "f4_hit": "hit",
        "f4_miss": "miss",
        "f4_note": "All three models score 8/9 and all fail on pack 5 "
                   "(pack 5 is labelled Voltage drop, outside the metric's scope).",
        "f4_second": "rank of 2nd true cell",
        "f5_title": "Fig. 5  Pack 5 false-alarm case: LOF score and the two alarm times",
        "f5_x": "Days since this pack's first record",
        "f5_y": "LOF score (log scale)",
        "f5_alarm_fixed": "fixed threshold alarm\nday {d:.2f}",
        "f5_alarm_base": "per-pack baseline alarm\nday {d:.2f}",
        "f5_outlier": "highest-scoring window\nday {d:.2f} (LOF={s:.1f})",
        "f5_note": "Day zero is this pack's first record, not the paper's "
                   "commissioning date — the mechanism is reproduced, not day 94.",
        "f5_corr": "corr(LOF score, mean current) = {r:+.3f}",
        "f6_title": "Fig. 6  Two-track false-alarm report",
        "f6_a": "Track A  true FAR",
        "f6_b": "Track B  pre-onset trigger rate",
        "f6_y_a": "Track A true FAR (%)",
        "f6_y_b": "Track B pre-onset trigger rate (%)",
        "f6_x": "Threshold rule",
        "f6_y_ab": "Percentage (%)",
        "f6_na": "n/a",
        "f6_note": "The two tracks have different denominators and scenarios; they must "
                   "never be subtracted or merged, only compared within a track.",
        "f6_note_a": "Track A is leave-one-pack-out (in-sample degenerates to 0).",
        "f6_note_na": "pack_baseline uses each pack's own first 10% as its reference; "
                      "there is no leave-one-out analogue.",
        "f6_note_disp": "Track A varies widely across packs; the cause is not attributed.",
        "f6_pack": "pack {p}",
        "f6_only_lfaae": "Note: baseline model only.",
        "m_fixed": "fixed\n(train 99th pct)",
        "m_quantile_regression": "condition-adaptive\nquantile (GBM)",
        "m_binned_quantile": "binned quantile\n+ monotone",
        "m_pack_baseline": "per-pack baseline\n(first 10%)",
    },
}

# 模型显示名
MODEL_LABEL = {
    "zh": {
        "lfaae": "LFAAE（论文基线）",
        "ours_full": "本方法（完整）",
        "ours_shared_latent": "本方法（共享潜向量）",
        "ours_no_late_no_kl": "本方法（无对齐·无KL）",
        "ours_no_kl": "本方法（无KL）",
        "ours_no_late": "本方法（无对齐）",
        "ours_shared_latent_late": "本方法（共享潜向量+对齐）",
    },
    "en": {
        "lfaae": "LFAAE (baseline)",
        "ours_full": "Ours (full)",
        "ours_shared_latent": "Ours (shared latent)",
        "ours_no_late_no_kl": "Ours (no align, no KL)",
        "ours_no_kl": "Ours (no KL)",
        "ours_no_late": "Ours (no alignment)",
        "ours_shared_latent_late": "Ours (shared latent + align)",
    },
}

# 消融配置显示名
CONFIG_LABEL = {
    "zh": {
        "no_late_no_kl": "无对齐\n无KL",
        "no_kl": "无KL",
        "no_late": "无对齐",
        "full": "完整",
        "shared_latent": "共享潜向量",
        "shared_latent_late": "共享潜向量\n+对齐",
    },
    "en": {
        "no_late_no_kl": "no align\nno KL",
        "no_kl": "no KL",
        "no_late": "no align",
        "full": "full",
        "shared_latent": "shared\nlatent",
        "shared_latent_late": "shared\nlatent+align",
    },
}


def L(lang: str, key: str, **kw) -> str:
    """取标签；``key`` 缺失时直接报错，避免静默出空标签。"""
    table = LABELS[lang]
    if key not in table:
        raise KeyError(f"标签 {key!r} 在语言 {lang!r} 中不存在")
    return table[key].format(**kw) if kw else table[key]


def model_name(lang: str, tag: str) -> str:
    return MODEL_LABEL[lang].get(tag, tag)


def config_name(lang: str, cfg: str) -> str:
    return CONFIG_LABEL[lang].get(cfg, cfg)


def save(fig, outdir, stem: str, lang: str, *, pdf: bool = True) -> list[str]:
    """存一份图：PNG（300dpi，看）+ PDF（矢量，投稿）。返回写出的路径。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    png = outdir / f"{stem}_{lang}.png"
    fig.savefig(png)
    written.append(str(png))
    if pdf:
        p = outdir / f"{stem}_{lang}.pdf"
        fig.savefig(p)
        written.append(str(p))
    plt.close(fig)
    return written
