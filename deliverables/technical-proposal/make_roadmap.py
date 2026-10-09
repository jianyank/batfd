# -*- coding: utf-8 -*-
"""
整体技术路线图 —— 供 technical-proposal.html / .pdf 使用。

图中所有文字与数字均取自 technical-proposal.html，未新造任何数据。
标注的行号对应当前 technical-proposal.html：

  L1 三处断裂                     -> :98-121    (§1.2)
  L1 定位（不追求更深模型）        -> :136-140   (§1.3)
  L2 输入 / 输出契约               -> :144-166   (§2.1)
  L2 peer / independent 特征模式   -> :205-230   (§2.3)
  L3 五阶段                        -> :168-196   (§2.2)
  L3 中位数 + 1.4826×MAD           -> :178, :229 (§2.2, §2.3)
  L3 连续 5 窗 / 不做 point adj.   -> :186-189, :255-269 (§2.2, §2.5)
  L3 检测内核（既有方法）           -> :170-173   (§2.2)
  L3 60/20/20 切分 · seed 42       -> :193-195   (§2.2)
  L4 393 行 / 5 模块 / 零重依赖     -> :294-301   (§3.1)
  L4 120 项测试 0 失败             -> :316       (§3.2)
  L4 49 产物 + 12 输入 哈希         -> :341       (§3.4)
  L5 26508 窗 / 80873 点           -> :330, :332 (§3.3)
  L5 40 组 / T=1 vs T=16 尺度对照   -> :334, :415 (§3.3, §4.3)
  L5 产出与下一步                  -> :495-529   (§5.1-5.2)

运行： D:/Python/miniconda3/envs/batfd/python.exe make_roadmap.py
"""

from __future__ import annotations

import warnings

import matplotlib as mpl

mpl.use("Agg")
# 与 results/figures 下的绘图脚本一致：缺字形直接报错，不静默出豆腐块
warnings.filterwarnings("error", message="Glyph .*missing from font")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = "fig_roadmap.png"

mpl.rcParams.update({
    "font.family": ["Microsoft YaHei"],
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# --------------------------------------------------------------------------
# 版面：数据坐标 x ∈ [0,100]，y ∈ [0,101.2]（自下而上）
# 正文宽度占 A4 的 174mm；图按 7.5in 宽出图后缩放到 174mm ≈ 91%，
# 因此 7pt 的字印在纸上约 6.4pt——下面所有字号都按这个换算过。
# --------------------------------------------------------------------------
X0, X1 = 2.0, 98.0
PAD = 1.6                       # 容器内左右留白
IX0, IX1 = X0 + PAD, X1 - PAD   # 3.6 / 96.4
IW = IX1 - IX0                  # 92.8

INK = "#1a1a1a"                 # 与 HTML body 同色
MUTED = "#555555"

# 每层一个色系：动机(金) → 契约(浅蓝) → 流程(主蓝) → 实现(灰蓝) → 验证(青)
LAYERS = {
    "prob":  {"edge": "#9a7b3f", "fill": "#fbf7ee"},
    "spec":  {"edge": "#5b7d9e", "fill": "#f1f6fb"},
    "pipe":  {"edge": "#173a56", "fill": "#e4eef8"},
    "build": {"edge": "#4d6274", "fill": "#eff2f5"},
    "proof": {"edge": "#2b6b63", "fill": "#e8f3f1"},
}

# 层序（自上而下）与纵向范围 (y_bottom, y_top)；层间留 3.0 缺口走箭头。
# 高度预算：14.0 + 15.5 + 26.5 + 12.0 + 20.0 = 88.0，加 4×3.0 缺口 = 100.0
ORDER = ["prob", "spec", "pipe", "build", "proof"]
BAND = {
    "prob":  (86.5, 100.5),
    "spec":  (68.0, 83.5),
    "pipe":  (38.5, 65.0),
    "build": (23.5, 35.5),
    "proof": (0.5, 20.5),
}
HEAD = 4.2                      # 层标题占用高度；内容自 y_top - HEAD 起算

TITLE = {
    "prob":  "问题与定位",
    "spec":  "统一契约与适配层",
    "pipe":  "五阶段流程（骨架固定，内核可替换）",
    "build": "工程实现与交付",
    "proof": "双场景验证 → 产出",
}


def rbox(ax, x, y, w, h, edge, fill, lw=1.2, radius=0.9, z=2):
    """圆角矩形。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=z,
        mutation_aspect=1.0,
    ))


def txt(ax, x, y, s, size, color=INK, weight="normal", ha="center", va="center",
        z=4, **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, **kw)


def arrow(ax, p0, p1, color="#4a6b87", lw=1.5, scale=13, z=5):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=scale,
        linewidth=lw, color=color, zorder=z, shrinkA=0, shrinkB=0,
    ))


def band(ax, key):
    """画一层容器 + 层标题，返回内容起始 y。"""
    y0, y1 = BAND[key]
    c = LAYERS[key]
    rbox(ax, X0, y0, X1 - X0, y1 - y0, c["edge"], c["fill"], lw=1.5, radius=1.2, z=1)
    # 左侧色条，强化层序
    ax.add_patch(Rectangle((X0 + 0.35, y0 + 1.2), 1.0, y1 - y0 - 2.4,
                           linewidth=0, facecolor=c["edge"], zorder=2))
    txt(ax, X0 + 2.6, y1 - 2.0, TITLE[key], 10.5, c["edge"],
        weight="bold", ha="left", va="top")
    return y1 - HEAD


def build_figure():
    fig = plt.figure(figsize=(7.5, 7.6), dpi=300, facecolor="white")
    ax = fig.add_axes([0.008, 0.006, 0.984, 0.988])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 101.2)
    ax.axis("off")

    # ================= L1 问题与定位 =================
    c = LAYERS["prob"]
    yc = band(ax, "prob")                      # 96.3

    pills = ["无故障标签", "通道语义异构", "因果性约束", "误报成本高"]
    pw, gap = 21.0, 2.9
    px = 50 - (len(pills) * pw + (len(pills) - 1) * gap) / 2
    for i, p in enumerate(pills):
        x = px + i * (pw + gap)
        rbox(ax, x, yc - 4.1, pw, 4.1, c["edge"], "#ffffff", lw=1.0, radius=0.8, z=3)
        txt(ax, x + pw / 2, yc - 2.05, p, 8.0, INK)

    txt(ax, 50, yc - 6.4,
        "现有三处断裂：特征断裂  ·  校准断裂  ·  评价断裂", 8.5, INK)
    txt(ax, 50, yc - 8.4,
        "定位：不追求更深的模型，交付可复用的检测流程与评估协议（不主张新算法）",
        8.5, c["edge"], weight="bold")

    # ================= L2 统一契约与适配层 =================
    c = LAYERS["spec"]
    yc = band(ax, "spec")                      # 79.3

    cw = (IW - 2.0) / 2                        # 45.4
    lx, rx = IX0, IX0 + cw + 2.0               # 3.6 / 51.0

    # 左：输入契约
    rbox(ax, lx, yc - 3.5, cw, 3.5, c["edge"], "#ffffff", lw=1.0, radius=0.8, z=3)
    txt(ax, lx + cw / 2, yc - 1.15, "输入契约  (N, T, C)", 8.2, c["edge"], weight="bold")
    txt(ax, lx + cw / 2, yc - 2.6, "非空 · 三维 · 全有限；NaN / Inf 显式拒绝", 7.4, MUTED)

    # 左：输出契约
    rbox(ax, lx, yc - 11.2, cw, 6.8, c["edge"], "#ffffff", lw=1.0, radius=0.8, z=3)
    txt(ax, lx + cw / 2, yc - 5.8, "输出契约（结构化对象）", 8.2, c["edge"], weight="bold")
    txt(ax, lx + cw / 2, yc - 7.7, "scores · threshold · confirmed", 7.4, MUTED)
    txt(ax, lx + cw / 2, yc - 9.5, "alarm 边沿 · channel_evidence · state", 7.4, MUTED)

    # 右：特征模式适配层（子框必须落在容器 68.1–79.3 内）
    rbox(ax, rx, yc - 11.2, cw, 11.2, c["edge"], "#ffffff", lw=1.0, radius=0.8, z=3)
    txt(ax, rx + cw / 2, yc - 1.3, "特征模式（学科语义适配层）", 8.2, c["edge"],
        weight="bold")

    for i, (name, desc) in enumerate([
        ("peer  —  同类可比通道的相对残差", "要求 ≥2 个物理可比较通道"),
        ("independent  —  逐通道独立统计", "适用于量纲互不相同的异构通道"),
    ]):
        sy = yc - 2.6 - i * 4.35               # 76.7 / 72.35
        rbox(ax, rx + 1.4, sy - 3.75, cw - 2.8, 3.75, c["edge"], c["fill"],
             lw=0.9, radius=0.7, z=4)
        txt(ax, rx + cw / 2, sy - 1.2, name, 7.4, INK)
        txt(ax, rx + cw / 2, sy - 2.7, desc, 6.9, MUTED)

    # ================= L3 五阶段流程 =================
    c = LAYERS["pipe"]
    yc = band(ax, "pipe")                      # 60.8

    txt(ax, 50, yc - 1.7,
        "时序切分固定为按时间顺序的 60% 拟合 / 20% 校准 / 20% 验证，seed 42",
        7.3, MUTED)

    stages = [
        ("正常参考拟合", "中位数 + 1.4826×MAD"),
        ("独立校准", "不重叠段 · q99"),
        ("因果连续报警", "连续 5 窗 · 不回填"),
        ("通道证据排序", "标准化偏离量"),
        ("严格逐点评价", "不做 point adjustment"),
    ]
    sgap = 1.4
    sbw = (IW - 4 * sgap) / 5                  # 17.44
    stop, sh = yc - 3.9, 10.0                  # 56.9 起，高 10.0
    for i, (name, desc) in enumerate(stages):
        x = IX0 + i * (sbw + sgap)
        cx = x + sbw / 2
        rbox(ax, x, stop - sh, sbw, sh, c["edge"], "#ffffff", lw=1.2, radius=0.9, z=3)
        ax.add_patch(Circle((cx, stop - 2.2), 1.1, facecolor=c["edge"],
                            edgecolor="none", zorder=4))
        txt(ax, cx, stop - 2.25, str(i + 1), 7.0, "#ffffff", weight="bold", z=5)
        txt(ax, cx, stop - 5.2, name, 8.0, INK, weight="bold")
        txt(ax, cx, stop - 8.4, desc, 6.8, MUTED)
        if i < 4:
            arrow(ax, (x + sbw + 0.1, stop - sh / 2), (x + sbw + sgap - 0.1, stop - sh / 2),
                  color=c["edge"], lw=1.2, scale=10)

    # 内核条
    ky, kh = yc - 20.2, 5.5                    # 40.6 – 46.1
    rbox(ax, IX0, ky, IW, kh, c["edge"], c["fill"], lw=1.2, radius=0.9, z=3)
    txt(ax, 50, ky + 3.85, "检测内核：四种既有方法，本作品不修改其算法本身", 7.8,
        c["edge"], weight="bold")
    txt(ax, 50, ky + 1.75,
        "robust（鲁棒绝对偏差） · LOF · Isolation Forest · PCA 重建残差", 7.3, INK)

    # ================= L4 工程实现与交付 =================
    c = LAYERS["build"]
    yc = band(ax, "build")                     # 31.3

    items = [
        ("交付形态", "chronoguard-ts 0.1.0", "393 行 · 5 模块 · 零重依赖"),
        ("隔离验证", "120 项测试，0 失败", "库外干净虚拟环境安装通过"),
        ("哈希溯源", "SHA256", "49 个产物 + 12 个输入核对"),
    ]
    bgap = 2.0
    bw = (IW - 2 * bgap) / 3                   # 29.6
    btop, bh = yc, 7.2                         # 31.3 – 24.1
    for i, (head, main, sub) in enumerate(items):
        x = IX0 + i * (bw + bgap)
        cx = x + bw / 2
        rbox(ax, x, btop - bh, bw, bh, c["edge"], "#ffffff", lw=1.1, radius=0.9, z=3)
        txt(ax, cx, btop - 1.9, head, 7.3, c["edge"], weight="bold")
        txt(ax, cx, btop - 4.0, main, 8.0, INK, weight="bold")
        txt(ax, cx, btop - 6.1, sub, 6.9, MUTED)

    # ================= L5 双场景验证 → 产出 =================
    c = LAYERS["proof"]
    yc = band(ax, "proof")                     # 16.3

    scen = [
        ("电池场景  ·  peer 特征", "4 个正常包 · 26508 窗"),
        ("服务器 SMD  ·  independent 特征", "3 台机器 · 80873 测试点"),
    ]
    bgap = 2.0
    bw = (IW - bgap) / 2                       # 45.4
    stop, sh = yc, 5.2                         # 16.3 – 11.1
    for i, (head, sub) in enumerate(scen):
        x = IX0 + i * (bw + bgap)
        cx = x + bw / 2
        rbox(ax, x, stop - sh, bw, sh, c["edge"], "#ffffff", lw=1.1, radius=0.9, z=3)
        txt(ax, cx, stop - 1.8, head, 7.8, INK, weight="bold")
        txt(ax, cx, stop - 3.9, sub, 7.2, MUTED)

    # 中间：实验链条
    my, mh = yc - 10.0, 4.0                    # 6.3 – 10.3
    rbox(ax, IX0, my, IW, mh, c["edge"], c["fill"], lw=1.0, radius=0.8, z=3)
    txt(ax, 50, my + mh / 2,
        "40 组固定协议实验   →   时序尺度对照（T=1 vs T=16）   →   "
        "定位特征空间距离对比度塌缩机理", 7.3, INK)
    arrow(ax, (50, stop - sh), (50, my + mh + 0.05), color=c["edge"], lw=1.1, scale=10)

    # 底部：产出与下一步
    oy, oh = yc - 15.3, 4.6                    # 1.0 – 5.6
    rbox(ax, IX0, oy, IW, oh, c["edge"], "#ffffff", lw=1.1, radius=0.8, z=3)
    txt(ax, IX0 + 3.0, oy + oh / 2,
        "产出：可复用流程与评估协议  ·  迁移失效诊断机理", 7.5, c["edge"],
        weight="bold", ha="left")
    txt(ax, IX1 - 2.4, oy + oh / 2, "下一步：事件级真值  ·  独立盲验", 7.5, MUTED,
        ha="right")
    arrow(ax, (50, my - 0.05), (50, oy + oh + 0.05), color=c["edge"], lw=1.1, scale=10)

    # ================= 层间箭头 =================
    for a, b in zip(ORDER[:-1], ORDER[1:]):
        arrow(ax, (50, BAND[a][0]), (50, BAND[b][1]))

    fig.savefig(OUT, dpi=300, facecolor="white")
    print(f"written: {OUT}")


if __name__ == "__main__":
    build_figure()
