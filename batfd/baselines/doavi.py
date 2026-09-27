"""DOAVI 基线 —— **未实现**，此处只留接口与原因说明。

状态：2026-09-26 决定**暂不实现**（用户确认）。

为什么不做，而不是"先做个近似"
--------------------------------
DOAVI 是本项目要对比的第二个方法，文献信息真实可查：

    Peifeng Huang, Shoutong Liu, Yinghui Ren, Yanyun He, Peipei Chao,
    Renlang Feng, Chuan Liu, Zhen Li, Zhonghao Bai.
    Safety risk assessment for automotive battery pack based on deviation
    and outlier analysis of voltage inconsistency.
    Journal of Cleaner Production, 466:142889, 2024.
    DOI: 10.1016/j.jclepro.2024.142889

从公开的摘要层面可以确认它的**方法轮廓**：

* 面向电芯（cell）与电池包（pack）两个层级，用**三个指标**做安全风险评估；
* 把单体电压的**相对偏差**做成散点图，展示在服役寿命的不同阶段；
* 在**不同的 SOC 区间内**量化**电压偏差**与**偏差角**，据此评估电芯级风险；
* 用**聚类算法**识别离群值；
* 包级风险用散点的**离散程度**刻画。

但**具体公式拿不到**。公开来源（X-MOL、NSTL、SearchWorks、Ovid 的收录页）只有上述
概念描述，没有任何公式；偏差角如何定义、SOC 如何分区、用哪种聚类、离散度如何量化、
阈值如何确定，全部在付费全文里。检索结果本身也确认了这一点。

本项目的硬性规则是**不编造方法细节**。如果按概念猜一套公式再挂上 DOAVI 的名字，
产出的对比数字既无法被审稿人复核，也无法证明公平 —— 这比"缺一个基线"更糟。
因此留空并注明，而不是做个近似充数。

另一个独立的技术约束（即便拿到原文也需要先解决）：DOAVI 依赖 **SOC 区间**分区。
而本项目的数据里 **只有 pack 6/8/9/10 带 SOC**（只有这四个文件的 `Hiddall` 含 SOC
通道，见 `configs/base.yaml` 的 `hidden` 段）。也就是说，忠实实现的 DOAVI 只能在这
4 个包上运行，另外 10 个包（pack 1–5、17–21）没有 SOC 可用于分区，这会直接削弱
"14 包对比"这张表。若要覆盖全部 14 包，需要先用等效电路模型自行估计 SOC —— 那是
另一项独立工作，且会把"模型估计误差"引入基线，影响公平性。

补齐它需要什么：1) 拿到原文全文（用户提供 PDF，或机构订阅）；或 2) 用户明确授权
「按摘要描述做概念复现」，并在论文中显著标注该基线为近似实现、且只在有 SOC 的
4 个包上报告。

在此之前，主对比基线为 :mod:`batfd.baselines.lfaae`（LFAAE，本项目所研究论文的
方法本身），它是更主要的对手，也足够支撑当前的方法对比。
"""

from __future__ import annotations

from dataclasses import dataclass

REASON = (
    "DOAVI 未实现：原文（DOI 10.1016/j.jclepro.2024.142889）为付费全文，"
    "公开来源只有概念描述、没有公式（偏差角定义、SOC 分区方式、聚类算法、"
    "离散度量化、阈值确定均不可得）。按项目规则不编造方法细节。"
    "另：该方法依赖 SOC 分区，而数据中仅 pack 6/8/9/10 带 SOC，"
    "忠实实现只能覆盖 4/14 个包。"
)

CITATION = (
    "Peifeng Huang, Shoutong Liu, Yinghui Ren, Yanyun He, Peipei Chao, "
    "Renlang Feng, Chuan Liu, Zhen Li, Zhonghao Bai. Safety risk assessment for "
    "automotive battery pack based on deviation and outlier analysis of voltage "
    "inconsistency. Journal of Cleaner Production, 466:142889, 2024. "
    "DOI: 10.1016/j.jclepro.2024.142889"
)

# 摘要层面可靠确认的方法轮廓，供将来实现时对照。
# ⚠ 这些不是公式，不能据此推演实现细节。
KNOWN_OUTLINE = {
    "levels": ["cell", "pack"],
    "n_indexes": 3,
    "cell_level": "在不同 SOC 区间内量化单体电压的相对偏差与偏差角，做成散点图",
    "outlier_step": "用聚类算法识别离群值",
    "pack_level": "用散点的离散程度评估包级风险",
    "validation": "含变形电芯的电池模组循环测试；5 台事故多发车辆的实车数据",
    "soc_dependency": True,  # 决定了它无法覆盖本项目全部 14 个包
}


@dataclass
class DOAVIUnavailable:
    """占位对象。任何试图调用它的代码都会立刻看到原因，而不是拿到假结果。"""

    reason: str = REASON
    citation: str = CITATION

    def __post_init__(self) -> None:
        raise NotImplementedError(self.reason)


def build(cfg: dict):  # pragma: no cover - 显式不可用
    """未实现。见模块 docstring。"""
    raise NotImplementedError(REASON)
