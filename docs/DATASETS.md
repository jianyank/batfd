# 数据集规格

本仓库的代码**不附带数据**。`datasets/` 被 `.gitignore` 排除，不随源码或交付包分发。
本文件只记录规格与获取方式，供拥有数据的使用方核对。

目录：

```text
datasets/
  battery/     电池电压数据（私有）
  smd/         服务器监控数据（第三方，OmniAnomaly）
  reference/   参考材料（未核验用途）
```

## 1. battery — 电池电压数据

来源为私有 MAT 文件，不在本仓库分发。原始文件与派生缓存分开放置。

### 1.1 派生缓存 `datasets/battery/`

`benchmark.py --battery-cache` 的默认目标是 `datasets/battery/StandTrainData`。

| 目录 | 窗口数 | 包号 | 时间戳 | 说明 |
|---|---:|---|---|---|
| `StandTrainData/` | 26,508 | 6, 8, 9, 10 | 无 | 训练数据，**保证无故障**，用于拟合与校准 |
| `StandTestData1/` | 8,110 | 6, 8, 9, 10 | 有 | 接近故障片段；包号与训练集相同 |
| `StandTestData2/` | 21,013 | 1, 2, 3, 4, 5 | 有 | 接近故障片段；无对应已训练模型 |
| `StandTestData3/` | 40,093 | 17, 18, 19, 20, 21 | 有 | 接近故障片段；无对应已训练模型 |
| `matlab_meta/` | — | — | — | 转换过程的元数据 |

三个测试缓存合计 **69,216 窗、14 个包**。

每个缓存目录的文件：

| 文件 | 形状 / 类型 | 含义 |
|---|---|---|
| `signal.npy` | `(N, 256, 20)` float32 | 逐窗口信号；**原始缩放值，未归一化** |
| `ids.npy` | `(N,)` | 与窗口对齐的包号 |
| `meta.json` | JSON | 来源文件、变量名、通道条件名、窗口数 |
| `cond.npy` | `(N, 10)` | 逐窗口工况协变量 |
| `hidden.npy` | — | 内阻相关变量（部分数据集存在） |
| `time.npy` | `(N,)` datetime64 | **仅测试缓存有**；训练缓存无时间戳 |

要点：

- 默认 `battery_windows()` 取 20 列中的 `0,2,...,14` 共 8 列作为可比单体通道，配合 `peer` 特征模式。
- `meta.json` 记录 `signal` 为原始缩放值；量纲换算是推断，仅在出图时应用。
- 训练缓存**没有时间戳与步长**，因此无法独立核实相邻预构造窗口是否共享原始采样点。
  这不等于已排除原始点级泄漏。
- `meta.json` 的 `circularity_guard` 声明工况协变量不含任何来自内阻的信息。
- `StandTestData1` 中约有 43 个完整窗口与训练缓存逐值重复，分析时应标记并给出剔除重复后的结果。

### 1.2 原始 MAT 文件 `datasets/battery/source/`

| 文件 | 大小 |
|---|---:|
| `StandTrainData.mat` | 267 MB |
| `StandTestData1.mat` | 99 MB |
| `StandTestData2.mat` | 79 MB |
| `StandTestData3.mat` | 347 MB |

MATLAB v5 格式。缓存由这些文件转换而来，转换链不在本仓库内。

## 2. smd — 服务器监控数据

第三方数据集，来自 [NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) 的 `ServerMachineDataset`。
上游为 MIT 许可，完整许可原文见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

`benchmark.py --smd-dir` 的默认目标是 `datasets/smd`。

```text
datasets/smd/
  train/<machine>.txt        训练序列（默认假设无故障，未经训练标签核实）
  test/<machine>.txt         测试序列
  test_label/<machine>.txt   逐点 0/1 标签，与 test 行对齐
  download_manifest.json     下载 URL、字节数与 SHA256
  LICENSE  README.md         上游原文
```

已下载的三台机器：

| 机器 | train 行 | test 行 | 测试异常点 | 占比 |
|---|---:|---:|---:|---:|
| machine-1-1 | 28,479 | 28,479 | 2,694 | 9.46% |
| machine-2-1 | 23,693 | 23,694 | 1,170 | 4.94% |
| machine-3-1 | 28,700 | 28,700 | 308 | 1.07% |

要点：

- 每行 38 个通道，逗号分隔的浮点文本。
- **SMD 共 28 台机器，本仓库只下载并验证了 3 台**，不代表完整数据集。
- `load_smd()` 在 `download_manifest.json` 存在时会核对三类文件的来源与 SHA256，
  不符或缺项即拒绝读取；无清单的手工输入仅做形状、有限值与标签检查。
- 上游 README 的运行说明包含在测试集上选 best F1 的做法，本项目未采用该步骤。
- `test_label` 只用于评价，不进入拟合、阈值校准或参数选择。

## 3. reference — 公开数据集的子集，来源已查明

```text
datasets/reference/sample_audit_20261003T091746Z/
  faults.details.-.LFP.batteries.xlsx   13 KB   故障明细表（见 3.3）
  field_data_test.zip                   39 MB   2 个系统的现场数据（见 3.2）
```

**来源已查明：这两个文件出自一套公开数据集。**

| | |
|---|---|
| 标题 | Lithium-Ion Battery Field Data: 28 LFP battery systems with 8 cells in series, up to 5 years of operation |
| DOI | [10.5281/zenodo.13715694](https://doi.org/10.5281/zenodo.13715694)（Zenodo，2024-09-14，v1.0.0） |
| 许可 | **CC-BY-NC-4.0** —— 必须署名，**不得商用** |
| 作者 | Schaeffer, Lenz, Gulla, Findeisen（TU Darmstadt）；Bazant, Braatz（MIT） |
| 论文 | [10.1016/j.xcrp.2024.102258](https://doi.org/10.1016/j.xcrp.2024.102258)，*Cell Reports Physical Science* 5(11), 2024，开放获取 |
| 参考代码 | [github.com/JoachimSchaeffer/BattGP](https://github.com/JoachimSchaeffer/BattGP) |
| 全量数据 | Zenodo `field_data.zip`，1591 MB，28 个系统、1.33 亿行、中位采样间隔 5 s |

本地 `field_data_test.zip` 是该数据集中 **2 个系统**的子集（`sys_3`、`sys_14`）。

数据集规格与本地现场数据列名**逐项吻合**，可确认同源：

| 数据集规格（官方 README） | 本地 `data_sys_*.csv` 列 |
|---|---|
| Current sensor 1 | `I_Battery` |
| Voltage sensors 9 | `U_Battery` + `U_Cell_1..8` |
| Temperature sensors 4 | `Temperature_1..4` |
| Cell balancing current sensors 8 | `I_CNV_Cell_1..8` |
| 24 V、≈160 Ah、LFP、每系统 8 单体串联 | 与论文正文一致 |

### 3.1 本项目电池数据与该数据集的关系

本项目 `datasets/battery/` 的包号（1–5、6、8、9、10、17–21）、每包 8 个可比单体、LFP 化学体系与 2013–2022 时间跨度，均与该数据集吻合。
**但训练缓存没有时间戳，未能逐值核实**，因此只作为高度一致的证据，不作为已证明的等同关系。

### 3.2 许可约束

`CC-BY-NC-4.0` 含 **NC（非商业）** 条款。用于校园算法竞赛（AI+学科交叉）属非商业用途，可以；**若转入创新创业类赛道或任何商业化场景，该数据不可使用**。任何引用必须按 CC-BY 要求署名，并给出 DOI。

### 3.3 `faults.details.-.LFP.batteries.xlsx`

单表 100 条记录，6 列：`vid` / `Fault type` / `Severity` / `Fault cell id` / `Fault Value` / `Unit`。

| Fault type | 数量 | 单位 |
|---|---:|---|
| Self-discharge | 40 | Ah/Day |
| High Resistance | 30 | Ratio: R/R95 |
| Low Capacity | 30 | Ratio: Q/Q95 |

Severity 为 Mild 52 / Moderate 30 / Severe 18。`vid` 取值 `vin_1`…`vin_100`；`Fault cell id` 取值 0–123（47 个不同值）。
文档属性显示由 openpyxl 于 2025-01-26 生成。

### 3.4 `field_data_test.zip`

解压后 2 个 CSV，各 26 列：`Timestamp, U_Battery, I_Battery, SOC_Battery, Temperature_1..4, U_CR, I_CR, U_Cell_1..8, I_CNV_Cell_1..8`。

| 文件 | 行数 | 时间跨度 |
|---|---:|---|
| `field_data/data_sys_3.csv` | 937,123 | 2013-03-07 → 2013-08-14 |
| `field_data/data_sys_14.csv` | 2,275,804 | 2017-08-05 → 2018-08-16 |

### 3.5 使用限制与合规要求

**可以使用。** 现场数据来自公开数据集，按 CC-BY 署名即可，非商业的校园竞赛用途符合许可。
引用时须给出数据集标题、作者、年份与 DOI：`10.5281/zenodo.13715694`。

**必须同时说明的约束：**

1. **NC 条款。** 不得用于商业化场景，也不得用于创新创业类赛道。若作品后续转入商业方向，须替换数据。
2. **数据集自带偏差。** 官方 README 声明全部 28 个系统均因"不满意行为"退回厂家，并明确写道
   "this data set is biased and not representative of the operational data of the entire population"。
   基于它的任何结论都应带上这条限制。
3. **`faults.details` 的字段语义未确认。** `Fault cell id` 取值 0–123，与每包 8 个单体的规模不符；
   `vid` 用 `vin_N` 而现场数据用 `sys_N`，两者命名体系不同。**该表目前不能直接当作单体索引或连接键使用。**
4. **逐包故障真值来自投稿中的稿件。** 论文附录按包与单体列出故障类型与受影响单体，但该稿件是
   2026-08-13 投 Elsevier 的预印本，尚未正式发表；引用它需要权利人确认授权。

本仓库**未使用这两个文件产出任何实验、指标或结论**。
若后续确认了字段语义与授权，应另立版本重新评价，并在此处补全引用信息。

> 参赛提醒：规则要求引用他人技术、算法或数据须明确标注并符合授权要求。
> 使用公开数据集是允许的，但必须署名并遵守 NC 条款。

## 4. 与评价的关系

- 电池数据**在本仓库内没有故障事件记录与可靠时间映射**，因此只能报告正常参考上的误报，
  不能计算真实故障召回率、F1、提前量或单体定位准确率。
- ⚠ **"训练数据保证无故障"这一前提未获证实，且现有证据对它不利。** 数据集官方 README 声明
  全部 28 个系统均因"不满意行为"退回厂家；论文附录的逐包故障表也把包 6、8、9、10 全部列为有故障。
  因此在取得时间映射之前，包 6/8/9/10 上的高报警率**不能自动解释为误报**，应表述为
  "未解释的跨包报警"，而不是"正常数据误报"。
- SMD 有逐点标签，可以计算严格逐点指标，但只覆盖 3 台机器，且训练段正常性是被假设而非被核实的。
- 两类的具体评价边界见 [../benchmarks/20261008/REPORT.md](../benchmarks/20261008/REPORT.md)
  与 [../research_system_20261007/REAL_ACCEPTANCE.md](../research_system_20261007/REAL_ACCEPTANCE.md)。
