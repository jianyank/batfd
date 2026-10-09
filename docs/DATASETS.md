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

## 3. reference — 第三方参考资料，来源与授权状态未知

```text
datasets/reference/sample_audit_20261003T091746Z/
  faults.details.-.LFP.batteries.xlsx   13 KB
  field_data_test.zip                   39 MB
```

**这两个文件由第三方提供，来源、授权与用途均未确认。本文件只记录只读观察到的结构，不作任何来源或可用性声明。**

### 3.1 `faults.details.-.LFP.batteries.xlsx`

单表 100 条记录，6 列：`vid` / `Fault type` / `Severity` / `Fault cell id` / `Fault Value` / `Unit`。

| Fault type | 数量 | 单位 |
|---|---:|---|
| Self-discharge | 40 | Ah/Day |
| High Resistance | 30 | Ratio: R/R95 |
| Low Capacity | 30 | Ratio: Q/Q95 |

Severity 为 Mild 52 / Moderate 30 / Severe 18。`vid` 取值 `vin_1`…`vin_100`；`Fault cell id` 取值 0–123（47 个不同值）。
文档属性显示由 openpyxl 于 2025-01-26 生成。

### 3.2 `field_data_test.zip`

解压后 2 个 CSV，各 26 列：`Timestamp, U_Battery, I_Battery, SOC_Battery, Temperature_1..4, U_CR, I_CR, U_Cell_1..8, I_CNV_Cell_1..8`。

| 文件 | 行数 | 时间跨度 |
|---|---:|---|
| `field_data/data_sys_3.csv` | 937,123 | 2013-03-07 → 2013-08-14 |
| `field_data/data_sys_14.csv` | 2,275,804 | 2017-08-05 → 2018-08-16 |

### 3.3 为什么不能用于本项目的评价

- **无法建立连接键。** 故障表用 `vin_N`，现场数据用 `sys_N`，本项目电池缓存用包号 1–21，三者命名体系互不相通。
- **单体索引规模不一致。** 故障表的 `Fault cell id` 取到 123，而现场数据只有 `U_Cell_1..8` 八个单体，本项目缓存为 20 列（取其中 8 列）。
- **来源与授权未知。** 提供方未说明出处，无法确认转载或展示授权。

因此本仓库**未使用这两个文件产出任何实验、指标或结论**，它们也不进入交付材料。
若后续确认了出处、授权与字段映射，应另立版本重新评价，并在此处补全引用信息。

> 参赛提醒：规则要求引用他人数据须明确标注并符合授权要求，伪造或来源不明地使用数据可导致取消资格。
> 在确认来源与授权之前，不得将本节材料作为作品的效果证据。

## 4. 与评价的关系

- 电池数据**没有故障事件记录与可靠时间映射**，因此只能报告正常参考上的误报，
  不能计算真实故障召回率、F1、提前量或单体定位准确率。
- SMD 有逐点标签，可以计算严格逐点指标，但只覆盖 3 台机器，且训练段正常性是被假设而非被核实的。
- 两类的具体评价边界见 [../benchmarks/20261008/REPORT.md](../benchmarks/20261008/REPORT.md)
  与 [../research_system_20261007/REAL_ACCEPTANCE.md](../research_system_20261007/REAL_ACCEPTANCE.md)。
