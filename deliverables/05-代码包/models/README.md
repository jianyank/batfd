# 训练模型（两个数据集）

本目录收录两个数据集上**已冻结的模型**，用于让评审无需重新拟合即可核对报告中的数字。

**总计 40 个模型，79.2 MB。**

```text
models/
  battery/     16 个：4 个电池包 × 4 种方法（T=256，peer 特征）
  smd_t1/      12 个：3 台机器 × 4 种方法（T=1 逐点基线）
  smd_t16/     12 个：3 台机器 × 4 种方法（T=16 开窗）
  verify_models.py   自检与重算脚本
```

每组目录内附该次运行的 `summary.json` / `summary.csv`，模型的阈值与指标均以它为准。

---

## ⚠️ 必须知情：模型里装了什么

**四类方法保存的内容差别很大，其中 LOF 会内嵌训练数据。**

| 方法 | 内部实际存储 | 单个体积 | 是否含训练样本 |
|---|---|---:|:--:|
| **LOF** | **`_fit_X`：6000 个训练窗口的标准化特征向量**（float64） | 9.7 MB（SMD）<br>2.5 MB（电池） | **是** |
| IForest | 100 棵孤立树的划分结构 | 0.9–1.4 MB | 结构，非样本 |
| PCA | 投影方向 `components_` 与均值 | 0.01–0.07 MB | 否，仅方向 |
| robust | 仅中位数 `center_` 与尺度 `scale_` | 0.00 MB | 否，仅统计量 |

**LOF 占本目录总量的 86%**（68.2 / 79.2 MB），原因就在这里：它不是"只存参数"，
而是把拟合时用到的 6000 个训练窗口原样保留下来，供 novelty 评分时做邻居查找。

这意味着：**发布 LOF 模型等同于发布训练数据的标准化特征样本。**

`verify_models.py` 会把每个模型实际存储的内容打印出来，不做隐藏。

## 数据前提

- **服务器 SMD**：来自公开数据集 OmniAnomaly / ServerMachineDataset，**MIT 许可**，可分发。
- **电池**：来自公开数据集 Zenodo `10.5281/zenodo.13715694`，**CC-BY-NC-4.0**。
  该许可要求**署名**且**不得商用**；本作品用于非商业的校园竞赛，符合许可条件。

  **需说明的边界**：本项目的电池缓存与该公开数据集在包号（1–28）、每包 8 个单体、
  LFP 化学体系与 2013–2022 时间跨度上高度吻合，但**训练缓存没有时间戳，未能逐值核实**。
  若后续确认电池数据实际受更严格的限制，**本目录中的电池 LOF 模型应一并撤回**，
  因为它们内嵌训练特征。

## 加载方式

模型经 `joblib`（即 pickle）序列化。**只加载可信来源的模型**——
哈希只能核对字节一致性，不能证明来源安全。

```python
from chronoguard import AnomalyDetector

model = AnomalyDetector.load("smd_t1/machine-1-1_lof.joblib")
print(model.method, model.threshold_)          # lof 5.53...
out = model.predict(windows, device_id="machine-1-1")   # windows 需为 (N,1,38)
```

`AnomalyDetector.load` 可传 `expected_sha256` 做字节核对。

## 自检与重算

```bash
# 仅核对身份与阈值（不需要数据）
python verify_models.py

# 带上 SMD 数据，重跑并逐项比对指标
python verify_models.py --smd-dir <SMD 目录>
```

**本次自检结果**（2026-10-09，Python 3.12.14）：

```text
身份 / 阈值检查：40 个模型，失败 0 项
SMD T=1  重跑核对：12 条记录，差异 0 条
SMD T=16 重跑核对：12 条记录，差异 0 条
```

即：**交付的模型能逐位复现报告中的每一个 F1 值。** 例如

```text
machine-1-1  lof  记录F1=0.404564  重算F1=0.404564
machine-3-1  pca  记录F1=0.420513  重算F1=0.420513
machine-1-1  pca  记录F1=0.608629  重算F1=0.608629   （T=16）
```

## 无法通过本目录验证的部分

- **电池模型的检测效果无法核对**。电池场景**没有独立故障事件记录**，
  只能核对报警计数与阈值，不能计算真实故障召回率、提前量或单体定位准确率。
- **模型不绑定当前源码哈希**。这些模型由 2026-10-08 的源码版本拟合；
  查阅 `summary.json` 与 `manifest.json` 可确认协议与版本。加载仅要求类结构兼容。
- **无数据时无法重算**。`summary.json` 记录了指标，但没有测试数据就无法独立复现；
  数据获取方式见技术方案的数据声明。

## 复现所需的代码

本目录只含模型。**测试与评估代码在源码包内**（`chronoguard_ts-0.1.0-source.zip`）：

| 脚本 | 用途 |
|---|---|
| `examples/benchmark.py` | 两数据集固定协议基准（电池 + SMD） |
| `examples/benchmark_smd_windows.py` | SMD 开窗（T>1）版本 |
| `examples/live_demo.py` | 流式推理回放动画 |
| `tests/` | 34 项回归测试 |
