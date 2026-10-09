# ChronoGuard ｜ chronoguard-ts

面向多通道时序数据的异常检测算法库。发行名 `chronoguard-ts`，导入名 `chronoguard`，版本 0.1.0。

同一套接口与评估协议可换用在不同来源的数据上：正常参考拟合 → 独立校准 → 因果连续报警 → 通道证据 → 严格评价。

内核是四种**既有**方法（鲁棒绝对偏差、LOF、Isolation Forest、PCA 重建残差），本库不修改其算法本身，只统一输入契约、校准方式与评价协议。**不主张新算法。**

## 安装

要求 Python >= 3.10。核心流程不需要 PyTorch、CUDA、MATLAB 或 Office。

```bash
python -m pip install -r requirements.txt   # 或 python -m pip install -e . 按 pyproject 解析
python -m pip install -e . --no-deps        # 依赖已匹配时可省去解析
```

重新生成图表需要可选的 `[plot]` 依赖。

## 快速开始

```python
import numpy as np
from chronoguard import AnomalyDetector

rng = np.random.default_rng(42)
fit = rng.normal(size=(240, 16, 6))
cal = rng.normal(size=(120, 16, 6))
test = rng.normal(size=(160, 16, 6))
test[50:90, :, 2] += 8

model = AnomalyDetector(
    method='lof', feature_mode='independent',
    quantile=.99, persistence=5, random_state=42,
).fit(fit).calibrate(cal)
out = model.predict(test, device_id='device-A')
print(out.scores, out.confirmed)
print(out.ranked_channels[:, :3])   # 0 基索引，不是原始列号映射
```

合成数据的接口演示，不是真实故障检出证据。完整运行：

```bash
python -X utf8 -B examples/demo.py
python -X utf8 -B -m unittest discover -s tests -v
```

## 接口要点

- **输入**为 `(N,T,C)` 三维有限数值；NaN 与无穷被拒绝，不静默填补。
- **特征模式**两种：`peer`（相对通道中位数残差，要求 ≥2 个物理可比较的同类通道）、`independent`（逐通道独立统计，适用于量纲各异的异构通道）。
- **拟合/校准分离**：标准化参数只来自拟合侧；阈值取独立正常校准段的 q99，不随测试数据变化。
- **因果连续报警**：严格 `score > threshold`；连续 5 窗超阈值才从第 5 窗起确认，不回填前 4 窗。
- **状态**：一次 `predict` 只能对应同一设备的一段连续有序序列；分块预测需原样传递 `state`，换设备或重新拟合后必须显式重置。
- **通道证据**是辅助偏离量，不是 LOF / Isolation Forest 分数的归因分解，不等同于根因定位。
- 通道顺序或窗口长度改变时必须重新拟合与校准；相同形状不代表相同物理含义。

完整契约、分块状态规则与保存加载的安全边界见 [docs/ALGORITHM_LIBRARY.md](docs/ALGORITHM_LIBRARY.md)。

## 目录

```text
src/chronoguard/
  detector.py      AnomalyDetector：拟合、校准、因果预测、分块状态、模型持久化
  features.py      independent / peer 两种特征模式；battery_windows 通道适配
  data.py          因果构窗 make_windows、按角色切分 split_train、SMD 读取与校验
  evaluation.py    严格逐点指标、报警区间、正常序列报警率
examples/
  demo.py                        合成数据接口演示
  benchmark.py                   固定协议基准（电池 + SMD）
  benchmark_smd_windows.py       同上，但 SMD 用 T>1 开窗
  live_demo.py                   流式回放演示，可导出 mp4 / gif
tests/                           33 项回归测试
```

## 文档

- [算法库接口与安全边界](docs/ALGORITHM_LIBRARY.md)
- [第三方材料与数据来源](docs/THIRD_PARTY_NOTICES.md)

本库不附带任何数据。SMD 可由 `examples/benchmark.py --download` 取得；电池缓存需要另行获取，可邮件联系 **youaweyk@outlook.com**。引用的公开电池数据集为 `CC-BY-NC-4.0`，含**非商业**条款，不得用于商业化场景。

---

# 边界与未验证内容

前面是用法，以下是能力边界。**这些不是免责套话，是实际测过或明确没测的结论。**

## 验证范围

库在两类学科数据上运行过：电池电压数据（正常包留出，`peer` 特征）与服务器监控数据 SMD（3 台机器，`independent` 特征）。详细实验报告、逐机器指标与修正前后对照不属于本仓库。

## 局限

- **检测效果有限。** 服务器场景的严格逐点 F1 明显偏低且机器间差异大；电池侧只有正常留出评价，**没有真实故障标签**，只能报告正常参考上的确认报警率，无法计算真实故障的 Recall、F1 或提前量。
- **尺度修正属开发复验。** MAD 退化修正发生在已查看同一测试数据结果之后，**不是全新未见测试集的盲验**。
- **SMD 只验证 3 台**而非上游全部 28 台；没有重复随机种子、置信区间或同协议先进方法对照，不能得出 SOTA 结论。
- **窗口重叠未能独立核实。** 电池输入是预构造的缓存窗口，缺少时间戳与步长，不能宣称已排除全部原始点级泄漏。
- **不承诺跨领域自动迁移。** 换场景需要正常参考数据、明确的通道语义、合适的特征模式，并**重新拟合与校准**。
- **不代表生产可用**，不支持安全控制结论；`q99` 不保证部署域 1% 误报，连续窗不等于固定时间。
- **SHA256 只说明字节一致性**，不证明数据真实性、无泄漏或模型来源可信。

## 创新点定位

创新性在**工程与评估层面**：把碎片化的场景代码提炼为统一契约的库、区分同类／异构通道两种特征语义、把独立校准与因果连续报警固化为流程、并坚持不做 point adjustment 的严格评价。

**不主张**新的检测算法，**不主张**对现有方法的改进，**不主张**跨领域自动迁移能力。
