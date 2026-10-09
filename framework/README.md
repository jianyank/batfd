# ChronoGuard

发行名 `chronoguard-ts`｜导入名 `chronoguard`｜版本 0.1.0

面向多通道时序数据的异常检测算法库：**正常参考拟合 → 独立校准 → 因果连续报警 → 通道证据 → 严格评价**，
同一套接口与评估协议可换用在不同来源的数据上。

**这不是一个跨领域通用模型，也不是新发明的 LOF、Isolation Forest 或 PCA。**
换数据需要正常参考数据、明确的通道语义、合适的特征模式，以及重新拟合与校准。
当前在电池电压与服务器监控两类数据上做过验证，检测效果有限，不得用于安全控制或宣称领先。

## 安装

要求 Python >= 3.10。核心流程不需要 PyTorch、CUDA、MATLAB 或 Office。

```bash
# 可移植安装：按 pyproject 的版本范围解析依赖
python -m pip install -e .

# 需要重新生成图表时
python -m pip install -e .[plot]

# 精确复现测试环境（Python 3.12，锁定版本）
python -m pip install -r requirements.txt
```

已验证环境为 Python 3.12.14、NumPy 2.5.2、SciPy 1.18.1、scikit-learn 1.9.1、joblib 1.6.0、threadpoolctl 3.7.0、matplotlib 3.11.2。
声明的版本范围不代表每个组合都已验证；复现结果优先使用 `requirements.txt`。

`-X utf8 -B` 不是必需项，代码中的文本 I/O 均显式指定编码；加上只是防御性做法。

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
  benchmark_smd_windows.py       同上，但 SMD 用 T>1 开窗（同一套流程）
tests/                           34 项回归测试
docs/ALGORITHM_LIBRARY.md        完整接口、输入契约与安全边界
```

## 输入契约（要点）

- 输入为 `(N, T, C)` 三维有限数值；NaN/无穷会被拒绝，不静默填补。
- 拟合至少 3 窗，校准至少 2 窗；这是 API 下限，不是统计充分性的保证。
- 一次 predict 只能对应同一设备的一段连续有序序列；库不检查时间戳。
- 通道顺序或窗口长度改变时必须重新拟合与校准；相同形状不代表相同物理含义。
- 通道证据是辅助偏离量，**不是 LOF / Isolation Forest 分数的归因分解**，不等同于根因定位。

完整契约、分块状态规则、保存加载的安全边界见 [ALGORITHM_LIBRARY.md](../docs/ALGORITHM_LIBRARY.md)。

## 关于测试

`tests/` 中有一项 peer 等价测试会加载原研究体系的 `features.py` 做逐值比对。
该文件只在源码仓库内存在；单独发布 `framework/` 时该测试会**跳过**而不是失败。

## 数据

本库不附带任何数据。数据集规格与获取方式见 [../docs/DATASETS.md](../docs/DATASETS.md)。
