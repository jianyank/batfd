# ChronoGuard：多通道时序异常检测算法库

发行名 `chronoguard-ts`｜导入名 `chronoguard`｜版本 0.1.0

这是可复用的算法库和固定评估协议，不是跨领域万能模型，也不是新发明的 LOF、Isolation Forest 或 PCA。

## 1. 能力范围

- 统一有限数值输入 `(N,T,C)`：N 为有序窗口数，T 为窗口长度，C 为通道数。
- 四种既有方法：`robust`、`lof`、`iforest`、`pca`；两种特征模式：`independent`、`peer`。
- 显式 fit → calibrate → predict，正常参考拟合与独立阈值校准分开。
- 输出异常分数、确认报警、报警起止边沿、通道证据及可传递的分块状态。
- 支持可变通道数，但每个拟合模型固定 T、C 与通道语义顺序。
- 模型保存/加载、可选 SHA256 检查、因果构窗、严格逐点与事件评价。
- 33 项库测试。

## 2. 安装与运行

要求 Python >= 3.10。核心流程不需要 PyTorch、CUDA、MATLAB 或 Office；matplotlib 仅在重新生成图表时需要，声明为可选依赖 `[plot]`。

已验证环境为 Python 3.12.14、NumPy 2.5.2、SciPy 1.18.1、scikit-learn 1.9.1、joblib 1.6.0、matplotlib 3.11.2。声明的版本范围不代表每个组合都已验证，复现实验优先使用 `requirements.txt`。

从仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -X utf8 -B -m unittest discover -s tests -v
python -X utf8 -B examples/demo.py
```

仅在依赖已匹配时使用 `--no-deps`。`-X utf8 -B` 不是必需项，代码中的文本 I/O 均显式指定编码，加上只是防御性做法。

## 3. 最小接口示例

下面是人工生成数据的接口演示，不能作为真实故障检出证据：

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
print(out.ranked_channels[:, :3])  # 0 基索引，不是原始列号自动映射
```

### 输入契约

- 输入必须非空、三维且全部有限；NaN/无穷被拒绝，不静默填补。
- 拟合至少 3 窗，校准至少 2 窗。这是 API 最低要求，不是统计充分性的保证；q99 尤其需要足够多的独立正常校准样本。
- 一个 `predict` 调用只能对应同一设备的一段连续有序序列。库不检查时间戳，连续性由调用方保证。
- 通道语义或顺序改变、窗口长度改变时须重新拟合与校准；相同形状不代表相同物理含义。
- 不自动识别单位、频率、时间戳或采样间隔，需调用方先做数据适配。
- 拟合与校准应使用业务确认的正常参考数据。API 不使用故障标签，也不能自行证实输入正常。

## 4. 因果窗口与防泄漏

原始序列应先划分角色，再分别构造窗口，避免拟合窗口与校准窗口跨边界共享原始采样点：

```python
from chronoguard.data import make_windows, split_train

# raw_training_series 和 raw_test_series 均为 (time, channels)
fit_raw, cal_raw, val_raw = split_train(raw_training_series)
fit_windows, _ = make_windows(fit_raw, window_size=16, stride=1)
cal_windows, _ = make_windows(cal_raw, window_size=16, stride=1)
val_windows, _ = make_windows(val_raw, window_size=16, stride=1)
test_windows, endpoints = make_windows(raw_test_series, window_size=16, stride=1)
# 输出对应窗口末端：对齐评价标签可用 test_labels[endpoints]
```

`split_train` 按顺序切为 60% 拟合 / 20% 校准 / 20% 验证，至少需要 10 行；每段构窗前还要检查其长度不小于窗口长度。

`make_windows` 的每个窗口只含末端及其之前的数据。T=16 时独立测试段前 15 点是预热区，不产生输出，评价必须明确这一覆盖范围，不能补为正常预测再混入指标。测试标签只用于评价，不用于选择阈值、方法或参数。

电池输入是预构造的缓存窗口，按包、按窗口索引分段。缓存没有原始时间戳或步长信息，**不能据此核实相邻窗口的原始点是否重叠，不能宣称已排除全部原始采样级泄漏**。

## 5. 特征与算法

| 特征模式 | 每通道统计量 | 使用边界 |
|---|---|---|
| `independent` | 均值、标准差、极差、归一化时间轴上的线性斜率、差分标准差 | 不直接混合不同通道的物理量，可用于异构监控通道；仍需每通道正常参考标准化 |
| `peer` | 相对通道中位数残差的均值、标准差、绝对残差 q95、残差差分标准差 | 至少 2 个物理可比较的同类通道，不能混合电压、温度、转速等不同单位 |

T=1 时 `independent` 的标准差、极差、斜率、差分标准差均为 0，主要使用逐点数值。因此 T=1 是明确的逐点基线，不代表已验证复杂时序建模能力。

拟合侧对特征计算中位数中心和 1.4826 × MAD 尺度；MAD < 1e-7 且拟合侧标准差 ≥ 1e-7 时改用该标准差，其他情况使用 MAD 并设 1e-7 下限。回退标记见 `model.scale_fallback_`。全部中心与尺度仅从拟合侧计算，校准与测试不会更新它们。常量通道突然变化、极小量纲或分布漂移仍可能造成异常放大。

| 方法 | 异常分数 | 固定实现参数 |
|---|---|---|
| `robust` | 全部标准化特征绝对值的最大值 | 中位数 / MAD，退化尺度按上述规则 |
| `lof` | novelty LOF 的负 `score_samples` | 邻居数 `min(20, N_fit-1)`，`n_jobs=1` |
| `iforest` | Isolation Forest 的负 `score_samples` | 100 棵树，`max_samples=min(256,N_fit)`，seed=42 |
| `pca` | 标准化特征重建残差平方的均值 | full SVD，保留 95% 拟合方差；完全常量拟合数据用全残差 |

PCA 的投影与重建使用固定逐行归约，避免 BLAS 随批大小变化的舍入差异在阈值边界改变报警；这不代表跨运行时或跨硬件的位级一致性。

所有分数都是越大越异常，但不同模型的分数没有统一量纲，不能直接比较或当作概率。

### 电池适配

```python
from chronoguard import battery_windows, AnomalyDetector

voltage = battery_windows(legacy_signal)  # (N,T,20) 中取 0,2,...,14 共 8 列
battery_model = AnomalyDetector('lof', feature_mode='peer')
# 自定义输入必须显式给出至少两列互不重复且有效的同类通道
voltage_custom = battery_windows(custom_signal, channel_indices=[0, 1, 2, 3])
```

排序返回的是适配后数组的 0 基通道号；要解释为原始传感器号，调用方须保存 `channel_indices` 映射。

## 6. 分块预测与报警状态

```python
first = model.predict(test[:60], device_id='device-A')
second = model.predict(test[60:], device_id='device-A', state=first.state)
# 换设备、时间断点、独立测试序列或重新开始时，显式重置
new_sequence = model.predict(test, device_id='device-B', state=None)
```

- 同一连续序列保持相同 `device_id`，并原样传递上一块返回的 `state`，分块报警与整段一致。
- `state` 绑定设备与模型/校准版本；重新 fit 或 calibrate 后旧状态被拒绝，不能跨设备复用。
- 不传 `state` 就是重新开始；用它接续连续序列会改变边界附近的连续确认。
- 默认严格 `scores > threshold`；连续 5 窗超阈值才从第 5 窗起确认，不回填前 4 窗。
- 一窗未超阈值即解除；`alarm_started` 为确认上升边沿，`alarm_cleared` 为下降边沿。
- 这里的 5 是窗口数量，不是秒，与窗口步长、采样率有关。
- 训练后不要直接修改构造参数；更换配置应新建检测器并重新拟合、校准。

## 7. 输出与评价

| 输出字段 | 含义 |
|---|---|
| `scores` | (N,) 异常分数 |
| `threshold` | 独立正常校准分数的 quantile，默认 q99 |
| `exceeded` / `confirmed` | (N,) 立即超阈值 / 连续确认标记 |
| `alarm_started` / `alarm_cleared` | (N,) 确认开始 / 解除边沿 |
| `channel_evidence` | (N,C) 通道证据；robust/LOF/IF 为标准化偏离，PCA 为通道重建残差平方均值 |
| `ranked_channels` | (N,C) 证据从大到小的 0 基索引，平局保持输入通道顺序 |
| `state` | 下一连续块可复用的状态 |

LOF 与 Isolation Forest 的通道证据是辅助偏离量，**不是对应模型分数的归因分解**。`peer` 残差依赖跨通道中位数；证据排序不等同于故障根因、因果解释或真实单体定位准确率。

```python
from chronoguard.evaluation import detection_metrics, alarm_intervals, normal_metrics

metrics = detection_metrics(labels, out.confirmed)  # 对齐的 0/1 一维标签
intervals = alarm_intervals(out.confirmed)          # [start,end) 半开区间
normal = normal_metrics(normal_out.confirmed)       # 仅正常序列的报警比例
```

严格逐点评价不做 point adjustment：事件内一次命中不会把整个事件补为检出。事件检出表示真实事件内至少出现一个确认点；延迟仅统计已检出事件，从事件开始到首次确认的样本偏移，不含未检出事件的"零延迟"。FPR 分母是实际正常点；只有正常数据时只能报告正常报警率，不能评价真实故障 Recall/F1。

## 8. 保存与安全边界

```python
digest = model.save('device_A_model.joblib')  # 路径必须不存在，避免覆盖
restored = AnomalyDetector.load('device_A_model.joblib', expected_sha256=digest)
```

joblib 使用 pickle。**只加载可信来源、在受控流程中生成的模型。** 哈希仅核对字节一致性，不证明来源安全；攻击者可以同时替换模型与哈希。不要加载附件、用户上传或网上下载的未知 joblib 文件。模型依赖 Python/NumPy/scikit-learn 等运行时，未保证跨版本兼容。

## 9. 复现实验

先完成依赖与本包安装。从仓库根目录执行；脚本自行定位仓库根，默认数据路径无需显式给出：

```bash
python -X utf8 -B examples/benchmark.py --scenario smd --download --output-dir results/algolib/my_smd_run
python -X utf8 -B examples/benchmark.py --scenario battery --output-dir results/algolib/my_battery_run
python -X utf8 -B examples/benchmark.py --scenario both --output-dir results/algolib/my_full_run
```

`--output-dir` 必须尚不存在，避免覆盖历史结果。

SMD 下载必须显式选择 `--download`，可能需要网络且上游 master 可变。本地下载清单保存 URL、大小与哈希；只要 `download_manifest.json` 存在，在线或离线的 `load_smd` 都会核对本机 train/test/test_label 的清单项、来源与哈希，不符或缺项时拒绝读取。没有清单的手工本地输入仍可读取，但仅做形状、有限值与标签检查，不宣称已核实原始字节身份。

电池缓存默认位于 `datasets/battery/StandTrainData`，需要 `signal.npy`、`ids.npy`、`meta.json`：signal 为 (N,256,20)，ids 与窗口对齐且包号为 6/8/9/10。拟合最多 6000 窗，按时间顺序均匀抽样；校准、验证、测试不抽样。q99 / 连续 5 窗 / seed 42 固定。

**本仓库不附带任何数据。**
