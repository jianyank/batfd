# ChronoGuard 实施与验收计划

日期：2026-10-08｜继续既有工作区，不创建分支、不提交或推送。

**目标：** 交付可安装、可复用、可检查的多通道时序异常检测库。
**架构：** WindowFeatures 负责通道特征，AnomalyDetector 负责拟合/校准/预测与状态，data/evaluation负责输入与严格评价。
**技术：** Python >=3.10、NumPy、scikit-learn、joblib、unittest。

## 全局约束

- 输入为有限 (N,T,C)，每个模型固定T/C/通道语义顺序。
- fit/calibrate独立，正常参考角色与测试评价分离。
- 状态绑定设备和模型/校准版本；时间断点必须重置。
- 通道排序为0基，证据不是因果根因。
- 只改需求所需文件，保留8份用户未提交修改及历史结果。

## 任务1：核心特征与检测器

文件：src/chronoguard/{__init__,features,detector}.py；tests/test_core.py。

- [x] independent与peer特征、电池显式列适配、原特征等价测试。
- [x] 四方法，fit/calibrate/predict分离，正常参考尺度与校准分位数。
- [x] 状态和报警边沿、批量/分块一致、重拟合/校准失效。
- [x] 保存、可信加载及可选哈希检查。
- [x] 稀疏零MAD非恒定特征先复现再修根因，保留前后结果。
- [x] PCA阈值边界分块不一致：先失败复现，再固定逐行投影/重建归约，保持严格超阈值规则。
- [x] SMD离线读取已有清单时必须核对三类文件哈希，先失败复现再修正。

## 任务2：数据、评价与安装入口

文件：src/chronoguard/{data,evaluation}.py；tests/test_data_evaluation.py；pyproject.toml；examples/demo.py。

- [x] 因果窗口、按时间60/20/20分段、端点对齐。
- [x] SMD显式下载/加载与缓存哈希，不静默覆盖损坏输入。
- [x] 严格逐点/事件评价，不做point adjustment。
- [x] 可安装配置、人工注入演示、34项新库测试。

## 任务3：固定实验与公开材料

文件：examples/benchmark.py；benchmarks/20261008/；docs/。

- [x] 固定q=.99、persistence=5、seed=42、最大6000拟合样本。
- [x] 电池四包正常留出与SMD三台逐机T=1基线，全部28组公开汇总。
- [x] 保存输入/源码/模型SHA256与源快照，复算12组SMD严格指标。
- [x] 不宣称真实电池故障召回、跨领域万能模型或原创既有算法。
- [x] 明示已观察同一测试集、尺度修正后的开发复验。

## 任务4：收尾交付（本轮）

- [x] 修复PowerShell管道ASCII与转义导致的中文/代码块损坏，直接UTF-8写入，保留失败复现及修复后检查。
- [x] 给仓库README追加入口，保留原研究内容；核对所有文档链接与代码块。
- [x] 重建wheel，检查中文元数据，仓库外独立安装运行四方法。
- [x] 白名单源码zip，不包含数据/模型/预测/日志；附原features.py用于等价测试。
- [x] 解压源码包独立安装，运行34项新测试、demo和benchmark --help。
- [x] 重跑原86项测试，核对8份用户初始哈希，git diff --check。
- [x] 更新VERIFICATION与交付哈希清单，按证据说明完成及效能局限。

## 最终验收记录

- 最终实验：outputs/algolib/checked_release_20261008；原始与复核前结果保留。
- 当前源码34项库测试 + 原86项回归，共120项通过；45组PCA分块检查与SMD清单红绿回归通过。
- 新wheel仓库外安装；候选源码包已解压/安装并通过34项、demo、benchmark --help。
- 最终源码包包含更新后的本计划与VERIFICATION，并再执行同样独立验收；完整包哈希保存在包外dist/DELIVERY.json和CHECKSUMS.sha256，避免自引用。
- 依赖环境复用已有batfd运行时；不声称全新依赖环境、盲验或生产可用。
