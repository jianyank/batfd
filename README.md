# BATFD：电池包无监督故障检测

当前仓库以 `research_system_20261007` 研究体系为主体，保留检测源码、基线比较、测试、阶段报告及汇总结果；旧深度学习框架与历史实验产物已从当前版本移除，不重写 Git 历史。

## 新增算法库入口（2026-10-08）

原研究体系保留不变；新增 **ChronoGuard 0.1.0** 多通道时序异常检测库，发行名 `chronoguard-ts`，导入名 `chronoguard`。

- [算法库安装、接口和安全边界](docs/ALGORITHM_LIBRARY.md)
- [参赛作品说明](docs/COMPETITION_BRIEF.md)
- [两场景28组实验报告与开发复验限制](benchmarks/20261008/REPORT.md)
- [第三方材料说明](docs/THIRD_PARTY_NOTICES.md)

四方法为 robust、LOF、Isolation Forest、PCA；接口可复用，不是万能模型或原创检测算法。电池只验证正常误报，SMD 仅3台逐点基线，效果有限。

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -X utf8 -B -m unittest discover -s tests -v
python -X utf8 -B examples/demo.py
```

## 任务与边界

- **训练数据保证无故障，测试数据为接近故障的时间片段。** 使用正常数据拟合检测器和校准参考阈值，推理不需要故障标签。
- 不将整个近故障片段标为阳性，也不要求故障后的每个数据点都报警。事件检出、提前量及真实单体定位评价需要独立事件记录与时间对应关系。
- 当前研究候选：**32维单体相对中位数残差 → MAD稳健标准化 → novelty LOF → 正常校准q99 → 连续5窗确认**。`lof_peer`是项目内简称，不是原创算法名称。
- 真实近故障测试集的冻结模型推理尚未完成，**不支持现场上线或安全控制结论**。

## 目录

```text
research_system_20261007/
  features.py / methods.py / protocol.py    特征、15种基线、分段与报警规则
  inject.py / evaluation.py / system.py     合成压力测试及检测系统
  run_benchmark.py                          四折按包留出实验与封存复验
  deploy.py / predict.py / smoke_demo.py    模型冻结、预测与接口演示
  diagnose_sequences.py                    连续序列诊断
  configs/screen.json / tests/              固定协议与回归测试
  MENTOR_REPORT.tex / MENTOR_REPORT.md       最新导师报告
  BEST_SCHEME.md / USAGE.md / REAL_ACCEPTANCE.md
  assets/                                  报告三幅图
  results/                                 历史研究的汇总表（不是逐窗数据）
requirements.txt                           核心依赖
```

## 快速开始

已测试环境为 Python 3.12.14。以下从仓库根目录执行；核心流程不需要 PyTorch、MiKTeX 或 Office。

```bash
python -m pip install -r requirements.txt
cd research_system_20261007
python -X utf8 -B -m unittest discover -s tests -v
python run_benchmark.py --help
```

当前精简版本已通过 **86项回归测试（2026-10-07，0失败）**；测试数较上一版本减少8项，对应已移出的历史审计模块，检测流程的回归测试保留。

回归测试只生成临时合成数据，不需要私有 MAT 文件或真实数据缓存。正式实验必须提供已有缓存，详见 `research_system_20261007/USAGE.md`。此版本以 NumPy 缓存为输入，不提供旧 MAT 预处理链。旧输入审计、报告转换工具、重复导航及过程日志已移出仓库；导师报告与研究汇总仍保留。

## 已有研究结果（不是本次重新训练）

四包26508窗、15种原始输入方法，拟合/校准/开发分开，阈值和参数不从挑战集择优。LOF-Peer合成开发域宏平均新增确认事件比例66.7%，Top-1目标命中61.8%；正常留出包6/8/9/10确认报警窗口比例为1.34%/20.38%/0.12%/16.30%。**合成指标不代表真实故障检出率，正常包报警表明跨包适配存在局限。**

最新方法与限制见 `research_system_20261007/MENTOR_REPORT.md`，精简结果见 `research_system_20261007/BEST_SCHEME.md`。汇总表保留原始研究数值；没有借清理仓库重算或提升指标。

## 数据与模型不公开

原始数据、NumPy缓存、逐窗输出、模型、历史源快照及原始日志不上传。本地 `data/`、`outputs/` 与未来 `research_system_20261007/runs/` 均被忽略。旧源码、未提交修改和全部实验已做仓库外可恢复备份。

旧模型绑定旧源码哈希，不能直接配合整理后的源码运行：应在完整旧环境中恢复使用，或用新源码重新运行实验并冻结新模型；不得修改旧 manifest 绕过验证。公开汇总表不足以单独完成历史实验的完整封存复验。
