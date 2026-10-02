# 四折 LOPO 检测评分修正与固定消融 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans；主代理TDD实现关键路径，只读代理并行审查，遵循先前批准的现有工作区边界。

**Goal:** 不重训，获得版本隔离的标准训练LOF四折结果及固定三组指标对照。

**Architecture:** v2继承旧LOF实现，只替换训练分数。单一入口读取封存特征与索引，拟合三包校准数据，在全新输出目录保存并语义复验结果。

**Tech Stack:** Python、NumPy、scikit-learn、unittest、PowerShell隐藏后台进程。

## Global Constraints
遵循配套设计；旧源码/产物/输入不得变化；不读取hidden/标签/测试；留出不参与拟合或模型选择；固定k20/q.99/m5；不自动Git提交。

## Task 1：版本隔离评分（独立可测试）
Files: Create batfd/detect/lof_v2.py、tests/test_lof_v2.py。
- [x] 编写训练分数等于标准拟合分数、新样本评分/scaler/旧实现不变、q99与小样本邻居边界测试。
- [x] 运行新增测试，确认缺失v2实现的红测试。
- [x] 最小继承实现，运行同一命令转绿。

## Task 2：特征复算入口（独立可验收）
Files: Create scripts/17_phase7_lof_replay.py、tests/test_phase7_lof_replay.py。
- [x] 测试交错列选择、fit只接收校准数据、真实小四折重放、拒绝覆盖/树重叠、源哈希/语义篡改拒绝。
- [x] 确认红测试，最小实现 run_replay/verify_replay 与CLI。
- [x] 运行针对性测试；启动后台全24维真实复算；独立verify及Phase6标准对照一致。

## Task 3：固定消融与交付
Files: 新outputs/diagnostics目录、新docs/changes记录；不修改既有封存文档或源码。
- [x] 全24维验收后后台 --ablation 跑固定all/sigma_v/reconstruction。
- [x] 新入口verify、Phase5 verify、Phase6 verify、完整unittest和CLI help。
- [x] 独立只读审查，有问题追根因并配复现测试；审查后才封存最终产物。
- [x] 记录真实计数、哈希、协议、限制和后续收益顺序。

## 2026-10-02 执行结果

- 正式全24维修正：D:/Matlab/project/pypack/outputs/diagnostics/phase7_lof_corrected_20261002T151200Z；运行和--verify均0。
- 固定三组消融：D:/Matlab/project/pypack/outputs/diagnostics/phase7_lof_ablation_20261002T152000Z；运行和--verify均0。
- 全量回归：97项通过；新增LOF v2专项13项、Phase7复算行为11项。
- 独立只读审查：无阻断项；未自动提交、合并或推送。
- 后续若追求真实收益，优先是补充有明确故障起点/健康标签的时间序列验证，而不是根据本轮无标签报警数量择优。
