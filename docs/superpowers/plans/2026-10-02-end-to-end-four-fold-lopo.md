# 端到端四折 LOPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 StandTrainData 的 pack 6/8/9/10 上实现严格隔离的端到端四折 LOPO，并以后台任务完成真实训练。

**Architecture:** 新增单一实验脚本复用现有缓存、训练器、重建、故障指标和 LOF 实现。每折构造不含留出包的训练视图，深度模型从随机初始化训练；训练完成后对校准包与留出包提取特征，使用校准包拟合 LOF/阈值，并输出独立可核验产物。

**Tech Stack:** Python 3、NumPy、PyTorch、scikit-learn、标准库 unittest、PowerShell 后台进程。

## Global Constraints
- 留出包不进入训练、验证、标准化、LOF 拟合、阈值校准或物理目标读取。
- 固定 seed=42、现有 skip_on 配置、LOF n_neighbors=20、q=0.99、持久性 m=5。
- 不读取 StandTestData1/2/3、labels.csv、旧冻结特征或旧模型权重。
- 不覆盖旧输出；新运行目录必须独占创建。
- 真实四折训练以一个隐藏后台进程顺序执行，避免单 GPU 显存争抢。

## Task 1: 隔离与诊断行为测试
- Create tests/test_phase5_e2e_lopo.py
- 覆盖留出包排除、训练/验证互斥、留出物理目标不读取、输出保护、失败状态和小型端到端。
- 先运行新增测试并确认实现尚不存在时按预期失败。

## Task 2: 端到端实验入口
- Create scripts/15_phase5_e2e_lopo.py
- 实现 split_outer_fold、build_fold_cache、run_fold、main。
- 复用现有 train_mod.prepare/train/load_checkpoint、inference.reconstruct_all、paper_metrics 和 LOFDetector。
- 原子更新 running/failed/complete manifest，保存独立权重、切分索引、特征、分数、逐包摘要及哈希。

## Task 3: 验证与记录
- 运行新增测试、全量 unittest、--help 和语法检查。
- 创建 docs/changes/2026-10-02-phase5-e2e-lopo.md 记录证据。

## Task 4: 后台真实四折
- 用 D:/Python/miniconda3/envs/batfd/python.exe 启动隐藏后台顺序训练。
- 核实 PID 存活、首折开始训练、运行目录和日志。
- 结束后核验四折 complete、产物哈希、窗口数和汇总。


## 最终验收状态（2026-10-02）
- [x] Task 1：隔离与诊断行为测试完成。
- [x] Task 2：端到端实验入口完成。
- [x] Task 3：全量66项测试通过；变更与验收证据已记录。
- [x] Task 4：pack 6/8/9/10后台顺序训练完成，共26,508个留出窗口；训练和自动验收退出码均0，并再次独立 --verify 通过。
- 真实结果及解释边界见 D:/Matlab/project/pypack/docs/changes/2026-10-02-phase5-e2e-lopo.md。
