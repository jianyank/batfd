# 四折 LOPO 高越限率诊断 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans。主代理TDD实现与数值计算，只读代理独立审查。

**Goal:** 从现有四折产物识别高越限率的可检验原因，形成可复算的证据和下一项最小实验。

**Architecture:** 新增单一只读诊断入口，复用原验收、LOF及工况函数。原始信号分块处理，不加载hidden；所有拟合只用每折校准包，新目录封存诊断结果。

**Tech Stack:** Python、NumPy、scikit-learn、SciPy（仅描述性秩相关）、unittest。

## Global Constraints
不改旧源码、配置或封存产物；不读取hidden/SOC/标签/测试集；不调参、不重训；校准拟合与留出评估分离；输出禁止覆盖或嵌入输入树。

## Task 1：测试与最小诊断入口
Files: Create scripts/16_phase6_lopo_shift_diagnostic.py、tests/test_phase6_lopo_shift_diagnostic.py。
- [x] 先编写真实行为测试：校准尺度和范围、分组几何、报警段、评分重放、旧树保护和真实小四折诊断。
- [x] python -X utf8 -B -m unittest discover -s tests -p test_phase6_lopo_shift_diagnostic.py -v，确认实现缺失的红测试。
- [x] 最小实现 distribution_rows(cal, held, names, center=None, scale=None)、alarm_shape(scores, threshold, persistence)、nearest_geometry(detector, cal, held, calibration_ids)、audit_scores(cal, held, cfg)、run_diagnostic(run_dir, output_dir)、main。
- [x] 同一测试命令转绿；没有与需求无关的代码变更。

## Task 2：真实四折诊断
- [x] 在全新phase6目录运行入口 --run-dir 已验收phase5目录 --output-dir 新目录。
- [x] 每折复算原分数、阈值和scaler；测量24维漂移、邻居几何、可观测工况、十等分报警形态和评分口径敏感性。
- [x] 生成CSV、摘要、报告和哈希；结束再次验证旧封存树未变。

## Task 3：审查与交付
- [x] 接收只读审查代理结论并据真实数据判断，不把假设当结论。
- [x] 全量unittest、入口--help、旧四折--verify及诊断SHA-256校验通过。
- [x] 更新变更记录及计划；输出证据、限制和下一步收益排序，不自作主张重训或采用新阈值。

## 2026-10-02 执行证据
- 新入口与测试已完成；实现缺失红测试退出1，补实现后6项PASS。
- 首次集成测试暴露Windows文件映射引用未释放；显式关闭signal._mmap后测试转绿，不用忽略清理错误掩盖根因。
- 真实诊断目录：D:/Matlab/project/pypack/outputs/diagnostics/phase6_lopo_shift_20261002T130500Z。
- 四折原校准/留出分数最大重放误差均0，scaler及原阈值精确一致。
- 全量回归72项PASS（15.821秒）；新旧独立--verify退出0，入口--help退出0。
- 仅做训练评分口径敏感性测量，未修改LOF既有源码或原封存结果；下一步优先纠正确定评分语义，再做预注册指标组消融，不立即重训。

## 审查后最小补强与最终验收
- 两路只读审查发现首版结果本身无阻断项，但总平方距离占比显著受尾部权重影响；补充逐窗口三组占比中位数及最高约1%窗口的总能量份额。
- 补充根配置与折内q/persistence一致性、必需6输出集合/schema、摘要四折/窗口/来源/只读协议、独立报警shape/bin复算及CSV折集合/行数检查。
- 新红测试先确认缺字段和重签名摘要篡改未被拒绝；补实现后7项PASS，完整回归73项PASS（16.455秒）。
- 最终增强版目录：D:/Matlab/project/pypack/outputs/diagnostics/phase6_lopo_shift_20261002T132000Z。仅此目录为当前入口的最终验收对象；首版目录保留、不覆盖，其绑定的是审查前实现指纹。
- 最终新诊断--verify及原Phase5--verify均退出0，原校准/留出分数最大重放误差仍为0。
- pack8总余弦距离占比98.60%，但逐窗中位数为37.38%，最高约1%窗口占能量93.20%；禁止将总和占比解释为多数报警的贡献。

- 最终只读复审确认：评分协议交叉核对及验收语义补强已覆盖审查意见，当前范围无阻断项；未修改原实验文件。代码保留当前codex/method-paper-rebuild分支，未自动提交、合并或推送。
