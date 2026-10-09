# ChronoGuard 算法库交付验收

时间：2026-10-08 18:58:30 +08:00｜工作区：D:/Matlab/project/pypack

## 目标与范围

- [confirmed] 续接已确认的 ChronoGuard 多通道时序异常检测库交付；不做界面、不将既有算法包装为原创。
- [executed] 仅更新库修复后的实验汇总、相关说明和交付产物；不提交、不推送、不改分支，保护 8 份用户原有未提交文件。
- [observed] 分支 codex/repository-cleanup，HEAD 2975d61d2858ae142e04abe668d3bb537f493274。

## 已完成阶段及依据

- [verified] 当前源码库测试 34 项通过；重建 wheel 时又运行 34 项与原研究 86 项，全共 120 项、0 失败。入口：outputs/algolib/delivery_verification_20261008/{all_tests.ps1,verify_delivery.ps1 build}。
- [verified] PCA 固定逐行投影/重建归约的阈值边界回归通过，数学等价性及 45 组分块压力检查见同目录 pca_* 日志/脚本；限定只读复核无阻断问题，代理已关闭。
- [verified] load_smd 在已有清单时核对离线来源及三类输入文件哈希；新增临时文件回归通过，没有改变真实输入或清单。
- [executed] 使用修正后的 7 份源码在 outputs/algolib/checked_release_20261008 重跑 16 电池 + 12 SMD，共 28 组；旧 release_20261008 与初始 MAD 结果保留。入口：rerun_benchmark.ps1，q99/连续5窗/seed42/最多6000拟合窗。
- [verified] verify_experiments.ps1 校验 49 产物、12 输入、7 源快照并独立 NumPy 复算 12 组 SMD 严格逐点/事件指标；公开 JSON/CSV 与新运行完全一致，8 文件初始 SHA256 未变。
- [observed] 新旧检测、误报、事件及验证比例指标一致；4 组 PCA 阈值有约 1e-16 差异，耗时不同，见 benchmark_comparison.json。这不构成性能提升证据。
- [verified] 新 wheel 重新构建，中文元数据检查及仓库外 target 安装通过；python -I 确认导入实际安装目录，四方法有限分数、分块报警、模型保存加载均通过。
- [verified] 7 份中文文档 UTF-8、控制字符、代码块与 15 处本地链接检查通过；README 只插入算法库入口，原研究全文未变。

## 失败、原因及限制

- [failed → verified] 此轮最初用 startsWith HEAD 检查 README 失败。根因为校验误以为新段落位于末尾，实际 Git diff 只在原任务节前新增18行，无删除；改为仅去除命名新增节后完整比较原文。没有改动 README 来迎合错误校验，详情见 readme_preservation_check.json。
- [observed] 前序 PowerShell ASCII 管道损坏中文、PCA BLAS 批大小舍入与 SMD 清单绕过的失败证据保留，未用阈值容差掩盖根因。
- [unresolved] 电池只有正常留出误报，预构造窗口缺时间戳/步长，未独立证明原始点不跨角色重叠；没有真实故障召回、提前量或定位验证。
- [observed] SMD 仅3台/T=1且同一测试数据已用于开发观察；结果是开发复验，不是独立盲验。未核对本次赛事最新规则或论证原创检测算法。
- [observed] 安装验证使用既有 batfd 依赖环境，不是新建干净依赖环境，也不保证跨版本/硬件一致或安全生产上线。

## 本阶段收尾

- [verified] 最终白名单源码包已完成解压/构建/仓库外安装，34项测试、demo和benchmark --help通过；源码包25文件（24内容文件+清单）均核验。
- [verified] 最终VERIFICATION、dist/DELIVERY.json与CHECKSUMS.sha256完成，final_audit.ps1确认源码zip内容与当前工作区一致、wheel中5个Python文件与zip/工作区一致，Git diff --check退出0。
- [unresolved] 后续若要申报算法创新，仍需核对赛事规则、明确真正方法贡献、冻结新独立测试；本阶段没有承诺或伪装这些科研验证已完成。

## 产物与来源

- 使用说明：docs/ALGORITHM_LIBRARY.md；参赛说明：docs/COMPETITION_BRIEF.md；报告/公开汇总：benchmarks/20261008/。
- 本地证据：outputs/algolib/delivery_verification_20261008/（build_result.json、wheel_result.json、experiments_result.json、documents_verified.json 等）。
- 本轮没有进行联网赛事/文献检索，不继承其他项目赛事结论。

## 最终交付（2026-10-08 19:05:27 +08:00）

- [verified] wheel：dist/chronoguard_ts-0.1.0-py3-none-any.whl，15131字节，SHA256 592c1f553c5d84ed5f8646c16cfd87b00a6d8318a70d920d7a79f7599f640939。
- [verified] source：dist/chronoguard_ts-0.1.0-source.zip，52640字节，SHA256 7624a15c5d8062c6aa7b57eeee5113a7ee55a8622f7c3d87a481d2364c5d743d。
- [verified] 最终源码包包含已更新的VERIFICATION与实施计划，并重新执行独立安装验收，详见 source_final_result.json；首次修正候选包日志另存 source_checked_candidate_*。
- [verified] verify_smd_regression.ps1通过存档旧源码独立模块再次复现12个子场景失败，新源码0失败；只读临时数据，没有回退工作区。pca_stress.ps1本轮重新通过45组逐位分块检查。
- [verified] 共120项仓库测试、28组实验、49产物/12输入溯源及12组SMD独立指标复算通过；8份用户初始哈希保留，不提交/推送或切分支。
- [observed] 最终审计脚本写入回读时发现三处JS到Python字符串换行转义，在运行前修正并回读；不影响打包源码，最终审计退出0。
