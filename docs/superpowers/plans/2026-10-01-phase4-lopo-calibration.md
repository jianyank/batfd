# Phase 4：冻结表示下训练包留一阈值诊断

日期：2026-10-01；分支：codex/method-paper-rebuild。

## 目标与范围
对第三轮 skip_on 已冻结的 StandTrainData 24 维特征，逐包留一拟合 	rain_novelty LOF，并以其余训练包的训练分数 q=0.99 校准阈值；在留出包上记录原始越限率、连续 5 窗口持久报警、首次确认报警位置。重点审视 pack 8。只新增诊断脚本与结果，不重训、不调参、不读取测试数据或任何故障起点标签、不改旧产物。

## 隐含前提与解释限制
- 特征源固定为 outputs/diagnostics/phase3_decoder_skip_20261001T144548Z_fd81c8b0c1/skip_on/features/StandTrainData.npy；运行前须对照第三轮完成态 manifest 校验文件 SHA-256。
- 包身份只由 outputs/cache/StandTrainData/ids.npy 提供，并与同目录 meta.json 中行数/计数核对；不访问 hidden、Hiddall、labels.csv 或 StandTestData*。
- LOF 参数继承第三轮：
_neighbors=20、稳健标准化开启、
orm_floor=0.001、q=0.99；持久性窗口 m=5。本实验只诊断固定协议，不搜索最佳阈值。
- 仅 LOF 与阈值拟合按包留一。深度表示此前见过全部四个训练包，所以结果不能称端到端跨包泛化；训练包留出越限率也不是现场虚警率。
- pack 顺序使用缓存中的原始行顺序（训练集没有时间列）；持久报警位置因此是包内窗口索引，不换算成天数。

## 文件职责
- 新增 scripts/14_phase4_lopo_calibration.py：验证固定输入来源，拟合四折 LOF，输出逐包汇总、逐窗口分数及输入/代码/产物指纹。
- 新增 	ests/test_phase4_lopo_calibration.py：覆盖严格排除留出包、参数/阈值和持久报警统计、错误输入拒绝、诊断输出不覆盖。
- 新增本计划及最终结果摘要到 D:\AI\Obsidian\pypack 的既有方法/实验笔记；不改旧实验文件或原始数据。
- 新结果单独落入 outputs/diagnostics/phase4_lopo_<UTC时间>_<身份摘要>/，不得复用或覆盖历史目录。

## 实施与验收
1. 先写行为测试并运行，确认因脚本/接口尚不存在而失败；再实现最小逻辑使测试通过。
2. 输入校验：特征维数为 N×24、全部有限；IDs 长度匹配；四个目标包均存在；训练缓存 meta 计数一致；冻结特征哈希与第三轮 manifest 一致。
3. 每折仅以其余三个 ID 的特征拟合 LOF；训练阈值只来自该折训练分数 q=0.99；在留出包分数上按 score > threshold 计越限，再应用 m=5 连续越限规则。
4. 报告按包窗口数、校准窗口数、训练阈值、越限窗口数/率、持久确认数/率、是否曾确认报警、首次确认报警窗口；逐窗口 CSV 保存原始分数和判断。
5. 使用项目指定 Python 运行新增测试、全量 unittest、脚本 --help 与真实四折诊断；逐项校验输入哈希、manifest 产物哈希、四个包及行数和不覆盖旧结果。
6. 结果只作训练分布偏移/阈值稳定性诊断。Obsidian 记录包含解释限制、pack 8 结果和可复现运行目录。

## 本轮禁止事项
- 不读取 StandTestData1/2/3、labels.csv、Hiddall 或测试故障起点。
- 不重训深度模型、不调整模型/LOF/阈值/持久性参数，不以 pack 8 结果回调阈值。
- 不改写 MAT、缓存、第三轮产物、历史 tables 或其他已有未跟踪文件。
