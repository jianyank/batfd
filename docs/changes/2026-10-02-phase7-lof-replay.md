# Phase7 LOF口径修正及固定指标组消融 — 执行记录

## 目标与协议
延续2026-10-02已批准的收益顺序：先标准训练LOF口径修正，再固定all/sigma_v/reconstruction对照，不重训深度模型。留出包不进入尺度、LOF或阈值拟合。k20/q.99/m5/median-MAD/.001下限不变，全部分组预先固定，不自动选赢家。

## 改动边界
仅新增D:/Matlab/project/pypack/batfd/detect/lof_v2.py、scripts/17_phase7_lof_replay.py及对应两个测试文件；不改batfd/detect/lof.py、Phase5/Phase6脚本、base配置或旧产物。不自动提交/合并/推送。

## 测试与根因记录（中间证据）
- LOF v2代理先红测试（13项中6次断言失败），最小fit覆盖后13项通过；主代理独立重跑13项通过。
- 新入口缺失时红测试退出1；实现后无报警情况下CSV验收失败，定位到DictWriter把None序列化为空字符串，而非字符串None。最小改正序列化对照并增加专用测试，9项通过，全量95项通过。
- 首轮后台D:/Matlab/project/pypack/outputs/diagnostics/phase7_lof_corrected_20261002T145430Z在第三折更新manifest时WinError5退出1，未验收；失败目录和日志原样保留，不覆盖、不作为正式结果。
- 文件属性Archive且ACL含Modify，未确定具体占用进程。通过CreateFileW持有不允许FILE_SHARE_DELETE的读句柄可确定性复现同样WinError5：旧目标replace失败、新目标成功、释放句柄后旧目标成功。
- 新入口改为终态manifest只创建一次，进度由stdout及独占折产物呈现；缺少complete终态的目录不能验收。拒绝任何既有目标替换的新增红测试先失败、修复后10项通过。不修改旧provenance，也不盲目sleep或重试。
- 第二轮非正式目录D:/Matlab/project/pypack/outputs/diagnostics/phase7_lof_corrected_20261002T150300Z已完成数值计算，但CLI调用不存在的console.say导致退出1，未执行自动验收。根因为未核对batfd.console实际API；按项目接口使用console.setup()和print()，增加真实main运行与--verify均返回0的回归测试。该目录绑定修正前源码指纹，原样保留，不作为当前正式交付或验收目标。

## 真实正式结果
正式全24维修正复算目录：
D:/Matlab/project/pypack/outputs/diagnostics/phase7_lof_corrected_20261002T151200Z

- 后台运行退出码：0；独立--verify退出码：0；stderr为空。
- 留出窗口：26,508；四包均有报警窗口，但本结果不等于故障检出证据。
- pack6：阈值1.5492453253137548；越限3425/9127（37.53%）；连续5窗确认105（1.15%）。
- pack8：阈值1.427162964959475；越限5987/11147（53.71%）；确认929（8.33%）。
- pack9：阈值1.5358598103057532；越限634/5093（12.45%）；确认10（0.20%）。
- pack10：阈值1.5704348348818569；越限289/1141（25.33%）；确认6（0.53%）。
- 宏平均越限率32.2532%，加权越限率38.9882%；宏平均确认率2.5517%，加权确认率3.9611%。
- 与Phase6标准训练口径对照逐折阈值、越限数、确认数均精确一致；旧Phase5和Phase6树未被写入。

固定三组消融目录：
D:/Matlab/project/pypack/outputs/diagnostics/phase7_lof_ablation_20261002T152000Z

- 后台运行退出码：0；独立--verify退出码：0；stderr为空；all结果与正式全24维目录逐折完全一致。
- all：宏平均越限率32.2532%，加权越限率38.9882%；宏平均确认率2.5517%，加权确认率3.9611%；4/4包有确认。
- sigma_v（0::3，8维）：宏平均越限率36.0560%，加权越限率40.5651%；宏平均确认率3.9264%，加权确认率5.0475%；4/4包有确认。
- reconstruction（1::3与2::3，16维）：宏平均越限率19.8145%，加权越限率26.2185%；宏平均确认率0.4126%，加权确认率0.6489%；2/4包有确认，pack9/10无连续5窗确认。
- 这些是无标签留出窗口上的报警形态对照，不是现场虚警率或故障检出率；不能仅因reconstruction报警较少就认定它是最优检测器。

## 解释边界
无独立故障标签，不把窗口越限率称现场虚警率。较少报警不能单独证明更好的检出，原行序不是时间轴，保存特征对照不代表新深度训练效果。旧condition_threshold配置未应用，本轮明确使用全局校准分位阈值。

## 最终验收与交付

- 最新完整回归命令：D:/Python/miniconda3/envs/batfd/python.exe -X utf8 -B -m unittest discover -s tests -v；97项通过，27.592秒，退出0。此前后台回归也退出0，日志保存在D:/Matlab/project/pypack/outputs/logs/phase7_lof_ablation_20261002T152000Z/regression.stdout.log、regression.stderr.log及regression.completion.json。
- Phase5正式产物、Phase6正式产物（132000Z）、Phase7正式全24维（151200Z）及三组消融（152000Z）各自--verify均退出0。两份Phase7正式终态均为complete，分别24和64个文件；后台run与verify的stderr均为0字节。
- Phase7入口及LOF v2语法编译、CLI --help退出0。独立只读代码审查无阻断项。
- 本轮--verify为独立进程执行的语义重放，复用生成函数重新拟合并核对产物，不是独立算法实现。
- 最后只读Git核对：分支codex/method-paper-rebuild，普通仓库；无已跟踪或暂存文件差异。本轮仅新增版本隔离实现、复算入口、测试、配套文档和独占输出/日志；既有未跟踪文件不清理，不提交、合并或推送。

### 正式实现与清单SHA256

| 文件（绝对路径） | SHA256 |
|---|---|
| D:/Matlab/project/pypack/batfd/detect/lof_v2.py | `72493f6ffe4d1ffc3fcaa792b913f32cd2e5322b7dc33062c2f7c1464d829f49` |
| D:/Matlab/project/pypack/scripts/17_phase7_lof_replay.py | `ba8d6986fab99e99fbd5b2239334b4d3d74576d554499f7d82ad544e47947ed2` |
| D:/Matlab/project/pypack/outputs/diagnostics/phase7_lof_corrected_20261002T151200Z/manifest.json | `a9ae498295953db39e92252795805ae0fe29eef2cf013593392673ce8a78be89` |
| D:/Matlab/project/pypack/outputs/diagnostics/phase7_lof_ablation_20261002T152000Z/manifest.json | `12c7e885232a589708e87d768024294ffb5fc932473b3bba2c14bcbefc135029` |

### 后续收益顺序

优先明确外部故障起点/健康标签与真实时间轴，先固定评估协议，再验证检出、健康段报警负担和提前量；本轮不擅自读取标签或测试数据。不要仅凭重建指标组报警较少删除sigma_v，也不通过抬阈值压报警，更不在无新增根因证据时盲目重训。
