# 运行、复验与预测

以下命令从仓库根目录进入 `research_system_20261007` 后执行。所有新实验/预测必须写入本研究目录下**尚不存在**的目录，工具拒绝覆盖已有结果。

## 1. 环境与无数据回归

```bash
python -m pip install -r requirements.txt
cd research_system_20261007
python -X utf8 -B -m unittest discover -s tests -v
```

安装命令在仓库根目录运行，后续命令在研究目录运行。使用 Python 3.12；依赖锁定到本地实际测试版本，不需要 PyTorch。回归仅使用临时合成数据，不访问真实缓存。

## 2. 真实缓存输入契约

这是特定电池数据研究源码，不是任意数据集的通用训练入口。正常缓存目录应包含：

| 文件 | 契约 |
| :---: | :---: |
| `signal.npy` | finite数值数组，形状为 `(26508,256,20)` |
| `ids.npy` | 与窗口逐行对齐的一维包ID |
| `meta.json` | 含 `has_time: false` 的已有缓存元信息 |

固定包规模：6号9127窗、8号11147窗、9号5093窗、10号1141窗。数组通道顺序和标准化域必须与既有缓存一致。代码取第1、3、5、7、9、11、13、15列为八路单体信号；不能将未确认单位的原始mV、其他通道排列或别的数据集直接代入。此版本不包含旧MAT转换链，请向数据提供方取得符合契约的缓存。

默认读取仓库内 `outputs/cache/StandTrainData/`；也可显式指定外部缓存。原始缓存只读，不会自动生成故障标签。

## 3. 新实验与模型冻结

```bash
python run_benchmark.py --cache-dir /path/to/StandTrainData --output-dir runs/screen_new
python run_benchmark.py --verify runs/screen_new
python deploy.py build --cache-dir /path/to/StandTrainData --run-dir runs/screen_new --output-dir runs/candidate_new
```

Windows将 `/path/to/StandTrainData` 换成缓存绝对路径（带空格时加引号）。实验固定四折按包留出、折内60%拟合/20%校准/20%开发、正常q99、连续5窗；选型只使用合成开发域。全局重拟合不是新的独立验证。

可选历史冻结特征重放：

```bash
python run_benchmark.py --cache-dir /path/to/StandTrainData --include-frozen --frozen-dir /path/to/phase5_e2e_lopo_20261002T105204Z --output-dir runs/screen_with_frozen
```

此功能需要完整历史manifest及四折特征文件，仓库不提供；只是既有特征的评分器重放，不是从头深度训练。缺这些资产时不使用 `--include-frozen`。

## 4. 预测和分段状态

```bash
python deploy.py predict --bundle-dir runs/candidate_new --input /path/to/pack_windows.npy --pack-id 10 --output-dir runs/prediction_new
python smoke_demo.py --cache-dir /path/to/StandTrainData --bundle-dir runs/candidate_new --output-dir runs/demo_new
```

预测数组必须为 `N×256×20` 且输入域与训练相同，一次CLI仅处理一个包的连续片段；不得把多包拼成一个序列。CLI每次重置状态。连续分块使用 `CandidateSystem.predict` 返回的 `state` 传入下一块，换包或缺测断点重新初始化。

`prediction.npz`保存分数、单体证据与排序、单窗超阈值、确认报警、报警开始/解除及就绪状态；`summary.json`记录输入/模型哈希及限制。单体排名是标准化残差证据，不是LOF因果贡献或已确认故障单体。烟雾演示是输入副本的固定注入与分段重放，不是真实故障性能验证。

仅加载自己生成或已核实来源与哈希的joblib模型；序列化模型不能按不可信数据对待。

## 5. 历史结果、复验与审计

- 仓库内 `results/` 只发布汇总表，不能代替模型、逐窗输出、源快照和输入文件完成完整复验。
- 旧实验封存记录包含绝对路径与输入哈希。迁移后不会自动重定位；恢复原始输入布局才能执行默认完整复验，不能修改manifest或关闭哈希检查让其通过。
- 旧v2模型绑定整理前的源码；路径参数修改也会使活动源码哈希变化。恢复完整旧环境或用当前源码重跑并生成新bundle，不绕过拒绝逻辑。
- `diagnose_sequences.py`需要完整封存run。`audit_real_inputs.py`是保留的历史只读审计入口，完整扫描还需四份MAT、缓存、时间侧车与相关旧审计资产；缺这些资产时不承诺成功。其独立辅助函数仍由合成单元测试覆盖。

## 6. 报告转换（可选）

```bash
python -m pip install -r requirements-doc.txt
python research_system_20261007/tools/report_to_markdown.py
python research_system_20261007/tools/report_to_word.py
```

本节从仓库根目录执行。第一步转换导师TeX并重绘三图，需Windows微软雅黑/宋体中文字体；第二步从Markdown生成含可编辑公式的Word。工具是本报告专用转换器，不是通用LaTeX解析器。生成的Word、诊断JSON和TeX编译中间文件不入库。原 `MENTOR_REPORT.tex` 保留原路径与内容，无需为了运行检测器编译它。
