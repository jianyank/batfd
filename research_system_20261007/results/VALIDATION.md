# 整理版验证记录

执行日期：2026-10-07。以下是整理后源码的新鲜验证，不代表真实故障性能。

## 核心回归

命令：

```text
python -X utf8 -B -m unittest discover -s tests -v
```

结果：**94项通过，0失败**，耗时33.129秒。测试只使用临时合成数组和临时目录，不读取或上传真实数据。

覆盖：特征与输入校验、15种候选方法、按包划分和q99阈值、连续5窗确认、状态重放、模型哈希防篡改、报告接口、真实输入审计辅助函数、外部缓存/冻结产物路径及CLI帮助。

## 无私有资产的独立发布副本

仅复制43份待发布文件，不包含`.git`、`data/`、`outputs/`、历史`runs/`或模型。相同回归命令再次执行：**94项通过，0失败**，耗时26.970秒，退出码0。

## 报告工具检查

- TeX→Markdown：执行成功，3表、3图、10个展示公式；Markdown规范化换行后与现有报告一致，三幅重绘PNG的SHA256全部一致。
- Markdown→Word：执行成功，结构审计为3表、3图、50个行内公式、10个展示公式；表格段落居中、无页首。未另行进行Word视觉渲染验收。
- 原`MENTOR_REPORT.tex`未修改，SHA256为`a232ed2f8f7ffac27517ed1ffe3a34fecb3027e72f68937138c2a5d7840eaba1`。

## 发布前静态检查

- `python -X utf8 -B -m compileall -q research_system_20261007`：执行成功，退出码0。生成的Python缓存被忽略，不入库。
- 43份待发布文件约0.42MB；路径及文本扫描未发现MAT/NPY/NPZ/Joblib/PT/PTH、原始日志、逐窗信号、Word/TeX中间文件、私钥/常见token或个人目录路径。
- `git diff --cached --check`：通过。GitHub Actions已配置回归，但远端CI状态以实际运行结果为准，不把本地测试当远端通过。
- 真实故障性能：未验证；`deployment_approved=false`。
