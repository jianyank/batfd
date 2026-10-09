# 代码包（网盘材料，待上传）

**当前状态：代码包已构建完成，未上传网盘。**
规则要求完整代码放入百度网盘，提交**永久有效、自动生成提取码**的分享链接。

## 代码包位置

**参赛用的代码包就在本目录**，可直接上传网盘：

| 文件 | 大小 | 说明 |
|---|---:|---|
| `chronoguard_ts-0.1.0-py3-none-any.whl` | 10,896 B | 可安装的库 |
| `chronoguard_ts-0.1.0-source.zip` | 48,966 B | 源码包（19 个条目） |
| `CHECKSUMS.sha256` | 204 B | 两个包的 SHA256 |
| `DELIVERY.json` | 728 B | 构建记录（来源目录、大小、哈希） |

已核对：两个包的 SHA256 与 `CHECKSUMS.sha256`、`DELIVERY.json` **三方一致**。

> 仓库根的 `dist/` 是同一个构建脚本的输出目录，本目录是**参赛投放副本**。
> 重新构建（仓库根执行）后，需再把新包复制过来：
>
> ```bash
> python tools/build_delivery.py
> cp dist/chronoguard_ts-0.1.0-*.{whl,zip} dist/CHECKSUMS.sha256 dist/DELIVERY.json \
>    deliverables/05-代码包/
> ```

## 源码包内容

```text
chronoguard_ts-0.1.0/
  README.md  pyproject.toml  requirements.txt
  src/chronoguard/          detector / features / data / evaluation
  examples/                 demo.py / benchmark.py /
                            benchmark_smd_windows.py / live_demo.py
  tests/                    34 项回归测试
  docs/                     算法库说明、数据集规格、第三方材料
  benchmarks/20261008/      实验报告
  DELIVERY_MANIFEST.json    包内每个文件的大小与 SHA256
```

## 代码包**不含**（有意排除）

私有数据、外部数据集、模型文件、逐点预测、Git 历史、缓存、日志。
`.gitignore` 与构建脚本共同保证这一点。

## 上传前检查

- [ ] 确认代码包**不含数据与模型**（`.npy` / `.npz` / `.joblib` / `.mat` 均为空）
- [ ] 网盘分享设置为**永久有效**
- [ ] 开启**自动生成提取码**
- [ ] 记录分享链接与提取码，填进提交表
- [ ] 链接在无登录状态下实测可访问

## 附加说明（建议随代码包一起放）

若希望评委能直接看到效果，可在网盘内附一段 `运行说明.md`，内容为：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -X utf8 -B -m unittest discover -s tests -v   # 34 项测试
python -X utf8 -B examples/demo.py                    # 合成数据接口演示
```

`examples/live_demo.py` 需要 SMD 数据与 ffmpeg，数据不随包分发，可注明获取方式。
