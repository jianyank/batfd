# 第三方材料与数据来源说明

## SMD / OmniAnomaly

本次服务器数据从 NetManAIOps/OmniAnomaly 仓库的 ServerMachineDataset 目录取得，实际下载的 URL、大小和 SHA256 保存在本地 data/external/SMD/download_manifest.json。
上游获取入口如下（master 为可变引用，本次字节身份由清单核对）：

```text
https://github.com/NetManAIOps/OmniAnomaly
https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/
```

本次只验证 machine-1-1、machine-2-1、machine-3-1。保存的上游 README 说明 SMD 有 28 台机器，应逐机独立训练/测试；其原运行说明包含在测试集找 best F1 的做法，本项目没有采用这一测试择优步骤。
库实现未复制 OmniAnomaly 模型代码。源码 zip 与 wheel 不包含下载数据，也不包含电池私有缓存。
下列是已下载仓库 LICENSE 的原文；本项目不据此额外声明所有外部数据的再分发授权，也不替第三方材料变更许可。

```text
MIT License

Copyright (c) 2021 NetManAIOps-OmniAnomaly

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Python 依赖与原项目材料

依赖 NumPy、scikit-learn、joblib；实验间接使用 SciPy、threadpoolctl。它们由包管理器单独安装，不嵌入本 wheel；各自许可与版权信息应随其原发行材料保留。

源码包附带 research_system_20261007/features.py，为当前工作仓库中原电池特征文件的未改动副本，只用于检验 peer 特征与原实现等价。ChronoGuard 核心运行不导入该文件。

本次没有擅自为用户项目指定新的开源许可证，也没有替用户授权私有数据。是否公开完整项目、数据或第三方材料，应由材料权利人和正式提交要求确认。
