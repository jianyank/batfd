# 第三方材料与数据来源说明

## SMD / OmniAnomaly

服务器监控数据取自 `NetManAIOps/OmniAnomaly` 仓库的 `ServerMachineDataset` 目录。上游入口如下（`master` 为可变引用，下载时的字节身份由本地 `download_manifest.json` 中的 URL、大小与 SHA256 核对）：

```text
https://github.com/NetManAIOps/OmniAnomaly
https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/
```

上游 README 说明 SMD 有 28 台机器，应逐机独立训练/测试；其原运行说明包含在测试集上找 best F1 的做法，本项目未采用这一测试择优步骤。

库实现未复制 OmniAnomaly 模型代码，也不包含下载数据。下列为上游仓库 LICENSE 原文；本项目不据此额外声明外部数据的再分发授权，也不替第三方材料变更许可。

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

## Python 依赖

依赖 NumPy、scikit-learn、joblib；实验间接使用 SciPy、threadpoolctl。它们由包管理器单独安装，不嵌入本库的发行包；各自许可与版权信息应随其原发行材料保留。
