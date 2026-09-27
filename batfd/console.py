"""控制台输出统一设置。

Windows 下 stdout 默认是 **GBK**：print 中文会乱码，遇到非 GBK 字节（例如
.mat 文件头）更会直接抛 ``UnicodeEncodeError`` 打断脚本。文档与日志用中文，
所以每个入口脚本第一件事调用 :func:`setup`，把 stdout/stderr 强制成 UTF-8。

用法: ``from batfd import console; console.setup()``
"""

from __future__ import annotations

import sys


def setup(*, force: bool = True) -> None:
    """把 stdout/stderr 切到 UTF-8；无法重设时静默跳过。

    ``errors='replace'`` 是刻意的：宁可个别字符显示成替代符号，也不让一条打印打断整轮实验。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 输出编码不该成为硬失败
            pass

    if force:
        # 让被重定向到文件时也走同一套设置
        try:  # pragma: no cover
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:  # noqa: BLE001
            pass
