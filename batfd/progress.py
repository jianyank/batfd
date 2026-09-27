"""进度条统一入口。

1. **输出被重定向到日志文件时必须自动关闭动画**：tqdm 靠回车符原地刷新，写进日志
   会变成几千行垃圾。传 ``disable=None`` 让 tqdm 自己判断 TTY：前台显示动画，
   重定向到文件只留一行。
2. **进度条和 print 不能混用**：bar 在刷新时再 print 会把画面撕开，周期性日志
   一律走 :func:`log`（内部用 ``tqdm.write``，在条上方另起一行）。

用法: ``progress.set_enabled(False)`` 强制关闭；循环体用 ``progress.bar(...)`` 包住。
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, TypeVar

from tqdm import tqdm

T = TypeVar("T")

# None 表示「自动判断」：是 TTY 就显示，被重定向就关闭
_enabled: bool | None = None


def set_enabled(value: bool | None) -> None:
    """True 强制开、False 强制关、None 交给 tqdm 自动判断。"""
    global _enabled
    _enabled = value


def bar(
    iterable: Iterable[T] | None = None, *, total: int | None = None, **kwargs: Any
) -> tqdm:
    """包一层 tqdm。``disable`` 默认自动判断，可被 :func:`set_enabled` 覆盖。"""
    kwargs.setdefault("disable", None if _enabled is None else (not _enabled))
    kwargs.setdefault("dynamic_ncols", True)   # 终端宽度变化时自适应，避免折行
    kwargs.setdefault("leave", True)
    return tqdm(iterable, total=total, **kwargs)


def log(msg: str = "") -> None:
    """在进度条上方输出一行，不破坏动画。"""
    tqdm.write(msg)
