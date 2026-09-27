"""绘图：字体/标签（``style``）与六张结果图（``plots``）。

标签集中在 ``style.LABELS``，绘图函数只认 ``L(lang, key)``，中英两版不会漂移。
入口脚本 ``scripts/10_figures.py``。
"""

from . import plots, style  # noqa: F401

__all__ = ["style", "plots"]
