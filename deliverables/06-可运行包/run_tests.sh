#!/usr/bin/env bash
# 直接运行测试代码，无需先安装库
#
# 自动把源码包的 src/ 加到 PYTHONPATH，因此解压后即可运行。
# 用法： bash run_tests.sh
# 若 python 不在默认位置： PY=/path/to/python bash run_tests.sh
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"

# 找 python：优先用 PY 环境变量，其次常见的 conda 环境，最后退回 PATH 上的 python
if [ -n "${PY:-}" ]; then
  :
elif [ -x /d/Python/miniconda3/envs/batfd/python.exe ]; then
  PY=/d/Python/miniconda3/envs/batfd/python.exe
else
  PY="$(command -v python3 || command -v python || true)"
fi

if [ -z "${PY:-}" ] || ! "$PY" -c "" 2>/dev/null; then
  echo "❌ 找不到可用的 Python。请显式指定："
  echo "     PY=/path/to/python bash run_tests.sh"
  exit 1
fi

# 找源码包目录（解压出来的那个）
PKG=""
for d in "$HERE"/chronoguard_ts-*; do
  [ -d "$d/src/chronoguard" ] && PKG="$d" && break
done

if [ -z "$PKG" ]; then
  echo "❌ 找不到源码包目录。请先解压："
  echo "     python -c \"import zipfile; zipfile.ZipFile('../05-代码包/chronoguard_ts-0.1.0-source.zip').extractall('.')\""
  exit 1
fi

MODELS="$HERE/../05-代码包/models/verify_models.py"

echo "测试代码直接运行（无需安装）"
echo "  python  : $PY"
echo "  源码包  : $PKG"
echo "  PYTHONPATH = $PKG/src"
echo ""

pass=0
fail=0

echo "──────────────────────────────────────────────────────────────"
echo "▶ 1. 单元测试（34 项）"
echo "──────────────────────────────────────────────────────────────"
if (cd "$PKG" && PYTHONPATH="$PKG/src" "$PY" -B -m unittest discover -s tests 2>&1 | tail -6); then
  echo "  ✅ 通过"; pass=$((pass+1))
else
  echo "  ❌ 失败"; fail=$((fail+1))
fi

if [ -f "$MODELS" ]; then
  echo ""
  echo "──────────────────────────────────────────────────────────────"
  echo "▶ 2. 交付模型自检（40 个）"
  echo "──────────────────────────────────────────────────────────────"
  if PYTHONPATH="$PKG/src" "$PY" -X utf8 -B "$MODELS" 2>&1 | tail -6; then
    echo "  ✅ 通过"; pass=$((pass+1))
  else
    echo "  ❌ 失败"; fail=$((fail+1))
  fi
else
  echo ""
  echo "（跳过模型自检：未找到 $MODELS）"
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  结果：$pass 项通过，$fail 项失败"
echo "  说明：本次运行仅设置 PYTHONPATH，未安装任何东西。"
echo "══════════════════════════════════════════════════════════════"
[ "$fail" -eq 0 ] || exit 1
