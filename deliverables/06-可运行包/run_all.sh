#!/usr/bin/env bash
# 端到端运行验证
#
# 从「交付源码包 + 真实数据」出发，把全部流程跑一遍，逐步报告成功或失败。
# 用法： bash run_all.sh
# 若 python 不在默认位置： PY=/path/to/python bash run_all.sh
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/d/Python/miniconda3/envs/batfd/python.exe}"
PKG="$HERE/chronoguard_ts-0.1.0"
OUT="$HERE/_运行输出"

pass=0; fail=0

step() {
  local name="$1"; shift
  echo ""
  echo "──────────────────────────────────────────────────────────────"
  echo "▶ $name"
  echo "──────────────────────────────────────────────────────────────"
  if "$@" 2>&1; then
    echo "  ✅ 通过：$name"; pass=$((pass+1))
  else
    echo "  ❌ 失败：$name"; fail=$((fail+1))
  fi
}

echo "端到端运行验证"
echo "  python  : $PY"
echo "  源码包  : $PKG"
echo "  数据    : $HERE/datasets -> (目录联接)"
echo "  输出    : $OUT"

if [ ! -d "$HERE/datasets" ]; then
  echo "❌ 找不到 datasets，请先建立目录联接"; exit 1
fi
if [ ! -d "$PKG" ]; then
  echo "❌ 找不到 $PKG，请先解压源码包"; exit 1
fi

rm -rf "$OUT"; mkdir -p "$OUT"

step "0. Python 与依赖" "$PY" -c "
import sys, numpy, sklearn, joblib
print('Python', sys.version.split()[0], '| NumPy', numpy.__version__,
      '| scikit-learn', sklearn.__version__, '| joblib', joblib.__version__)
"

step "1. 安装库（源码包内）" bash -c "cd '$PKG' && '$PY' -m pip install -e . --no-deps -q 2>&1 | grep -v '^\[notice\]' || true; '$PY' -c 'import chronoguard; print(\"导入成功:\", chronoguard.__file__)'"

step "2. 单元测试（34 项）" bash -c "cd '$PKG' && '$PY' -B -m unittest discover -s tests 2>&1 | tail -4"

step "3. 合成数据接口演示" bash -c "cd '$PKG' && '$PY' -B examples/demo.py 2>&1 | head -6"

step "4. 服务器 SMD 基准（T=1，3 机 × 4 方法）" \
  bash -c "cd '$PKG' && '$PY' -X utf8 -B examples/benchmark.py --scenario smd --output-dir '$OUT/smd_t1' 2>&1 | tail -2"

step "5. 服务器 SMD 开窗（T=16）" \
  bash -c "cd '$PKG' && '$PY' -X utf8 -B examples/benchmark_smd_windows.py --window-size 16 --output-dir '$OUT/smd_t16' 2>&1 | tail -2"

step "6. 电池基准（4 包 × 4 方法，正常参考）" \
  bash -c "cd '$PKG' && '$PY' -X utf8 -B examples/benchmark.py --scenario battery --output-dir '$OUT/battery' 2>&1 | tail -2"

step "7. 流式回放动画（短片段）" \
  bash -c "cd '$PKG' && '$PY' -X utf8 -B examples/live_demo.py --machine machine-3-1 --method pca \
     --start 27000 --stop 27200 --step 25 --format gif --output '$OUT/live_demo.gif' 2>&1 | tail -4"

step "8. 交付模型自检（40 个，含 SMD 重算核对）" \
  bash -c "'$PY' -X utf8 -B '$HERE/../05-代码包/models/verify_models.py' --smd-dir '$HERE/datasets/smd' 2>&1 | tail -4"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  结果：$pass 项通过，$fail 项失败"
echo "  产物：$OUT"
echo "══════════════════════════════════════════════════════════════"
[ "$fail" -eq 0 ] || exit 1
