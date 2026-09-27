#!/usr/bin/env bash
# 创建 pypack 专用 conda 环境 batfd，并安装全部依赖。
# 用法（在 git bash 中，从 pypack/ 目录执行）：
#     bash scripts/setup_env.sh 2>&1 | tee outputs/logs/setup.log
#
# 说明：
#   - 独立环境，不复用 conda `data`，避免污染用户既有环境。
#   - PyTorch 选 cu126 轮子：本机驱动 610.88 / CUDA UMD 13.3 向下兼容 cu12x。
#   - 若 cu126 不可用，脚本会自动回退 cu124，再回退 CPU 版。
#   - 回滚：conda env remove -n batfd -y

set -uo pipefail

ENV_NAME=batfd
PY_VER=3.12

echo "=== [1/4] 创建 conda 环境 ${ENV_NAME} (python ${PY_VER}) ==="
conda create -n "${ENV_NAME}" python="${PY_VER}" -y
if [ $? -ne 0 ]; then echo "!! conda create 失败"; exit 1; fi

echo "=== [2/4] 安装 PyTorch (尝试 cu126 -> cu124 -> cpu) ==="
TORCH_OK=0
for CU in cu126 cu124; do
    echo "--- 尝试 ${CU} ---"
    conda run -n "${ENV_NAME}" python -m pip install \
        torch torchvision --index-url "https://download.pytorch.org/whl/${CU}"
    if [ $? -eq 0 ]; then
        # 装上了不代表能跑，实际验证一次
        if conda run -n "${ENV_NAME}" python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"; then
            TORCH_OK=1
            echo "--- ${CU} 安装成功 ---"
            break
        fi
    fi
    echo "--- ${CU} 不可用，回退 ---"
done

if [ "${TORCH_OK}" -eq 0 ]; then
    echo "--- 回退 CPU 版 torch（管线仍可跑通，只是训练慢）---"
    conda run -n "${ENV_NAME}" python -m pip install torch torchvision
fi

echo "=== [3/4] 安装其余依赖 ==="
conda run -n "${ENV_NAME}" python -m pip install \
    numpy scipy pandas scikit-learn matplotlib pyarrow pyyaml tqdm psutil

echo "=== [4/4] 环境自检 ==="
conda run -n "${ENV_NAME}" python - <<'PYEOF'
import sys, importlib
print("python", sys.version.split()[0])
for m in ["numpy","scipy","pandas","sklearn","matplotlib","pyarrow","yaml","tqdm","psutil","torch"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  FAIL {m:14s} {type(e).__name__}: {e}")
try:
    import torch
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu =", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch cuda check failed:", e)
try:
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(novelty=True, n_neighbors=5)
    import numpy as np
    X = np.random.RandomState(0).randn(50,4); lof.fit(X); print("LOF novelty 打分可用 =", lof.score_samples(X[:3]).round(3))
except Exception as e:
    print("LOF check failed:", e)
PYEOF

echo "=== 完成。激活方式：conda activate ${ENV_NAME} ==="
