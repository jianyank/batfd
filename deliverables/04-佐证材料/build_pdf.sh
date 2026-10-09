#!/usr/bin/env bash
# 构建佐证材料 PDF（HTML -> Edge 无头打印）
# 用法： bash build_pdf.sh
set -euo pipefail

EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
DIR="/d/Matlab/project/pypack/deliverables/04-佐证材料"

cd "$DIR"

# Git Bash 的 /d/... 路径必须转成 Windows 路径，否则 Edge 报 ERR_FILE_NOT_FOUND
WIN="$(pwd -W)"

echo "[1/2] HTML -> PDF"
"$EDGE" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$WIN\\supporting-evidence.pdf" \
  "file:///$WIN/supporting-evidence.html"

echo "[2/2] HTML -> PNG 自查截图"
"$EDGE" --headless=new --disable-gpu --no-pdf-header-footer \
  --screenshot="$WIN\\preview.png" --window-size=1200,1600 \
  "file:///$WIN/supporting-evidence.html"

echo "完成： $DIR/supporting-evidence.pdf"
ls -la "$DIR/supporting-evidence.pdf"
