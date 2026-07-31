#!/usr/bin/env bash
# 建 conda 环境（python 3.10）并用清华镜像装依赖。
set -euo pipefail
cd "$(dirname "$0")/.."

PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
ENV_NAME="${STOCK_PREDICT_ENV:-stock-predict}"

if ! command -v conda >/dev/null 2>&1; then
  echo "❌ 未找到 conda，请先安装 miniconda"; exit 1
fi

# 建议一次性配好清华源（conda）
if ! grep -q "mirrors.tuna.tsinghua.edu.cn" ~/.condarc 2>/dev/null; then
  echo ">>> 配置 ~/.condrac 清华镜像..."
  cat >> ~/.condarc <<'YAML'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
YAML
fi

echo ">>> 创建 conda 环境 $ENV_NAME (python=3.10)..."
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" python=3.10 pip
else
  echo "    环境 $ENV_NAME 已存在，跳过创建。"
fi

echo ">>> 安装依赖（清华 PyPI 源）..."
conda run -n "$ENV_NAME" python -m pip install -i "$PIP_INDEX" --upgrade pip
conda run -n "$ENV_NAME" python -m pip install -i "$PIP_INDEX" -r requirements.txt
conda run -n "$ENV_NAME" python -m pip install -i "$PIP_INDEX" -e .

echo ""
echo "✅ 完成。请激活环境后使用："
echo "   conda activate $ENV_NAME"
echo "   stock-predict --help"
