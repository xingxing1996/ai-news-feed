#!/usr/bin/env bash
# 跑量化命令（run=全流程训练 / refresh=刷新日报）。容器内被 cron 调用，也可手动跑。
set -e
cd /app
export PYTHONPATH=/app:/app/quant
export STOCK_PREDICT_CONFIG=/app/quant/config/settings.cn.yaml
mkdir -p /app/quant/data/output
echo "===== $(date '+%F %T %Z') $1 开始 ====="
python -m stock_predict.cli "$1"
echo "===== $(date '+%F %T %Z') $1 完成 ====="
