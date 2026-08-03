#!/usr/bin/env bash
# 容器入口：cron 调度（每天训练 + 每2h刷新）+ FastAPI 查询服务 常驻。
# 用法：
#   默认            → cron 调度 + API 常驻
#   docker run ... once   → 只跑一次训练（不启动 API/cron）
#   docker run ... run|refresh|train   → 跑指定命令后退出
set -e
cd /app
export PYTHONPATH=/app:/app/quant
export STOCK_PREDICT_CONFIG=/app/quant/config/settings.cn.yaml
mkdir -p /app/quant/data/output

API_PORT="${API_PORT:-8000}"
CRON_TRAIN="${CRON_TRAIN:-0 9 * * 1-5}"      # 默认工作日 北京17:00(UTC 09:00) 训练
CRON_REFRESH="${CRON_REFRESH:-0 */2 * * *}"   # 默认每 2h 刷新日报

# 单次模式：跑完退出（不启动 API/cron）
case "${1:-serve}" in
  run|refresh|train|ingest|features|backtest|news|report|universe|compare)
    exec python -m stock_predict.cli "$@"
    ;;
esac

# ---- serve 模式：cron 调度 + API ----
printf 'CRON_TZ=%s\n%s root /app/docker/run_cn.sh run >> /app/quant/data/output/cron.log 2>&1\n%s root /app/docker/run_cn.sh refresh >> /app/quant/data/output/cron.log 2>&1\n' \
  "$TZ" "$CRON_TRAIN" "$CRON_REFRESH" > /etc/cron.d/quant
chmod 0644 /etc/cron.d/quant && crontab /etc/cron.d/quant
echo "[$(date)] 调度: train='$CRON_TRAIN' refresh='$CRON_REFRESH' (TZ=$TZ)"

cron   # 启动 cron 守护（后台）
# 后台先跑一次训练，让 API 一启动就有结果
nohup /app/docker/run_cn.sh run > /app/quant/data/output/startup.log 2>&1 &

# 前台常驻 API
echo "[$(date)] API 启动 :$API_PORT"
exec uvicorn api:app --host 0.0.0.0 --port "$API_PORT" --app-dir /app
