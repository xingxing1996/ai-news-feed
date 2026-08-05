#!/usr/bin/env bash
# 容器入口：
#   1) 设了 CRON_SCHEDULE → 安装定时任务并常驻（每日自动 run）
#   2) 否则按参数执行（默认 run：一次性全流程）
set -e
cd /app
mkdir -p /app/data/output

if [ -n "$CRON_SCHEDULE" ]; then
  echo "[$(date)] 定时模式：$CRON_SCHEDULE (TZ=$TZ)"
  printf "CRON_TZ=%s\n%s root /app/docker/run_daily.sh >> /app/data/output/cron.log 2>&1\n" \
    "$TZ" "$CRON_SCHEDULE" > /etc/cron.d/stockpredict
  chmod 0644 /etc/cron.d/stockpredict
  crontab /etc/cron.d/stockpredict
  touch /app/data/output/cron.log
  # 启动时先跑一次，再常驻 cron
  /app/docker/run_daily.sh || true
  exec cron -f
fi

case "${1:-run}" in
  run|ingest|features|train|backtest|walkforward|evaluate|news|report|universe)
    exec stock-predict "$@" ;;
  shell) exec bash ;;
  *) exec stock-predict "$@" ;;
esac
