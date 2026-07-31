#!/usr/bin/env bash
# 每日定时全流程：ingest(拉新数据) → features → train(重训模型) → backtest → news → report
# 即「每天用最新数据重训模型 + 刷新推荐」。
set -e
cd /app
mkdir -p /app/data/output
echo "===== $(date '+%F %T %Z') 每日运行开始 ====="
stock-predict run
echo "===== $(date '+%F %T %Z') 完成，日报见 /app/data/output/daily_report.md ====="
