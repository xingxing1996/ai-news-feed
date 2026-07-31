#!/usr/bin/env bash
# 每 2h 刷新 A股+港股日报：复用每日训练模型，只刷新闻 + 重出 recommendations_cn。
# 由 launchd 触发（StartInterval=7200）。不重训、不重拉历史行情。
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV="${STOCK_PREDICT_ENV:-stock-predict}"
BIN="$HOME/miniconda3/envs/$ENV/bin/stock-predict"
BIN_DIR="$(dirname "$BIN")"
KEY="${ARK_API_KEY:-}"

export PATH="$BIN_DIR:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/..:$ROOT"
export STOCK_PREDICT_CONFIG="$ROOT/config/settings.cn.yaml"
[ -n "$KEY" ] && export ARK_API_KEY="$KEY"

exec "$BIN" refresh
