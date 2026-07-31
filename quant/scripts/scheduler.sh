#!/usr/bin/env bash
# macOS launchd 定时任务：每天早上指定时间自动「stock-predict run」（重训模型 + 刷新日报）。
# 不依赖 docker / cron。用 launchd：合盖睡眠错过时间会在唤醒后补跑一次。
#
# 用法：
#   bash scripts/scheduler.sh install      # 安装（默认每天 08:00）
#   CRON_HOUR=8 CRON_MINUTE=0 bash scripts/scheduler.sh install   # 自定义时间
#   bash scripts/scheduler.sh test         # 立即手动触发一次
#   bash scripts/scheduler.sh status       # 查看加载状态
#   bash scripts/scheduler.sh uninstall    # 卸载
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.stockpredict.daily"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
ENV="${STOCK_PREDICT_ENV:-stock-predict}"
HOUR="${CRON_HOUR:-8}"
MIN="${CRON_MINUTE:-0}"

# 解析 conda 环境里的 stock-predict 绝对路径（launchd 无 shell 环境，必须绝对路径）
BIN="$(conda run -n "$ENV" which stock-predict 2>/dev/null || echo "$HOME/miniconda3/envs/$ENV/bin/stock-predict")"
BIN_DIR="$(dirname "$BIN")"

# 从 settings.local.yaml 读 ARK_API_KEY（不把 key 写进脚本）
KEY="$(
  python3 - "$ROOT/config/settings.local.yaml" <<'PY' 2>/dev/null || true
import sys, yaml
try:
    d = yaml.safe_load(open(sys.argv[1])) or {}
    print((d.get("llm") or {}).get("api_key", ""))
except Exception:
    pass
PY
)"
[ -z "$KEY" ] && KEY="${ARK_API_KEY:-}"

gen_plist() {
  local keyline=""
  [ -n "$KEY" ] && keyline="    <key>ARK_API_KEY</key><string>$KEY</string>"
  cat > "$PLIST" <<XML
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${BIN}</string>
    <string>run</string>
  </array>
  <key>WorkingDirectory</key><string>${ROOT}</string>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>${HOUR}</integer><key>Minute</key><integer>${MIN}</integer></dict>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>${BIN_DIR}:/usr/local/bin:/usr/bin:/bin</string>
    <key>PYTHONPATH</key><string>${ROOT}/..:${ROOT}</string>
    <key>STOCK_PREDICT_CONFIG</key><string>${ROOT}/config/settings.cn.yaml</string>
${keyline}
  </dict>
  <key>StandardOutPath</key><string>${ROOT}/data/output/scheduler.log</string>
  <key>StandardErrorPath</key><string>${ROOT}/data/output/scheduler.err.log</string>
  <key>RunAtLoad</key><false/>
</dict>
</plist>
XML
}

case "${1:-install}" in
  install)
    mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/data/output"
    launchctl unload "$PLIST" 2>/dev/null || true
    gen_plist
    launchctl load "$PLIST"
    echo "✅ 已安装：每天 ${HOUR}:${MIN} 自动运行 'stock-predict run'（拉最新数据 + 重训 + 日报）"
    echo "   入口:   $BIN"
    echo "   配置:   $ROOT/config/settings.cn.yaml  (A股+港股, 增量采集)"
    echo "   日志:   $ROOT/data/output/scheduler.log"
    echo "   日报:   $ROOT/data/output/daily_report.md"
    echo "   立即测试: bash scripts/scheduler.sh test"
    ;;
  uninstall)
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "已卸载定时任务"
    ;;
  test)
    launchctl start "$LABEL" && echo "已触发，看日志: tail -f $ROOT/data/output/scheduler.log"
    ;;
  status)
    launchctl list | grep stockpredict || echo "未加载"
    ;;
  *)
    echo "用法: bash scripts/scheduler.sh [install|uninstall|test|status]"; exit 1 ;;
esac
