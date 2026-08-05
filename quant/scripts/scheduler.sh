#!/usr/bin/env bash
# macOS launchd 定时（A股+港股，本地跑，AKShare 稳）：
#   1) 每天 08:00（可改）→ stock-predict run   （拉数据 + 重训模型 + 日报）
#   2) 每 2 小时        → stock-predict refresh（复用当天模型 + 最新新闻，刷新日报，不重训）
# 不依赖 docker/cron。launchd：合盖睡眠错过会在唤醒后补跑。
#
# 用法：
#   bash scripts/scheduler.sh install                       # 安装两个任务（每天8点训练 + 每2h刷新）
#   CRON_HOUR=8 CRON_MINUTE=0 bash scripts/scheduler.sh install   # 自定义训练时间
#   bash scripts/scheduler.sh test train|refresh            # 立即手动触发
#   bash scripts/scheduler.sh status                        # 查看加载状态
#   bash scripts/scheduler.sh uninstall                     # 卸载全部
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"          # quant/
ENV="${STOCK_PREDICT_ENV:-stock-predict}"
HOUR="${CRON_HOUR:-8}"
MIN="${CRON_MINUTE:-0}"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-7200}"       # 刷新间隔(秒)，默认 2h

# conda 环境的 python 绝对路径（launchd 无 shell 环境）
PY="$(conda run -n "$ENV" which python 2>/dev/null || echo "$HOME/miniconda3/envs/$ENV/bin/python")"
PY_DIR="$(dirname "$PY")"

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
  # $1=label  $2=cmd(run/refresh)  $3=schedule-xml
  local label="$1" cmd="$2" sched="$3"
  local keyline=""
  [ -n "$KEY" ] && keyline="    <key>ARK_API_KEY</key><string>$KEY</string>"
  cat > "$HOME/Library/LaunchAgents/${label}.plist" <<XML
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${label}</string>
  <key>ProgramArguments</key>
  <array><string>${PY}</string><string>-m</string><string>stock_predict.cli</string><string>${cmd}</string></array>
  <key>WorkingDirectory</key><string>${ROOT}</string>
  ${sched}
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>${PY_DIR}:/usr/local/bin:/usr/bin:/bin</string>
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

LABEL_T="com.stockpredict.cn.train"
LABEL_R="com.stockpredict.cn.refresh"
SCHED_TRAIN="<key>StartCalendarInterval</key><dict><key>Hour</key><integer>${HOUR}</integer><key>Minute</key><integer>${MIN}</integer></dict>"
SCHED_REFRESH="<key>StartInterval</key><integer>${REFRESH_INTERVAL}</integer>"

case "${1:-install}" in
  install)
    mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/data/output"
    launchctl unload "$HOME/Library/LaunchAgents/${LABEL_T}.plist" 2>/dev/null || true
    launchctl unload "$HOME/Library/LaunchAgents/${LABEL_R}.plist" 2>/dev/null || true
    gen_plist "$LABEL_T" run     "$SCHED_TRAIN"
    gen_plist "$LABEL_R" refresh "$SCHED_REFRESH"
    launchctl load "$HOME/Library/LaunchAgents/${LABEL_T}.plist"
    launchctl load "$HOME/Library/LaunchAgents/${LABEL_R}.plist"
    echo "✅ 已安装："
    echo "   ① 每天 ${HOUR}:${MIN}  → 训练(run)：拉数据+重训模型+日报"
    echo "   ② 每 $((REFRESH_INTERVAL/3600))h → 刷新(refresh)：复用模型+最新新闻，刷新日报"
    echo "   配置: $ROOT/config/settings.cn.yaml  |  python: $PY"
    echo "   日志: $ROOT/data/output/scheduler.log"
    echo "   测试: bash scripts/scheduler.sh test train   或   test refresh"
    ;;
  uninstall)
    launchctl unload "$HOME/Library/LaunchAgents/${LABEL_T}.plist" 2>/dev/null || true
    launchctl unload "$HOME/Library/LaunchAgents/${LABEL_R}.plist" 2>/dev/null || true
    rm -f "$HOME/Library/LaunchAgents/${LABEL_T}.plist" "$HOME/Library/LaunchAgents/${LABEL_R}.plist"
    echo "已卸载全部 A股 定时任务"
    ;;
  test)
    lbl="$LABEL_T"; [ "${2:-}" = "refresh" ] && lbl="$LABEL_R"
    launchctl start "$lbl" && echo "已触发 $lbl，看日志: tail -f $ROOT/data/output/scheduler.log"
    ;;
  status)
    launchctl list | grep stockpredict || echo "未加载"
    ;;
  *)
    echo "用法: bash scripts/scheduler.sh [install|uninstall|test|status]"; exit 1 ;;
esac
