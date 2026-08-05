#!/usr/bin/env bash
# 用合成数据（无网络）端到端跑一遍 Phase 1 流程：ingest→features→train→backtest→report。
set -euo pipefail
cd "$(dirname "$0")/.."

# 用 settings.local.yaml 覆盖：开启合成数据（不入库）
cat > config/settings.local.yaml <<'YAML'
data:
  synthetic: true
# 演示用：缩短训练区间以适配合成数据起点
model:
  split:
    train_end: "2022-06-30"
    valid_start: "2022-07-01"
    valid_end: "2023-06-30"
    test_start: "2023-07-01"
    test_end: ""
YAML

echo ">>> 运行全流程（合成数据，data.synthetic=true）..."
if command -v stock-predict >/dev/null 2>&1; then
  stock-predict run
else
  python -m stock_predict.cli run
fi

echo ""
echo "📄 日报见：data/output/daily_report.md"
