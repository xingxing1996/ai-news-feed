"""命令行入口：``stock-predict <command>``。

命令：
  universe   查看/写入股票池
  ingest     采集行情/财务/估值 → SQLite + Parquet + Qlib bin
  features   构建因子矩阵 + 超额收益 label
  train      LightGBM 训练 + Rank IC 评估
  backtest   Top-N 持有20天回测
  report     生成 AI 投资日报
  run        顺序执行以上全部
"""
from __future__ import annotations

import json
import logging

import typer
from rich.console import Console
from rich.table import Table

from .config import load_settings, reset_settings

app = typer.Typer(add_completion=False, help="stock-predict: AI 量化投资研究系统")
console = Console()


def _setup(config: str | None, verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    reset_settings()
    if config:
        load_settings(config)  # 预加载覆盖


@app.callback()
def main(
    ctx: typer.Context,
    config: str = typer.Option(None, "--config", "-c", help="配置文件路径"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    _setup(config, verbose)


@app.command()
def universe(show: bool = typer.Option(True, help="打印股票池")):
    """查看 / 写入股票池。"""
    from .data.universe import resolve_universe, universe_to_db

    df = resolve_universe()
    n = universe_to_db(df)
    if show:
        tbl = Table(title=f"股票池（共 {len(df)} 只）")
        for col in ("code", "name", "market", "industry"):
            tbl.add_column(col)
        for _, r in df.iterrows():
            tbl.add_row(str(r["code"]), str(r["name"]), str(r["market"]), str(r["industry"]))
        console.print(tbl)
    console.print(f"[green]已写入 stock 表 {n} 行[/green]")


@app.command()
def ingest():
    """采集行情 / 财务 / 估值。"""
    from .data.loaders import fetch_and_store
    from .data.universe import resolve_universe

    stats = fetch_and_store(resolve_universe())
    console.print("[green]ingest 完成[/green]")
    console.print_json(data=stats)

    # Qlib bin 导入（best-effort，失败不影响后续 pandas 因子兜底）
    from .data.qlib_ingest import ingest_to_qlib, write_instruments_file
    from .data.universe import resolve_universe as _ru

    q = ingest_to_qlib()
    write_instruments_file(_ru())
    console.print(f"[cyan]Qlib 导入: {q}[/cyan]")


@app.command()
def features():
    """构建因子矩阵 + label。"""
    from .features.build import build_feature_matrix

    _, stats = build_feature_matrix()
    console.print("[green]features 完成[/green]")
    console.print_json(data=stats)


@app.command()
def train():
    """LightGBM 训练 + 评估。"""
    from .model.lgbm import train_and_predict

    metrics = train_and_predict()
    console.print("[green]train 完成[/green]")
    for seg in ("valid", "test"):
        if seg in metrics:
            console.print(f"[cyan]{seg}[/cyan]")
            console.print_json(data=metrics[seg])


@app.command()
def backtest(ranking: str = typer.Option(None, help="排序信号: label|abs|bench|blend（默认 blend）")):
    """Top-N 持有回测。默认用综合分(blend)排序选股。"""
    from .backtest.strategy import run_backtest

    report = run_backtest(ranking=ranking)
    # 落盘一份纯文本摘要供日报引用
    from pathlib import Path

    from .config import get_settings

    out = Path(get_settings().paths.output_dir) / "backtest_metrics.txt"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    console.print(f"[green]backtest 完成（ranking={report.get('ranking')}）[/green]")
    console.print_json(data=report)


@app.command()
def compare():
    """四排序对比：跑赢行业/上涨/跑赢大盘/综合分，看哪个 ranking 回测最好。"""
    from .backtest.strategy import run_backtest_compare

    res = run_backtest_compare()
    tbl = Table(title="四排序回测对比")
    for col in ("ranking", "ann_return", "sharpe", "max_drawdown", "excess_ann", "avg_turnover"):
        tbl.add_column(col)
    for mode, m in res.items():
        if "error" in m:
            tbl.add_row(mode, "—", "—", "—", "—", f"err: {m['error'][:20]}")
        else:
            tbl.add_row(mode, f"{m.get('ann_return', 0):.2%}", f"{m.get('sharpe', 0):.2f}",
                        f"{m.get('max_drawdown', 0):.2%}", f"{m.get('excess_ann', 0):.2%}",
                        f"{m.get('avg_turnover', 0):.2f}")
    console.print(tbl)
    console.print_json(data=res)




@app.command()
def news(max_codes: int = typer.Option(20, help="最多抽取事件的股票数（控 LLM 成本）"),
         per_stock: int = typer.Option(3, help="每只股票最多分析的新闻条数")):
    """采集新闻 → DeepSeek 事件抽取 → news_score（供日报引用）。"""
    from .news.pipeline import run_news_pipeline

    stats = run_news_pipeline(max_codes=max_codes, per_stock=per_stock)
    console.print("[green]news 完成[/green]")
    console.print_json(data=stats)


@app.command()
def walkforward(train_days: int = typer.Option(756, help="滚动训练窗（交易日近似）"),
                step: int = typer.Option(21, help="调仓/重训步长（交易日）")):
    """Walk-forward 滚动回测（机构级 OOS 评估，比单段切分更可信）。"""
    from .backtest.walkforward import run_walkforward

    report = run_walkforward(train_days=train_days, step=step)
    console.print("[green]walk-forward 完成[/green]")
    console.print_json(data=report)


@app.command()
def evaluate(n_quantiles: int = typer.Option(5, help="分位数")):
    """因子评估套件：分位多空 / 单调性 / IC 衰减。"""
    import pandas as pd

    from .data.warehouse import read_parquet
    from .model.evaluate import ic_decay, quantile_analysis

    pred = read_parquet("predictions")
    daily = read_parquet("daily_price")
    if pred.empty:
        raise RuntimeError("predictions 为空，请先 train 或 walkforward。")
    qa = quantile_analysis(pred, n_quantiles)
    console.print("[cyan]分位多空[/cyan]"); console.print_json(data={k: v for k, v in qa.items() if k != "group_ann_returns"})
    if qa.get("group_ann_returns"):
        console.print("各组年化:", qa["group_ann_returns"])
    decay = ic_decay(pred, daily)
    console.print("[cyan]IC 衰减[/cyan]"); console.print_json(data=decay)


@app.command()
def report(as_of: str = typer.Option(None, help="指定报告日 YYYY-MM-DD，默认最近一日")):
    """生成 AI 投资日报。"""
    from .report.daily import generate_daily_report

    text = generate_daily_report(as_of=as_of)
    console.print("[green]report 完成[/green]")
    console.rule("AI 投资日报")
    console.print(text)


@app.command()
def refresh():
    """每 2h 刷新日报：复用每日训练的模型 + 最新特征快照 + 实时新闻，不重训。"""
    from .report.daily import refresh_report

    text = refresh_report()
    console.print("[green]refresh 完成[/green]（复用模型，仅刷新新闻+日报）")
    console.rule("AI 投资日报(刷新)")
    console.print(text)


@app.command(name="run")
def run_all():
    """顺序执行：ingest → features → train → backtest → report。"""
    console.rule("[bold]1/5 ingest")
    from .data.loaders import fetch_and_store
    from .data.qlib_ingest import ingest_to_qlib, write_instruments_file
    from .data.universe import resolve_universe

    universe = resolve_universe()
    console.print_json(data=fetch_and_store(universe))
    console.print(f"[cyan]Qlib 导入: {ingest_to_qlib()}[/cyan]")
    write_instruments_file(universe)

    console.rule("[bold]2/5 features")
    from .features.build import build_feature_matrix

    console.print_json(data=build_feature_matrix()[1])

    console.rule("[bold]3/5 train")
    from .model.lgbm import train_and_predict

    console.print_json(data=train_and_predict())

    console.rule("[bold]4/5 backtest (walk-forward OOS，诚实样本外)")
    from .backtest.walkforward import run_walkforward

    bt = run_walkforward()
    console.print_json(data=bt)

    console.rule("[bold]4.5/6 news（best-effort，需网络 + LLM）")
    try:
        from .news.pipeline import run_news_pipeline

        console.print_json(data=run_news_pipeline())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]news 步骤跳过：{exc}[/yellow]")

    console.rule("[bold]5/6 report")
    from .report.daily import generate_daily_report

    console.rule("AI 投资日报")
    console.print(generate_daily_report())


if __name__ == "__main__":
    app()
