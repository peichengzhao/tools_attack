"""
结果分析与可视化。

读取实验结果 JSON 文件，生成:
1. 汇总表格（对标论文 Table 1）
2. 柱状图对比
3. Pareto Front 分析

使用方式:
    python scripts/analyze_results.py --results-dir results/
    python scripts/analyze_results.py --results-file results/results_20240101_120000.json
"""

import json
import os
import sys
from pathlib import Path

import click
import pandas as pd
from rich import print as rprint
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()

# 论文 Table 1 中的参考结果 (GPT-4o)
PAPER_RESULTS = {
    "no_defense": {
        "utility": 0.7333,
        "asr_important_instructions": 0.4719,
        "asr_injecagent": 0.1483,
        "asr_tool_knowledge": 0.3483,
    },
    "task_shield": {
        "utility": 0.6979,
        "asr_important_instructions": 0.0207,
        "asr_injecagent": 0.0103,
        "asr_tool_knowledge": 0.0207,
    },
}


def load_results(results_dir: str = None, results_file: str = None) -> list[dict]:
    """加载实验结果。"""
    if results_file:
        with open(results_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if results_dir:
        results_path = Path(results_dir)
        all_results = []
        for json_file in sorted(results_path.glob("results_*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_results.extend(data)
        return all_results

    return []


def analyze_results(results: list[dict]):
    """分析实验结果。"""
    if not results:
        rprint("[red]没有找到实验结果[/red]")
        return

    df = pd.DataFrame(results)

    # 打印原始数据
    rprint("\n[bold]原始实验数据:[/bold]")
    rprint(df.to_string(index=False))

    # 按场景和防御方式分组统计
    rprint("\n[bold]按场景统计:[/bold]")
    _print_suite_summary(df)

    # 整体统计
    rprint("\n[bold]整体统计:[/bold]")
    _print_overall_summary(df)

    # 与论文结果对比
    rprint("\n[bold]与论文结果对比 (论文使用 GPT-4o):[/bold]")
    _print_paper_comparison(df)


def _print_suite_summary(df: pd.DataFrame):
    """按场景打印统计结果。"""
    table = Table(title="分场景结果")
    table.add_column("场景", style="cyan")
    table.add_column("防御", style="green")
    table.add_column("攻击", style="magenta")
    table.add_column("Utility ↑", justify="right")
    table.add_column("ASR ↓", justify="right", style="red")

    for _, row in df.iterrows():
        utility_str = f"{row['utility']:.2%}" if pd.notna(row.get("utility")) else "N/A"
        asr_str = f"{row['asr']:.2%}" if pd.notna(row.get("asr")) else "N/A"

        table.add_row(
            str(row.get("suite", "")),
            str(row.get("defense", "")),
            str(row.get("attack", "")),
            utility_str,
            asr_str,
        )

    console.print(table)


def _print_overall_summary(df: pd.DataFrame):
    """打印整体统计。"""
    table = Table(title="整体统计")
    table.add_column("防御方式", style="green")
    table.add_column("平均 Utility ↑", justify="right")
    table.add_column("平均 ASR ↓", justify="right", style="red")

    for defense in df["defense"].unique():
        subset = df[df["defense"] == defense]

        utility_vals = subset["utility"].dropna()
        asr_vals = subset["asr"].dropna()

        avg_utility = utility_vals.mean() if len(utility_vals) > 0 else float("nan")
        avg_asr = asr_vals.mean() if len(asr_vals) > 0 else float("nan")

        table.add_row(
            str(defense),
            f"{avg_utility:.2%}" if pd.notna(avg_utility) else "N/A",
            f"{avg_asr:.2%}" if pd.notna(avg_asr) else "N/A",
        )

    console.print(table)


def _print_paper_comparison(df: pd.DataFrame):
    """与论文结果进行对比。"""
    table = Table(title="与论文结果对比")
    table.add_column("指标", style="cyan")
    table.add_column("论文 (GPT-4o)", justify="right", style="dim")
    table.add_column("复现 (DeepSeek)", justify="right", style="bold")
    table.add_column("差异", justify="right")

    # 比较无防御的 Utility
    paper_no_def_utility = PAPER_RESULTS["no_defense"]["utility"]
    our_no_def = df[df["defense"] == "no_defense"]
    our_no_def_utility = our_no_def["utility"].dropna().mean() if len(our_no_def) > 0 else None

    if our_no_def_utility is not None:
        diff = our_no_def_utility - paper_no_def_utility
        diff_str = f"{diff:+.2%}"
        table.add_row(
            "无防御 Utility",
            f"{paper_no_def_utility:.2%}",
            f"{our_no_def_utility:.2%}",
            diff_str,
        )

    # 比较 Task Shield 的 Utility
    paper_ts_utility = PAPER_RESULTS["task_shield"]["utility"]
    our_ts = df[df["defense"] == "task_shield"]
    our_ts_utility = our_ts["utility"].dropna().mean() if len(our_ts) > 0 else None

    if our_ts_utility is not None:
        diff = our_ts_utility - paper_ts_utility
        diff_str = f"{diff:+.2%}"
        table.add_row(
            "Task Shield Utility",
            f"{paper_ts_utility:.2%}",
            f"{our_ts_utility:.2%}",
            diff_str,
        )

    # 比较 ASR
    for attack_key, attack_label in [
        ("important_instructions", "ASR (important_inst)"),
        ("injecagent", "ASR (injecagent)"),
        ("tool_knowledge", "ASR (tool_knowledge)"),
    ]:
        paper_asr_no_def = PAPER_RESULTS["no_defense"].get(
            f"asr_{attack_key}", None
        )
        paper_asr_ts = PAPER_RESULTS["task_shield"].get(
            f"asr_{attack_key}", None
        )

        our_asr_no_def = (
            our_no_def[our_no_def["attack"] == attack_key]["asr"].dropna().mean()
            if len(our_no_def[our_no_def["attack"] == attack_key]) > 0
            else None
        )
        our_asr_ts = (
            our_ts[our_ts["attack"] == attack_key]["asr"].dropna().mean()
            if len(our_ts[our_ts["attack"] == attack_key]) > 0
            else None
        )

        if paper_asr_no_def is not None and our_asr_no_def is not None:
            diff = our_asr_no_def - paper_asr_no_def
            table.add_row(
                f"无防御 {attack_label}",
                f"{paper_asr_no_def:.2%}",
                f"{our_asr_no_def:.2%}",
                f"{diff:+.2%}",
            )

        if paper_asr_ts is not None and our_asr_ts is not None:
            diff = our_asr_ts - paper_asr_ts
            table.add_row(
                f"Task Shield {attack_label}",
                f"{paper_asr_ts:.2%}",
                f"{our_asr_ts:.2%}",
                f"{diff:+.2%}",
            )

    console.print(table)


def generate_plots(results: list[dict], output_dir: str = "results"):
    """生成可视化图表。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except ImportError:
        rprint("[yellow]matplotlib 未安装，跳过图表生成[/yellow]")
        return

    df = pd.DataFrame(results)
    output_path = Path(output_dir)

    # 1. Utility 对比柱状图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, metric in enumerate(["utility", "asr"]):
        ax = axes[idx]
        pivot = df.pivot_table(
            values=metric, index="suite", columns="defense", aggfunc="mean"
        )

        if not pivot.empty:
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"{'Utility ↑' if metric == 'utility' else 'ASR ↓'}")
            ax.set_ylabel(metric.upper())
            ax.set_xlabel("测试场景")
            ax.legend(title="防御方式")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plot_path = output_path / "comparison_chart.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    rprint(f"[green]图表已保存至: {plot_path}[/green]")
    plt.close()


@click.command()
@click.option("--results-dir", "-d", default="results", help="结果目录")
@click.option("--results-file", "-f", default=None, help="指定结果文件")
@click.option("--plot", is_flag=True, help="生成图表")
def main(results_dir: str, results_file: str | None, plot: bool):
    """分析实验结果。"""
    results = load_results(results_dir=results_dir, results_file=results_file)

    if not results:
        rprint("[red]未找到实验结果。请先运行 run_benchmark.py[/red]")
        return

    analyze_results(results)

    if plot:
        generate_plots(results, results_dir)


if __name__ == "__main__":
    main()
