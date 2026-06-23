"""
运行完整的 AgentDojo 基准测试。

对应论文 Section 4 (Experiments) 的实验设置。

使用方式:
    # 运行所有场景和攻击
    python scripts/run_benchmark.py --all

    # 运行指定场景和攻击
    python scripts/run_benchmark.py --suite workspace --attack important_instructions

    # 只测试 Utility（无攻击）
    python scripts/run_benchmark.py --suite workspace --no-attack

    # 运行基线实验（无防御）
    python scripts/run_benchmark.py --suite workspace --attack important_instructions --no-defense
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console
from rich.table import Table

# 将项目根目录添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from agentdojo.benchmark import (
    SuiteResults,
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
)
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.task_suite.load_suites import get_suite, get_suites

# AgentDojo benchmark 版本（论文使用 v1）
BENCHMARK_VERSION = "v1"

from task_shield.task_shield_pipeline import (
    build_task_shield_pipeline,
    build_baseline_pipeline,
)

# 论文中使用的四个测试场景
ALL_SUITES = ["workspace", "travel", "banking", "slack"]

# 论文中使用的三种攻击方式
ALL_ATTACKS = ["important_instructions", "injecagent", "tool_knowledge"]

console = Console()


def run_single_experiment(
    suite_name: str,
    attack_name: str | None,
    use_defense: bool,
    model: str | None,
    results_dir: Path,
    force_rerun: bool = False,
) -> dict:
    """
    运行单个实验（一个场景 + 一种攻击 + 是否使用防御）。

    Args:
        suite_name: 测试场景名称
        attack_name: 攻击方式名称（None 表示无攻击）
        use_defense: 是否使用 Task Shield 防御
        model: 模型名称
        results_dir: 结果保存目录
        force_rerun: 是否强制重新运行

    Returns:
        实验结果字典
    """
    defense_tag = "task_shield" if use_defense else "no_defense"
    attack_tag = attack_name or "no_attack"

    rprint(f"\n{'=' * 60}")
    rprint(f"[bold]场景: {suite_name} | 攻击: {attack_tag} | 防御: {defense_tag}[/bold]")
    rprint(f"{'=' * 60}")

    # 获取测试套件
    suite = get_suite(BENCHMARK_VERSION, suite_name)

    # 构建 Pipeline
    if use_defense:
        pipeline = build_task_shield_pipeline(model=model)
        rprint("[green]✓ 使用 Task Shield 防御[/green]")
    else:
        pipeline = build_baseline_pipeline(model=model)
        rprint("[yellow]⚠ 无防御（基线）[/yellow]")

    # 设置日志目录
    logdir = results_dir / f"{suite_name}_{attack_tag}_{defense_tag}"
    logdir.mkdir(parents=True, exist_ok=True)

    # 运行基准测试
    try:
        if attack_name is None:
            # 无攻击 → 只测试 Utility
            results = benchmark_suite_without_injections(
                pipeline,
                suite,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=BENCHMARK_VERSION,
            )
            utility = _calculate_utility(results)
            rprint(f"[blue]Utility: {utility:.2%}[/blue]")
            return {
                "suite": suite_name,
                "attack": "none",
                "defense": defense_tag,
                "utility": utility,
                "asr": None,
            }
        else:
            # 有攻击 → 测试 Utility + Security
            attacker = load_attack(attack_name, suite, pipeline)
            results = benchmark_suite_with_injections(
                pipeline,
                suite,
                attacker,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=BENCHMARK_VERSION,
            )
            utility = _calculate_utility_from_injection_results(results)
            asr = _calculate_asr(results)
            rprint(f"[blue]Utility: {utility:.2%} | ASR: {asr:.2%}[/blue]")
            return {
                "suite": suite_name,
                "attack": attack_name,
                "defense": defense_tag,
                "utility": utility,
                "asr": asr,
            }
    except Exception as e:
        rprint(f"[red]实验失败: {e}[/red]")
        logging.exception(f"实验失败: {suite_name}/{attack_tag}/{defense_tag}")
        return {
            "suite": suite_name,
            "attack": attack_tag,
            "defense": defense_tag,
            "utility": None,
            "asr": None,
            "error": str(e),
        }


def _calculate_utility(results: SuiteResults) -> float:
    """从无攻击的结果中计算 Utility。"""
    utility_results = results.get("utility_results", {})
    if not utility_results:
        return 0.0
    values = list(utility_results.values())
    return sum(values) / len(values) if values else 0.0


def _calculate_utility_from_injection_results(results: SuiteResults) -> float:
    """从有攻击的结果中计算 Utility。"""
    utility_results = results.get("utility_results", {})
    if not utility_results:
        return 0.0
    values = list(utility_results.values())
    return sum(values) / len(values) if values else 0.0


def _calculate_asr(results: SuiteResults) -> float:
    """计算攻击成功率 (Attack Success Rate)。"""
    security_results = results.get("security_results", {})
    if not security_results:
        return 0.0
    # security_results[key] == True 意味着安全（攻击失败）
    # ASR = 1 - (安全的比例) = 攻击成功的比例
    values = list(security_results.values())
    safe_rate = sum(values) / len(values) if values else 0.0
    return 1.0 - safe_rate


@click.command()
@click.option("--suite", "-s", multiple=True, help="测试场景 (可多选)")
@click.option("--attack", "-a", multiple=True, help="攻击方式 (可多选)")
@click.option("--all", "run_all", is_flag=True, help="运行所有场景和攻击")
@click.option("--no-attack", is_flag=True, help="只运行无攻击测试 (Utility)")
@click.option("--no-defense", is_flag=True, help="不使用防御 (基线)")
@click.option("--model", "-m", default=None, help="模型名称")
@click.option(
    "--results-dir",
    "-o",
    type=click.Path(),
    default="results",
    help="结果保存目录",
)
@click.option("--force-rerun", is_flag=True, help="强制重新运行")
def main(
    suite: tuple,
    attack: tuple,
    run_all: bool,
    no_attack: bool,
    no_defense: bool,
    model: str | None,
    results_dir: str,
    force_rerun: bool,
):
    """运行 Task Shield 基准测试。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # 确定要运行的场景
    suites = list(suite) if suite else (ALL_SUITES if run_all else ["workspace"])
    # 确定要运行的攻击
    attacks = (
        list(attack)
        if attack
        else (ALL_ATTACKS if run_all else ["important_instructions"])
    )

    all_results = []

    for suite_name in suites:
        if no_attack:
            # 只测试 Utility
            result = run_single_experiment(
                suite_name=suite_name,
                attack_name=None,
                use_defense=not no_defense,
                model=model,
                results_dir=results_path,
                force_rerun=force_rerun,
            )
            all_results.append(result)
        else:
            for attack_name in attacks:
                # 有防御的实验
                if not no_defense:
                    result = run_single_experiment(
                        suite_name=suite_name,
                        attack_name=attack_name,
                        use_defense=True,
                        model=model,
                        results_dir=results_path,
                        force_rerun=force_rerun,
                    )
                    all_results.append(result)

                # 无防御的基线实验
                result = run_single_experiment(
                    suite_name=suite_name,
                    attack_name=attack_name,
                    use_defense=False,
                    model=model,
                    results_dir=results_path,
                    force_rerun=force_rerun,
                )
                all_results.append(result)

    # 打印汇总表格
    _print_results_table(all_results)

    # 保存结果到 JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_path / f"results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    rprint(f"\n[green]结果已保存至: {results_file}[/green]")


def _print_results_table(results: list[dict]):
    """打印实验结果汇总表格。"""
    table = Table(title="Task Shield 实验结果汇总")
    table.add_column("场景", style="cyan")
    table.add_column("攻击", style="magenta")
    table.add_column("防御", style="green")
    table.add_column("Utility", justify="right")
    table.add_column("ASR", justify="right", style="red")

    for r in results:
        utility_str = f"{r['utility']:.2%}" if r.get("utility") is not None else "N/A"
        asr_str = f"{r['asr']:.2%}" if r.get("asr") is not None else "N/A"
        error = r.get("error", "")
        if error:
            utility_str = f"[red]ERR[/red]"
            asr_str = f"[red]ERR[/red]"

        table.add_row(
            r.get("suite", ""),
            r.get("attack", ""),
            r.get("defense", ""),
            utility_str,
            asr_str,
        )

    console.print(table)


if __name__ == "__main__":
    main()
