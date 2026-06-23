"""
运行单个测试用例（用于调试和快速验证）。

使用方式:
    # 基本测试
    python scripts/run_single_test.py

    # 指定场景和任务
    python scripts/run_single_test.py --suite workspace --user-task user_task_0

    # 带攻击的测试
    python scripts/run_single_test.py --suite workspace --user-task user_task_0 \
        --attack important_instructions

    # 不使用防御
    python scripts/run_single_test.py --suite workspace --no-defense
"""

import json
import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console

# 将项目根目录添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from agentdojo.task_suite.load_suites import get_suite
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.functions_runtime import FunctionsRuntime

# AgentDojo benchmark 版本
BENCHMARK_VERSION = "v1"

from task_shield.task_shield_pipeline import (
    build_task_shield_pipeline,
    build_baseline_pipeline,
)

console = Console()


@click.command()
@click.option("--suite", "-s", default="workspace", help="测试场景")
@click.option("--user-task", "-ut", default="user_task_0", help="用户任务 ID")
@click.option("--attack", "-a", default=None, help="攻击方式")
@click.option("--injection-task", "-it", default=None, help="注入任务 ID")
@click.option("--model", "-m", default=None, help="模型名称")
@click.option("--no-defense", is_flag=True, help="不使用防御")
@click.option("--verbose", "-v", is_flag=True, help="详细输出")
def main(
    suite: str,
    user_task: str,
    attack: str | None,
    injection_task: str | None,
    model: str | None,
    no_defense: bool,
    verbose: bool,
):
    """运行单个测试用例进行调试。"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    rprint(f"\n[bold cyan]Task Shield 单个测试[/bold cyan]")
    rprint(f"场景: {suite}")
    rprint(f"用户任务: {user_task}")
    rprint(f"攻击: {attack or '无'}")
    rprint(f"防御: {'无' if no_defense else 'Task Shield'}")
    rprint()

    # 获取测试套件
    test_suite = get_suite(BENCHMARK_VERSION, suite)

    # 获取用户任务
    task = test_suite.get_user_task_by_id(user_task)
    if task is None:
        rprint(f"[red]找不到用户任务: {user_task}[/red]")
        rprint(f"可用任务: {list(test_suite.user_tasks.keys())}")
        return

    # 构建 Pipeline
    if no_defense:
        pipeline = build_baseline_pipeline(model=model)
        rprint("[yellow]⚠ 不使用防御[/yellow]")
    else:
        pipeline = build_task_shield_pipeline(model=model)
        rprint("[green]✓ 使用 Task Shield 防御[/green]")

    # 设置环境（使用默认注入向量）
    default_injections = test_suite.get_injection_vector_defaults()

    # 如果有攻击，注入恶意内容
    inj_task = None
    if attack and injection_task:
        inj_task = test_suite.get_injection_task_by_id(injection_task)
        if inj_task is None:
            rprint(f"[red]找不到注入任务: {injection_task}[/red]")
            return

        attacker = load_attack(attack, test_suite, pipeline)
        attack_injections = attacker.attack(inj_task)
        default_injections.update(attack_injections)
        rprint(f"[red]已注入攻击: {attack} / {injection_task}[/red]")
    elif attack:
        # 使用第一个注入任务
        first_inj_task_id = next(iter(test_suite.injection_tasks.keys()))
        inj_task = test_suite.get_injection_task_by_id(first_inj_task_id)
        attacker = load_attack(attack, test_suite, pipeline)
        attack_injections = attacker.attack(inj_task)
        default_injections.update(attack_injections)
        rprint(f"[red]已注入攻击: {attack} / {first_inj_task_id}[/red]")

    env = test_suite.load_and_inject_default_environment(default_injections)

    # 获取用户查询
    user_query = task.PROMPT
    rprint(f"\n[bold]用户查询:[/bold] {user_query}")
    rprint()

    # 创建 runtime
    runtime = FunctionsRuntime(test_suite.tools)

    # 运行 Pipeline
    rprint("[bold]开始执行...[/bold]")
    rprint("-" * 60)

    try:
        _, _, result_env, messages, _ = pipeline.query(user_query, runtime, env)

        # 显示对话历史
        rprint("\n[bold]对话历史:[/bold]")
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "unknown")
                content = getattr(msg, "content", "")

            if isinstance(content, list):
                # AgentDojo 的 content blocks 格式
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("content", ""))
                content = " ".join(text_parts)

            # 截断过长的内容
            display_content = (
                content[:500] + "..." if len(str(content)) > 500 else content
            )

            color = {
                "system": "dim",
                "user": "green",
                "assistant": "blue",
                "tool": "yellow",
            }.get(role, "white")

            rprint(f"  [{color}][{role}] {display_content}[/{color}]")

            # 显示工具调用
            if isinstance(msg, dict):
                tool_calls = msg.get("tool_calls", [])
            else:
                tool_calls = getattr(msg, "tool_calls", []) or []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    name = func.get("name", "?") if isinstance(func, dict) else "?"
                    args = func.get("arguments", "{}") if isinstance(func, dict) else "{}"
                else:
                    name = getattr(tc, "function", "?")
                    args = json.dumps(getattr(tc, "args", {}), ensure_ascii=False)
                rprint(f"    [magenta]→ 工具调用: {name}({args})[/magenta]")

        # 检查结果
        rprint("\n[bold]任务完成检查:[/bold]")
        try:
            utility_result = task.utility(result_env, messages)
            rprint(
                f"  Utility: {'[green]✓ PASS[/green]' if utility_result else '[red]✗ FAIL[/red]'}"
            )
        except Exception as e:
            rprint(f"  Utility 检查失败: {e}")

        if attack and inj_task is not None:
            rprint("\n[bold]安全检查:[/bold]")
            try:
                security_result = inj_task.security(result_env, messages)
                rprint(
                    f"  Security: {'[green]✓ 攻击被防御[/green]' if security_result else '[red]✗ 攻击成功[/red]'}"
                )
            except Exception as e:
                rprint(f"  Security 检查失败: {e}")

    except Exception as e:
        rprint(f"\n[red]执行失败: {e}[/red]")
        logging.exception("执行失败")


if __name__ == "__main__":
    main()
