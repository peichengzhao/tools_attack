#!/usr/bin/env python3
"""
Minimal AppWorld benign runner for Exp7.

Uses AppWorld test_normal tasks with a ReAct-style loop and optional max_defense
system addendum (unified system layers only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EXP7_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ATTACK_ROOT = EXP7_ROOT.parent
if str(TOOLS_ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ATTACK_ROOT))

from paths import ensure_on_sys_path

ensure_on_sys_path()
from model_api import OPENAI_API_KEY, OPENAI_BASE_URL  # noqa: E402

from exp7.harness import apply_unified_system_addendum  # noqa: E402


def call_llm(client, model: str, messages: list[dict]) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=800,
    )
    if not resp.choices:
        return ""
    return resp.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--harness", choices=("baseline", "max_defense"), default="baseline")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from openai import OpenAI

    try:
        from appworld import AppWorld, load_task_ids
    except ImportError as exc:
        raise SystemExit("Install appworld: pip install appworld") from exc

    split = os.environ.get("APPWORLD_SPLIT", "test_normal")
    limit = int(os.environ.get("APPWORLD_TASK_LIMIT", "0") or "0")
    task_ids = load_task_ids(split)
    if limit > 0:
        task_ids = task_ids[:limit]

    base_system = (
        "You are an AppWorld agent. Write Python code in a REPL to complete the "
        "user task using apis.* helpers. Call apis.supervisor.complete_task() when done."
    )
    system_prompt = apply_unified_system_addendum(
        base_system,
        enabled=(args.harness == "max_defense"),
    )

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    successes = 0
    total = 0

    for task_id in task_ids:
        total += 1
        with AppWorld(task_id=task_id) as world:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Task: {world.task.instruction}\n"
                        f"Supervisor: {world.task.supervisor}\n"
                        "Respond with Python code only."
                    ),
                },
            ]
            done = False
            for _ in range(8):
                reply = call_llm(client, args.model, messages)
                messages.append({"role": "assistant", "content": reply})
                try:
                    exec_globals: dict = {}
                    exec(reply, exec_globals)
                except Exception as exc:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Execution error: {exc}. Try again.",
                        }
                    )
                    continue
                if world.task_completed():
                    done = True
                    break
            if done:
                successes += 1

    utility = (successes / total * 100.0) if total else 0.0
    print(
        json.dumps(
            {
                "utility": round(utility, 2),
                "successes": successes,
                "total": total,
                "split": split,
                "harness": args.harness,
                "model": args.model,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
