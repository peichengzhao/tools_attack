#!/usr/bin/env python3
"""Run a PIArena defense on the paper's 211-case InjecAgent DS split."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
PIARENA_ROOT = REPO_ROOT / "related_work" / "baselines" / "PIArena"


def ensure_conda_runtime() -> None:
    lib_dir = str(Path(sys.prefix) / "lib")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir in current.split(os.pathsep):
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        part for part in (lib_dir, current) if part
    )
    os.environ["_AAAI27_CONDA_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--defense",
        default="pisanitizer",
        help="PIArena representative defense.",
    )
    parser.add_argument(
        "--test-data",
        default=str(
            REPO_ROOT / "InjecAgent" / "data" / "splits" / "test_cases_211.json"
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument(
        "--output-root",
        default=str(
            REPO_ROOT
            / "results"
            / "aaai27_supplement"
            / "defense_baselines"
            / "piarena"
        ),
    )
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def prepare_workspace(args: argparse.Namespace) -> Path:
    workspace = Path(args.output_root) / "runtime" / args.defense
    data_dir = workspace / "agents" / "InjecAgent" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    source_data = PIARENA_ROOT / "agents" / "InjecAgent" / "data"
    for name in ("tools.json", "attacker_simulated_responses.json"):
        source = source_data / name
        if not source.is_file():
            raise SystemExit(f"Missing PIArena data file: {source}")
        shutil.copy2(source, data_dir / name)

    cases = json.loads(Path(args.test_data).read_text(encoding="utf-8"))
    if args.limit > 0:
        cases = cases[: args.limit]
    for name in (
        "test_cases_ds_base.json",
        "test_cases_ds_enhanced.json",
        "test_cases_dh_base.json",
        "test_cases_dh_enhanced.json",
    ):
        subset = cases if name == "test_cases_ds_base.json" else []
        (data_dir / name).write_text(
            json.dumps(subset, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return workspace


DEFENSE_MODULES = {
    "none": ("piarena.defenses.defense_none", "NoDefense"),
    "datafilter": ("piarena.defenses.datafilter.defense_datafilter", "DataFilterDefense"),
    "pisanitizer": ("piarena.defenses.pisanitizer.defense_pisanitizer", "PISanitizerDefense"),
    "promptguard": ("piarena.defenses.promptguard.defense_promptguard", "PromptGuardDefense"),
    "datasentinel": ("piarena.defenses.datasentinel.defense_datasentinel", "DataSentinelDefense"),
    "piguard": ("piarena.defenses.piguard.defense_piguard", "PIGuardDefense"),
    "attentiontracker": (
        "piarena.defenses.attentiontracker.defense_attentiontracker",
        "AttentionTrackerDefense",
    ),
    "promptarmor": ("piarena.defenses.promptarmor.defense_promptarmor", "PromptArmorDefense"),
    "promptlocate": ("piarena.defenses.promptlocate.defense_promptlocate", "PromptLocateDefense"),
}


def lazy_get_defense(name: str, config: dict | None = None):
    if name not in DEFENSE_MODULES:
        raise ValueError(
            f"Unknown PIArena defense {name!r}; choose from {sorted(DEFENSE_MODULES)}"
        )
    module_name, class_name = DEFENSE_MODULES[name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(config=config)


def install_piarena_modules() -> None:
    """Avoid importing every optional defense (and vLLM) for a single run."""
    import piarena

    stub = types.ModuleType("piarena.defenses")
    stub.__path__ = [str(PIARENA_ROOT / "piarena" / "defenses")]
    stub.get_defense = lazy_get_defense
    sys.modules["piarena.defenses"] = stub
    setattr(piarena, "defenses", stub)


def install_openai_compatible_model(main_injecagent) -> None:
    from openai import OpenAI

    class OpenAICompatibleModel:
        def __init__(self, model_name_or_path: str):
            self.model_name_or_path = model_name_or_path
            self.model_name = model_name_or_path.split("/")[-1]
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )

        def query(self, messages, **kwargs):
            response = self.client.chat.completions.create(
                model=self.model_name_or_path,
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""

    main_injecagent.Model = OpenAICompatibleModel


def main() -> int:
    if not os.environ.get("_AAAI27_CONDA_REEXEC"):
        ensure_conda_runtime()
    args = parse_args()
    if not (PIARENA_ROOT / "main_injecagent.py").is_file():
        raise SystemExit(f"PIArena source not found: {PIARENA_ROOT}")
    workspace = prepare_workspace(args)
    if args.export_only:
        print(workspace)
        return 0

    os.chdir(workspace)
    sys.path.insert(0, str(PIARENA_ROOT))
    install_piarena_modules()
    import main_injecagent

    main_injecagent.get_defense = lazy_get_defense
    if args.check:
        defense = lazy_get_defense(args.defense)
        print(
            json.dumps(
                {
                    "ok": True,
                    "defense": args.defense,
                    "class": type(defense).__name__,
                    "workspace": str(workspace),
                },
                indent=2,
            )
        )
        return 0
    install_openai_compatible_model(main_injecagent)
    main_injecagent.main(
        SimpleNamespace(
            model=args.model,
            defense=args.defense,
            name=f"tce211-{args.defense}",
            seed=args.seed,
            checkpoint_interval=args.checkpoint_interval,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
