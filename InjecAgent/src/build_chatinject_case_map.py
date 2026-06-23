#!/usr/bin/env python3
"""Build / refresh ChatInject GOAL mapping for InjecAgent 544 S1."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cross_eval.chatinject_mapping import (  # noqa: E402
    build_case_goal_map,
    save_case_goal_map,
)


def main() -> None:
    data_path = os.path.join("data", "test_cases_ds_base.json")
    with open(data_path, encoding="utf-8") as f:
        cases = json.load(f)
    mapping = build_case_goal_map(cases)
    path = save_case_goal_map(mapping)
    print(f"Mapped {len(mapping)} cases -> {path}")


if __name__ == "__main__":
    main()
