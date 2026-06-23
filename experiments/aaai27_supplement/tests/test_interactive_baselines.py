from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class InteractiveBaselineConfigTests(unittest.TestCase):
    def test_config_contains_only_interactive_baselines(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "interactive_baselines.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            [item["id"] for item in config["defenses"]],
            ["task_shield", "melon", "camel"],
        )
        self.assertEqual(config["benchmark_version"], "v1.2")
        self.assertEqual(
            set(config["suites"]),
            {"workspace", "banking", "travel", "slack"},
        )

    def test_pending_table_can_be_generated_without_results(self) -> None:
        from scripts.build_interactive_table import load_cells

        config = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "configs"
                / "interactive_baselines.json"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as tmp:
            rows = load_cells(
                config,
                Path(tmp),
                ["deepseek-v4.1-flash-1017"],
            )
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["status"] == "pending" for row in rows))


if __name__ == "__main__":
    unittest.main()
