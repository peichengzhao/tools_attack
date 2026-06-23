from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aaai27_supplement.baselines import (
    BaselineSpec,
    load_registry,
    read_model_runtime_config,
    validate_baseline,
)


class BaselineRegistryTests(unittest.TestCase):
    def test_registry_has_required_baselines(self) -> None:
        registry = load_registry(
            Path(__file__).resolve().parents[1] / "configs" / "baselines.json"
        )
        self.assertEqual(
            set(registry),
            {"task_shield", "melon", "camel", "piarena"},
        )
        self.assertEqual(registry["task_shield"].provenance, "third_party_reproduction")
        self.assertEqual(registry["melon"].provenance, "official")
        self.assertEqual(registry["piarena"].representative_defense, "pisanitizer")

    def test_validation_checks_commit_and_entrypoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "related_work" / "baselines" / "demo"
            repo.mkdir(parents=True)
            (repo / "VENDORED_COMMIT.txt").write_text("abc123\n", encoding="utf-8")
            (repo / "main.py").write_text("", encoding="utf-8")
            spec = BaselineSpec(
                baseline_id="demo",
                display_name="Demo",
                source_url="https://example.invalid",
                commit="abc123",
                provenance="official",
                repo_dir="related_work/baselines/demo",
                benchmarks=("demo",),
                entrypoints=("main.py",),
                runner="demo",
                notes="",
            )
            report = validate_baseline(root, spec)
            self.assertTrue(report["ok"])
            (repo / "VENDORED_COMMIT.txt").write_text("different\n", encoding="utf-8")
            self.assertFalse(validate_baseline(root, spec)["ok"])

    def test_read_model_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.py"
            path.write_text(
                'OPENAI_API_KEY = "secret"\nOPENAI_BASE_URL = "https://example/v1"\n',
                encoding="utf-8",
            )
            values = read_model_runtime_config(path)
            self.assertEqual(values["OPENAI_BASE_URL"], "https://example/v1")
            self.assertEqual(values["OPENAI_API_KEY"], "secret")


if __name__ == "__main__":
    unittest.main()
