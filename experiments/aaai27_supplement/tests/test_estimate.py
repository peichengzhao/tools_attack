from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aaai27_supplement.estimate import (  # noqa: E402
    estimate_coevolution_calls,
    estimate_frozen_attack_calls,
    estimate_harness_calls,
    estimate_pipeline_calls,
)


class EstimateTest(unittest.TestCase):
    def test_coevolution(self):
        result = estimate_coevolution_calls(632, 8)
        self.assertEqual(result["victim"], 5056)
        self.assertEqual(result["updates"], 4424)
        self.assertEqual(result["total"], 9480)

    def test_heldout(self):
        self.assertEqual(estimate_frozen_attack_calls(211)["total"], 422)
        self.assertEqual(estimate_harness_calls(211, 8)["total"], 1688)

    def test_pipeline(self):
        result = estimate_pipeline_calls(
            train_cases=632,
            heldout_cases=211,
            rounds=8,
            harness_candidates=8,
        )
        self.assertEqual(result["train"], 9480)
        self.assertEqual(result["validation_attack"], 422)
        self.assertEqual(result["validation_harness"], 1688)
        self.assertEqual(result["test_attack"], 422)
        self.assertEqual(result["test_harness"], 211)


if __name__ == "__main__":
    unittest.main()
