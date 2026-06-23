from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aaai27_supplement.metrics import parse_case, summarize  # noqa: E402


class MetricsTest(unittest.TestCase):
    def test_fixed_denominator(self):
        records = []
        fixtures = [
            {"eval": "succ", "utility_eval": "succ", "case_id": "a"},
            {"eval": "unsucc", "utility_eval": "succ", "case_id": "b"},
            {"eval": "invalid", "utility_eval": "invalid", "case_id": "c"},
        ]
        for item in fixtures:
            parsed = parse_case(item, source="test", method="m")
            self.assertIsNotNone(parsed)
            records.append(parsed)
        result = summarize(records, raw_count=3, duplicates=0)
        self.assertEqual(result["total_cases"], 3)
        self.assertEqual(result["valid"], 2)
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["attack_success_final"], 1)
        self.assertEqual(result["asr_all_pct"], 33.33)
        self.assertEqual(result["asr_valid_pct"], 50.0)
        self.assertEqual(result["invalid_rate_pct"], 33.33)
        self.assertEqual(result["utility_success"], 2)
        self.assertEqual(result["svu"], 1)


if __name__ == "__main__":
    unittest.main()
