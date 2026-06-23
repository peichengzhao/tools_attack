from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aaai27_supplement.metrics import canonical_content_sha256  # noqa: E402


class SplitManifestTest(unittest.TestCase):
    def test_counts(self):
        path = ROOT / "data" / "split_manifest.csv"
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1054)
        counts = {"train": 0, "validation": 0, "test": 0}
        for row in rows:
            counts[row["split"]] += 1
        self.assertEqual(counts, {"train": 632, "validation": 211, "test": 211})

    def test_manifest_hash(self):
        path = ROOT / "data" / "split_manifest.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "695ccb17ad3ea55fc6d9e51016605912e91511a9e4187f3497cfa9e3477f051d",
        )

    def test_source_hashes(self):
        path = ROOT / "data" / "split_manifest.csv"
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        cache: dict[str, list[dict]] = {}
        for row in rows[:10]:
            source = REPO_ROOT / "InjecAgent" / row["source_file"]
            if not source.is_file():
                self.skipTest(f"source data unavailable: {source}")
            if row["source_file"] not in cache:
                cache[row["source_file"]] = json.loads(
                    source.read_text(encoding="utf-8")
                )
            case = cache[row["source_file"]][int(row["source_index"]) - 1]
            self.assertEqual(
                canonical_content_sha256(case),
                row["content_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
