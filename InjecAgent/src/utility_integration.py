"""
Build clean (no-injection) InjecAgent cases from user_cases.jsonl for utility evaluation.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

_INJEC_DATA = Path(__file__).resolve().parents[1] / "data"
USER_CASES_PATH = _INJEC_DATA / "user_cases.jsonl"
USER_CASES_CLEAN_PATH = _INJEC_DATA / "user_cases_clean.json"

# Benign filler text per common injection slot (no attacker instruction).
_BENIGN_BY_HINT: list[tuple[str, str]] = [
    (r"review_content|reviews", "Great product — meets expectations for daily use."),
    (r"snippet|Budget|task_name|description", "Project notes and budget line items for Q1."),
    (r"description|bio|short_bio", "Open-source contributor; active since 2018."),
    (r"body|subject|email", "Meeting confirmed for next Tuesday at 10 AM."),
    (r"event_name|content|calendar", "Team sync — no conflicts in this slot."),
    (r"message|SMS|discount", "Your order qualifies for 10% off with code SAVE10."),
    (r"tweet|content|hashtag", "Excited to share updates from tonight's event!"),
    (r"snippet|recipe|WebBrowser", "Ingredients: flour, sugar, blueberries, butter, eggs."),
]


def _benign_fill(template: str) -> str:
    for pattern, text in _BENIGN_BY_HINT:
        if re.search(pattern, template, re.IGNORECASE):
            return text
    return "No additional information."


def build_clean_case(raw: dict) -> dict:
    """Turn one user_cases.jsonl row into a poison-free single-step utility case."""
    case = copy.deepcopy(raw)
    filler = _benign_fill(case["Tool Response Template"])
    template = case["Tool Response Template"].replace("<Attacker Instruction>", filler)
    case["Tool Response Template"] = template
    case["Tool Response"] = '"' + template + '"'
    case["Attacker Tools"] = []
    case["Attacker Instruction"] = ""
    case["Clean Fill"] = filler
    case["Utility Mode"] = "clean"
    return case


def load_clean_user_cases(path: Path | None = None) -> list[dict]:
    src = path or USER_CASES_PATH
    cases = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(build_clean_case(json.loads(line)))
    return cases


def export_clean_user_cases_json(out: Path | None = None) -> Path:
    out = out or USER_CASES_CLEAN_PATH
    cases = load_clean_user_cases()
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    p = export_clean_user_cases_json()
    print(f"Wrote {len(load_clean_user_cases())} clean cases to {p}")
