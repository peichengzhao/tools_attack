"""Call-volume estimates for the AAAI-27 supplement pipeline."""

from __future__ import annotations


def estimate_coevolution_calls(cases: int, rounds: int) -> dict[str, int]:
    """Upper-bound logical model calls for one full co-evolution run."""
    victim = cases * rounds
    updates = cases * max(0, rounds - 1)
    return {
        "victim": victim,
        "updates": updates,
        "total": victim + updates,
    }


def estimate_frozen_attack_calls(cases: int) -> dict[str, int]:
    return {
        "rule_generation": cases,
        "victim": cases,
        "total": cases * 2,
    }


def estimate_harness_calls(cases: int, candidates: int) -> dict[str, int]:
    return {
        "victim": cases * candidates,
        "total": cases * candidates,
    }


def estimate_pipeline_calls(
    *,
    train_cases: int,
    heldout_cases: int,
    rounds: int,
    harness_candidates: int,
) -> dict[str, int]:
    train = estimate_coevolution_calls(train_cases, rounds)
    validation_attack = estimate_frozen_attack_calls(heldout_cases)
    harness = estimate_harness_calls(heldout_cases, harness_candidates)
    test_attack = estimate_frozen_attack_calls(heldout_cases)
    test_harness = estimate_harness_calls(heldout_cases, 1)
    total = (
        train["total"]
        + validation_attack["total"]
        + harness["total"]
        + test_attack["total"]
        + test_harness["total"]
    )
    return {
        "train": train["total"],
        "validation_attack": validation_attack["total"],
        "validation_harness": harness["total"],
        "test_attack": test_attack["total"],
        "test_harness": test_harness["total"],
        "total": total,
    }
