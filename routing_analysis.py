"""Check saved routing decisions and compare the gates with a paired bootstrap."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from prepare import ROOT
from run_experiment import curve

POLICIES = ("learned_gate", "score_only_gate", "prob_margin")
SEED = 42
REPLICATES = 10_000


def analyze(rows, thresholds, rng):
    index = [int(row["row"]) for row in rows]
    if index != list(range(len(index))) or not index:
        raise ValueError("Expected one prediction per test row in test order")
    errors = np.asarray([row["fusion"] != row["gold"] for row in rows])
    scores = {name: np.asarray([float(row[name]) for row in rows]) for name in POLICIES}
    accepted = {}
    counts = {}
    for name in POLICIES:
        threshold = thresholds[name]
        accepted[name] = scores[name] >= threshold if threshold is not None else np.zeros_like(errors)
        saved = np.asarray([row[f"{name}_auto"] == "1" for row in rows])
        np.testing.assert_array_equal(accepted[name], saved)
        counts[name] = {"accepted": int(saved.sum()), "errors": int(np.sum(saved & errors))}

    first, second = POLICIES[:2]
    coverage_delta = accepted[first].astype(int) - accepted[second]
    draws = np.empty((REPLICATES, 2))
    for i in range(REPLICATES):
        sample = rng.integers(len(rows), size=len(rows))
        sampled_errors = errors[sample]
        draws[i, 0] = coverage_delta[sample].mean()
        draws[i, 1] = curve(scores[first][sample], sampled_errors).mean() - curve(
            scores[second][sample], sampled_errors,
        ).mean()
    intervals = np.percentile(draws, [2.5, 97.5], axis=0).T

    return {
        "rows": len(rows),
        "counts": counts,
        "margin_vs_gate_without_neighbors": {
            "margin_only_accepted": int(np.sum(accepted["prob_margin"] & ~accepted[second])),
            "gate_only_accepted": int(np.sum(accepted[second] & ~accepted["prob_margin"])),
            "unequal_scores": int(np.sum(scores["prob_margin"] != scores[second])),
        },
        "coverage_delta": {
            "estimate": float(coverage_delta.mean()), "ci95": intervals[0].tolist(),
        },
        "aurc_delta": {
            "estimate": float(curve(scores[first], errors).mean() - curve(scores[second], errors).mean()),
            "ci95": intervals[1].tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    metrics = json.loads((args.results / "metrics.json").read_text(encoding="utf-8"))
    with (args.results / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    rng = np.random.default_rng(SEED)
    result = {"seed": SEED, "replicates": REPLICATES, "test": {}}
    for name in ("original", "wrapped"):
        rows = [row for row in predictions if row["variant"] == name]
        analysis = analyze(rows, metrics["thresholds"], rng)
        if analysis["rows"] != metrics["data"][name]["rows"]:
            raise ValueError(f"Unexpected prediction count for {name}")
        for policy, counts in analysis["counts"].items():
            expected = metrics["test"][name]["routing"][policy]
            if any(expected[key] != value for key, value in counts.items()):
                raise ValueError(f"Routing counts disagree for {name}/{policy}")
        result["test"][name] = analysis
    (args.results / "routing_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
