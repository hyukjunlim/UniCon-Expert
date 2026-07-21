"""Join blinded responses with sampling metadata and summarize flagging evaluation."""

import argparse
import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"


def annotation_issues(row):
    current = row.get("annotation_issues")
    if current is not None and not pd.isna(current):
        return [
            "Artifact agent" if part == "Misassigned agent" else part
            for part in str(current).split(";")
            if part
        ]
    value = row.get("component_annotations")
    if value is not None and not pd.isna(value):
        try:
            return [
                "Artifact agent" if item["issue"] == "Misassigned agent" else item["issue"]
                for item in json.loads(value)
            ]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            pass
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA_DIR / "annotation_inputs.csv")
    parser.add_argument("--key", type=Path, default=DATA_DIR / "annotation_inputs_key.csv")
    parser.add_argument("--annotations", type=Path, default=DATA_DIR / "human_annotations.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "analysis_results.csv")
    args = parser.parse_args()

    samples = pd.read_csv(args.input)
    key = pd.read_csv(args.key)
    annotations = pd.read_csv(args.annotations)
    samples = samples.merge(key, on=["evaluation_id", "source_index"], validate="one_to_one")
    results = samples.merge(
        annotations,
        on=["evaluation_id", "source_index", "reaction_smiles"],
        how="left",
        validate="one_to_one",
    )
    results.to_csv(args.output, index=False)

    results["issues"] = results.apply(annotation_issues, axis=1)
    completed = results[results["issues"].apply(bool)].copy()
    print(f"Completed: {len(completed)}/{len(results)}")
    if completed.empty:
        return
    completed["any_annotation_issue"] = completed["issues"].apply(
        lambda values: "Missing agent" in values or "Artifact agent" in values
    )
    print("\nAnnotation-issue rates by hidden sampling group:")
    print(
        completed.groupby("sampling_group")[["any_annotation_issue"]]
        .agg(["sum", "count", "mean"])
    )
    for issue in ("Missing agent", "Artifact agent", "No obvious annotation issue"):
        selected = completed["issues"].apply(lambda values: issue in values)
        rates = selected.groupby(completed["sampling_group"]).agg(["sum", "count", "mean"])
        print(f"\n{issue}:")
        print(rates)
    print(f"\nJoined row-level results: {args.output}")


if __name__ == "__main__":
    main()
