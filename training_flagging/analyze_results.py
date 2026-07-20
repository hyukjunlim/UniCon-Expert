"""Join blinded responses with sampling metadata and summarize flagging evaluation."""

import argparse
import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"


def annotation_issues(row):
    value = row.get("component_annotations")
    if value is not None and not pd.isna(value):
        try:
            return [item["issue"] for item in json.loads(value)]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            pass
    legacy = row.get("annotation_issues", "")
    return [] if pd.isna(legacy) else [part for part in str(legacy).split(";") if part]


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

    completed = results[results["overall_assessment"].notna()].copy()
    print(f"Completed: {len(completed)}/{len(results)}")
    if completed.empty:
        return
    print("\nOverall assessment by hidden sampling group:")
    print(pd.crosstab(completed["sampling_group"], completed["overall_assessment"], margins=True))
    completed["issues"] = completed.apply(annotation_issues, axis=1)
    completed["any_annotation_issue"] = completed["issues"].apply(
        lambda values: "Missing reagent" in values or "Misassigned reagent" in values
    )
    completed["questionable_archive"] = (
        completed["overall_assessment"] == "Archived protocol is incomplete/questionable"
    )
    print("\nPrimary flagging rates by hidden sampling group:")
    print(
        completed.groupby("sampling_group")[["any_annotation_issue", "questionable_archive"]]
        .agg(["sum", "count", "mean"])
    )
    for issue in ("Missing reagent", "Misassigned reagent", "No obvious annotation issue"):
        selected = completed["issues"].apply(lambda values: issue in values)
        rates = selected.groupby(completed["sampling_group"]).agg(["sum", "count", "mean"])
        print(f"\n{issue}:")
        print(rates)
    print(f"\nJoined row-level results: {args.output}")


if __name__ == "__main__":
    main()
