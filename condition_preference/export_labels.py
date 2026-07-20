"""Convert randomized condition-preference responses to source-index labels."""

import argparse
import csv
import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
CHOICE_TO_LABELS = {
    "cannot determine": "cannot_determine",
}


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def label_from_row(row, row_number):
    choice = row.get("user_choice", "").strip()
    choice_key = choice.lower()
    if choice_key in CHOICE_TO_LABELS:
        return CHOICE_TO_LABELS[choice_key]

    is_option_1_gt = parse_bool(row.get("is_option_1_GT", ""))
    if choice_key == "option 1":
        return "gt" if is_option_1_gt else "baseline"
    if choice_key == "option 2":
        return "baseline" if is_option_1_gt else "gt"
    raise ValueError(f"Row {row_number}: unsupported user_choice {choice!r}")


def export_labels(annotations_path, output_path, allow_partial=False):
    with annotations_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{annotations_path}: no annotation rows found")

    required = {"source_index", "user_choice", "is_option_1_GT"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{annotations_path}: missing columns: {', '.join(sorted(missing))}")

    if not allow_partial:
        inputs_path = annotations_path.with_name("annotation_inputs.csv")
        if inputs_path.exists():
            with inputs_path.open(newline="") as handle:
                expected = sum(1 for _ in csv.DictReader(handle))
            if len(rows) != expected:
                raise ValueError(
                    f"Expected {expected} completed annotations, found {len(rows)}. "
                    "Use --allow-partial to export an incomplete evaluation."
                )

    labels = {}
    for row_number, row in enumerate(rows, start=2):
        source_index = str(int(row["source_index"]))
        label = label_from_row(row, row_number)
        if source_index in labels and labels[source_index] != label:
            raise ValueError(f"Conflicting labels for test index {source_index}")
        labels[source_index] = label

    ordered = {str(index): labels[str(index)] for index in sorted(map(int, labels))}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(ordered, handle, indent=2)
        handle.write("\n")
    return len(ordered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DATA_DIR / "human_annotations.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "expert_labels.json")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    count = export_labels(args.annotations, args.output, args.allow_partial)
    print(f"Wrote {count} labels to {args.output}")


if __name__ == "__main__":
    main()
