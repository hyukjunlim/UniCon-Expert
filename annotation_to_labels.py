import argparse
import csv
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
HUMAN_PATTERN = "annotation_human_*.csv"
LABEL_PATTERN = "expert_labels_*.json"
CHOICE_TO_LABELS = {
    "tie": "tie",
    "bad": "neither",
    "invalid reaction": "invalid_reaction",
    "invalid_reaction": "invalid_reaction",
}
RANGE_RE = re.compile(r"_(\d+)_(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert annotation_human_*.csv files to expert_labels_*.json files "
            "and merge expert label JSON files into expert_labels.json."
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help="Specific annotation_human_*.csv files to convert. Defaults to all in this directory.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory containing annotation CSVs and label JSONs.",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert CSVs; do not write the merged expert_labels.json.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing expert_labels_*.json files; do not convert CSVs.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write a JSON even when the CSV has fewer rows than its filename range.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing expert_labels_*.json files.",
    )
    return parser.parse_args()


def parse_range(path):
    match = RANGE_RE.search(path.stem)
    if not match:
        return 0, None
    start, end = (int(value) for value in match.groups())
    return start, end


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


def load_existing_label_count(path):
    if not path.exists():
        return None
    with path.open() as f:
        return len(json.load(f))


def convert_csv(csv_path, allow_partial=False, force=False):
    start, end = parse_range(csv_path)
    expected_count = None if end is None else end - start
    output_path = csv_path.with_name(csv_path.name.replace("annotation_human", "expert_labels")).with_suffix(
        ".json"
    )

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{csv_path}: no annotation rows found")

    if expected_count is not None and len(rows) != expected_count:
        existing_count = load_existing_label_count(output_path)
        if existing_count == expected_count and not force:
            return output_path, "skipped partial CSV; existing complete JSON kept"
        if not allow_partial and not force:
            raise ValueError(
                f"{csv_path}: expected {expected_count} rows from filename range, found {len(rows)}. "
                "Use --allow-partial to write it anyway or --force to overwrite."
            )

    labels = {str(start + index): label_from_row(row, index + 2) for index, row in enumerate(rows)}

    if output_path.exists() and not force:
        existing_count = load_existing_label_count(output_path)
        if existing_count is not None and existing_count > len(labels):
            return output_path, "skipped; existing JSON has more labels"

    with output_path.open("w") as f:
        json.dump(labels, f, indent=2)
        f.write("\n")

    return output_path, f"wrote {len(labels)} labels"


def numeric_label_items(path):
    with path.open() as f:
        labels = json.load(f)

    for key, value in labels.items():
        try:
            numeric_key = int(key)
        except ValueError as exc:
            raise ValueError(f"{path}: non-numeric label key {key!r}") from exc
        yield numeric_key, str(key), value


def merge_labels(annotation_dir):
    merged = {}
    sources = {}

    for path in sorted(annotation_dir.glob(LABEL_PATTERN), key=lambda p: parse_range(p)[0]):
        if path.name == "expert_labels.json":
            continue
        for numeric_key, key, value in numeric_label_items(path):
            if key in merged and merged[key] != value:
                raise ValueError(
                    f"Conflicting label for index {key}: {sources[key]} has {merged[key]!r}, "
                    f"{path} has {value!r}"
                )
            merged[key] = value
            sources[key] = str(path)

    output_path = annotation_dir / "expert_labels.json"
    ordered = {str(key): merged[str(key)] for key in sorted(int(key) for key in merged)}
    with output_path.open("w") as f:
        json.dump(ordered, f, indent=2)
        f.write("\n")
    return output_path, len(ordered)


def main():
    args = parse_args()
    annotation_dir = args.annotation_dir.resolve()

    if args.convert_only and args.merge_only:
        raise SystemExit("--convert-only and --merge-only cannot be used together")

    if not args.merge_only:
        csv_files = args.csv_files or sorted(annotation_dir.glob(HUMAN_PATTERN), key=lambda p: parse_range(p)[0])
        for csv_path in csv_files:
            output_path, status = convert_csv(
                csv_path.resolve(),
                allow_partial=args.allow_partial,
                force=args.force,
            )
            print(f"{output_path.name}: {status}")

    if not args.convert_only:
        output_path, count = merge_labels(annotation_dir)
        print(f"{output_path.name}: wrote {count} labels")


if __name__ == "__main__":
    main()
