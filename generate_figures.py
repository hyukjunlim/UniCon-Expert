"""Generate high-resolution static images used by the annotation UIs."""

import argparse
import sys
from pathlib import Path


EXPERT_ANNOTATION_DIR = Path(__file__).resolve().parent
if str(EXPERT_ANNOTATION_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERT_ANNOTATION_DIR))

from common.figure_assets import DEFAULT_FIGURE_DIR, generate_csv_figures


WORKFLOWS = {
    "condition_preference": {
        "input": EXPERT_ANNOTATION_DIR / "condition_preference/data/annotation_inputs.csv",
        "fields": [
            ("condition_a", "condition_a_slots", "condition_a"),
            ("condition_b", "condition_b_slots", "condition_b"),
        ],
    },
    "training_flagging": {
        "input": EXPERT_ANNOTATION_DIR / "training_flagging/data/annotation_inputs.csv",
        "fields": [
            ("archived_condition", "archived_condition_slots", "archived_condition"),
        ],
    },
}


def generate_workflow(workflow, input_path=None, figure_dir=DEFAULT_FIGURE_DIR, overwrite=False):
    config = WORKFLOWS[workflow]
    row_count, failures = generate_csv_figures(
        input_path or config["input"],
        workflow,
        config["fields"],
        figure_dir,
        overwrite,
    )
    return row_count, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow", choices=["all", *WORKFLOWS], default="all"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Custom input CSV; only valid when one workflow is selected.",
    )
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.input is not None and args.workflow == "all":
        parser.error("--input requires a specific --workflow")

    selected = WORKFLOWS if args.workflow == "all" else [args.workflow]
    total_failures = []
    for workflow in selected:
        row_count, failures = generate_workflow(
            workflow, args.input, args.figure_dir, args.overwrite
        )
        total_failures.extend(failures)
        print(f"{workflow}: generated assets for {row_count} rows in {args.figure_dir}")
    if total_failures:
        preview = ", ".join(total_failures[:10])
        raise SystemExit(
            f"Could not render {len(total_failures)} value(s): {preview}"
        )


if __name__ == "__main__":
    main()
