"""Streamlit UI for evaluating UniCon training-data flags."""

import json
import os
import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.csv_store import read_csv_if_exists, upsert_row
from common.figure_assets import (
    DEFAULT_FIGURE_DIR,
    load_pre_rendered_condition_paths,
    reaction_figure_path,
    svg_data_uri,
)
from common.ui_style import (
    ANNOTATION_APP_CSS,
    display_condition_slot,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_INPUT = DATA_DIR / "annotation_inputs.csv"
DEFAULT_OUTPUT = DATA_DIR / "human_annotations.csv"
ISSUE_OPTIONS = ["Missing agent", "Misassigned agent", "No obvious annotation issue"]
NO_ISSUE_OPTION = "No obvious annotation issue"
ASSESSMENT_OPTIONS = [
    "Archived protocol is plausible",
    "Archived protocol is implausible",
    "Cannot determine",
]

st.set_page_config(page_title="Training-set flag evaluation", layout="wide")
st.markdown(ANNOTATION_APP_CSS, unsafe_allow_html=True)


def saved_reaction_issues(saved):
    """Load the current reaction-level response or migrate an older per-component one."""
    if saved is None:
        return []
    value = saved.get("annotation_issues")
    if value is not None and not pd.isna(value):
        return [issue for issue in str(value).split(";") if issue in ISSUE_OPTIONS]
    value = saved.get("component_annotations")
    if value is None or pd.isna(value):
        return []
    try:
        issues = [item["issue"] for item in json.loads(value)]
        return list(dict.fromkeys(issue for issue in issues if issue in ISSUE_OPTIONS))
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return []


def normalize_issue_selection(issues):
    """Keep the no-issue choice mutually exclusive from actual issue choices."""
    issues = [issue for issue in issues if issue in ISSUE_OPTIONS]
    actual_issues = [issue for issue in issues if issue != NO_ISSUE_OPTION]
    if actual_issues:
        return actual_issues
    return [NO_ISSUE_OPTION] if NO_ISSUE_OPTION in issues else []


def resolve_issue_selection(selected, previous):
    """Resolve a multi-select click while preserving the intended exclusivity."""
    selected = list(selected)
    added = set(selected) - set(previous)
    if NO_ISSUE_OPTION in added:
        return [NO_ISSUE_OPTION]
    if NO_ISSUE_OPTION in selected and added:
        selected = [issue for issue in selected if issue != NO_ISSUE_OPTION]
    return normalize_issue_selection(selected)


def enforce_issue_exclusivity(widget_key, previous_key):
    """Immediately clear the logically incompatible selection after a click."""
    selected = list(st.session_state.get(widget_key, []))
    previous = list(st.session_state.get(previous_key, []))
    selected = resolve_issue_selection(selected, previous)
    st.session_state[widget_key] = selected
    st.session_state[previous_key] = selected


def save_response(path, response):
    upsert_row(path, response, key_column="evaluation_id")


st.sidebar.header("Data settings")
input_path = st.sidebar.text_input("Input data", str(DEFAULT_INPUT))
output_path = st.sidebar.text_input("Human annotations", str(DEFAULT_OUTPUT))
figure_dir = st.sidebar.text_input("Pre-rendered figures", str(DEFAULT_FIGURE_DIR))

data = read_csv_if_exists(input_path)
responses = read_csv_if_exists(output_path)
required = {"evaluation_id", "source_index", "reaction_smiles", "archived_condition", "candidate_condition"}
if data.empty or not required.issubset(data.columns):
    st.error(f"Input must be a non-empty CSV containing: {', '.join(sorted(required))}")
    st.stop()

data = data.sort_values("evaluation_id").reset_index(drop=True)
completed = set(responses["evaluation_id"].astype(int)) if not responses.empty else set()
data_key = (os.path.abspath(input_path), os.path.abspath(output_path))
if "current_index" not in st.session_state or st.session_state.get("data_key") != data_key:
    st.session_state.current_index = next((i for i, value in enumerate(data["evaluation_id"]) if int(value) not in completed), len(data))
    st.session_state.data_key = data_key

if st.session_state.current_index >= len(data):
    st.success(f"All {len(data)} reactions have been annotated. Results: {output_path}")
    if st.button("Review previous"):
        st.session_state.current_index = len(data) - 1
        st.rerun()
    st.stop()

index = st.session_state.current_index
row = data.iloc[index]
evaluation_id = int(row["evaluation_id"])
if responses.empty or "evaluation_id" not in responses:
    saved = None
else:
    saved_rows = responses.loc[responses["evaluation_id"].astype(int) == evaluation_id]
    saved = saved_rows.iloc[-1] if not saved_rows.empty else None

st.title("🧪 Reaction-Condition Dataset Flagging")
st.progress(index / len(data))
left, center, right = st.columns([1, 2, 1])
if left.button("< Previous", disabled=index == 0, width="stretch"):
    st.session_state.current_index -= 1
    st.rerun()
center.write(f"**Sample:** {index + 1} / {len(data)}")
if right.button("Next >", disabled=index == len(data) - 1, width="stretch"):
    st.session_state.current_index += 1
    st.rerun()

reaction_path = reaction_figure_path(
    figure_dir,
    "training_flagging",
    evaluation_id,
    row["reaction_smiles"],
)
if reaction_path.exists():
    st.image(str(reaction_path), width="stretch")
else:
    st.warning(
        "Pre-rendered reaction figure is missing. Run "
        "`python expert_annotation/generate_figures.py`."
    )
    st.code(row["reaction_smiles"])

st.subheader("Archived protocol")
condition_assets = load_pre_rendered_condition_paths(
    row,
    "training_flagging",
    "archived_condition",
    "archived_condition_slots",
    "archived_condition",
    figure_dir,
)
condition_columns = st.columns(min(2, max(1, len(condition_assets))))
missing_assets = []
if not condition_assets:
    st.caption("No listed condition components")
for component_index, (path, smiles, role) in enumerate(condition_assets):
    with condition_columns[component_index % len(condition_columns)]:
        if path.exists():
            st.markdown(
                f'<div class="condition-component-image">'
                f'<img src="{svg_data_uri(path)}" alt="{escape(smiles)}"></div>',
                unsafe_allow_html=True,
            )
        else:
            missing_assets.append(path)
            st.code(smiles)
        st.markdown(
            f'<div class="condition-component-label">'
            f'{escape(display_condition_slot(role))}: {escape(smiles)}</div>',
            unsafe_allow_html=True,
        )
if missing_assets:
    st.warning(
        "Missing pre-rendered condition figure(s). Run "
        "`python expert_annotation/generate_figures.py`."
    )

previous_issues = normalize_issue_selection(saved_reaction_issues(saved))
issues_key = f"issues_{evaluation_id}"
issues_previous_key = f"issues_previous_{evaluation_id}"
if issues_previous_key not in st.session_state:
    st.session_state[issues_previous_key] = previous_issues
st.markdown("### A. Annotation issues for this reaction")
selected_issues = st.segmented_control(
    "Annotation issues",
    ISSUE_OPTIONS,
    selection_mode="multi",
    default=previous_issues,
    key=issues_key,
    on_change=enforce_issue_exclusivity,
    args=(issues_key, issues_previous_key),
    label_visibility="collapsed",
    width="stretch",
)

default_assessment = None if saved is None else str(saved["overall_assessment"])
st.markdown("### B. Overall assessment")
assessment = st.segmented_control(
    "Overall assessment",
    ASSESSMENT_OPTIONS,
    default=default_assessment if default_assessment in ASSESSMENT_OPTIONS else None,
    key=f"assessment_{evaluation_id}",
    label_visibility="collapsed",
    width="stretch",
)
notes = st.text_area(
    "Notes",
    value="" if saved is None or pd.isna(saved.get("notes")) else str(saved["notes"]),
    placeholder="If you choose implausible, briefly explain why.",
)

if st.button("Save and continue", type="primary"):
    error = None
    if not selected_issues:
        error = "Select at least one annotation-issue response."
    elif NO_ISSUE_OPTION in selected_issues and len(selected_issues) > 1:
        error = "No obvious annotation issue cannot be combined with another issue."
    elif assessment is None:
        error = "Select an overall assessment."
    if error:
        st.error(error)
    else:
        save_response(
            output_path,
            {
                "evaluation_id": evaluation_id,
                "source_index": int(row["source_index"]),
                "reaction_smiles": row["reaction_smiles"],
                "annotation_issues": ";".join(selected_issues),
                "overall_assessment": assessment,
                "notes": notes,
            },
        )
        st.session_state.current_index = min(index + 1, len(data))
        st.rerun()
