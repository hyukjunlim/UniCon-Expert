"""Streamlit UI for evaluating UniCon training-data flags."""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.csv_store import read_csv_if_exists, upsert_row
from common.reaction_rendering import render_reaction
from common.ui_style import (
    ANNOTATION_APP_CSS,
    CONDITION_IMAGE_SIZE,
    CONDITION_LEGEND_FONT_SIZE,
    CONDITION_MOLECULE_IMAGE_SIZE,
    IMAGE_RENDER_SCALE,
    display_condition_slot,
    fit_image_to_canvas,
    scale_size,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_INPUT = DATA_DIR / "annotation_inputs.csv"
DEFAULT_OUTPUT = DATA_DIR / "human_annotations.csv"
ISSUE_OPTIONS = ["Missing reagent", "Misassigned reagent", "No obvious annotation issue"]
ASSESSMENT_OPTIONS = [
    "Archived protocol is plausible",
    "Archived protocol is implausible",
    "Cannot determine",
]

st.set_page_config(page_title="Training-set flag evaluation", layout="wide")
st.markdown(ANNOTATION_APP_CSS, unsafe_allow_html=True)


def render_conditions(value, slots):
    if pd.isna(value) or str(value).strip().lower() in {"", "none", "nan"}:
        return None, []
    slot_values = [] if pd.isna(slots) else [part.strip() for part in str(slots).split(",")]
    molecules, legends, invalid = [], [], []
    for index, smiles in enumerate(part.strip() for part in str(value).split(",") if part.strip()):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            invalid.append(smiles)
            continue
        molecules.append(molecule)
        role = display_condition_slot(slot_values[index]) if index < len(slot_values) else ""
        legends.append(f"{smiles}\n{role}" if role else smiles)
    if not molecules:
        return None, invalid
    draw_options = rdMolDraw2D.MolDrawOptions()
    draw_options.legendFontSize = CONDITION_LEGEND_FONT_SIZE * IMAGE_RENDER_SCALE
    return Draw.MolsToGridImage(
        molecules,
        molsPerRow=min(3, len(molecules)),
        subImgSize=scale_size(CONDITION_MOLECULE_IMAGE_SIZE),
        legends=legends,
        drawOptions=draw_options,
    ), invalid


def condition_components(value, slots):
    values = [] if pd.isna(value) or str(value).strip().lower() in {"", "none", "nan"} else [
        part.strip() for part in str(value).split(",") if part.strip()
    ]
    roles = [] if pd.isna(slots) else [part.strip() for part in str(slots).split(",")]
    if not values:
        return [{"smiles": "No listed component", "slot": "Archived protocol"}]
    return [
        {"smiles": smiles, "slot": roles[index] if index < len(roles) else "Condition component"}
        for index, smiles in enumerate(values)
    ]


def saved_component_issues(saved):
    if saved is None or "component_annotations" not in saved or pd.isna(saved["component_annotations"]):
        return {}
    try:
        annotations = json.loads(saved["component_annotations"])
        return {int(item["component_index"]): item["issue"] for item in annotations}
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return {}


def save_response(path, response):
    upsert_row(path, response, key_column="evaluation_id")


st.sidebar.header("Data settings")
input_path = st.sidebar.text_input("Input data", str(DEFAULT_INPUT))
output_path = st.sidebar.text_input("Human annotations", str(DEFAULT_OUTPUT))

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

st.title("Training-set annotation review")
st.progress(index / len(data))
st.write(f"Sample {index + 1} of {len(data)}")
left, _, right = st.columns([1, 4, 1])
if left.button("Previous", disabled=index == 0):
    st.session_state.current_index -= 1
    st.rerun()
if right.button("Next", disabled=index == len(data) - 1):
    st.session_state.current_index += 1
    st.rerun()

reaction_image = render_reaction(row["reaction_smiles"])
if reaction_image is not None:
    st.image(reaction_image, width="stretch", caption=row["reaction_smiles"])
else:
    st.code(row["reaction_smiles"])

st.subheader("Archived protocol")
image, invalid = render_conditions(row["archived_condition"], row.get("archived_condition_slots", ""))
if image is not None:
    st.image(fit_image_to_canvas(image), width=CONDITION_IMAGE_SIZE[0])
else:
    st.write(row["archived_condition"])
if invalid:
    st.caption(f"Could not render: {', '.join(invalid)}")

archived_components = condition_components(row["archived_condition"], row.get("archived_condition_slots", ""))
previous_issues = saved_component_issues(saved)
st.markdown("### A. Annotation issue for each archived component")
component_annotations = []
for component_index, component in enumerate(archived_components):
    previous_issue = previous_issues.get(component_index, "No obvious annotation issue")
    display_slot = display_condition_slot(component["slot"])
    st.markdown(f"**{display_slot}: {component['smiles']}**")
    issue = st.segmented_control(
        "Annotation issue",
        ISSUE_OPTIONS,
        default=previous_issue if previous_issue in ISSUE_OPTIONS else ISSUE_OPTIONS[2],
        key=f"issue_{evaluation_id}_{component_index}",
        label_visibility="collapsed",
        width="stretch",
    )
    component_annotations.append(
        {
            "component_index": component_index,
            "slot": component["slot"],
            "smiles": component["smiles"],
            "issue": issue,
        }
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
notes = st.text_area("Optional notes", value="" if saved is None or pd.isna(saved.get("notes")) else str(saved["notes"]))

if st.button("Save and continue", type="primary"):
    error = None
    if assessment is None:
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
                "component_annotations": json.dumps(component_annotations),
                "overall_assessment": assessment,
                "notes": notes,
            },
        )
        st.session_state.current_index = min(index + 1, len(data))
        st.rerun()
