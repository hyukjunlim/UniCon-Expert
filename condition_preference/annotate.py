# streamlit run annotate.py

import streamlit as st
import pandas as pd
import random
import os
import sys
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.csv_store import upsert_row
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
INPUT_DATA = str(DATA_DIR / "annotation_inputs.csv")
HUMAN_FILE = str(DATA_DIR / "human_annotations.csv")
# Set page config for a professional look
st.set_page_config(page_title="Reaction Condition Preference", layout="wide")

# Custom CSS for better UI
st.markdown(ANNOTATION_APP_CSS, unsafe_allow_html=True)

def show_condition_option(title, row, value_column, slot_column, figure_dir):
    """Display the pre-rendered image for each component in one option."""
    st.markdown(f"### {title}")
    assets = load_pre_rendered_condition_paths(
        row,
        "condition_preference",
        value_column,
        slot_column,
        value_column,
        figure_dir,
    )
    if not assets:
        st.caption("No condition SMILES")
        return
    # Keep every component at half-row width, including a lone component.
    columns = st.columns(2)
    missing = []
    for index, (path, smiles, role) in enumerate(assets):
        with columns[index % len(columns)]:
            if path.exists():
                st.markdown(
                    f'<div class="condition-component-image">'
                    f'<img src="{svg_data_uri(path)}" alt="{escape(smiles)}"></div>',
                    unsafe_allow_html=True,
                )
            else:
                missing.append(path)
                st.code(smiles)
            st.markdown(
                f'<div class="condition-component-label">'
                f'{escape(display_condition_slot(role))}: {escape(smiles)}</div>',
                unsafe_allow_html=True,
            )
    if missing:
        st.warning(
            "Missing pre-rendered figure(s). Run `python expert_annotation/generate_figures.py`."
        )

# Load dataset function
def load_data(file_path):
    """
    Loads the input dataset. Creates a dummy one if it doesn't exist for demo purposes.
    """
    if not os.path.exists(file_path):
        st.warning(f"Input file '{file_path}' not found. Loading dummy data for demonstration.")
        dummy_data = pd.DataFrame({
            'evaluation_id': [0, 1, 2],
            'source_index': [0, 1, 2],
            'reaction_smiles': [
                '[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][CH2:6][OH:7]>>[CH3:1][C:2](=[O:3])[O:4][CH2:6][CH3:5].[OH2:7]',
                'c1ccccc1.BrBr>>c1ccccc1Br.Br',
                'CC(=O)Cl.N>>CC(=O)N.Cl'
            ],
            'condition_a': [
                'H2SO4 (cat.), Ethanol, 80°C, 4h (Ground Truth)',
                'FeBr3, Br2, CHCl3, RT, 12h',
                'Et3N, DCM, 0°C to RT'
            ],
            'condition_b': [
                'HCl, Methanol, reflux, 10h (Baseline)',
                'AlCl3, Br2, No solvent, 50°C',
                'Pyridine, THF, 60°C'
            ]
        })
        return dummy_data
    
    try:
        df = pd.read_csv(file_path)
        required_cols = ['evaluation_id', 'source_index', 'reaction_smiles', 'condition_a', 'condition_b']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Dataset must contain: {required_cols}")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

def load_existing_annotations(file_path):
    """
    Loads previous annotations, including older files without slot columns.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)
    except pd.errors.ParserError:
        # Older annotation files may have fewer header columns than newer rows.
        # Only reaction_smiles is needed for resume, so read that field robustly.
        return pd.read_csv(file_path, usecols=["reaction_smiles"], engine="python")
    except Exception as e:
        st.warning(f"Could not read existing annotations for resume: {e}")
        return pd.DataFrame()

def find_resume_index(data, output_file):
    """
    Returns the first input row that has not already been annotated.
    """
    annotations = load_existing_annotations(output_file)
    if annotations.empty or "evaluation_id" not in annotations.columns:
        return 0

    completed = set(annotations["evaluation_id"].dropna().astype(int))
    return next(
        (index for index, value in enumerate(data["evaluation_id"]) if int(value) not in completed),
        len(data),
    )

def find_existing_annotation(annotations, evaluation_id):
    """
    Returns the latest saved annotation for a reaction, if present.
    """
    if annotations.empty or "evaluation_id" not in annotations.columns:
        return None

    matches = annotations[annotations["evaluation_id"].astype(int) == int(evaluation_id)]
    if matches.empty:
        return None

    return matches.iloc[-1].to_dict()

def get_annotation_value(annotation, column, default=""):
    if annotation is None or column not in annotation:
        return default

    value = annotation[column]
    if value is None or pd.isna(value):
        return default

    return value

def get_option_slots(row, option_text):
    """
    Reconstructs slot labels for older annotations that only saved option text.
    """
    if str(option_text) == str(row.get("condition_a", "")):
        return row.get("condition_a_slots", "")
    if str(option_text) == str(row.get("condition_b", "")):
        return row.get("condition_b_slots", "")
    return ""

def parse_saved_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}

def get_output_columns(file_path, default_columns):
    """
    Keeps existing columns while allowing the annotation schema to grow.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return default_columns

    try:
        existing_columns = pd.read_csv(file_path, nrows=0).columns.tolist()
        return existing_columns + [
            column for column in default_columns if column not in existing_columns
        ]
    except Exception:
        return default_columns

def save_annotation(file_path, annotation):
    """
    Saves one annotation, updating an existing row for the same reaction.
    """
    default_columns = list(annotation.keys())
    output_columns = get_output_columns(file_path, default_columns)

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        existing = load_existing_annotations(file_path)
    else:
        existing = pd.DataFrame(columns=output_columns)

    upsert_row(
        file_path,
        annotation,
        key_column="evaluation_id",
        existing=existing,
        columns=output_columns,
    )

# --- App State Initialization ---
if 'randomized_data' not in st.session_state:
    # We store the randomization per index to ensure consistency across re-runs 
    # for the same sample (e.g. if user resizes window)
    st.session_state.randomized_data = {} 

# --- Sidebar ---
st.sidebar.header("Data Settings")
input_file = st.sidebar.text_input("Input Data Path", INPUT_DATA)
output_file = st.sidebar.text_input("Output Annotations Path", HUMAN_FILE)
figure_dir = st.sidebar.text_input("Pre-rendered Figures", str(DEFAULT_FIGURE_DIR))

# Load data
data = load_data(input_file)
annotations = load_existing_annotations(output_file) if not data.empty else pd.DataFrame()

if data.empty:
    st.info("Please provide a valid CSV file with 'reaction_smiles', 'condition_a', and 'condition_b' columns.")
else:
    data_key = (os.path.abspath(input_file), os.path.abspath(output_file))
    if (
        'current_index' not in st.session_state
        or st.session_state.get('data_key') != data_key
    ):
        st.session_state.current_index = find_resume_index(data, output_file)
        st.session_state.randomized_data = {}
        st.session_state.data_key = data_key

if not data.empty and st.session_state.current_index < len(data):
    # Current Sample
    idx = st.session_state.current_index
    row = data.iloc[idx]
    saved_annotation = find_existing_annotation(annotations, row['evaluation_id'])
    saved_choice = get_annotation_value(saved_annotation, "user_choice", "")
    
    # Randomization Logic
    # We check if we already randomized this specific index
    if saved_annotation is not None:
        opt1 = get_annotation_value(saved_annotation, "shown_option_1", row['condition_a'])
        opt2 = get_annotation_value(saved_annotation, "shown_option_2", row['condition_b'])
        opt1_slots = get_annotation_value(
            saved_annotation,
            "shown_option_1_slots",
            get_option_slots(row, opt1),
        )
        opt2_slots = get_annotation_value(
            saved_annotation,
            "shown_option_2_slots",
            get_option_slots(row, opt2),
        )
        st.session_state.randomized_data[idx] = {
            'is_opt1_a': parse_saved_bool(get_annotation_value(saved_annotation, "is_option_1_GT", False)),
            'opt1_text': opt1,
            'opt2_text': opt2,
            'opt1_slots': opt1_slots,
            'opt2_slots': opt2_slots
        }
    elif idx not in st.session_state.randomized_data:
        # Condition A is typically the Ground Truth / reference in the input
        # Condition B is typically the Baseline / prediction
        condition_a_slots = row.get('condition_a_slots', '')
        condition_b_slots = row.get('condition_b_slots', '')
        is_opt1_a = random.choice([True, False])
        if is_opt1_a:
            opt1, opt2 = row['condition_a'], row['condition_b']
            opt1_slots, opt2_slots = condition_a_slots, condition_b_slots
        else:
            opt1, opt2 = row['condition_b'], row['condition_a']
            opt1_slots, opt2_slots = condition_b_slots, condition_a_slots
        
        # Store metadata for internal tracking
        st.session_state.randomized_data[idx] = {
            'is_opt1_a': is_opt1_a,
            'opt1_text': opt1,
            'opt2_text': opt2,
            'opt1_slots': opt1_slots,
            'opt2_slots': opt2_slots
        }
    
    r_data = st.session_state.randomized_data[idx]
    is_opt1_a = r_data['is_opt1_a']
    opt1_text = r_data['opt1_text']
    opt2_text = r_data['opt2_text']
    opt1_slots = r_data.get('opt1_slots', '')
    opt2_slots = r_data.get('opt2_slots', '')
    opt1_column = "condition_a" if is_opt1_a else "condition_b"
    opt2_column = "condition_b" if is_opt1_a else "condition_a"

    # --- UI Header ---
    st.title("🧪 Chemical Reaction Condition Preference")
    
    # Progress
    progress = (idx) / len(data)
    st.progress(progress)
    st.write(f"**Progress:** {idx} / {len(data)} annotated ({int(progress*100)}%)")

    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    if nav_col1.button("< Previous", disabled=idx == 0, width="stretch"):
        st.session_state.current_index -= 1
        st.rerun()

    nav_col2.write(f"**Sample:** {idx + 1} / {len(data)}")

    if nav_col3.button("Next >", disabled=idx >= len(data) - 1, width="stretch"):
        st.session_state.current_index += 1
        st.rerun()

    # --- Reaction Image ---
    st.subheader("Reaction SMILES")
    reaction_path = reaction_figure_path(
        figure_dir,
        "condition_preference",
        row["evaluation_id"],
        row["reaction_smiles"],
    )
    if reaction_path.exists():
        # Use more width for the reaction image
        st.image(str(reaction_path), width="stretch")
    else:
        st.error(
            "Pre-rendered reaction figure is missing. Run "
            "`python expert_annotation/generate_figures.py`."
        )
        st.code(row['reaction_smiles'])

    st.divider()

    # --- Reagent Choices ---
    st.subheader("Which condition set do you prefer?")
    col_left, col_right = st.columns(2)
    
    with col_left:
        show_condition_option(
            "Option 1", row, opt1_column, f"{opt1_column}_slots", figure_dir
        )
        
    with col_right:
        show_condition_option(
            "Option 2", row, opt2_column, f"{opt2_column}_slots", figure_dir
        )

    # --- Buttons Section ---
    st.markdown("---")
    st.markdown("### Decision")
    if saved_choice:
        st.success(f"Saved choice for this sample: {saved_choice}")
    else:
        st.info("No saved choice for this sample yet.")

    saved_notes = str(get_annotation_value(saved_annotation, "notes", ""))
    notes_key = f"notes_{int(row['evaluation_id'])}"
    
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    def handle_click(choice):
        annotation = {
            'evaluation_id': int(row['evaluation_id']),
            'source_index': int(row['source_index']),
            'reaction_smiles': row['reaction_smiles'],
            'shown_option_1': opt1_text,
            'shown_option_1_slots': opt1_slots,
            'shown_option_2': opt2_text,
            'shown_option_2_slots': opt2_slots,
            'user_choice': choice,
            'is_option_1_GT': is_opt1_a,
            'notes': st.session_state.get(notes_key, saved_notes),
        }
        
        save_annotation(output_file, annotation)
        
        # Increment index
        st.session_state.current_index = min(st.session_state.current_index + 1, len(data))

    if btn_col1.button("Prefer Option 1", width="stretch", type="primary" if saved_choice == "Option 1" else "secondary"):
        handle_click("Option 1")
        st.rerun()
        
    if btn_col2.button("Prefer Option 2", width="stretch", type="primary" if saved_choice == "Option 2" else "secondary"):
        handle_click("Option 2")
        st.rerun()
        
    if btn_col3.button("Cannot determine", width="stretch", type="primary" if saved_choice == "Cannot determine" else "secondary"):
        handle_click("Cannot determine")
        st.rerun()

    st.text_area(
        "Notes",
        value=saved_notes,
        placeholder="If you choose Cannot determine, briefly explain why.",
        key=notes_key,
    )

elif not data.empty:
    # All samples completed
    st.balloons()
    st.success(f"### 🎉 All {len(data)} samples have been annotated!")
    st.write(f"Results saved to: `{output_file}`")

    if st.button("< Previous sample", disabled=len(data) == 0):
        st.session_state.current_index = max(len(data) - 1, 0)
        st.rerun()
    
    # Summary of current annotations
    if os.path.exists(output_file):
        results_df = pd.read_csv(output_file)
        st.write("### Current Results Summary")
        st.dataframe(results_df.tail(10))
        
        # Download button
        csv_download = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Annotations as CSV",
            data=csv_download,
            file_name="annotations_export.csv",
            mime="text/csv",
        )

    if st.button("Restart Session"):
        st.session_state.current_index = 0
        st.session_state.randomized_data = {}
        st.rerun()
