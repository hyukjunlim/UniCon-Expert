# streamlit run annotation_run.py

import streamlit as st
import pandas as pd
import random
from datetime import datetime
import os
import sys
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont
import io

DEFAULT_START = 0
DEFAULT_END = 50

def get_range_from_args(default_start=DEFAULT_START, default_end=DEFAULT_END):
    numeric_args = []
    for arg in sys.argv[1:]:
        try:
            numeric_args.append(int(arg))
        except ValueError:
            continue

    if len(numeric_args) >= 2:
        return numeric_args[0], numeric_args[1]
    return default_start, default_end

START, END = get_range_from_args()
suffix = f"_{START}_{END}"
INPUT_DATA = f"annotation_input_data{suffix}.csv"
HUMAN_FILE = f"annotation_human{suffix}.csv"
IMAGE_RENDER_SCALE = 2
REACTION_MOLECULE_IMAGE_SIZE = (500, 300)
REACTION_SEPARATOR_WIDTH = 54
REACTION_ARROW_WIDTH = 105
REACTION_CANVAS_PADDING = 18
CONDITION_IMAGE_SIZE = (500, 300)
CONDITION_MOLECULE_IMAGE_SIZE = (180, 180)
CONDITION_LEGEND_FONT_SIZE = 50
CONDITION_SLOT_DISPLAY_LABELS = {
    "cat": "Catalyst",
    "catalyst": "Catalyst",
    "solv0": "Solvent",
    "solv1": "Solvent",
    "solv2": "Solvent",
    "solvent": "Solvent",
    "reag0": "Reagent",
    "reag1": "Reagent",
    "reag2": "Reagent",
    "reag3": "Reagent",
    "reagent": "Reagent",
}

# Set page config for a professional look
st.set_page_config(page_title="Reaction Condition Preference", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
    .stButton > button {
        height: 3em;
        font-weight: bold;
    }
    .option-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #f9f9f9;
        min-height: 100px;
        text-align: center;
        font-size: 1.2em;
    }
    </style>
""", unsafe_allow_html=True)

def split_reaction_smiles(smiles):
    """
    Splits reaction SMILES into reactant and product component SMILES.
    """
    parts = smiles.split(">")
    if len(parts) == 2:
        reactants, products = parts
    elif len(parts) == 3:
        reactants, _, products = parts
    else:
        return [], []

    return (
        [part for part in reactants.split(".") if part],
        [part for part in products.split(".") if part],
    )

def load_reaction_font(size):
    """
    Loads a readable separator font, falling back to PIL's default if needed.
    """
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()

def scale_size(size, scale=IMAGE_RENDER_SCALE):
    return tuple(int(value * scale) for value in size)

def scale_value(value, scale=IMAGE_RENDER_SCALE):
    return int(value * scale)

def draw_centered_text(draw, box, text, font, fill="black"):
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    x = box[0] + (box[2] - box[0] - text_width) // 2
    y = box[1] + (box[3] - box[1] - text_height) // 2
    draw.text((x, y), text, font=font, fill=fill)

def draw_reaction_arrow(draw, box):
    mid_y = (box[1] + box[3]) // 2
    start_x = box[0] + scale_value(12)
    end_x = box[2] - scale_value(14)
    draw.line((start_x, mid_y, end_x, mid_y), fill="black", width=scale_value(3))
    draw.polygon(
        [
            (end_x, mid_y),
            (end_x - scale_value(14), mid_y - scale_value(8)),
            (end_x - scale_value(14), mid_y + scale_value(8)),
        ],
        fill="black",
    )

def render_reaction_components(reactants, products):
    """
    Renders reaction components with explicit separators to avoid RDKit plus overlap.
    """
    components = [("mol", smiles) for smiles in reactants]
    for smiles in products:
        components.append(("product", smiles))

    mol_images = []
    invalid_smiles = []
    for _, component_smiles in components:
        mol = Chem.MolFromSmiles(component_smiles)
        if mol is None:
            invalid_smiles.append(component_smiles)
            continue
        mol_images.append(Draw.MolToImage(mol, size=scale_size(REACTION_MOLECULE_IMAGE_SIZE)))

    if invalid_smiles or len(mol_images) != len(components):
        return None

    molecule_width, molecule_height = scale_size(REACTION_MOLECULE_IMAGE_SIZE)
    separator_width = scale_value(REACTION_SEPARATOR_WIDTH)
    arrow_width = scale_value(REACTION_ARROW_WIDTH)
    canvas_padding = scale_value(REACTION_CANVAS_PADDING)
    total_molecules = len(components)
    separator_count = max(0, len(reactants) - 1) + max(0, len(products) - 1)
    total_width = (
        canvas_padding * 2
        + total_molecules * molecule_width
        + separator_count * separator_width
        + arrow_width
    )
    total_height = molecule_height + canvas_padding * 2
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    separator_font = load_reaction_font(scale_value(52))

    x = canvas_padding
    y = canvas_padding
    image_index = 0
    for reactant_index, _ in enumerate(reactants):
        canvas.paste(mol_images[image_index].convert("RGB"), (x, y))
        image_index += 1
        x += molecule_width
        if reactant_index < len(reactants) - 1:
            draw_centered_text(
                draw,
                (x, y, x + separator_width, y + molecule_height),
                "+",
                separator_font,
            )
            x += separator_width

    draw_reaction_arrow(draw, (x, y, x + arrow_width, y + molecule_height))
    x += arrow_width

    for product_index, _ in enumerate(products):
        canvas.paste(mol_images[image_index].convert("RGB"), (x, y))
        image_index += 1
        x += molecule_width
        if product_index < len(products) - 1:
            draw_centered_text(
                draw,
                (x, y, x + separator_width, y + molecule_height),
                "+",
                separator_font,
            )
            x += separator_width

    return canvas

# Helper function to render reaction SMILES to a PIL Image
def render_reaction(smiles):
    """
    Converts a reaction SMILES into a PIL Image using RDKit.
    Handles standard reaction SMILES (reactants>>products).
    """
    if not smiles or not isinstance(smiles, str):
        return None
        
    # Pre-processing to handle common invalid SMILES like HBr or HCl
    # Standard SMILES for these are just Br and Cl (hydrogens are implicit)
    clean_smiles = smiles.replace("HBr", "Br").replace("HCl", "Cl")
    
    try:
        reactants, products = split_reaction_smiles(clean_smiles)
        if reactants and products:
            img = render_reaction_components(reactants, products)
            if img is not None:
                return img

        # Use ReactionFromSmarts with useSmiles=True for standard SMILES strings
        rxn = rdChemReactions.ReactionFromSmarts(clean_smiles, useSmiles=True)
        if rxn is None:
            # Try without useSmiles just in case it's actually SMARTS
            rxn = rdChemReactions.ReactionFromSmarts(clean_smiles)
            
        if rxn is None:
            return None
        
        # Draw the reaction
        img = Draw.ReactionToImage(rxn, subImgSize=scale_size((900, 900)))
        return img
    except Exception as e:
        st.error(f"Error rendering reaction: {e}")
        return None

def split_condition_smiles(condition_smiles):
    """
    Splits a condition field containing comma-separated SMILES into clean entries.
    """
    condition_smiles = normalize_condition_smiles(condition_smiles)
    if not condition_smiles:
        return []
    return [part.strip() for part in condition_smiles.split(",") if part.strip()]

def split_condition_slots(condition_slots):
    """
    Splits a comma-separated condition slot field into clean entries.
    """
    condition_slots = normalize_condition_smiles(condition_slots)
    if not condition_slots:
        return []
    return [part.strip() for part in condition_slots.split(",")]

def normalize_condition_smiles(condition_smiles):
    """
    Normalizes empty CSV values so pandas NaN/None are not shown as text.
    """
    if condition_smiles is None or pd.isna(condition_smiles):
        return ""

    condition_smiles = str(condition_smiles).strip()
    if condition_smiles.lower() in {"nan", "none", "null"}:
        return ""

    return condition_smiles

def get_condition_slot(slots, index):
    """
    Returns the condition role label aligned with a displayed SMILES.
    """
    if index >= len(slots):
        return ""
    slot = slots[index]
    return CONDITION_SLOT_DISPLAY_LABELS.get(slot.lower(), slot)

def render_condition_molecules(condition_smiles, condition_slots=None):
    """
    Converts comma-separated condition SMILES into a grid image using RDKit.
    Returns the rendered image and any entries RDKit could not parse.
    """
    smiles_list = split_condition_smiles(condition_smiles)
    slots = split_condition_slots(condition_slots)
    mols = []
    legends = []
    invalid_smiles = []

    for index, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.append(smiles)
            continue
        mols.append(mol)
        slot = get_condition_slot(slots, index)
        legends.append(f"{smiles}\n{slot}" if slot else smiles)

    if not mols:
        return None, invalid_smiles

    draw_options = rdMolDraw2D.MolDrawOptions()
    draw_options.legendFontSize = scale_value(CONDITION_LEGEND_FONT_SIZE)

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(3, len(mols)),
        subImgSize=scale_size(CONDITION_MOLECULE_IMAGE_SIZE),
        legends=legends,
        useSVG=False,
        drawOptions=draw_options,
    )
    return img, invalid_smiles

def fit_image_to_canvas(img, size=CONDITION_IMAGE_SIZE):
    """
    Fits an RDKit image into a fixed-size white canvas so both options render
    at the same height in Streamlit.
    """
    canvas_size = scale_size(size)
    canvas = Image.new("RGB", canvas_size, "white")
    resized = img.convert("RGB")
    resized.thumbnail(canvas_size, Image.Resampling.LANCZOS)

    x = (canvas_size[0] - resized.width) // 2
    y = (canvas_size[1] - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas

def show_condition_option(title, condition_smiles, condition_slots=None):
    """
    Displays condition text and its RDKit molecule image when possible.
    """
    st.markdown(f"### {title}")
    condition_smiles = normalize_condition_smiles(condition_smiles)

    img, invalid_smiles = render_condition_molecules(condition_smiles, condition_slots)
    if img:
        st.image(fit_image_to_canvas(img), use_container_width=True)
    elif condition_smiles:
        st.write(condition_smiles)
    else:
        st.caption("No condition SMILES")

    if invalid_smiles:
        st.caption(f"Could not render as SMILES: {', '.join(invalid_smiles)}")

# Load dataset function
def load_data(file_path):
    """
    Loads the input dataset. Creates a dummy one if it doesn't exist for demo purposes.
    """
    if not os.path.exists(file_path):
        st.warning(f"Input file '{file_path}' not found. Loading dummy data for demonstration.")
        dummy_data = pd.DataFrame({
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
        required_cols = ['reaction_smiles', 'condition_a', 'condition_b']
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
    if annotations.empty or "reaction_smiles" not in annotations.columns:
        return 0

    completed = annotations["reaction_smiles"].dropna().astype(str).tolist()
    input_reactions = data["reaction_smiles"].astype(str).tolist()

    resume_index = 0
    for input_smiles, completed_smiles in zip(input_reactions, completed):
        if input_smiles != completed_smiles:
            break
        resume_index += 1

    return min(resume_index, len(data))

def find_existing_annotation(annotations, reaction_smiles):
    """
    Returns the latest saved annotation for a reaction, if present.
    """
    if annotations.empty or "reaction_smiles" not in annotations.columns:
        return None

    matches = annotations[
        annotations["reaction_smiles"].astype(str) == str(reaction_smiles)
    ]
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
    Keeps appends compatible with the annotation file already on disk.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return default_columns

    try:
        return pd.read_csv(file_path, nrows=0).columns.tolist()
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

    new_row = pd.DataFrame([annotation], columns=output_columns)
    if existing.empty or "reaction_smiles" not in existing.columns:
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        reaction_smiles = str(annotation["reaction_smiles"])
        matches = existing["reaction_smiles"].astype(str) == reaction_smiles
        if matches.any():
            updated = existing.copy()
            row_index = matches[matches].index[-1]
            for column in output_columns:
                if column in new_row.columns:
                    updated.loc[row_index, column] = new_row.iloc[0][column]
        else:
            updated = pd.concat([existing, new_row], ignore_index=True)

    updated.to_csv(file_path, index=False)

# --- App State Initialization ---
if 'randomized_data' not in st.session_state:
    # We store the randomization per index to ensure consistency across re-runs 
    # for the same sample (e.g. if user resizes window)
    st.session_state.randomized_data = {} 

# --- Sidebar ---
st.sidebar.header("Data Settings")
input_file = st.sidebar.text_input("Input Data Path", INPUT_DATA)
output_file = st.sidebar.text_input("Output Annotations Path", HUMAN_FILE)

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
    saved_annotation = find_existing_annotation(annotations, row['reaction_smiles'])
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

    # --- UI Header ---
    st.title("🧪 Chemical Reaction Preference Annotation")
    
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
    img = render_reaction(row['reaction_smiles'])
    if img:
        # Use more width for the reaction image
        st.image(img, use_container_width=True, caption=row['reaction_smiles'], width='stretch')
    else:
        st.error("Failed to render reaction image. Please check the SMILES format.")

    st.divider()

    # --- Reagent Choices ---
    st.subheader("Which reagent set is better?")
    col_left, col_right = st.columns(2)
    
    with col_left:
        show_condition_option("Option 1", opt1_text, opt1_slots)
        
    with col_right:
        show_condition_option("Option 2", opt2_text, opt2_slots)

    # --- Buttons Section ---
    st.markdown("---")
    st.markdown("### Decision")
    if saved_choice:
        st.success(f"Saved choice for this sample: {saved_choice}")
    else:
        st.info("No saved choice for this sample yet.")
    
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    
    def handle_click(choice):
        annotation = {
            'reaction_smiles': row['reaction_smiles'],
            'shown_option_1': opt1_text,
            'shown_option_1_slots': opt1_slots,
            'shown_option_2': opt2_text,
            'shown_option_2_slots': opt2_slots,
            'user_choice': choice,
            'is_option_1_GT': is_opt1_a
        }
        
        save_annotation(output_file, annotation)
        
        # Increment index
        st.session_state.current_index = min(st.session_state.current_index + 1, len(data))

    if btn_col1.button("👈 Prefer Option 1", width="stretch", type="primary" if saved_choice == "Option 1" else "secondary"):
        handle_click("Option 1")
        st.rerun()
        
    if btn_col2.button("👉 Prefer Option 2", width="stretch", type="primary" if saved_choice == "Option 2" else "secondary"):
        handle_click("Option 2")
        st.rerun()
        
    if btn_col3.button("🤝 Both Valid / Equivalent", width="stretch", type="primary" if saved_choice == "Tie" else "secondary"):
        handle_click("Tie")
        st.rerun()
        
    if btn_col4.button("❌ Both Bad / Invalid", width="stretch", type="primary" if saved_choice == "Bad" else "secondary"):
        handle_click("Bad")
        st.rerun()

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
