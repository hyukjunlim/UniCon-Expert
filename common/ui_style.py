"""Shared visual tokens for the Streamlit annotation apps.

The values in this module come from the condition-preference interface, which
is the visual reference for the other expert-annotation tools.
"""

from PIL import Image


IMAGE_RENDER_SCALE = 1
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

ANNOTATION_APP_CSS = """
<style>
/* Keep the annotation canvas proportional to the browser width. */
div[data-testid="stMainBlockContainer"],
.stMain .block-container {
    max-width: 85%;
    width: 85%;
}
@media (max-width: 768px) {
    div[data-testid="stMainBlockContainer"],
    .stMain .block-container {
        max-width: 100%;
        width: 100%;
    }
}
.stButton > button {
    height: 3em;
    font-weight: bold;
}
/*
 * Use restrained, non-alarming accents for annotation actions.  Streamlit's
 * default primary color can read as an error state when several choices are
 * visible at once, so selected choices use red and commit actions use green.
 */
.stButton > button[kind="primary"] {
    background-color: #d9534f !important;
    border-color: #d9534f !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #c4423e !important;
    border-color: #c4423e !important;
}
.stButton > button[kind="primary"]:focus,
.stButton > button[kind="primary"]:active {
    background-color: #c4423e !important;
    border-color: #c4423e !important;
    box-shadow: 0 0 0 0.2rem rgb(217 83 79 / 22%) !important;
}
div[class*="st-key-save_"] button[kind="primary"] {
    background-color: #2e8b57 !important;
    border-color: #2e8b57 !important;
}
div[class*="st-key-save_"] button[kind="primary"]:hover {
    background-color: #247447 !important;
    border-color: #247447 !important;
}
div[class*="st-key-save_"] button[kind="primary"]:focus,
div[class*="st-key-save_"] button[kind="primary"]:active {
    background-color: #247447 !important;
    border-color: #247447 !important;
    box-shadow: 0 0 0 0.2rem rgb(46 139 87 / 22%) !important;
}
/* Status markers are bare, tightly packed clickable squares. */
div[class*="st-key-question_status"] div[data-testid="stVerticalBlock"] {
    gap: 0.25rem !important;
}
div[class*="st-key-question_status"] div[data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important;
    gap: 0.4rem !important;
    justify-content: center !important;
}
div[class*="st-key-question_status"] div[data-testid="stColumn"] {
    flex: 0 0 1.75rem !important;
    min-width: 1.75rem !important;
    width: 1.75rem !important;
}
div[class*="st-key-status_"] button,
div[class*="st-key-status_"] button:hover,
div[class*="st-key-status_"] button:focus,
div[class*="st-key-status_"] button:active {
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    align-items: center !important;
    height: 1.7rem !important;
    justify-content: center !important;
    min-height: 1.7rem !important;
    min-width: 1.7rem !important;
    padding: 0 !important;
    width: 1.7rem !important;
}
div[class*="st-key-status_"] button p {
    font-size: 1.3rem;
    line-height: 1;
    text-align: center;
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
div[data-testid="stSegmentedControl"] button {
    min-height: 3em;
    font-weight: bold;
}
.condition-component-label {
    color: var(--text-color, inherit);
    font-size: 1.2rem;
    font-weight: 600;
    line-height: 1.35;
    margin-top: 0.35rem;
    margin-bottom: 0.8rem;
}
.condition-component-image {
    align-items: center;
    display: flex;
    height: 240px;
    justify-content: flex-start;
    overflow: hidden;
    width: 100%;
}
.condition-component-image img {
    height: 100%;
    object-fit: contain;
    object-position: left center;
    width: 100%;
}
</style>
"""


def scale_size(size, scale=IMAGE_RENDER_SCALE):
    """Scale an image size for sharp downsampling in Streamlit."""
    return tuple(int(value * scale) for value in size)


def display_condition_slot(slot):
    """Use the condition-preference app's human-readable role labels."""
    slot = str(slot).strip()
    return CONDITION_SLOT_DISPLAY_LABELS.get(slot.lower(), slot)


def fit_image_to_canvas(img, size=CONDITION_IMAGE_SIZE):
    """Center an image on the condition-preference app's standard canvas."""
    canvas_size = scale_size(size)
    canvas = Image.new("RGB", canvas_size, "white")
    resized = img.convert("RGB")
    resized.thumbnail(canvas_size, Image.Resampling.LANCZOS)

    x = (canvas_size[0] - resized.width) // 2
    y = (canvas_size[1] - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas
