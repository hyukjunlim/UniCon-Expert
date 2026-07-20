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
/* Keep the annotation canvas at a comfortable reading width on ultrawide
   displays. It remains fluid below this limit, so smaller windows and mobile
   layouts still use all of the available space. */
div[data-testid="stMainBlockContainer"],
.stMain .block-container {
    max-width: 1280px;
}
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
div[data-testid="stSegmentedControl"] button {
    min-height: 3em;
    font-weight: bold;
}
.condition-component-label {
    color: white;
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
    justify-content: center;
    overflow: hidden;
    width: 100%;
}
.condition-component-image img {
    height: 100%;
    object-fit: contain;
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
