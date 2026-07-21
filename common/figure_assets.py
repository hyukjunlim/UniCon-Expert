"""Pre-rendered image assets for the expert-annotation interfaces."""

import base64
import hashlib
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D

from common.reaction_rendering import (
    clear_molecule_atom_maps,
    clear_reaction_atom_maps,
    split_reaction_smiles,
)


EXPERT_ANNOTATION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = EXPERT_ANNOTATION_DIR / "figs"
MOLECULE_IMAGE_SIZE = (500, 300)
REACTION_IMAGE_SIZE = (1800, 300)
REACTION_MOLECULE_SIZE = (340, 180)
REACTION_SEPARATOR_WIDTH = 44
REACTION_ARROW_WIDTH = 132
REACTION_CANVAS_PADDING = 8
ASSET_VERSION = "svg-v6"


def normalize_text(value):
    """Normalize empty CSV cells without displaying pandas sentinel values."""
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"nan", "none", "null"} else value


def split_csv_field(value):
    value = normalize_text(value)
    return [] if not value else [part.strip() for part in value.split(",") if part.strip()]


def canonical_smiles(value):
    """Return a stable canonical representation for ordering and comparison."""
    molecule = Chem.MolFromSmiles(value)
    return Chem.MolToSmiles(molecule, canonical=True) if molecule else str(value)


def _content_hash(*values):
    content = "\x1f".join(normalize_text(value) for value in values)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def _sample_dir(figure_dir, workflow, evaluation_id):
    return Path(figure_dir) / workflow / f"sample_{int(evaluation_id):04d}"


def reaction_figure_path(figure_dir, workflow, evaluation_id, reaction_smiles):
    digest = _content_hash(ASSET_VERSION, reaction_smiles)
    return _sample_dir(figure_dir, workflow, evaluation_id) / f"reaction_{digest}.svg"


def condition_figure_paths(
    figure_dir, workflow, evaluation_id, field_name, values, slots
):
    components = split_csv_field(values)
    roles = split_csv_field(slots)
    paths = []
    for index, smiles in enumerate(components):
        role = roles[index] if index < len(roles) else "Condition component"
        digest = _content_hash(ASSET_VERSION, smiles, role)
        path = _sample_dir(figure_dir, workflow, evaluation_id) / (
            f"{field_name}_{index:02d}_{digest}.svg"
        )
        paths.append((path, smiles, role))
    return paths


def _save_svg(svg, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def svg_data_uri(path):
    """Encode a pre-rendered SVG for embedding in a fixed-height UI container."""
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _molecule_svg_body(molecule, size, padding=0.025):
    clear_molecule_atom_maps(molecule)
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    drawer.drawOptions().padding = padding
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.split("<!-- END OF HEADER -->", 1)[1].rsplit("</svg>", 1)[0]


def _render_tight_reaction_svg(reaction_smiles):
    """Lay out reaction components with large vector separators and minimal margins."""
    reactant_smiles, product_smiles = split_reaction_smiles(reaction_smiles)
    if not reactant_smiles or not product_smiles:
        return None
    reactants = [Chem.MolFromSmiles(value) for value in reactant_smiles]
    products = [Chem.MolFromSmiles(value) for value in product_smiles]
    if any(molecule is None for molecule in [*reactants, *products]):
        return None

    molecule_width, molecule_height = REACTION_MOLECULE_SIZE
    plus_count = max(0, len(reactants) - 1) + max(0, len(products) - 1)
    width = (
        2 * REACTION_CANVAS_PADDING
        + (len(reactants) + len(products)) * molecule_width
        + plus_count * REACTION_SEPARATOR_WIDTH
        + REACTION_ARROW_WIDTH
    )
    height = molecule_height + 2 * REACTION_CANVAS_PADDING
    middle_y = height / 2
    elements = [
        "<?xml version='1.0' encoding='utf-8'?>",
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}px" '
        f'height="{height}px" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    x = REACTION_CANVAS_PADDING

    def add_molecules(molecules):
        nonlocal x
        for index, molecule in enumerate(molecules):
            body = _molecule_svg_body(molecule, REACTION_MOLECULE_SIZE)
            elements.append(f'<g transform="translate({x},{REACTION_CANVAS_PADDING})">{body}</g>')
            x += molecule_width
            if index < len(molecules) - 1:
                plus_x = x + REACTION_SEPARATOR_WIDTH / 2
                elements.append(
                    f'<text x="{plus_x}" y="{middle_y}" text-anchor="middle" '
                    'dominant-baseline="central" font-family="DejaVu Sans, sans-serif" '
                    'font-size="36" font-weight="600" fill="black">+</text>'
                )
                x += REACTION_SEPARATOR_WIDTH

    add_molecules(reactants)
    arrow_start = x + 20
    arrow_end = x + REACTION_ARROW_WIDTH - 25
    elements.extend(
        [
            f'<line x1="{arrow_start}" y1="{middle_y}" x2="{arrow_end}" '
            f'y2="{middle_y}" stroke="black" stroke-width="3"/>',
            f'<polygon points="{arrow_end},{middle_y} {arrow_end - 12},{middle_y - 7} '
            f'{arrow_end - 12},{middle_y + 7}" fill="black"/>',
        ]
    )
    x += REACTION_ARROW_WIDTH
    add_molecules(products)
    elements.append("</svg>")
    return "\n".join(elements)


def generate_reaction_figure(path, reaction_smiles, overwrite=False):
    if path.exists() and not overwrite:
        return True
    clean_smiles = normalize_text(reaction_smiles).replace("HBr", "Br").replace("HCl", "Cl")
    svg = _render_tight_reaction_svg(clean_smiles)
    if svg is not None:
        _save_svg(svg, path)
        return True
    try:
        reaction = rdChemReactions.ReactionFromSmarts(clean_smiles, useSmiles=True)
        if reaction is None:
            reaction = rdChemReactions.ReactionFromSmarts(clean_smiles)
    except Exception:
        reaction = None
    if reaction is None:
        return False
    clear_reaction_atom_maps(reaction)
    drawer = rdMolDraw2D.MolDraw2DSVG(*REACTION_IMAGE_SIZE)
    drawer.drawOptions().padding = 0.04
    drawer.DrawReaction(reaction)
    drawer.FinishDrawing()
    _save_svg(drawer.GetDrawingText(), path)
    return True


def generate_condition_figure(path, smiles, role, overwrite=False):
    if path.exists() and not overwrite:
        return True
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return False
    body = _molecule_svg_body(molecule, MOLECULE_IMAGE_SIZE, padding=0.04)
    width, height = MOLECULE_IMAGE_SIZE
    svg = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}px" '
        f'height="{height}px" viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="white"/>{body}</svg>'
    )
    _save_svg(svg, path)
    return True


def generate_row_figures(
    row,
    workflow,
    condition_fields,
    figure_dir=DEFAULT_FIGURE_DIR,
    overwrite=False,
):
    """Generate the reaction and individual condition-component images for one row."""
    evaluation_id = row["evaluation_id"]
    failures = []
    reaction_path = reaction_figure_path(
        figure_dir, workflow, evaluation_id, row["reaction_smiles"]
    )
    if not generate_reaction_figure(reaction_path, row["reaction_smiles"], overwrite):
        failures.append(normalize_text(row["reaction_smiles"]))

    for value_column, slot_column, field_name in condition_fields:
        for path, smiles, role in condition_figure_paths(
            figure_dir,
            workflow,
            evaluation_id,
            field_name,
            row.get(value_column, ""),
            row.get(slot_column, ""),
        ):
            if not generate_condition_figure(path, smiles, role, overwrite):
                failures.append(smiles)
    return failures


def generate_csv_figures(
    input_path,
    workflow,
    condition_fields,
    figure_dir=DEFAULT_FIGURE_DIR,
    overwrite=False,
):
    """Generate all UI image assets referenced by an annotation-input CSV."""
    frame = pd.read_csv(input_path)
    required = {"evaluation_id", "reaction_smiles"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{input_path}: missing columns: {', '.join(sorted(missing))}")

    failures = []
    for _, row in frame.iterrows():
        failures.extend(
            generate_row_figures(row, workflow, condition_fields, figure_dir, overwrite)
        )
    return len(frame), failures


def load_pre_rendered_condition_paths(
    row,
    workflow,
    value_column,
    slot_column,
    field_name,
    figure_dir=DEFAULT_FIGURE_DIR,
    priority_values=None,
):
    """Return component assets, optionally placing shared values first."""
    assets = condition_figure_paths(
        figure_dir,
        workflow,
        row["evaluation_id"],
        field_name,
        row.get(value_column, ""),
        row.get(slot_column, ""),
    )
    priority_keys = {
        canonical_smiles(value) for value in split_csv_field(priority_values)
    }

    def canonical_key(asset):
        _, smiles, _ = asset
        canonical = canonical_smiles(smiles)
        return canonical not in priority_keys, canonical, smiles

    return sorted(assets, key=canonical_key)
