"""Shared helpers for rendering reactions in annotation apps."""

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw, rdChemReactions

from expert_annotation.common.ui_style import IMAGE_RENDER_SCALE, scale_size


REACTION_MOLECULE_IMAGE_SIZE = (500, 300)
REACTION_SEPARATOR_WIDTH = 54
REACTION_ARROW_WIDTH = 105
REACTION_CANVAS_PADDING = 18


def clear_molecule_atom_maps(molecule):
    """Remove atom-map labels from an RDKit molecule in place."""
    if molecule is None:
        return None

    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return molecule


def clear_reaction_atom_maps(reaction):
    """Remove atom-map labels from every template in an RDKit reaction."""
    if reaction is None:
        return None

    template_groups = (
        reaction.GetReactants(),
        reaction.GetAgents(),
        reaction.GetProducts(),
    )
    for templates in template_groups:
        for molecule in templates:
            clear_molecule_atom_maps(molecule)
    return reaction


def scale_value(value, scale=IMAGE_RENDER_SCALE):
    return int(value * scale)


def split_reaction_smiles(smiles):
    """Split reaction SMILES into reactant and product components."""
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
    """Load the font used for reaction separators."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


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
    """Render components with explicit, consistently sized separators."""
    components = list(reactants) + list(products)
    molecule_images = []
    for component_smiles in components:
        molecule = Chem.MolFromSmiles(component_smiles)
        if molecule is None:
            return None
        clear_molecule_atom_maps(molecule)
        molecule_images.append(
            Draw.MolToImage(molecule, size=scale_size(REACTION_MOLECULE_IMAGE_SIZE))
        )

    molecule_width, molecule_height = scale_size(REACTION_MOLECULE_IMAGE_SIZE)
    separator_width = scale_value(REACTION_SEPARATOR_WIDTH)
    arrow_width = scale_value(REACTION_ARROW_WIDTH)
    canvas_padding = scale_value(REACTION_CANVAS_PADDING)
    separator_count = max(0, len(reactants) - 1) + max(0, len(products) - 1)
    total_width = (
        canvas_padding * 2
        + len(components) * molecule_width
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
        canvas.paste(molecule_images[image_index].convert("RGB"), (x, y))
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
        canvas.paste(molecule_images[image_index].convert("RGB"), (x, y))
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


def render_reaction(smiles):
    """Render a reaction with the condition-preference visual style."""
    if not smiles or not isinstance(smiles, str):
        return None

    clean_smiles = smiles.replace("HBr", "Br").replace("HCl", "Cl")
    try:
        reactants, products = split_reaction_smiles(clean_smiles)
        if reactants and products:
            image = render_reaction_components(reactants, products)
            if image is not None:
                return image

        reaction = rdChemReactions.ReactionFromSmarts(clean_smiles, useSmiles=True)
        if reaction is None:
            reaction = rdChemReactions.ReactionFromSmarts(clean_smiles)
        if reaction is None:
            return None
        clear_reaction_atom_maps(reaction)
        return Draw.ReactionToImage(reaction, subImgSize=scale_size((900, 900)))
    except Exception:
        return None
