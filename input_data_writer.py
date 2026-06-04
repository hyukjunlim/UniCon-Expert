import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CHEMPROP_ROOT = REPO_ROOT / "chemprop"
for path in (REPO_ROOT, CHEMPROP_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


DEFAULT_TEST_PATH = str(REPO_ROOT / "data/MPNN_data/GCN_data_test.csv")
DEFAULT_LABEL_PATH = str(REPO_ROOT / "data/labels")
DEFAULT_LIBRARY_PATH = str(REPO_ROOT / "data/condition_library")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "expert_annotation")
DEFAULT_VOCAB_CACHE_DIR = str(REPO_ROOT / "expert_annotation/_cache/unicon_vocab")
TARGET_COLUMNS = ["cat", "solv0", "solv1", "reag0", "reag1", "reag2"]
DISPLAY_SLOT_LABELS = {
    "cat": "Catalyst",
    "solv0": "Solvent",
    "solv1": "Solvent",
    "reag0": "Reagent",
    "reag1": "Reagent",
    "reag2": "Reagent",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Write expert annotation input CSV rows from the test set. "
            "The selected interval is zero-indexed and end-exclusive."
        )
    )
    parser.add_argument("start_pos", nargs="?", type=int, help="First test-set index to include.")
    parser.add_argument("end_pos", nargs="?", type=int, help="Stop before this test-set index.")
    parser.add_argument("--start", dest="start_opt", type=int, help="First test-set index to include.")
    parser.add_argument("--end", dest="end_opt", type=int, help="Stop before this test-set index.")
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH, help="Path to the test CSV.")
    parser.add_argument("--label-path", default=DEFAULT_LABEL_PATH, help="Path to condition label CSVs.")
    parser.add_argument(
        "--library-path",
        default=DEFAULT_LIBRARY_PATH,
        help="Path to template condition libraries.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where annotation_input_data_{start}_{end}.csv is written.",
    )
    parser.add_argument(
        "--vocab-cache-dir",
        default=DEFAULT_VOCAB_CACHE_DIR,
        help="Directory where the generated annotation vocabulary cache is written.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic generation.")
    args = parser.parse_args()
    args.start = args.start_opt if args.start_opt is not None else args.start_pos
    args.end = args.end_opt if args.end_opt is not None else args.end_pos
    return args


def validate_args(args):
    if args.start is None or args.end is None:
        raise ValueError("Provide start and end, e.g. python input_data_writer.py 50 100")
    if args.start < 0:
        raise ValueError("start must be >= 0")
    if args.end <= args.start:
        raise ValueError("end must be greater than start")
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"Test CSV not found: {args.test_path}")


def setup_prediction_context(end, test_path, label_path, library_path, vocab_cache_dir, seed):
    try:
        from chemprop.features import (
            set_adding_hs,
            set_explicit_h,
            set_keeping_atom_map,
            set_reaction,
        )
        from eval_attention import _load_baseline_models, _setup_args
        from eval_template_baseline import get_condition_labels, load_libraries
        from utils import load_unified_vocab, set_seed
    except ImportError as exc:
        raise SystemExit(
            "Could not import the model dependencies needed to generate baseline "
            "conditions. Activate the UniCon/Chemprop environment and try again. "
            f"Original error: {exc}"
        ) from exc

    set_seed(seed)

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        args = _setup_args(end)
    finally:
        sys.argv = original_argv
    args.separate_test_path = test_path
    args.label_dir = label_path
    args.save_dir = vocab_cache_dir
    args.checkpoint_path = None
    args.smiles_columns = ["reaction"]
    args.number_of_molecules = 1
    args.target_columns = TARGET_COLUMNS
    args.ignore_columns = ["tpl_SMARTS_r1", "tpl_SMARTS_r0", "tpl_SMARTS_r0*"]
    args.reaction = True

    set_reaction(args.reaction, args.reaction_mode)
    set_explicit_h(args.explicit_h)
    set_adding_hs(args.adding_h)
    set_keeping_atom_map(args.keeping_atom_map)

    total_vocab_size, vocab_mappings, col_to_file, _, idx_to_entity = load_unified_vocab(
        args,
        logger=None,
        allow_load=True,
        use_dmpnn=getattr(args, "use_dmpnn_reagents", False),
    )
    args.pad_idx = total_vocab_size
    args.vocab_size = total_vocab_size + 1

    condition_key = get_condition_labels(label_path)
    libraries = load_libraries(library_path)
    baseline_models, _ = _load_baseline_models(args, col_to_file, vocab_mappings, idx_to_entity)

    return args, vocab_mappings, col_to_file, idx_to_entity, condition_key, libraries, baseline_models


def display_slot(col_name):
    return DISPLAY_SLOT_LABELS.get(col_name, col_name)


def format_gt_condition_with_slots(targets, args, col_to_file, vocab_mappings, idx_to_entity):
    names = []
    slots = []
    global_indices = []

    if targets[0] is None:
        return "", "", []

    for i, val in enumerate(targets[0]):
        if val is None or pd.isna(val):
            continue

        col_name = args.target_columns[i]
        if col_name not in col_to_file:
            continue

        fname = col_to_file[col_name]
        local_idx = int(val)
        if local_idx not in vocab_mappings[fname]:
            continue

        global_idx = vocab_mappings[fname][local_idx]
        names.append(idx_to_entity.get(global_idx, f"Unknown-{global_idx}"))
        slots.append(display_slot(col_name))
        global_indices.append(global_idx)

    return ", ".join(names), ", ".join(slots), global_indices


def format_condition_list_with_slots(condition, condition_key, col_to_file, vocab_mappings):
    cat_list, solv_list, reag_list = condition_key
    role_lists = [cat_list, solv_list, solv_list, reag_list, reag_list, reag_list]
    names = []
    slots = []
    global_indices = []

    for i, smiles in enumerate(condition):
        if i >= len(TARGET_COLUMNS):
            break
        if smiles in (None, "None") or pd.isna(smiles):
            continue

        col_name = TARGET_COLUMNS[i]
        names.append(smiles)
        slots.append(display_slot(col_name))

        if smiles in role_lists[i] and col_name in col_to_file:
            local_idx = role_lists[i].index(smiles)
            fname = col_to_file[col_name]
            if local_idx in vocab_mappings[fname]:
                global_indices.append(vocab_mappings[fname][local_idx])

    return ", ".join(names), ", ".join(slots), global_indices


def get_baseline_info_with_slots(
    baseline_models,
    graph_input,
    templates,
    libraries,
    condition_key,
    col_to_file,
    vocab_mappings,
):
    import ast
    import torch
    from eval_template_baseline import naive_condition_selector, select_conditions_baseline

    mpnn_out = []
    with torch.no_grad():
        for target in TARGET_COLUMNS:
            if target in baseline_models:
                probs = baseline_models[target](graph_input)
                if probs.dim() == 3:
                    probs = probs.squeeze(1)
                probs = probs.cpu().numpy()[0]
                mpnn_out.append(list(probs))
            else:
                vocab_size = len(vocab_mappings.get(col_to_file.get(target, ""), {}))
                mpnn_out.append([1.0 / vocab_size] * vocab_size if vocab_size > 0 else [1.0])

    current_tpl_r1, current_tpl_r0, current_tpl_r_1 = templates
    _, baseline_libraries = libraries

    if current_tpl_r1 in baseline_libraries[2]:
        candidates = select_conditions_baseline(current_tpl_r1, mpnn_out, baseline_libraries[2], condition_key)
    elif current_tpl_r0 in baseline_libraries[1]:
        candidates = select_conditions_baseline(current_tpl_r0, mpnn_out, baseline_libraries[1], condition_key)
    elif current_tpl_r_1 in baseline_libraries[0]:
        candidates = select_conditions_baseline(current_tpl_r_1, mpnn_out, baseline_libraries[0], condition_key)
    else:
        candidates = naive_condition_selector(mpnn_out, [3, 3, 3, 3, 3, 3], condition_key)

    if not candidates:
        return "None", "", []

    condition = ast.literal_eval(candidates[0][0])
    display_str, slot_str, indices = format_condition_list_with_slots(
        condition,
        condition_key,
        col_to_file,
        vocab_mappings,
    )
    return display_str if display_str else "None", slot_str, indices


def build_rows(start, end, test_path, label_path, library_path, vocab_cache_dir, seed):
    try:
        from chemprop.data import MoleculeDataLoader, get_data
    except ImportError as exc:
        raise SystemExit(
            "Could not import the model dependencies needed to generate baseline "
            "conditions. Activate the UniCon/Chemprop environment and try again. "
            f"Original error: {exc}"
        ) from exc

    (
        args,
        vocab_mappings,
        col_to_file,
        idx_to_entity,
        condition_key,
        libraries,
        baseline_models,
    ) = setup_prediction_context(end, test_path, label_path, library_path, vocab_cache_dir, seed)

    df_data = pd.read_csv(test_path)
    data = get_data(path=test_path, args=args)
    loader = MoleculeDataLoader(dataset=data, batch_size=1, num_workers=0, shuffle=False)

    rows = []
    for index, batch in enumerate(loader):
        if index >= end:
            break
        if index < start:
            continue

        graph_input = batch.batch_graph()
        targets = batch.targets()
        reaction_smiles = batch.smiles()[0][0]
        templates = (
            df_data["tpl_SMARTS_r1"].iloc[index],
            df_data["tpl_SMARTS_r0"].iloc[index],
            df_data["tpl_SMARTS_r0*"].iloc[index],
        )

        gt_str, gt_slots, _ = format_gt_condition_with_slots(
            targets,
            args,
            col_to_file,
            vocab_mappings,
            idx_to_entity,
        )
        baseline_str, baseline_slots, _ = get_baseline_info_with_slots(
            baseline_models,
            graph_input,
            templates,
            libraries,
            condition_key,
            col_to_file,
            vocab_mappings,
        )

        rows.append(
            {
                "reaction_smiles": reaction_smiles,
                "condition_a": gt_str if gt_str else "None",
                "condition_a_slots": gt_slots,
                "condition_b": baseline_str if baseline_str else "None",
                "condition_b_slots": baseline_slots,
            }
        )
        print(
            f"[{index}] GT={rows[-1]['condition_a']} ({rows[-1]['condition_a_slots']}) | "
            f"Baseline={rows[-1]['condition_b']} ({rows[-1]['condition_b_slots']})"
        )

    return rows


def write_annotation_input(start, end, test_path, label_path, library_path, output_dir, vocab_cache_dir, seed):
    rows = build_rows(start, end, test_path, label_path, library_path, vocab_cache_dir, seed)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"annotation_input_data_{start}_{end}.csv")
    pd.DataFrame(
        rows,
        columns=[
            "reaction_smiles",
            "condition_a",
            "condition_a_slots",
            "condition_b",
            "condition_b_slots",
        ],
    ).to_csv(
        output_path,
        index=False,
    )
    return output_path, len(rows)


def main():
    args = parse_args()
    validate_args(args)
    output_path, row_count = write_annotation_input(
        args.start,
        args.end,
        args.test_path,
        args.label_path,
        args.library_path,
        args.output_dir,
        args.vocab_cache_dir,
        args.seed,
    )
    print(f"Wrote {row_count} rows to {output_path}")


if __name__ == "__main__":
    main()
