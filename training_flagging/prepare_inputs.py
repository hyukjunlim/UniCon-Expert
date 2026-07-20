"""Build a blinded human-evaluation set for UniCon training-data flags."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


WORKFLOW_DIR = Path(__file__).resolve().parent
DATA_DIR = WORKFLOW_DIR / "data"
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "chemprop"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chemprop.data import MoleculeDataLoader, MoleculeDataset, get_data
from chemprop.features import set_adding_hs, set_explicit_h, set_keeping_atom_map, set_reaction
from eval_template_baseline import get_condition_labels, load_libraries
from run_capfilt import _get_baseline_top1_indices, _load_baseline_models
from train_unicon import SLOT_ORDER, UniConArgs
from unicon_model import UniCon
from utils import RxnFitCollator, load_unified_vocab, set_seed


SLOT_LABELS = {
    "cat": "Catalyst",
    "solv0": "Solvent",
    "solv1": "Solvent",
    "reag0": "Reagent",
    "reag1": "Reagent",
    "reag2": "Reagent",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample flagged and control training reactions for blinded human evaluation."
    )
    parser.add_argument("--n", type=int, default=50, help="Samples per group (default: 50).")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Optional prefix limit for a smoke run.")
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="Score the entire input instead of stopping once both sample pools are full.",
    )
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data/MPNN_data/GCN_data_train.csv")
    parser.add_argument("--label-path", type=Path, default=REPO_ROOT / "data/labels")
    parser.add_argument("--library-path", type=Path, default=REPO_ROOT / "data/condition_library")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DATA_DIR / "annotation_inputs.csv",
    )
    parser.add_argument(
        "--key-path",
        type=Path,
        default=None,
        help="Hidden score/cohort key (default: <output stem>_key.csv).",
    )
    return parser.parse_args()


def condition_text(indices, idx_to_entity, pad_idx):
    values, slots = [], []
    for slot, index in zip(SLOT_ORDER, indices):
        index = int(index)
        if index == pad_idx:
            continue
        values.append(idx_to_entity.get(index, f"Unknown-{index}"))
        slots.append(SLOT_LABELS[slot])
    return ", ".join(values) or "None", ", ".join(slots)


def select_evaluation_rows(rows, n, threshold, seed):
    """Choose equal, disjoint non-exact flagged/control groups and blind their order."""
    if n <= 0:
        raise ValueError("--n must be positive")
    eligible = [row for row in rows if not row["candidate_matches_archived"]]
    flagged = [row for row in eligible if row["unicon_score"] > threshold]
    controls = [row for row in eligible if row["unicon_score"] <= threshold]
    if len(flagged) < n or len(controls) < n:
        raise ValueError(
            f"Cannot sample {n} per group: found {len(flagged)} flagged and {len(controls)} controls."
        )

    rng = np.random.default_rng(seed)
    flagged_idx = rng.choice(len(flagged), size=n, replace=False)
    control_idx = rng.choice(len(controls), size=n, replace=False)
    selected = []
    for index in flagged_idx:
        selected.append({**flagged[int(index)], "sampling_group": "flagged"})
    for index in control_idx:
        selected.append({**controls[int(index)], "sampling_group": "control"})
    rng.shuffle(selected)
    for display_index, row in enumerate(selected):
        row["evaluation_id"] = display_index
    return selected


@torch.no_grad()
def score_training_set(cli_args):
    args = UniConArgs().parse_args([])
    args.data_path = str(cli_args.data_path.resolve())
    args.label_dir = str(cli_args.label_path.resolve())
    if cli_args.checkpoint_path is not None:
        args.checkpoint_path = str(cli_args.checkpoint_path.resolve())
    elif not os.path.isabs(args.checkpoint_path):
        args.checkpoint_path = str(REPO_ROOT / args.checkpoint_path)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.return_smiles = True
    args.num_workers = 0

    set_seed(args.seed)
    set_reaction(args.reaction, args.reaction_mode)
    set_explicit_h(args.explicit_h)
    set_adding_hs(args.adding_h)
    set_keeping_atom_map(args.keeping_atom_map)

    total_vocab_size, vocab_mappings, col_to_file, fingerprint_matrix, idx_to_entity = load_unified_vocab(
        args, allow_load=True
    )
    args.pad_idx = total_vocab_size
    args.vocab_size = total_vocab_size + 1

    model = UniCon(
        args,
        fingerprint_matrix,
        vocab_size=args.vocab_size,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
    )
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"UniCon checkpoint not found: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()

    condition_key = get_condition_labels(args.label_dir)
    libraries = load_libraries(str(cli_args.library_path.resolve()))
    baseline_models = _load_baseline_models(args)
    missing_models = sorted(set(SLOT_ORDER) - set(baseline_models))
    if missing_models:
        raise FileNotFoundError(f"Missing ReaCon D-MPNN models for: {', '.join(missing_models)}")

    frame = pd.read_csv(args.data_path)
    dataset = get_data(path=args.data_path, args=args)
    source_indices = np.arange(len(dataset))
    if cli_args.limit is not None:
        frame = frame.iloc[: cli_args.limit].reset_index(drop=True)
        dataset = dataset[: cli_args.limit]
        source_indices = source_indices[: cli_args.limit]

    # A random traversal followed by taking the first N members of each pool is
    # equivalent to uniform sampling without replacement within those pools.
    traversal_rng = np.random.default_rng(cli_args.sample_seed)
    traversal_order = traversal_rng.permutation(len(dataset))
    dataset = MoleculeDataset([dataset[int(index)] for index in traversal_order])
    frame = frame.iloc[traversal_order].reset_index(drop=True)
    source_indices = source_indices[traversal_order]

    collator = RxnFitCollator(args, col_to_file, vocab_mappings, total_vocab_size)
    loader = MoleculeDataLoader(
        dataset=dataset,
        batch_size=cli_args.batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=collator,
    )
    single_loader = MoleculeDataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)
    single_iter = iter(single_loader)

    rows = []
    traversal_index = 0
    flagged_seen = 0
    controls_seen = 0
    pools_full = False
    for graph_input, archived_batch, _, smiles_batch in tqdm(loader, desc="Scoring training reactions"):
        archived_batch = archived_batch.to(args.device)
        memory, memory_mask, _ = model.encode_reaction(graph_input)
        archived_embeds = model.encode_reagents(archived_batch)
        archived_scores = model.compute_itm_score(memory, memory_mask, archived_embeds, archived_batch)

        for batch_index in range(archived_batch.size(0)):
            source_row = frame.iloc[traversal_index]
            source_index = int(source_indices[traversal_index])
            templates = tuple(source_row.get(name, np.nan) for name in ("tpl_SMARTS_r1", "tpl_SMARTS_r0", "tpl_SMARTS_r0*"))
            single_batch = next(single_iter)
            candidate_indices, _ = _get_baseline_top1_indices(
                baseline_models,
                single_batch.batch_graph(),
                templates,
                libraries,
                condition_key,
                col_to_file,
                vocab_mappings,
                total_vocab_size,
            )
            has_candidate = bool(candidate_indices)
            candidate_indices = candidate_indices[: args.max_reagents]
            candidate_indices += [total_vocab_size] * (args.max_reagents - len(candidate_indices))
            candidate_tensor = torch.tensor([candidate_indices], dtype=torch.long, device=args.device)
            if has_candidate:
                candidate_embeds = model.encode_reagents(candidate_tensor)
                candidate_score = model.compute_itm_score(
                    memory[batch_index : batch_index + 1],
                    memory_mask[batch_index : batch_index + 1] if memory_mask is not None else None,
                    candidate_embeds,
                    candidate_tensor,
                )[0]
                score = torch.sigmoid(candidate_score - archived_scores[batch_index]).item()
            else:
                # Match run_capfilt.py: an absent proposal cannot flag the archive.
                score = 0.0
            archived_indices = archived_batch[batch_index].tolist()
            archived_text, archived_slots = condition_text(archived_indices, idx_to_entity, total_vocab_size)
            candidate_text, candidate_slots = condition_text(candidate_indices, idx_to_entity, total_vocab_size)
            archived_set = {i for i in archived_indices if i != total_vocab_size}
            candidate_set = {i for i in candidate_indices if i != total_vocab_size}
            rows.append(
                {
                    "source_index": source_index,
                    "reaction_smiles": smiles_batch[batch_index][0],
                    "archived_condition": archived_text,
                    "archived_condition_slots": archived_slots,
                    "candidate_condition": candidate_text,
                    "candidate_condition_slots": candidate_slots,
                    "unicon_score": score,
                    "candidate_matches_archived": archived_set == candidate_set,
                }
            )
            is_non_exact = archived_set != candidate_set
            is_flagged = is_non_exact and score > cli_args.threshold
            flagged_seen += int(is_flagged)
            controls_seen += int(is_non_exact and not is_flagged)
            traversal_index += 1
            if not cli_args.scan_all and flagged_seen >= cli_args.n and controls_seen >= cli_args.n:
                pools_full = True
                break
        if pools_full:
            break
    return rows


def main():
    args = parse_args()
    rows = score_training_set(args)
    selected = select_evaluation_rows(rows, args.n, args.threshold, args.sample_seed)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_frame = pd.DataFrame(selected)
    selected_frame["threshold"] = args.threshold
    selected_frame["sample_seed"] = args.sample_seed
    selected_frame["candidate_source"] = "reacon_top1"
    public_columns = [
        "evaluation_id",
        "source_index",
        "reaction_smiles",
        "archived_condition",
        "archived_condition_slots",
        "candidate_condition",
        "candidate_condition_slots",
    ]
    key_columns = [
        "evaluation_id",
        "source_index",
        "unicon_score",
        "candidate_matches_archived",
        "sampling_group",
        "threshold",
        "sample_seed",
        "candidate_source",
    ]
    key_path = args.key_path or args.output_path.with_name(f"{args.output_path.stem}_key.csv")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    selected_frame[public_columns].to_csv(args.output_path, index=False)
    selected_frame[key_columns].to_csv(key_path, index=False)
    flagged_total = sum(not row["candidate_matches_archived"] and row["unicon_score"] > args.threshold for row in rows)
    print(f"Inspected {len(rows)} randomly ordered training reactions; {flagged_total} exceed {args.threshold:g}.")
    print(f"Wrote {len(selected)} blinded evaluation rows to {args.output_path}")
    print(f"Wrote the hidden score/cohort key to {key_path}")


if __name__ == "__main__":
    main()
