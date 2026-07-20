# Condition preference evaluation

This workflow compares recorded test-set conditions with a ReaCon top-1
proposal for a seeded random sample in a blinded Streamlit interface.

## Files

- `prepare_inputs.py`: randomly samples N non-exact test pairs without replacement.
- `annotate.py`: displays reactions and randomly ordered condition options.
- `export_labels.py`: converts UI choices to labels keyed by test-set index.
- `../generate_figures.py`: pre-renders scalable reaction and component SVGs.
- `data/`: prepared inputs, raw human responses, and exported labels.

## Data flow

```text
GCN_data_test.csv
  -> annotation_inputs.csv
  -> human_annotations.csv
  -> expert_labels.json
```

Input rows contain `evaluation_id`, original `source_index`, `reaction_smiles`,
conditions A/B, and their slot labels. Defaults are N=50 and sample seed 42.
The generator traverses a seeded random permutation of the test set, skips rows
where the archived and ReaCon condition sets match exactly, and stops after N
non-exact pairs have been collected.
Condition A is the archived label and condition B is the ReaCon proposal. The
UI hides that identity by randomizing the displayed options and recording
`is_option_1_GT` with each response. Experts may also save an optional `notes`
field, particularly to explain a **Cannot determine** decision.

Exported labels use the original test-set indices and are `gt`, `baseline`,
`cannot_determine`. The annotation interface presents exactly **Prefer Option
1**, **Prefer Option 2**, and **Cannot determine**. Incomplete annotations are
rejected by default; use `--allow-partial` only for an intentionally incomplete
export.

`data/historical_ranges/` preserves the earlier first-150 evaluation and is not
read by the randomized pipeline.

The Streamlit app does not invoke RDKit to draw displayed rows. Preparation
creates the required files in `../figs/condition_preference/`; missing or stale
assets are reported in the UI and can be regenerated with
`python expert_annotation/generate_figures.py --workflow condition_preference`.
