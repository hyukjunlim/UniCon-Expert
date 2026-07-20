# Training-set flag evaluation

This workflow evaluates the curation rule described in the paper: ReaCon
proposes an alternative condition set and UniConScore compares it with the
archived training label.

## Sampling

`prepare_inputs.py` visits training reactions in seeded random order and
excludes exact archive/ReaCon condition-set matches from both cohorts. Among
the remaining non-exact rows, a sample is flagged when its
candidate-vs-archive UniConScore is greater than the threshold; controls have
scores at or below the threshold. Defaults are:

- 50 flagged samples
- 50 non-flagged controls
- threshold 0.75
- sample seed 42

Traversal stops when both pools are full. Use `--scan-all` to score the entire
input dataset.

## Output files

All generated files live in `data/`:

- `annotation_inputs.csv`: blinded rows used by the Streamlit app
- `annotation_inputs_key.csv`: hidden scores and cohort membership
- `human_annotations.csv`: saved human responses
- `analysis_results.csv`: joined, unblinded row-level results

For every archived condition component, the annotator selects exactly one issue
label. Each component defaults to "No obvious annotation issue." The annotator
also selects one overall assessment for the archived protocol.
Only the archived protocol is shown; the proposed protocol is not displayed in
the annotation interface.
`analyze_results.py` reports issue and questionable-archive rates for the
flagged and control groups.
