# Expert annotation workflows

This directory contains two independent human-evaluation workflows. Each one
keeps its scripts and generated data together so that input preparation, human
annotation, and result processing are easy to follow.

```text
expert_annotation/
├── common/                  shared CSV annotation storage
├── condition_preference/    archived conditions vs. ReaCon conditions
│   ├── prepare_inputs.py
│   ├── annotate.py
│   ├── export_labels.py
│   └── data/
├── training_flagging/       evaluation of UniCon training-data flags
│   ├── prepare_inputs.py
│   ├── annotate.py
│   ├── analyze_results.py
│   └── data/
├── archive/                 archived experimental code
├── figs/                    generated high-resolution UI images (gitignored)
└── _cache/                  generated UniCon vocabulary cache
```

## Condition preference evaluation

Use this workflow when a chemist should choose between the archived condition
set and a ReaCon proposal. The app randomizes their left/right positions.

```bash
python expert_annotation/condition_preference/prepare_inputs.py
streamlit run expert_annotation/condition_preference/annotate.py
python expert_annotation/condition_preference/export_labels.py
```

The preparation step randomly samples non-exact archived/ReaCon pairs using
`--sample-seed 42` by default. Exact condition matches are skipped. The final labels are written to
`condition_preference/data/expert_labels.json`. `eval_attention.py` reads that
file directly.

Input preparation also pre-renders each reaction and condition component as a
scalable SVG under `figs/condition_preference/`. To regenerate static images for
an existing CSV without running model inference:

```bash
python expert_annotation/generate_figures.py --workflow condition_preference
```

See [condition_preference/README.md](condition_preference/README.md) for schemas
and conversion behavior.

## Training-set flag evaluation

Use this workflow to test whether high UniConScore flags are enriched for
questionable archived conditions.

```bash
python expert_annotation/training_flagging/prepare_inputs.py
streamlit run expert_annotation/training_flagging/annotate.py
python expert_annotation/training_flagging/analyze_results.py
```

The default preparation samples 50 flags above 0.75 and 50 controls from
non-exact archived/ReaCon pairs. It writes the annotator-facing rows and the
hidden score/cohort key separately. Do not give `annotation_inputs_key.csv` to
the annotator before evaluation finishes.

The preparation command writes the corresponding static images under
`figs/training_flagging/`. Generate or refresh both workflows' existing inputs
with:

```bash
python expert_annotation/generate_figures.py --overwrite
```

See [training_flagging/README.md](training_flagging/README.md) for the sampling
rule, response schema, and output files.

## Github publishing

The cleanest approach is `git subtree`, which publishes only `expert_annotation` while preserving its relevant Git history.

Create the destination repository as an empty repo—don’t initialize it with a README or license—then use:

```bash
# Run from the root of the current repository
git remote add expert-annotation <NEW_REPOSITORY_URL>

git subtree split \
  --prefix=expert_annotation \
  --branch=expert-annotation-publish

git push -u expert-annotation expert-annotation-publish:main
```

In the new repository, the contents of `expert_annotation` will appear at the repository root; no other folders from the original repo are included.

For later updates:

```bash
git subtree split \
  --prefix=expert_annotation \
  --branch=expert-annotation-publish

git push expert-annotation expert-annotation-publish:main
```

Optionally remove the local publishing branch afterward:

```bash
git branch -D expert-annotation-publish
```

Before publishing publicly, check the folder’s complete history for secrets or large files—the split preserves historical commits affecting that folder. If you don’t want any history, copy the folder into a fresh local repository and make a new initial commit instead.
