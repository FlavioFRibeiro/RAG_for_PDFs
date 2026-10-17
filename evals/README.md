# Evaluation

`datasets/manual_qa.yaml` contains 48 human-checked cases: 32 development cases and 16 reserved cases, evenly split across the two manuals. Physical PDF page numbers are used as retrieval relevance labels. Passing this finite set is not a guarantee of universal accuracy.

Validate the dataset without credentials:

```bash
python scripts/evaluate.py --validate-only
```

After indexes have been built, run retrieval evaluation (OpenAI query embeddings incur usage):

```bash
python scripts/evaluate.py --split development
```

Add `--generation` to run the bounded answer evaluation. Reports are written to `evals/reports/` and are intentionally ignored by Git because they represent environment- and model-specific measurements.
