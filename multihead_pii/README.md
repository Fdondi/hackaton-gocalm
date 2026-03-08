# Multi-Head PII

## What it implements

- Shared encoder (`ModernBERT`) with three heads:
  - proposal head (BIO token tags)
  - type head (`NONE` + PII label taxonomy)
  - sensitivity head (`REDACT|KEEP`) with continuous `redact_probability`
- Deterministic decoder for overlap resolution and final redaction spans.
- Standalone train/infer/evaluate entrypoints.

## Files

- `multihead_pii/config.py` - config dataclass + JSON load/save
- `multihead_pii/labels.py` - label vocabularies and mappings
- `multihead_pii/dataset.py` - JSONL dataset adapters + candidate generation
- `multihead_pii/model.py` - shared encoder + multi-head architecture
- `multihead_pii/losses.py` - multitask losses
- `multihead_pii/decoder.py` - postprocessing and conflict handling
- `multihead_pii/train.py` - training CLI
- `multihead_pii/infer.py` - inference CLI
- `multihead_pii/evaluate.py` - evaluation CLI
- `configs/multihead_v1.json` - default config

## Data formats

### Main dataset JSONL

Each row can be either:

1) Span-only format:

```json
{"text":"...", "spans":[{"start":10,"end":20,"label":"EMAIL"}]}
```

2) Rich item format:

```json
{"text":"...", "items":[{"start":10,"end":20,"category":"REAL_PII","label":"EMAIL"}]}
```

Sensitivity targets from `items` are mapped as:

- `REAL_PII` -> `REDACT`
- `PII_LOOKALIKE` -> `KEEP`

### Optional sensitivity companion JSONL

Use this when main rows only have `spans` and you want explicit stage-C supervision.
Rows must align 1:1 by line number with the main dataset:

```json
{"sensitivity_spans":[{"start":10,"end":20,"sensitivity":"REDACT"}]}
```

Allowed sensitivity values: `REDACT`, `KEEP`.

## BIO proposal tags

`BIO` is the token-level format used by the proposal head to mark candidate boundaries.

- `B-ENTITY`: first token of an entity-like span
- `I-ENTITY`: continuation token in the same span
- `O`: outside any entity-like span

Example:

```text
Text:   Jane  Doe   called  support@example.com
Tags:   B     I     O       B
```

How this is used in the pipeline:

1. The proposal head predicts BIO tags for each token.
2. BIO spans are decoded into candidate spans (high recall).
3. Candidate spans are merged with regex candidates.
4. Type and sensitivity heads score each candidate.
5. Decoder outputs final redaction decisions and confidence values.

This means BIO is only for boundary proposal, not final type/sensitivity decisions.

## Train

```bash
python -m multihead_pii.train --train train.gpt-5-nano.jsonl --valid valid.gpt-5-nano.jsonl --config configs/multihead_v1.json --output outputs_multihead
```

Add companion labels if needed:

```bash
python -m multihead_pii.train  --train train.gpt-5-nano.jsonl  --valid valid.gpt-5-nano.jsonl --train-sensitivity train.sensitivity.jsonl --valid-sensitivity valid.sensitivity.jsonl
```

## Inference

```bash
python -m multihead_pii.infer --input valid.gpt-5-nano.jsonl --checkpoint outputs_multihead/multihead_model.pt --output outputs_multihead/predictions.jsonl
```

`typed_predictions` contains per-candidate `redact_probability` (0-1).
`redactions` contains spans chosen by decoder thresholding.

## Evaluate

```bash
python -m multihead_pii.evaluate  --valid valid.gpt-5-nano.jsonl --checkpoint outputs_multihead/multihead_model.pt --output outputs_multihead/eval_report.json
```

Evaluation includes both discrete and continuous sensitivity metrics:

- `sensitivity_candidate_accuracy` (thresholded/argmax)
- `sensitivity_redact_probability_mae`
- `sensitivity_redact_probability_brier`

## Notes

- Existing project files are untouched by this rollout.
- Checkpoints and reports are isolated under `outputs_multihead/`.
