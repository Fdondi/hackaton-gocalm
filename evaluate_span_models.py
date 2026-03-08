import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from pii_labels import TRAINING_LABELS
from train_modernbert_span_classifier import (
    MAX_LENGTH,
    MAX_SPAN_LEN,
    JsonlSpanDataset,
    SpanClassifier,
    collate_fn,
    set_seed,
)


@dataclass
class ModelSpec:
    name: str
    base_model: str
    checkpoint: Optional[str] = None
    encoder_path: Optional[str] = None
    max_span_len: Optional[int] = None
    max_length: Optional[int] = None


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def _token_span_to_char_span(
    offsets: List[Tuple[int, int]], token_start: int, token_end: int
) -> Optional[Tuple[int, int]]:
    if token_start < 0 or token_end >= len(offsets) or token_start > token_end:
        return None
    char_start = offsets[token_start][0]
    char_end = offsets[token_end][1]
    if char_end <= char_start:
        return None
    return char_start, char_end


def _load_model_specs(path: Path) -> List[ModelSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("models config must be a non-empty JSON array")

    specs: List[ModelSpec] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"model spec at index {idx} must be a JSON object")
        if "name" not in row or "base_model" not in row:
            raise ValueError(
                f"model spec at index {idx} requires 'name' and 'base_model'"
            )
        specs.append(
            ModelSpec(
                name=row["name"],
                base_model=row["base_model"],
                checkpoint=row.get("checkpoint"),
                encoder_path=row.get("encoder_path"),
                max_span_len=row.get("max_span_len"),
                max_length=row.get("max_length"),
            )
        )
    return specs


def _model_encoder_source(spec: ModelSpec) -> str:
    return spec.encoder_path or spec.base_model


def _load_classifier_weights(
    model: SpanClassifier,
    checkpoint_path: Path,
    expected_label2id: Dict[str, int],
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["classifier"] if isinstance(checkpoint, dict) and "classifier" in checkpoint else checkpoint
    if isinstance(checkpoint, dict) and "label2id" in checkpoint:
        saved_label2id = checkpoint["label2id"]
        if saved_label2id != expected_label2id:
            raise ValueError(
                f"label2id mismatch in checkpoint {checkpoint_path}. "
                "Use matching label taxonomy."
            )
    model.load_state_dict(state_dict, strict=True)


@torch.no_grad()
def evaluate_model(
    spec: ModelSpec,
    valid_path: Path,
    label2id: Dict[str, int],
    batch_size: int,
    device: str,
    skip_missing: bool,
) -> Optional[Dict]:
    none_id = label2id["NONE"]
    id2label = {v: k for k, v in label2id.items()}
    labels_no_none = [l for l in TRAINING_LABELS if l != "NONE"]

    encoder_source = _model_encoder_source(spec)
    tokenizer = AutoTokenizer.from_pretrained(encoder_source, use_fast=True)
    dataset = JsonlSpanDataset(
        path=str(valid_path),
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=spec.max_length or MAX_LENGTH,
        max_span_len=spec.max_span_len or MAX_SPAN_LEN,
        negative_sample_rate=1.0,  # deterministic eval over full candidate space
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = SpanClassifier(
        model_name=encoder_source,
        num_labels=len(label2id),
        max_span_len=spec.max_span_len or MAX_SPAN_LEN,
    ).to(device)

    if spec.checkpoint:
        checkpoint_path = Path(spec.checkpoint)
        if not checkpoint_path.exists():
            msg = (
                f"checkpoint not found for model '{spec.name}': "
                f"{checkpoint_path.resolve()}"
            )
            if skip_missing:
                print(f"[skip] {msg}")
                return None
            raise FileNotFoundError(msg)
        _load_classifier_weights(model, checkpoint_path, expected_label2id=label2id)

    model.eval()

    total_loss = 0.0
    total_batches = 0
    total_candidates = 0
    total_candidates_correct = 0
    positive_candidates = 0
    positive_candidates_correct = 0

    docs = 0
    exact_match_docs = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    skipped_docs = 0
    dataset_example_idx = 0

    per_label = {
        label: {"tp": 0, "fp": 0, "fn": 0, "support": 0, "predicted": 0}
        for label in labels_no_none
    }

    for batch in loader:
        batch_tensors = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        outputs = model(
            input_ids=batch_tensors["input_ids"],
            attention_mask=batch_tensors["attention_mask"],
            candidate_spans=batch_tensors["candidate_spans"],
            labels=batch_tensors["labels"],
        )
        total_loss += outputs["loss"].item()
        total_batches += 1

        logits = outputs["logits"]
        flat_pred = logits.argmax(dim=-1)

        cursor = 0
        for i, raw_example in enumerate(batch["raw"]):
            spans_i = batch_tensors["candidate_spans"][i]
            labels_i = batch_tensors["labels"][i]
            valid_mask = spans_i[:, 0] >= 0
            n_valid = int(valid_mask.sum().item())
            if n_valid == 0:
                docs += 1
                dataset_example_idx += 1
                continue

            pred_i = flat_pred[cursor : cursor + n_valid]
            gold_i = labels_i[valid_mask]
            spans_i_valid = spans_i[valid_mask]
            cursor += n_valid

            total_candidates += n_valid
            total_candidates_correct += int((pred_i == gold_i).sum().item())

            pos_mask = gold_i != none_id
            pos_count = int(pos_mask.sum().item())
            positive_candidates += pos_count
            if pos_count:
                positive_candidates_correct += int((pred_i[pos_mask] == gold_i[pos_mask]).sum().item())

            offsets = raw_example.get("offsets")
            gold_spans = raw_example.get("spans")
            if gold_spans is None and dataset_example_idx < len(dataset.examples):
                # collate_fn carries __getitem__ output as "raw"; recover spans from source examples.
                gold_spans = dataset.examples[dataset_example_idx].spans
            if not isinstance(offsets, list) or not isinstance(gold_spans, list):
                skipped_docs += 1
                print(
                    f"[skip] model={spec.name} doc_index={dataset_example_idx} "
                    f"skipped_total={skipped_docs} reason=missing_or_invalid_offsets_or_spans"
                )
                dataset_example_idx += 1
                continue

            pred_set: Set[Tuple[int, int, str]] = set()
            for j in range(n_valid):
                pred_label_id = int(pred_i[j].item())
                if pred_label_id == none_id:
                    continue
                tok_s = int(spans_i_valid[j, 0].item())
                tok_e = int(spans_i_valid[j, 1].item())
                char_span = _token_span_to_char_span(offsets, tok_s, tok_e)
                if char_span is None:
                    continue
                pred_set.add((char_span[0], char_span[1], id2label[pred_label_id]))

            gold_set: Set[Tuple[int, int, str]] = set()
            for gold_span in gold_spans:
                if not isinstance(gold_span, dict):
                    continue
                label = gold_span.get("label")
                if label == "NONE" or label not in label2id:
                    continue
                start = gold_span.get("start")
                end = gold_span.get("end")
                if not isinstance(start, int) or not isinstance(end, int):
                    continue
                gold_set.add((start, end, label))

            docs += 1
            dataset_example_idx += 1
            if pred_set == gold_set:
                exact_match_docs += 1

            tp = len(pred_set & gold_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            for label in labels_no_none:
                gold_l = {x for x in gold_set if x[2] == label}
                pred_l = {x for x in pred_set if x[2] == label}
                per_label[label]["tp"] += len(gold_l & pred_l)
                per_label[label]["fp"] += len(pred_l - gold_l)
                per_label[label]["fn"] += len(gold_l - pred_l)
                per_label[label]["support"] += len(gold_l)
                per_label[label]["predicted"] += len(pred_l)

    micro = _prf(total_tp, total_fp, total_fn)

    per_label_metrics = {}
    macro_f1_all: List[float] = []
    macro_f1_present: List[float] = []
    for label in labels_no_none:
        counts = per_label[label]
        m = _prf(counts["tp"], counts["fp"], counts["fn"])
        m["support"] = counts["support"]
        m["predicted"] = counts["predicted"]
        per_label_metrics[label] = m
        macro_f1_all.append(m["f1"])
        if counts["support"] > 0:
            macro_f1_present.append(m["f1"])

    report = {
        "model_name": spec.name,
        "base_model": spec.base_model,
        "encoder_source": encoder_source,
        "checkpoint": spec.checkpoint,
        "num_docs": docs,
        "avg_loss": _safe_div(total_loss, total_batches),
        "candidate_accuracy": _safe_div(total_candidates_correct, total_candidates),
        "candidate_accuracy_positive_only": _safe_div(
            positive_candidates_correct, positive_candidates
        ),
        "span_micro_precision": micro["precision"],
        "span_micro_recall": micro["recall"],
        "span_micro_f1": micro["f1"],
        "span_macro_f1_all_labels": statistics.mean(macro_f1_all) if macro_f1_all else 0.0,
        "span_macro_f1_present_labels": statistics.mean(macro_f1_present)
        if macro_f1_present
        else 0.0,
        "exact_match_rate": _safe_div(exact_match_docs, docs),
        "span_counts": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "per_label": per_label_metrics,
    }
    return report


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def print_report(report: Dict) -> None:
    print("=" * 88)
    print(f"model: {report['model_name']}")
    print(f"base_model: {report['base_model']}")
    print(f"encoder_source: {report['encoder_source']}")
    print(f"checkpoint: {report['checkpoint'] or 'none (untrained classifier)'}")
    print("-" * 88)
    print(
        f"docs={report['num_docs']} "
        f"loss={_fmt(report['avg_loss'])} "
        f"cand_acc={_fmt(report['candidate_accuracy'])} "
        f"cand_pos_acc={_fmt(report['candidate_accuracy_positive_only'])}"
    )
    print(
        f"span_micro_p={_fmt(report['span_micro_precision'])} "
        f"span_micro_r={_fmt(report['span_micro_recall'])} "
        f"span_micro_f1={_fmt(report['span_micro_f1'])}"
    )
    print(
        f"span_macro_f1_all={_fmt(report['span_macro_f1_all_labels'])} "
        f"span_macro_f1_present={_fmt(report['span_macro_f1_present_labels'])} "
        f"doc_exact_match={_fmt(report['exact_match_rate'])}"
    )
    counts = report["span_counts"]
    print(f"span_counts tp={counts['tp']} fp={counts['fp']} fn={counts['fn']}")
    print("-" * 88)
    print("per-label metrics:")
    for label, metrics in report["per_label"].items():
        print(
            f"  {label:14s} "
            f"p={_fmt(metrics['precision'])} "
            f"r={_fmt(metrics['recall'])} "
            f"f1={_fmt(metrics['f1'])} "
            f"support={metrics['support']} "
            f"pred={metrics['predicted']}"
        )
    print("=" * 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more span-classifier models on JSONL validation data."
    )
    parser.add_argument(
        "--valid",
        default="valid.jsonl",
        help="Path to validation JSONL.",
    )
    parser.add_argument(
        "--models-config",
        default="models_to_eval.json",
        help="JSON file with a list of model specs to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run evaluation on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        default="evaluation_report.json",
        help="Path to write the full JSON report.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip model entries whose checkpoint path does not exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    if device == "auto":
        device = "cpu"

    label2id = {label: i for i, label in enumerate(TRAINING_LABELS)}
    specs = _load_model_specs(Path(args.models_config))
    valid_path = Path(args.valid)

    reports: List[Dict] = []
    for spec in specs:
        report = evaluate_model(
            spec=spec,
            valid_path=valid_path,
            label2id=label2id,
            batch_size=args.batch_size,
            device=device,
            skip_missing=args.skip_missing,
        )
        if report is None:
            continue
        reports.append(report)
        print_report(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "device": device,
                "valid_path": str(valid_path),
                "results": reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved report to {output_path.resolve()}")


if __name__ == "__main__":
    main()
