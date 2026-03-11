import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import MultiHeadConfig
from .dataset import JsonlMultiHeadDataset, collate_fn
from .decoder import decode_final_spans, select_non_overlapping_typed_spans
from .labels import TYPE_ID_TO_LABEL
from .model import MultiHeadPiiModel
from .type_comparison import ValueKey, classify_value_relationships, make_value_key
from .train import resolve_device


def _extract_value(text: str, start: int, end: int) -> str:
    safe_start = max(0, min(start, len(text)))
    safe_end = max(safe_start, min(end, len(text)))
    return text[safe_start:safe_end]


def _value_dict(label: str, value: str) -> Dict:
    return {
        "value": value,
        "label": label,
    }


def _load_checkpoint(checkpoint_path: str, device: str) -> Dict:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def _build_model_from_checkpoint(
    checkpoint: Dict,
    device: str,
) -> MultiHeadPiiModel:
    cfg = MultiHeadConfig(**checkpoint["config"])
    model = MultiHeadPiiModel(
        model_name=cfg.model_name,
        max_span_len=cfg.max_span_len,
        span_width_vocab_size=cfg.span_width_vocab_size,
        dropout=cfg.dropout,
        proposal_loss_weight=cfg.proposal_loss_weight,
        type_loss_weight=cfg.type_loss_weight,
        sensitivity_loss_weight=cfg.sensitivity_loss_weight,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-head PII inference.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--output", default="outputs_multihead/predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--sensitivity", default=None, help="Optional sensitivity companion JSONL.")
    parser.add_argument(
        "--pretty-output",
        default=None,
        help=(
            "Optional pretty JSON output path. Defaults to <output stem>.pretty.jsonl "
            "when not provided."
        ),
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, device=device)
    config = MultiHeadConfig(**checkpoint["config"])

    encoder_source = str((Path(args.checkpoint).parent / "encoder").resolve())
    if not (Path(encoder_source).exists()):
        encoder_source = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(encoder_source, use_fast=True)

    dataset = JsonlMultiHeadDataset(
        path=args.input,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_span_len=config.max_span_len,
        negative_sample_rate=1.0,
        training=False,
        sensitivity_path=args.sensitivity,
        include_regex_candidates=config.include_regex_candidates,
        lookalike_redact_target=config.lookalike_redact_target,
        no_info_keep_target=config.no_info_keep_target,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = _build_model_from_checkpoint(checkpoint, device=device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    for batch in loader:
        batch_tensors = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        outputs = model(
            input_ids=batch_tensors["input_ids"],
            attention_mask=batch_tensors["attention_mask"],
            candidate_spans=batch_tensors["candidate_spans"],
        )

        type_logits = outputs["type_logits"]
        sens_logits = outputs["sensitivity_logits"]
        cursor = 0
        for i, raw in enumerate(batch["raw"]):
            spans_i = batch_tensors["candidate_spans"][i]
            valid_mask = spans_i[:, 0] >= 0
            n_valid = int(valid_mask.sum().item())
            spans_valid = spans_i[valid_mask]
            if n_valid == 0 or type_logits is None or sens_logits is None:
                all_rows.append(
                    {
                        "text": raw["text"],
                        "redactions": [],
                        "type_comparison": {
                            "true_positives": [],
                            "false_positives": [],
                            "false_negatives": [],
                        },
                    }
                )
                continue

            type_i = type_logits[cursor : cursor + n_valid]
            sens_i = sens_logits[cursor : cursor + n_valid]
            cursor += n_valid

            final_redact = decode_final_spans(
                text=raw["text"],
                offsets=raw["offsets"],
                candidate_spans=spans_valid,
                type_logits=type_i,
                sensitivity_logits=sens_i,
                redact_score_threshold=config.redact_score_threshold,
                nms_iou_threshold=config.nms_iou_threshold,
            )

            type_probs = torch.softmax(type_i, dim=-1)
            pred_type_candidates: List[Tuple[int, int, str, float]] = []
            for j in range(n_valid):
                span = spans_valid[j]
                token_start = int(span[0].item())
                token_end = int(span[1].item())
                char_start = raw["offsets"][token_start][0]
                char_end = raw["offsets"][token_end][1]
                if char_end <= char_start:
                    continue
                type_id = int(type_probs[j].argmax(dim=-1).item())
                label = TYPE_ID_TO_LABEL[type_id]
                if label == "NONE":
                    continue
                score = float(type_probs[j, type_id].item())
                pred_type_candidates.append((char_start, char_end, label, score))

            # Remove nested/overlapping predictions to avoid contradictory TP/FP accounting.
            pred_type_set: Set[Tuple[int, int, str]] = set(
                select_non_overlapping_typed_spans(pred_type_candidates)
            )

            pred_type_value_keys: Set[ValueKey] = set()
            for start, end, label in pred_type_set:
                value = _extract_value(raw["text"], start, end)
                pred_type_value_keys.add(make_value_key(label, value))

            gold_type_value_keys: Set[ValueKey] = set()
            for span in raw.get("gold_type_spans", []):
                label = span.get("label")
                value = span.get("value")
                start = span.get("start")
                end = span.get("end")
                if not isinstance(label, str) or label == "NONE":
                    continue
                if isinstance(value, str):
                    gold_type_value_keys.add(make_value_key(label, value))
                    continue
                if isinstance(start, int) and isinstance(end, int):
                    gold_value = _extract_value(raw["text"], start, end)
                    gold_type_value_keys.add(make_value_key(label, gold_value))

            comparison = classify_value_relationships(
                pred_keys=pred_type_value_keys,
                gold_keys=gold_type_value_keys,
            )

            all_rows.append(
                {
                    "text": raw["text"],
                    "redactions": [
                        {
                            "start": span.start,
                            "end": span.end,
                            "value": _extract_value(raw["text"], span.start, span.end),
                            "label": span.label,
                            "decision": span.decision,
                            "redact_score": span.redact_score,
                            "redact_probability": span.redact_probability,
                            "type_confidence": span.type_confidence,
                        }
                        for span in final_redact
                    ],
                    "type_comparison": {
                        "true_positives": [
                            _value_dict(label=row["label"], value=row["value"])
                            for row in comparison["exact_tp"]
                        ],
                        "false_positives": [
                            _value_dict(label=row["label"], value=row["value"])
                            for row in comparison["exact_fp"]
                        ],
                        "false_negatives": [
                            _value_dict(label=row["label"], value=row["value"])
                            for row in comparison["exact_fn"]
                        ],
                        "type_mismatches": comparison["type_mismatches"],
                        "value_subsets": comparison["value_subsets"],
                        "value_supersets": comparison["value_supersets"],
                    },
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=True, indent=2) + "\n")
    print(f"saved predictions to {output_path.resolve()}")


if __name__ == "__main__":
    main()
