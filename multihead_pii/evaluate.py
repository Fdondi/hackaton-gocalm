import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import MultiHeadConfig
from .dataset import JsonlMultiHeadDataset, collate_fn
from .decoder import decode_final_spans
from .labels import SENSITIVITY_LABEL_TO_ID, TYPE_ID_TO_LABEL
from .model import MultiHeadPiiModel
from .train import resolve_device


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-head PII model.")
    parser.add_argument("--valid", required=True, help="Validation JSONL path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--output", default="outputs_multihead/eval_report.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--sensitivity", default=None, help="Optional sensitivity companion JSONL.")
    return parser.parse_args()


def _load_model(checkpoint_path: str, device: str) -> Tuple[MultiHeadPiiModel, MultiHeadConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = MultiHeadConfig(**checkpoint["config"])
    model = MultiHeadPiiModel(
        model_name=config.model_name,
        max_span_len=config.max_span_len,
        span_width_vocab_size=config.span_width_vocab_size,
        dropout=config.dropout,
        proposal_loss_weight=config.proposal_loss_weight,
        type_loss_weight=config.type_loss_weight,
        sensitivity_loss_weight=config.sensitivity_loss_weight,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, config = _load_model(args.checkpoint, device=device)

    encoder_source = str((Path(args.checkpoint).parent / "encoder").resolve())
    if not Path(encoder_source).exists():
        encoder_source = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(encoder_source, use_fast=True)

    dataset = JsonlMultiHeadDataset(
        path=args.valid,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_span_len=config.max_span_len,
        negative_sample_rate=1.0,
        training=False,
        sensitivity_path=args.sensitivity,
        include_regex_candidates=config.include_regex_candidates,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    losses = {"loss": 0.0, "proposal_loss": 0.0, "type_loss": 0.0, "sensitivity_loss": 0.0}
    total_batches = 0

    proposal_correct = 0
    proposal_total = 0
    sens_correct = 0
    sens_total = 0
    sens_mae_sum = 0.0
    sens_brier_sum = 0.0

    type_tp = type_fp = type_fn = 0
    redact_tp = redact_fp = redact_fn = 0
    docs = 0

    for batch in loader:
        batch_tensors = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        outputs = model(
            input_ids=batch_tensors["input_ids"],
            attention_mask=batch_tensors["attention_mask"],
            candidate_spans=batch_tensors["candidate_spans"],
            proposal_labels=batch_tensors["proposal_labels"],
            type_labels=batch_tensors["type_labels"],
            sensitivity_labels=batch_tensors["sensitivity_labels"],
        )
        for key in losses:
            losses[key] += float(outputs[key].item())
        total_batches += 1

        proposal_logits = outputs["proposal_logits"]
        proposal_labels = batch_tensors["proposal_labels"]
        proposal_pred = proposal_logits.argmax(dim=-1)
        valid_tokens = proposal_labels != -100
        proposal_correct += int((proposal_pred[valid_tokens] == proposal_labels[valid_tokens]).sum().item())
        proposal_total += int(valid_tokens.sum().item())

        type_logits = outputs["type_logits"]
        sensitivity_logits = outputs["sensitivity_logits"]
        cursor = 0
        for i, raw in enumerate(batch["raw"]):
            docs += 1
            spans_i = batch_tensors["candidate_spans"][i]
            valid_mask = spans_i[:, 0] >= 0
            n_valid = int(valid_mask.sum().item())
            if n_valid == 0 or type_logits is None or sensitivity_logits is None:
                continue

            spans_valid = spans_i[valid_mask]
            type_i = type_logits[cursor : cursor + n_valid]
            sens_i = sensitivity_logits[cursor : cursor + n_valid]
            type_labels_i = batch_tensors["type_labels"][i][valid_mask]
            sens_labels_i = batch_tensors["sensitivity_labels"][i][valid_mask]
            cursor += n_valid

            sens_pred = sens_i.argmax(dim=-1)
            sens_mask = sens_labels_i != -100
            sens_correct += int((sens_pred[sens_mask] == sens_labels_i[sens_mask]).sum().item())
            sens_total += int(sens_mask.sum().item())
            if int(sens_mask.sum().item()) > 0:
                sens_probs = torch.softmax(sens_i, dim=-1)
                redact_probs = sens_probs[:, SENSITIVITY_LABEL_TO_ID["REDACT"]]
                targets = (sens_labels_i == SENSITIVITY_LABEL_TO_ID["REDACT"]).float()
                masked_preds = redact_probs[sens_mask]
                masked_targets = targets[sens_mask]
                sens_mae_sum += float(torch.abs(masked_preds - masked_targets).sum().item())
                sens_brier_sum += float(((masked_preds - masked_targets) ** 2).sum().item())

            pred_type_set: Set[Tuple[int, int, str]] = set()
            type_probs = torch.softmax(type_i, dim=-1)
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
                pred_type_set.add((char_start, char_end, label))

            gold_type_set: Set[Tuple[int, int, str]] = set()
            for span in raw.get("gold_type_spans", []):
                start = span.get("start")
                end = span.get("end")
                label = span.get("label")
                if isinstance(start, int) and isinstance(end, int) and isinstance(label, str):
                    gold_type_set.add((start, end, label))

            type_tp += len(pred_type_set & gold_type_set)
            type_fp += len(pred_type_set - gold_type_set)
            type_fn += len(gold_type_set - pred_type_set)

            final_redact = decode_final_spans(
                offsets=raw["offsets"],
                candidate_spans=spans_valid,
                type_logits=type_i,
                sensitivity_logits=sens_i,
                redact_score_threshold=config.redact_score_threshold,
                nms_iou_threshold=config.nms_iou_threshold,
            )
            pred_redact_set = {(x.start, x.end) for x in final_redact}
            gold_redact_set = {
                (int(x["start"]), int(x["end"]))
                for x in raw.get("gold_redact_spans", [])
                if isinstance(x, dict) and isinstance(x.get("start"), int) and isinstance(x.get("end"), int)
            }
            redact_tp += len(pred_redact_set & gold_redact_set)
            redact_fp += len(pred_redact_set - gold_redact_set)
            redact_fn += len(gold_redact_set - pred_redact_set)

    report = {
        "num_docs": docs,
        "avg_losses": {k: _safe_div(v, total_batches) for k, v in losses.items()},
        "proposal_accuracy": _safe_div(proposal_correct, proposal_total),
        "sensitivity_candidate_accuracy": _safe_div(sens_correct, sens_total),
        "sensitivity_redact_probability_mae": _safe_div(sens_mae_sum, sens_total),
        "sensitivity_redact_probability_brier": _safe_div(sens_brier_sum, sens_total),
        "type_metrics": _prf(type_tp, type_fp, type_fn),
        "redaction_metrics": _prf(redact_tp, redact_fp, redact_fn),
        "span_counts": {
            "type": {"tp": type_tp, "fp": type_fp, "fn": type_fn},
            "redaction": {"tp": redact_tp, "fp": redact_fp, "fn": redact_fn},
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved report to {out_path.resolve()}")


if __name__ == "__main__":
    main()
