import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


SpanKey = Tuple[int, int, str]
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert inference JSONL to per-row type_comparison "
            "(true_positives/false_positives/false_negatives)."
        )
    )
    parser.add_argument(
        "--predictions",
        default="outputs_multihead/predictions.with_value.jsonl",
        help="Inference JSONL produced by multihead inference.",
    )
    parser.add_argument(
        "--gold",
        default="valid.gpt-5-nano.jsonl",
        help="Gold JSONL aligned 1:1 with predictions rows.",
    )
    parser.add_argument(
        "--output",
        default="outputs_multihead/predictions.tp_fp_fn.jsonl",
        help="Destination JSONL path.",
    )
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: invalid JSON ({exc})") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path} line {line_no}: row is not an object")
            yield obj


def _extract_value(text: str, start: int, end: int) -> str:
    safe_start = max(0, min(start, len(text)))
    safe_end = max(safe_start, min(end, len(text)))
    return text[safe_start:safe_end]


def _build_span_key_rows(rows: List[Dict]) -> List[SpanKey]:
    out: List[SpanKey] = []
    for span in rows:
        if not isinstance(span, dict):
            continue
        start = span.get("start")
        end = span.get("end")
        label = span.get("label")
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(label, str):
            continue
        if label == "NONE":
            continue
        out.append((start, end, label))
    return out


def _gold_rows(gold_row: Dict) -> List[SpanKey]:
    spans = gold_row.get("spans", [])
    items = gold_row.get("items", [])
    return _build_span_key_rows(spans if isinstance(spans, list) else []) + _build_span_key_rows(
        items if isinstance(items, list) else []
    )


def _pred_rows(pred_row: Dict) -> List[SpanKey]:
    typed = pred_row.get("typed_predictions", [])
    return _build_span_key_rows(typed if isinstance(typed, list) else [])


def _span_dict(text: str, start: int, end: int, label: str) -> Dict:
    return {
        "start": start,
        "end": end,
        "value": _extract_value(text, start, end),
        "label": label,
    }


def _is_suspicious_span(text: str, start: int, end: int, label: str) -> bool:
    value = _extract_value(text, start, end)
    if not value.strip():
        return True

    if label == "EMAIL":
        return EMAIL_RE.fullmatch(value.strip()) is None
    if label == "IP_ADDRESS":
        return IPV4_RE.fullmatch(value.strip()) is None
    if label == "CREDIT_CARD":
        digits = sum(ch.isdigit() for ch in value)
        if digits < 13:
            return True
        left_char = text[start - 1] if start > 0 else ""
        right_char = text[end] if end < len(text) else ""
        # If span stops mid digit-run, this likely indicates a broken endpoint.
        return bool(left_char.isdigit() or right_char.isdigit())
    if label == "ADDRESS":
        if end < len(text) and value and value[-1].isdigit() and text[end].isdigit():
            return True
    return False


def _compute_comparison(
    pred_rows: List[SpanKey],
    gold_rows: List[SpanKey],
) -> Tuple[List[SpanKey], List[SpanKey], List[SpanKey]]:
    pred_set = set(pred_rows)
    gold_set = set(gold_rows)
    # Exact matching only: comparison should reflect exact char spans.
    tp = sorted(pred_set & gold_set)
    fp = sorted(pred_set - gold_set)
    fn = sorted(gold_set - pred_set)
    return tp, fp, fn


def main() -> None:
    args = parse_args()
    pred_path = Path(args.predictions)
    gold_path = Path(args.gold)
    out_path = Path(args.output)

    pred_rows = list(_iter_jsonl(pred_path))
    gold_rows = list(_iter_jsonl(gold_path))
    if len(pred_rows) != len(gold_rows):
        raise ValueError(
            "Row-count mismatch between prediction and gold files: "
            f"{len(pred_rows)} vs {len(gold_rows)}"
        )

    converted: List[Dict] = []
    suspicious_gold_spans = 0
    for pred_row, gold_row in zip(pred_rows, gold_rows):
        text = pred_row.get("text")
        if not isinstance(text, str):
            text = gold_row.get("text", "")
            if not isinstance(text, str):
                text = ""

        pred_rows_i = _pred_rows(pred_row)
        gold_rows_i = _gold_rows(gold_row)
        for start, end, label in gold_rows_i:
            if _is_suspicious_span(text, start, end, label):
                suspicious_gold_spans += 1

        tp, fp, fn = _compute_comparison(
            pred_rows=pred_rows_i,
            gold_rows=gold_rows_i,
        )

        pred_row.pop("typed_predictions", None)
        pred_row["type_comparison"] = {
            "true_positives": [_span_dict(text, s, e, label) for s, e, label in tp],
            "false_positives": [_span_dict(text, s, e, label) for s, e, label in fp],
            "false_negatives": [_span_dict(text, s, e, label) for s, e, label in fn],
        }
        converted.append(pred_row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {len(converted)} rows to {out_path.resolve()}")
    if suspicious_gold_spans > 0:
        print(
            "WARNING: detected suspicious gold spans "
            f"(likely boundary/indexing issues): {suspicious_gold_spans}"
        )


if __name__ == "__main__":
    main()
