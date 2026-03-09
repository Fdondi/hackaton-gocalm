import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add/refresh `value` fields for index-based span annotations in JSONL. "
            "Useful for inspecting exact text slices used by labels."
        )
    )
    parser.add_argument("--input", required=True, help="Source JSONL path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Destination JSONL path. Defaults to <input stem>.with_values.jsonl",
    )
    parser.add_argument(
        "--fields",
        default="spans,items,sensitivity_spans",
        help="Comma-separated row fields to enrich when present.",
    )
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_no}: row is not an object")
            yield row


def _extract_value(text: str, start: int, end: int) -> str:
    safe_start = max(0, min(start, len(text)))
    safe_end = max(safe_start, min(end, len(text)))
    return text[safe_start:safe_end]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}.with_values{input_path.suffix}")
    )
    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    if not fields:
        raise ValueError("No fields requested. Provide at least one field name in --fields.")

    rows = list(_iter_jsonl(input_path))
    updated_rows: List[Dict] = []
    updated_spans = 0
    mismatched_existing_values = 0

    for idx, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"row {idx}: missing/invalid `text`")
        for field in fields:
            spans = row.get(field)
            if not isinstance(spans, list):
                continue
            for span in spans:
                if not isinstance(span, dict):
                    continue
                start = span.get("start")
                end = span.get("end")
                if not isinstance(start, int) or not isinstance(end, int):
                    continue
                extracted = _extract_value(text, start, end)
                existing_value = span.get("value")
                if isinstance(existing_value, str) and existing_value != extracted:
                    mismatched_existing_values += 1
                span["value"] = extracted
                updated_spans += 1
        updated_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in updated_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"wrote {len(updated_rows)} rows to {output_path.resolve()}")
    print(f"updated value fields: {updated_spans}")
    if mismatched_existing_values > 0:
        print(
            "WARNING: existing value text mismatched span indices for "
            f"{mismatched_existing_values} spans; overwritten with index slices."
        )


if __name__ == "__main__":
    main()
