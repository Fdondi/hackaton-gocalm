import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

LABEL_ALIASES = {
    "PHONE_NUMBER": "PHONE",
    "TELEPHONE": "PHONE",
    "ORGANIZATION": "ORG",
    "IP": "IP_ADDRESS",
    "IPADDRESS": "IP_ADDRESS",
    "DATE_TIME": "OTHER",
    "LOCATION": "OTHER",
    "CITY": "OTHER",
    "COUNTRY": "OTHER",
    "STATE": "OTHER",
    "ZIPCODE": "ADDRESS",
    "ZIP_CODE": "ADDRESS",
    "POSTAL_CODE": "ADDRESS",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "NAME": "PERSON",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract declared span text and labels from JSON/JSONL files "
            "for external review."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input dataset files (.json or .jsonl).",
    )
    parser.add_argument(
        "--output",
        default="declared_spans_for_review.txt",
        help="Output text file path.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Keep only unique 'STRING: LABEL' lines in output.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_no}: each JSONL row must be an object")
            rows.append(row)
        return rows

    if suffix == ".json":
        raw_text = path.read_text(encoding="utf-8")
        try:
            raw_obj = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            # Some datasets are newline-delimited JSON but use .json extension.
            rows = []
            for line_no, raw in enumerate(raw_text.splitlines(), start=1):
                text = raw.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"{path}: invalid JSON and invalid JSONL-style content "
                        f"(first failing line {line_no})"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path} line {line_no}: each row must be an object")
                rows.append(row)
            if rows:
                return rows
            raise
        if isinstance(raw_obj, list):
            if not all(isinstance(x, dict) for x in raw_obj):
                raise ValueError(f"{path}: JSON array must contain only objects")
            return raw_obj
        if isinstance(raw_obj, dict):
            return [raw_obj]
        raise ValueError(f"{path}: JSON must be an object or an array of objects")

    raise ValueError(f"{path}: unsupported file extension (expected .json or .jsonl)")


def _format_malformed(
    src: Path,
    row_idx: int,
    span_idx: int,
    reason: str,
    start: object,
    end: object,
    label: object,
) -> str:
    return (
        f"[MALFORMED_SPAN {src.name} row={row_idx} span={span_idx} "
        f"start={start!r} end={end!r} label={label!r} reason={reason}]: MALFORMED"
    )


def _sanitize_text_for_markdown(text: str) -> str:
    # Prevent accidental fence breaks if source text contains backticks.
    return text.replace("```", "'''")


def build_markdown_examples(
    rows: Iterable[dict],
    src: Path,
    start_index: int,
    dedupe: bool,
) -> Tuple[List[str], int, int, int]:
    sections: List[str] = []
    malformed_count = 0
    emitted_spans = 0
    example_index = start_index

    for row_idx, row in enumerate(rows, start=1):
        text = row.get("text")
        if not isinstance(text, str):
            text = row.get("full_text")
        spans = row.get("spans")
        example_index += 1
        span_lines: List[str] = []
        seen_spans = set()

        if not isinstance(text, str):
            span_lines.append(
                _format_malformed(
                    src=src,
                    row_idx=row_idx,
                    span_idx=-1,
                    reason="row.text is not a string",
                    start=None,
                    end=None,
                    label=None,
                )
            )
            malformed_count += 1
            text = "<INVALID row.text>"
            spans = []
        elif not isinstance(spans, list):
            span_lines.append(
                _format_malformed(
                    src=src,
                    row_idx=row_idx,
                    span_idx=-1,
                    reason="row.spans is not a list",
                    start=None,
                    end=None,
                    label=None,
                )
            )
            malformed_count += 1
            spans = []

        if isinstance(spans, list):
            for span_idx, span in enumerate(spans, start=1):
                if not isinstance(span, dict):
                    span_lines.append(
                        _format_malformed(
                            src=src,
                            row_idx=row_idx,
                            span_idx=span_idx,
                            reason="span is not an object",
                            start=None,
                            end=None,
                            label=None,
                        )
                    )
                    malformed_count += 1
                    continue

                start = span.get("start")
                end = span.get("end")
                label = span.get("label")
                if not isinstance(start, int) or not isinstance(end, int):
                    start = span.get("start_position")
                    end = span.get("end_position")
                if not isinstance(label, str):
                    label = span.get("entity_type")

                if not isinstance(start, int) or not isinstance(end, int):
                    span_lines.append(
                        _format_malformed(
                            src=src,
                            row_idx=row_idx,
                            span_idx=span_idx,
                            reason="start/end must be integers",
                            start=start,
                            end=end,
                            label=label,
                        )
                    )
                    malformed_count += 1
                    continue
                if start < 0 or end <= start or end > len(text):
                    span_lines.append(
                        _format_malformed(
                            src=src,
                            row_idx=row_idx,
                            span_idx=span_idx,
                            reason="span bounds invalid for text length",
                            start=start,
                            end=end,
                            label=label,
                        )
                    )
                    malformed_count += 1
                    continue
                if not isinstance(label, str):
                    span_lines.append(
                        _format_malformed(
                            src=src,
                            row_idx=row_idx,
                            span_idx=span_idx,
                            reason="label is not a string",
                            start=start,
                            end=end,
                            label=label,
                        )
                    )
                    malformed_count += 1
                    continue
                label = LABEL_ALIASES.get(label.strip().upper(), label.strip().upper())

                snippet = text[start:end].replace("\n", "\\n").strip()
                if not snippet:
                    span_lines.append(
                        _format_malformed(
                            src=src,
                            row_idx=row_idx,
                            span_idx=span_idx,
                            reason="extracted span text is empty after strip",
                            start=start,
                            end=end,
                            label=label,
                        )
                    )
                    malformed_count += 1
                    continue

                line = f"{snippet}: {label}"
                if dedupe and line in seen_spans:
                    continue
                seen_spans.add(line)
                span_lines.append(line)
                emitted_spans += 1

        rendered_spans = "\n".join(f"- {line}" for line in span_lines) if span_lines else "- (none)"
        section = (
            f"# example {example_index}\n\n"
            f"Original text:\n"
            f"```\n{_sanitize_text_for_markdown(text)}\n```\n\n"
            f"Declared spans:\n"
            f"{rendered_spans}\n"
        )
        sections.append(section)

    return sections, malformed_count, emitted_spans, example_index


def main() -> None:
    args = parse_args()

    all_sections: List[str] = []
    total_rows = 0
    total_emitted_spans = 0
    malformed_total = 0
    example_index = 0

    for input_path in args.inputs:
        path = Path(input_path)
        rows = load_rows(path)
        file_sections, malformed_count, emitted_spans, example_index = build_markdown_examples(
            rows=rows,
            src=path,
            start_index=example_index,
            dedupe=args.dedupe,
        )
        all_sections.extend(file_sections)
        total_rows += len(rows)
        total_emitted_spans += emitted_spans
        malformed_total += malformed_count

    output_path = Path(args.output)
    output_text = "\n\n".join(all_sections)
    if output_text:
        output_text += "\n"
    output_path.write_text(output_text, encoding="utf-8")

    print(f"Wrote {len(all_sections)} examples to {output_path.resolve()}")
    print(
        f"Processed rows={total_rows}, emitted_spans={total_emitted_spans}, malformed_spans={malformed_total}"
    )


if __name__ == "__main__":
    main()
