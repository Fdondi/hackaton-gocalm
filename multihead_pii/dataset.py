import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset

from .labels import (
    BIO_LABEL_TO_ID,
    SENSITIVITY_LABEL_TO_ID,
    TYPE_ID_TO_LABEL,
    TYPE_LABEL_TO_ID,
    sensitivity_from_item_category,
)


EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\+?\d[\d\-\s().]{6,}\d")
IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9 ]{10,34}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

REGEX_TYPE_MAP = {
    "EMAIL": EMAIL_PATTERN,
    "PHONE": PHONE_PATTERN,
    "IP_ADDRESS": IPV4_PATTERN,
    "IBAN": IBAN_PATTERN,
    "CREDIT_CARD": CREDIT_CARD_PATTERN,
}


@dataclass
class MultiHeadExample:
    text: str
    spans: List[Dict]
    items: List[Dict]
    sensitivity_spans: List[Dict]


def is_valid_token(offset_pair: Tuple[int, int]) -> bool:
    start, end = offset_pair
    return not (start == 0 and end == 0)


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def char_span_to_token_span(
    offsets: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> Optional[Tuple[int, int]]:
    token_start = None
    token_end = None
    for i, (start, end) in enumerate(offsets):
        if not is_valid_token((start, end)):
            continue
        if overlap(start, end, char_start, char_end):
            if token_start is None:
                token_start = i
            token_end = i
    if token_start is None or token_end is None:
        return None
    return token_start, token_end


def token_span_to_char_span(
    offsets: List[Tuple[int, int]],
    token_start: int,
    token_end: int,
) -> Optional[Tuple[int, int]]:
    if token_start < 0 or token_end < token_start or token_end >= len(offsets):
        return None
    start_off = offsets[token_start]
    end_off = offsets[token_end]
    if not is_valid_token(start_off) or not is_valid_token(end_off):
        return None
    char_start = start_off[0]
    char_end = end_off[1]
    if char_end <= char_start:
        return None
    return char_start, char_end


def _span_key(start: int, end: int) -> Tuple[int, int]:
    return (start, end)


def _load_jsonl_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
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
            rows.append(row)
    return rows


def _normalize_rows(
    main_rows: List[Dict],
    sensitivity_rows: Optional[List[Dict]] = None,
) -> List[MultiHeadExample]:
    sensitivity_rows = sensitivity_rows or []
    sensitivity_by_index = {i: row for i, row in enumerate(sensitivity_rows)}
    examples: List[MultiHeadExample] = []
    for idx, row in enumerate(main_rows):
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"row {idx}: missing/invalid text")
        spans = row.get("spans")
        items = row.get("items")
        if spans is None:
            spans = []
        if items is None:
            items = []
        if not isinstance(spans, list):
            raise ValueError(f"row {idx}: spans must be a list")
        if not isinstance(items, list):
            raise ValueError(f"row {idx}: items must be a list")
        companion = sensitivity_by_index.get(idx, {})
        companion_spans = companion.get("sensitivity_spans", []) if isinstance(companion, dict) else []
        if not isinstance(companion_spans, list):
            raise ValueError(f"sensitivity row {idx}: sensitivity_spans must be a list")
        examples.append(
            MultiHeadExample(
                text=text,
                spans=spans,
                items=items,
                sensitivity_spans=companion_spans,
            )
        )
    return examples


def _regex_char_candidates(text: str) -> List[Tuple[int, int]]:
    found: Set[Tuple[int, int]] = set()
    for pattern in REGEX_TYPE_MAP.values():
        for match in pattern.finditer(text):
            found.add((match.start(), match.end()))
    return sorted(found)


def _build_supervision_maps(
    example: MultiHeadExample,
) -> Tuple[Dict[Tuple[int, int], str], Dict[Tuple[int, int], str], Set[Tuple[int, int]]]:
    type_by_char_span: Dict[Tuple[int, int], str] = {}
    sens_by_char_span: Dict[Tuple[int, int], str] = {}
    non_pii_spans: Set[Tuple[int, int]] = set()

    # Prefer richer item annotations when present.
    if example.items:
        for item in example.items:
            if not isinstance(item, dict):
                continue
            start = item.get("start")
            end = item.get("end")
            label = item.get("label")
            category = item.get("category")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            key = _span_key(start, end)
            if category == "NON_PII_NUMBER":
                non_pii_spans.add(key)
                continue
            if isinstance(label, str) and label in TYPE_LABEL_TO_ID:
                type_by_char_span[key] = label
            sensitivity = sensitivity_from_item_category(category)
            if sensitivity is not None:
                sens_by_char_span[key] = sensitivity

    # Span-only annotations still train type and proposal.
    for span in example.spans:
        if not isinstance(span, dict):
            continue
        start = span.get("start")
        end = span.get("end")
        label = span.get("label")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(label, str)
            or label not in TYPE_LABEL_TO_ID
            or label == "NONE"
        ):
            continue
        key = _span_key(start, end)
        type_by_char_span.setdefault(key, label)

    # Companion file gives explicit sensitivity labels for span-only datasets.
    for span in example.sensitivity_spans:
        if not isinstance(span, dict):
            continue
        start = span.get("start")
        end = span.get("end")
        sensitivity = span.get("sensitivity")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(sensitivity, str)
            or sensitivity not in SENSITIVITY_LABEL_TO_ID
        ):
            continue
        sens_by_char_span[_span_key(start, end)] = sensitivity

    return type_by_char_span, sens_by_char_span, non_pii_spans


def _build_proposal_labels(
    offsets: List[Tuple[int, int]],
    positive_tok_spans: Set[Tuple[int, int]],
) -> List[int]:
    labels = [-100 for _ in offsets]
    for i, off in enumerate(offsets):
        if is_valid_token(off):
            labels[i] = BIO_LABEL_TO_ID["O"]
    for start, end in positive_tok_spans:
        if start < 0 or end >= len(labels) or end < start:
            continue
        labels[start] = BIO_LABEL_TO_ID["B-ENTITY"]
        for idx in range(start + 1, end + 1):
            labels[idx] = BIO_LABEL_TO_ID["I-ENTITY"]
    return labels


def _enumerate_candidates(
    offsets: List[Tuple[int, int]],
    max_span_len: int,
) -> List[Tuple[int, int]]:
    valid_idxs = [i for i, off in enumerate(offsets) if is_valid_token(off)]
    candidates: List[Tuple[int, int]] = []
    for start in valid_idxs:
        max_end = min(start + max_span_len - 1, len(offsets) - 1)
        for end in range(start, max_end + 1):
            if not is_valid_token(offsets[end]):
                break
            candidates.append((start, end))
    return candidates


class JsonlMultiHeadDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int,
        max_span_len: int,
        negative_sample_rate: float = 0.2,
        training: bool = True,
        sensitivity_path: Optional[str] = None,
        include_regex_candidates: bool = True,
    ):
        main_rows = _load_jsonl_rows(path)
        sensitivity_rows = _load_jsonl_rows(sensitivity_path) if sensitivity_path else None
        if sensitivity_rows and len(sensitivity_rows) != len(main_rows):
            raise ValueError(
                "Sensitivity companion row count mismatch: "
                f"{len(sensitivity_rows)} vs {len(main_rows)}"
            )
        self.examples = _normalize_rows(main_rows, sensitivity_rows)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_span_len = max_span_len
        self.negative_sample_rate = negative_sample_rate
        self.training = training
        self.include_regex_candidates = include_regex_candidates

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]

        type_by_char, sens_by_char, _ = _build_supervision_maps(ex)
        type_by_tok: Dict[Tuple[int, int], int] = {}
        sens_by_tok: Dict[Tuple[int, int], int] = {}
        positive_tok_spans: Set[Tuple[int, int]] = set()

        for (start, end), label in type_by_char.items():
            tok_span = char_span_to_token_span(offsets, start, end)
            if tok_span is None:
                continue
            type_by_tok[tok_span] = TYPE_LABEL_TO_ID[label]
            positive_tok_spans.add(tok_span)

        for (start, end), sensitivity in sens_by_char.items():
            tok_span = char_span_to_token_span(offsets, start, end)
            if tok_span is None:
                continue
            sens_by_tok[tok_span] = SENSITIVITY_LABEL_TO_ID[sensitivity]

        proposal_labels = _build_proposal_labels(offsets, positive_tok_spans)
        all_candidates = _enumerate_candidates(offsets, max_span_len=self.max_span_len)

        candidate_spans: List[List[int]] = []
        type_labels: List[int] = []
        sensitivity_labels: List[int] = []

        positive_set = set(type_by_tok.keys()) | set(sens_by_tok.keys())
        for candidate in all_candidates:
            is_positive = candidate in positive_set
            if self.training and not is_positive and random.random() > self.negative_sample_rate:
                continue
            candidate_spans.append([candidate[0], candidate[1]])
            type_labels.append(type_by_tok.get(candidate, TYPE_LABEL_TO_ID["NONE"]))
            sensitivity_labels.append(sens_by_tok.get(candidate, -100))

        if self.include_regex_candidates:
            existing = {tuple(x) for x in candidate_spans}
            for char_start, char_end in _regex_char_candidates(ex.text):
                tok_span = char_span_to_token_span(offsets, char_start, char_end)
                if tok_span is None or tok_span in existing:
                    continue
                if tok_span[1] - tok_span[0] + 1 > self.max_span_len:
                    continue
                candidate_spans.append([tok_span[0], tok_span[1]])
                type_labels.append(type_by_tok.get(tok_span, TYPE_LABEL_TO_ID["NONE"]))
                sensitivity_labels.append(sens_by_tok.get(tok_span, -100))
                existing.add(tok_span)

        if not candidate_spans:
            # Keep shape-valid fallback.
            candidate_spans = [[0, 0]]
            type_labels = [TYPE_LABEL_TO_ID["NONE"]]
            sensitivity_labels = [-100]

        gold_redact_char_spans = sorted(
            [
                {"start": char_span[0], "end": char_span[1]}
                for (tok_start, tok_end), sens in sens_by_tok.items()
                if sens == SENSITIVITY_LABEL_TO_ID["REDACT"]
                for char_span in [token_span_to_char_span(offsets, tok_start, tok_end)]
                if char_span is not None
            ],
            key=lambda row: (row["start"], row["end"]),
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "candidate_spans": torch.tensor(candidate_spans, dtype=torch.long),
            "proposal_labels": torch.tensor(proposal_labels, dtype=torch.long),
            "type_labels": torch.tensor(type_labels, dtype=torch.long),
            "sensitivity_labels": torch.tensor(sensitivity_labels, dtype=torch.long),
            "raw": {
                "text": ex.text,
                "offsets": offsets,
                "gold_type_spans": [
                    {"start": char_span[0], "end": char_span[1], "label": label}
                    for (tok_start, tok_end), type_id in sorted(type_by_tok.items())
                    for label in [TYPE_ID_TO_LABEL[type_id]]
                    for char_span in [token_span_to_char_span(offsets, tok_start, tok_end)]
                    if char_span is not None
                ],
                "gold_redact_spans": gold_redact_char_spans,
            },
        }


def collate_fn(batch: List[Dict]) -> Dict:
    max_toks = max(x["input_ids"].shape[0] for x in batch)
    max_spans = max(x["candidate_spans"].shape[0] for x in batch)

    input_ids = []
    attention_mask = []
    candidate_spans = []
    proposal_labels = []
    type_labels = []
    sensitivity_labels = []

    for item in batch:
        pad_t = max_toks - item["input_ids"].shape[0]
        input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_t), value=0))
        attention_mask.append(
            torch.nn.functional.pad(item["attention_mask"], (0, pad_t), value=0)
        )
        proposal_labels.append(
            torch.nn.functional.pad(item["proposal_labels"], (0, pad_t), value=-100)
        )

        pad_s = max_spans - item["candidate_spans"].shape[0]
        candidate_spans.append(
            torch.nn.functional.pad(item["candidate_spans"], (0, 0, 0, pad_s), value=-1)
        )
        type_labels.append(torch.nn.functional.pad(item["type_labels"], (0, pad_s), value=-100))
        sensitivity_labels.append(
            torch.nn.functional.pad(item["sensitivity_labels"], (0, pad_s), value=-100)
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "candidate_spans": torch.stack(candidate_spans),
        "proposal_labels": torch.stack(proposal_labels),
        "type_labels": torch.stack(type_labels),
        "sensitivity_labels": torch.stack(sensitivity_labels),
        "raw": [item["raw"] for item in batch],
    }
