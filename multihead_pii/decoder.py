import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .labels import (
    BIO_ID_TO_LABEL,
    SENSITIVITY_LABEL_TO_ID,
    TYPE_ID_TO_LABEL,
)


EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\+?\d[\d\-\s().]{6,}\d")
IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9 ]{10,34}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

REGEX_PATTERNS = [
    EMAIL_PATTERN,
    PHONE_PATTERN,
    IPV4_PATTERN,
    IBAN_PATTERN,
    CREDIT_CARD_PATTERN,
]


@dataclass
class DecodedSpan:
    start: int
    end: int
    label: str
    decision: str
    redact_score: float
    redact_probability: float
    type_confidence: float


def _iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def _token_span_to_char_span(
    offsets: List[Tuple[int, int]], token_start: int, token_end: int
) -> Optional[Tuple[int, int]]:
    if token_start < 0 or token_end >= len(offsets) or token_end < token_start:
        return None
    char_start = offsets[token_start][0]
    char_end = offsets[token_end][1]
    if char_end <= char_start:
        return None
    return char_start, char_end


def _trim_char_span(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    safe_start = max(0, min(start, len(text)))
    safe_end = max(safe_start, min(end, len(text)))
    while safe_start < safe_end and text[safe_start].isspace():
        safe_start += 1
    while safe_end > safe_start and text[safe_end - 1].isspace():
        safe_end -= 1
    if safe_end <= safe_start:
        return None
    return safe_start, safe_end


def decode_proposal_bio(
    proposal_logits: torch.Tensor,
    offsets: List[Tuple[int, int]],
    attention_mask: torch.Tensor,
) -> List[Tuple[int, int]]:
    labels = proposal_logits.argmax(dim=-1).tolist()
    spans: List[Tuple[int, int]] = []
    start = None
    end = None
    seq_len = int(attention_mask.sum().item())
    for idx in range(min(seq_len, len(labels), len(offsets))):
        if offsets[idx] == (0, 0):
            continue
        tag = BIO_ID_TO_LABEL[labels[idx]]
        if tag == "B-ENTITY":
            if start is not None:
                spans.append((start, end))
            start = idx
            end = idx
        elif tag == "I-ENTITY" and start is not None:
            end = idx
        else:
            if start is not None:
                spans.append((start, end))
                start = None
                end = None
    if start is not None:
        spans.append((start, end))
    return spans


def regex_char_candidates(text: str) -> List[Tuple[int, int]]:
    seen = set()
    for pattern in REGEX_PATTERNS:
        for match in pattern.finditer(text):
            seen.add((match.start(), match.end()))
    return sorted(seen)


def non_max_suppression(
    spans: List[DecodedSpan],
    iou_threshold: float,
) -> List[DecodedSpan]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda x: x.redact_score, reverse=True)
    selected: List[DecodedSpan] = []
    for candidate in ordered:
        keep = True
        for existing in selected:
            candidate_bounds = (candidate.start, candidate.end)
            existing_bounds = (existing.start, existing.end)
            # Never emit overlapping final spans. Keep highest-score candidate.
            if _overlap(candidate_bounds, existing_bounds):
                keep = False
                break
            if _iou(candidate_bounds, existing_bounds) >= iou_threshold:
                keep = False
                break
        if keep:
            selected.append(candidate)
    return sorted(selected, key=lambda x: (x.start, x.end))


def select_non_overlapping_typed_spans(
    typed_spans: List[Tuple[int, int, str, float]]
) -> List[Tuple[int, int, str]]:
    """Select a deterministic non-overlapping typed span set.

    Each input tuple is (start, end, label, score). Selection is score-first,
    with longer spans preferred on ties so container spans beat nested spans.
    """
    ordered = sorted(
        typed_spans,
        key=lambda item: (-item[3], -(item[1] - item[0]), item[0], item[1], item[2]),
    )
    selected: List[Tuple[int, int, str]] = []
    selected_bounds: List[Tuple[int, int]] = []
    for start, end, label, _score in ordered:
        bounds = (start, end)
        if any(_overlap(bounds, existing) for existing in selected_bounds):
            continue
        selected.append((start, end, label))
        selected_bounds.append(bounds)
    return sorted(selected, key=lambda item: (item[0], item[1], item[2]))


def decode_final_spans(
    text: Optional[str],
    offsets: List[Tuple[int, int]],
    candidate_spans: torch.Tensor,
    type_logits: Optional[torch.Tensor],
    sensitivity_logits: Optional[torch.Tensor],
    redact_score_threshold: float,
    nms_iou_threshold: float,
) -> List[DecodedSpan]:
    if type_logits is None or sensitivity_logits is None or type_logits.numel() == 0:
        return []

    type_probs = torch.softmax(type_logits, dim=-1)
    sensitivity_probs = torch.softmax(sensitivity_logits, dim=-1)

    decoded: List[DecodedSpan] = []
    for idx in range(type_logits.size(0)):
        span = candidate_spans[idx]
        token_start = int(span[0].item())
        token_end = int(span[1].item())
        char_span = _token_span_to_char_span(offsets, token_start, token_end)
        if char_span is None:
            continue
        if text is not None:
            trimmed = _trim_char_span(text, char_span[0], char_span[1])
            if trimmed is None:
                continue
            char_span = trimmed

        type_id = int(type_probs[idx].argmax(dim=-1).item())
        sens_id = int(sensitivity_probs[idx].argmax(dim=-1).item())

        label = TYPE_ID_TO_LABEL[type_id]
        if label == "NONE":
            continue

        type_conf = float(type_probs[idx, type_id].item())
        redact_prob = float(
            sensitivity_probs[idx, SENSITIVITY_LABEL_TO_ID["REDACT"]].item()
        )
        redact_score = type_conf * redact_prob

        if sens_id != SENSITIVITY_LABEL_TO_ID["REDACT"]:
            continue
        if math.isnan(redact_score) or redact_score < redact_score_threshold:
            continue

        decoded.append(
            DecodedSpan(
                start=char_span[0],
                end=char_span[1],
                label=label,
                decision="REDACT",
                redact_score=redact_score,
                redact_probability=redact_prob,
                type_confidence=type_conf,
            )
        )

    return non_max_suppression(decoded, iou_threshold=nms_iou_threshold)


def attach_regex_candidates(
    text: str,
    offsets: List[Tuple[int, int]],
    candidate_spans: List[Tuple[int, int]],
    max_span_len: int,
) -> List[Tuple[int, int]]:
    existing = set(candidate_spans)
    for char_start, char_end in regex_char_candidates(text):
        token_start = None
        token_end = None
        for idx, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:
                continue
            if max(start, char_start) < min(end, char_end):
                if token_start is None:
                    token_start = idx
                token_end = idx
        if token_start is None or token_end is None:
            continue
        if token_end - token_start + 1 > max_span_len:
            continue
        existing.add((token_start, token_end))
    return sorted(existing)
