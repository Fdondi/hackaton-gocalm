import unittest

from multihead_pii.decoder import (
    DecodedSpan,
    _trim_char_span,
    non_max_suppression,
    select_non_overlapping_typed_spans,
)


class TypedSpanSelectionTests(unittest.TestCase):
    def test_select_non_overlapping_keeps_highest_scoring_overlap(self) -> None:
        spans = [
            (10, 30, "ADDRESS", 0.91),
            (17, 22, "ADDRESS", 0.70),  # nested lower-score span
            (40, 45, "FORM_ID", 0.88),  # non-overlapping
        ]
        selected = select_non_overlapping_typed_spans(spans)
        self.assertEqual(
            selected,
            [
                (10, 30, "ADDRESS"),
                (40, 45, "FORM_ID"),
            ],
        )

    def test_select_non_overlapping_prefers_larger_span_on_score_tie(self) -> None:
        spans = [
            (10, 15, "MONEY", 0.80),
            (10, 22, "MONEY", 0.80),
        ]
        selected = select_non_overlapping_typed_spans(spans)
        self.assertEqual(selected, [(10, 22, "MONEY")])


class RedactionNmsTests(unittest.TestCase):
    def test_non_max_suppression_rejects_any_overlap(self) -> None:
        spans = [
            DecodedSpan(10, 30, "ADDRESS", "REDACT", 0.90, 0.95, 0.95),
            DecodedSpan(20, 40, "ADDRESS", "REDACT", 0.89, 0.94, 0.95),
        ]
        selected = non_max_suppression(spans, iou_threshold=0.99)
        self.assertEqual(len(selected), 1)
        self.assertEqual((selected[0].start, selected[0].end), (10, 30))

    def test_trim_char_span_removes_edge_whitespace(self) -> None:
        text = "  Maria Rossi  "
        trimmed = _trim_char_span(text, 0, len(text))
        self.assertEqual(trimmed, (2, 13))


if __name__ == "__main__":
    unittest.main()
