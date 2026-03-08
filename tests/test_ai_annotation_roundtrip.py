import unittest
from pathlib import Path

from tokenizers import Tokenizer

from generate_challenging_span_data import _parse_annotated_text_to_example
from multihead_pii.dataset import char_span_to_token_span, token_span_to_char_span
from pii_labels import PII_LABELS


class AiAnnotationRoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tokenizer_path = Path("outputs_multihead/encoder/tokenizer.json")
        cls.tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def test_marker_parse_then_char_token_char_roundtrip_is_exact(self) -> None:
        # Marker syntax used by current generator/parser is [[ROOT:SUB|value]].
        examples = [
            ("EMAIL", "jane.doe+qa@acme.example"),
            ("OTHER", "v4.5.12"),
            ("OTHER", "1299.99"),
            ("ADDRESS", "123 Oak St, Springfield, IL 62704"),
            ("PASSWORD", "P@ssw0rd!#2026"),
            ("IP_ADDRESS", "192.168.0.42"),
            ("CREDIT_CARD", "4111 1111 1111 1111"),
            ("USERNAME", "jane_doe-ops"),
            ("ID_NUMBER", "ID-12-ABCD"),
            ("ACCOUNT_NUMBER", "ACC-0012399988"),
        ]
        annotated_text = " || ".join(
            # No space between ':' and marker keeps value start token-aligned.
            f"field{idx}:[[REAL_PII:{label}|{value}]]"
            for idx, (label, value) in enumerate(examples)
        )
        parsed = _parse_annotated_text_to_example(annotated_text, labels=set(PII_LABELS))
        text = parsed["text"]
        items = parsed["items"]
        offsets = self.tokenizer.encode(text).offsets

        self.assertEqual(len(items), len(examples))
        for idx, item in enumerate(items):
            expected_label, expected_value = examples[idx]
            start = item["start"]
            end = item["end"]
            actual_value = text[start:end]

            self.assertEqual(item["label"], expected_label)
            self.assertEqual(actual_value, expected_value)

            token_span = char_span_to_token_span(offsets, start, end)
            self.assertIsNotNone(token_span)
            roundtrip_char_span = token_span_to_char_span(offsets, token_span[0], token_span[1])  # type: ignore[index]
            self.assertEqual(roundtrip_char_span, (start, end))

    def test_space_prefixed_values_are_canonicalized_not_dropped(self) -> None:
        # This mirrors real generated text where marker value is preceded by punctuation + space.
        examples = [
            ("EMAIL", "jane.doe+qa@acme.example"),
            ("PASSWORD", "P@ssw0rd!#2026"),
            ("ADDRESS", "123 Oak St, Springfield, IL 62704"),
            ("OTHER", "v4.5.12"),
            ("IP_ADDRESS", "192.168.0.42"),
        ]
        annotated_text = " || ".join(
            f"field{idx}: [[REAL_PII:{label}|{value}]]"
            for idx, (label, value) in enumerate(examples)
        )
        parsed = _parse_annotated_text_to_example(annotated_text, labels=set(PII_LABELS))
        text = parsed["text"]
        items = parsed["items"]
        offsets = self.tokenizer.encode(text).offsets

        for idx, item in enumerate(items):
            expected_label, expected_value = examples[idx]
            start = item["start"]
            end = item["end"]
            token_span = char_span_to_token_span(offsets, start, end)

            # Critical property for training: span is still usable, not excluded.
            self.assertIsNotNone(token_span)
            self.assertEqual(item["label"], expected_label)
            self.assertEqual(text[start:end], expected_value)

            roundtrip_char_span = token_span_to_char_span(offsets, token_span[0], token_span[1])  # type: ignore[index]
            self.assertIsNotNone(roundtrip_char_span)
            canonical_value = text[roundtrip_char_span[0] : roundtrip_char_span[1]]  # type: ignore[index]

            # Canonical token-boundary text may include leading whitespace; value content must survive intact.
            self.assertEqual(canonical_value.strip(), expected_value)


if __name__ == "__main__":
    unittest.main()
