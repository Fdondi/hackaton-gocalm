import unittest

from multihead_pii.dataset import char_span_to_token_span, token_span_to_char_span


class TokenCharRoundtripTests(unittest.TestCase):
    def test_roundtrip_can_expand_to_token_boundaries(self) -> None:
        # Simulates a tokenizer whose token includes a leading space.
        offsets = [(0, 5), (6, 10), (0, 0)]
        gold_char_span = (1, 5)
        tok_span = char_span_to_token_span(offsets, *gold_char_span)
        self.assertEqual(tok_span, (0, 0))
        char_span = token_span_to_char_span(offsets, tok_span[0], tok_span[1])  # type: ignore[index]
        self.assertEqual(char_span, (0, 5))

    def test_roundtrip_preserves_exact_when_aligned(self) -> None:
        offsets = [(0, 4), (5, 8), (9, 13)]
        gold_char_span = (5, 8)
        tok_span = char_span_to_token_span(offsets, *gold_char_span)
        self.assertEqual(tok_span, (1, 1))
        char_span = token_span_to_char_span(offsets, tok_span[0], tok_span[1])  # type: ignore[index]
        self.assertEqual(char_span, gold_char_span)

    def test_token_to_char_rejects_invalid_token_indices(self) -> None:
        offsets = [(0, 4), (5, 8)]
        self.assertIsNone(token_span_to_char_span(offsets, -1, 0))
        self.assertIsNone(token_span_to_char_span(offsets, 0, 9))
        self.assertIsNone(token_span_to_char_span(offsets, 1, 0))


if __name__ == "__main__":
    unittest.main()
