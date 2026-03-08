import unittest

import convert_inference_to_tp_fp_fn as conv


class ConvertInferenceComparisonTests(unittest.TestCase):
    def test_exact_matching_marks_off_by_one_as_mismatch(self) -> None:
        pred = [(8, 30, "EMAIL")]
        gold = [(9, 30, "EMAIL")]

        tp, fp, fn = conv._compute_comparison(
            pred_rows=pred,
            gold_rows=gold,
        )
        self.assertEqual(tp, [])
        self.assertEqual(len(fp), 1)
        self.assertEqual(len(fn), 1)

    def test_exact_matching_marks_identical_span_as_tp(self) -> None:
        pred = [(9, 30, "EMAIL")]
        gold = [(9, 30, "EMAIL")]
        tp, fp, fn = conv._compute_comparison(pred_rows=pred, gold_rows=gold)
        self.assertEqual(tp, [(9, 30, "EMAIL")])
        self.assertEqual(fp, [])
        self.assertEqual(fn, [])


class SuspiciousSpanDetectionTests(unittest.TestCase):
    def test_detects_truncated_credit_card(self) -> None:
        text = "Card 4111 1111 1111 1111 on file."
        self.assertTrue(conv._is_suspicious_span(text, 5, 15, "CREDIT_CARD"))

    def test_detects_truncated_address_zip_tail(self) -> None:
        text = "Lives at 123 Oak St, Springfield, IL 62704."
        self.assertTrue(conv._is_suspicious_span(text, 9, 41, "ADDRESS"))

    def test_valid_email_not_flagged(self) -> None:
        text = "Contact jane.doe@acme.example for details."
        self.assertFalse(conv._is_suspicious_span(text, 8, 29, "EMAIL"))


if __name__ == "__main__":
    unittest.main()
