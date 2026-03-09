import unittest

import convert_inference_to_tp_fp_fn as conv


class ConvertInferenceComparisonTests(unittest.TestCase):
    def test_value_matching_ignores_boundary_mismatch(self) -> None:
        pred = [("EMAIL", "jane.doe@acme.example")]
        gold = [("EMAIL", "jane.doe@acme.example")]

        tp, fp, fn = conv._compute_comparison(
            pred_rows=pred,
            gold_rows=gold,
        )
        self.assertEqual(tp, [("EMAIL", "jane.doe@acme.example")])
        self.assertEqual(len(fp), 0)
        self.assertEqual(len(fn), 0)

    def test_value_matching_trims_predicted_value(self) -> None:
        pred = [("EMAIL", " jane.doe@acme.example ")]
        gold = [("EMAIL", "jane.doe@acme.example")]
        tp, fp, fn = conv._compute_comparison(pred_rows=pred, gold_rows=gold)
        self.assertEqual(tp, [("EMAIL", "jane.doe@acme.example")])
        self.assertEqual(fp, [])
        self.assertEqual(fn, [])

    def test_value_matching_marks_different_string_as_mismatch(self) -> None:
        pred = [("EMAIL", "jane.doe@acme.example")]
        gold = [("EMAIL", "john.doe@acme.example")]
        tp, fp, fn = conv._compute_comparison(pred_rows=pred, gold_rows=gold)
        self.assertEqual(tp, [])
        self.assertEqual(fp, [("EMAIL", "jane.doe@acme.example")])
        self.assertEqual(fn, [("EMAIL", "john.doe@acme.example")])


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
