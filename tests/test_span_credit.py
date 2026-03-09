import unittest

from multihead_pii.span_credit import overlap_credit, soft_match_total_credit


class SpanCreditTests(unittest.TestCase):
    def test_overlap_credit_exact_is_one(self) -> None:
        self.assertEqual(overlap_credit((5, 8), (5, 8)), 1.0)

    def test_overlap_credit_partial_uses_requested_formula(self) -> None:
        # M=3, N=3, K=2 -> 1 / 2^(3+3-2) = 1/16.
        score = overlap_credit((10, 12), (11, 13))
        self.assertAlmostEqual(score, 1.0 / 16.0)

    def test_soft_match_prevents_double_counting_same_gold(self) -> None:
        pred = [(10, 12, "EMAIL"), (10, 12, "EMAIL")]
        gold = [(10, 12, "EMAIL")]
        score = soft_match_total_credit(pred, gold, require_same_label=True)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
