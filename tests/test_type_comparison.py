import unittest

from multihead_pii.type_comparison import (
    classify_value_relationships,
    compute_value_comparison,
    make_value_key,
)


class TypeComparisonTests(unittest.TestCase):
    def test_make_value_key_trims_whitespace(self) -> None:
        self.assertEqual(
            make_value_key("PERSON", " Maria Rossi "),
            ("PERSON", "Maria Rossi"),
        )

    def test_compute_value_comparison_uses_normalized_sets(self) -> None:
        pred = [make_value_key("ORG", " NovaTech ")]
        gold = [make_value_key("ORG", "NovaTech")]
        tp, fp, fn = compute_value_comparison(pred, gold)
        self.assertEqual(tp, [("ORG", "NovaTech")])
        self.assertEqual(fp, [])
        self.assertEqual(fn, [])

    def test_classifies_type_mismatch_when_value_matches(self) -> None:
        pred = [("ORG", "Maria Rossi")]
        gold = [("PERSON", "Maria Rossi")]
        out = classify_value_relationships(pred, gold)
        self.assertEqual(len(out["exact_tp"]), 0)
        self.assertEqual(len(out["exact_fn"]), 0)
        self.assertEqual(len(out["type_mismatches"]), 1)
        self.assertEqual(out["type_mismatches"][0]["predicted"]["label"], "ORG")
        self.assertEqual(out["type_mismatches"][0]["gold"]["label"], "PERSON")

    def test_classifies_subset_and_superset(self) -> None:
        pred = [("ADDRESS", "North Ave"), ("ADDRESS", "56 North Ave, Springfield")]
        gold = [("ADDRESS", "56 North Ave, Springfield"), ("ADDRESS", "North Ave")]
        out = classify_value_relationships(pred, gold)
        self.assertEqual(len(out["exact_tp"]), 2)

        pred_subset = [("ADDRESS", "North Ave")]
        gold_subset = [("ADDRESS", "56 North Ave, Springfield")]
        out_subset = classify_value_relationships(pred_subset, gold_subset)
        self.assertEqual(len(out_subset["value_subsets"]), 1)

        pred_superset = [("ADDRESS", "56 North Ave, Springfield")]
        gold_superset = [("ADDRESS", "North Ave")]
        out_superset = classify_value_relationships(pred_superset, gold_superset)
        self.assertEqual(len(out_superset["value_supersets"]), 1)


if __name__ == "__main__":
    unittest.main()
