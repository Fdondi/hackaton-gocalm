import unittest

from multihead_pii.dataset import (
    MultiHeadExample,
    _build_sensitivity_soft_target,
    _build_supervision_maps,
    _build_type_soft_target,
)
from multihead_pii.labels import SENSITIVITY_LABEL_TO_ID, TYPE_LABEL_TO_ID


class DatasetSensitivityDefaultsTests(unittest.TestCase):
    def test_spans_only_defaults_typed_spans_to_redact(self) -> None:
        ex = MultiHeadExample(
            text="Jane Doe jane@example.com",
            spans=[
                {"start": 0, "end": 8, "label": "PERSON"},
                {"start": 9, "end": 25, "label": "EMAIL"},
            ],
            items=[],
            sensitivity_spans=[],
        )
        type_by_char, sens_by_char, _, _ = _build_supervision_maps(ex)

        self.assertEqual(type_by_char[(0, 8)], "PERSON")
        self.assertEqual(type_by_char[(9, 25)], "EMAIL")
        self.assertEqual(sens_by_char[(0, 8)], "REDACT")
        self.assertEqual(sens_by_char[(9, 25)], "REDACT")

    def test_explicit_sensitivity_is_not_overridden(self) -> None:
        ex = MultiHeadExample(
            text="Public hotline 800-111-2222",
            spans=[{"start": 15, "end": 27, "label": "PHONE"}],
            items=[],
            sensitivity_spans=[{"start": 15, "end": 27, "sensitivity": "KEEP"}],
        )
        _, sens_by_char, _, _ = _build_supervision_maps(ex)
        self.assertEqual(sens_by_char[(15, 27)], "KEEP")

    def test_overlap_soft_targets_assign_partial_credit(self) -> None:
        type_target = _build_type_soft_target(
            candidate=(3, 5),
            exact_type_id=TYPE_LABEL_TO_ID["NONE"],
            gold_type_spans=[((3, 6), TYPE_LABEL_TO_ID["EMAIL"])],
        )
        self.assertGreater(type_target[TYPE_LABEL_TO_ID["EMAIL"]], 0.0)
        self.assertLess(type_target[TYPE_LABEL_TO_ID["EMAIL"]], 1.0)

        sens_target = _build_sensitivity_soft_target(
            candidate=(3, 5),
            exact_sens_id=-100,
            gold_sens_spans=[((3, 6), SENSITIVITY_LABEL_TO_ID["REDACT"])],
        )
        self.assertGreater(sens_target[SENSITIVITY_LABEL_TO_ID["REDACT"]], 0.0)
        self.assertLess(sens_target[SENSITIVITY_LABEL_TO_ID["REDACT"]], 1.0)

    def test_overlapping_gold_spans_are_merged_for_soft_credit(self) -> None:
        type_target = _build_type_soft_target(
            candidate=(1, 5),
            exact_type_id=TYPE_LABEL_TO_ID["NONE"],
            gold_type_spans=[
                ((2, 3), TYPE_LABEL_TO_ID["ADDRESS"]),
                ((4, 6), TYPE_LABEL_TO_ID["ADDRESS"]),
            ],
        )
        # Candidate tokens: {1..5}; merged overlapping gold tokens: {2..6}
        # IoU = |{2,3,4,5}| / |{1,2,3,4,5,6}| = 4/6.
        self.assertAlmostEqual(type_target[TYPE_LABEL_TO_ID["ADDRESS"]], 4.0 / 6.0)
        self.assertAlmostEqual(type_target[TYPE_LABEL_TO_ID["NONE"]], 2.0 / 6.0)

        sens_target = _build_sensitivity_soft_target(
            candidate=(1, 5),
            exact_sens_id=-100,
            gold_sens_spans=[
                ((2, 3), SENSITIVITY_LABEL_TO_ID["REDACT"]),
                ((4, 6), SENSITIVITY_LABEL_TO_ID["REDACT"]),
            ],
        )
        self.assertAlmostEqual(sens_target[SENSITIVITY_LABEL_TO_ID["REDACT"]], 4.0 / 6.0)
        self.assertAlmostEqual(sens_target[SENSITIVITY_LABEL_TO_ID["KEEP"]], 2.0 / 6.0)


if __name__ == "__main__":
    unittest.main()
