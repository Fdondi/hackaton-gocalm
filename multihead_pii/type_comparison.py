from typing import Dict, Iterable, List, Set, Tuple


ValueKey = Tuple[str, str]


def normalize_value(value: str) -> str:
    return value.strip()


def make_value_key(label: str, value: str) -> ValueKey:
    return (label, normalize_value(value))


def compute_value_comparison(
    pred_keys: Iterable[ValueKey],
    gold_keys: Iterable[ValueKey],
) -> Tuple[List[ValueKey], List[ValueKey], List[ValueKey]]:
    pred_set: Set[ValueKey] = {make_value_key(label, value) for label, value in pred_keys}
    gold_set: Set[ValueKey] = {make_value_key(label, value) for label, value in gold_keys}
    tp = sorted(pred_set & gold_set)
    fp = sorted(pred_set - gold_set)
    fn = sorted(gold_set - pred_set)
    return tp, fp, fn


def _pick_best_gold_match(
    pred_label: str,
    pred_value: str,
    candidates: List[ValueKey],
) -> ValueKey:
    return sorted(
        candidates,
        key=lambda gold: (
            0 if gold[0] == pred_label else 1,
            abs(len(gold[1]) - len(pred_value)),
            gold[0],
            gold[1],
        ),
    )[0]


def classify_value_relationships(
    pred_keys: Iterable[ValueKey],
    gold_keys: Iterable[ValueKey],
) -> Dict[str, List[Dict]]:
    """Classify non-exact prediction outcomes for richer analysis.

    Returns:
      - exact_tp / exact_fp / exact_fn as lists of {"label","value"}
      - type_mismatches where value matches but label differs
      - value_subsets where predicted value is strict subset of gold value
      - value_supersets where predicted value strictly contains gold value
    """
    pred_set: Set[ValueKey] = {make_value_key(label, value) for label, value in pred_keys}
    gold_set: Set[ValueKey] = {make_value_key(label, value) for label, value in gold_keys}

    exact_tp_set = pred_set & gold_set
    remaining_pred = sorted(pred_set - exact_tp_set)
    remaining_gold = set(gold_set - exact_tp_set)

    type_mismatches: List[Dict] = []
    value_subsets: List[Dict] = []
    value_supersets: List[Dict] = []
    exact_fp: List[Dict] = []
    consumed_gold_non_exact: Set[ValueKey] = set()

    for pred_label, pred_value in remaining_pred:
        same_value_different_type = [
            (gold_label, gold_value)
            for gold_label, gold_value in remaining_gold
            if gold_value == pred_value and gold_label != pred_label
        ]
        if same_value_different_type:
            best = _pick_best_gold_match(pred_label, pred_value, same_value_different_type)
            type_mismatches.append(
                {
                    "predicted": {"label": pred_label, "value": pred_value},
                    "gold": {"label": best[0], "value": best[1]},
                }
            )
            consumed_gold_non_exact.add(best)
            remaining_gold.discard(best)
            continue

        subset_candidates = [
            (gold_label, gold_value)
            for gold_label, gold_value in remaining_gold
            if pred_value and pred_value != gold_value and pred_value in gold_value
        ]
        if subset_candidates:
            best = _pick_best_gold_match(pred_label, pred_value, subset_candidates)
            value_subsets.append(
                {
                    "predicted": {"label": pred_label, "value": pred_value},
                    "gold": {"label": best[0], "value": best[1]},
                }
            )
            consumed_gold_non_exact.add(best)
            remaining_gold.discard(best)
            continue

        superset_candidates = [
            (gold_label, gold_value)
            for gold_label, gold_value in remaining_gold
            if gold_value and pred_value != gold_value and gold_value in pred_value
        ]
        if superset_candidates:
            best = _pick_best_gold_match(pred_label, pred_value, superset_candidates)
            value_supersets.append(
                {
                    "predicted": {"label": pred_label, "value": pred_value},
                    "gold": {"label": best[0], "value": best[1]},
                }
            )
            consumed_gold_non_exact.add(best)
            remaining_gold.discard(best)
            continue

        exact_fp.append({"label": pred_label, "value": pred_value})

    exact_tp = [{"label": label, "value": value} for label, value in sorted(exact_tp_set)]
    exact_fn = [
        {"label": label, "value": value}
        for label, value in sorted(gold_set - {(x["label"], x["value"]) for x in exact_tp} - consumed_gold_non_exact)
    ]
    return {
        "exact_tp": exact_tp,
        "exact_fp": exact_fp,
        "exact_fn": exact_fn,
        "type_mismatches": type_mismatches,
        "value_subsets": value_subsets,
        "value_supersets": value_supersets,
    }
