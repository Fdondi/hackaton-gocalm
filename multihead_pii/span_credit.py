from typing import Iterable, List, Sequence, Tuple


TokenSpan = Tuple[int, int]
LabeledTokenSpan = Tuple[int, int, str]


def token_span_len(span: TokenSpan) -> int:
    return max(0, span[1] - span[0] + 1)


def token_intersection_len(a: TokenSpan, b: TokenSpan) -> int:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    return max(0, right - left + 1)


def overlap_credit(pred_span: TokenSpan, gold_span: TokenSpan) -> float:
    """
    Exact token span match gets full credit 1.0.
    Partial overlap gets 1 / 2^(M + N - K), where:
      - M is gold span length in tokens
      - N is predicted span length in tokens
      - K is token intersection length
    """
    if pred_span == gold_span:
        return 1.0
    m = token_span_len(gold_span)
    n = token_span_len(pred_span)
    k = token_intersection_len(pred_span, gold_span)
    if m <= 0 or n <= 0 or k <= 0:
        return 0.0
    return 2.0 ** (-(m + n - k))


def best_overlap_credit(
    pred_span: TokenSpan,
    gold_spans: Iterable[TokenSpan],
) -> float:
    best = 0.0
    for gold in gold_spans:
        score = overlap_credit(pred_span, gold)
        if score > best:
            best = score
    return best


def soft_match_total_credit(
    pred_spans: Sequence[LabeledTokenSpan],
    gold_spans: Sequence[LabeledTokenSpan],
    require_same_label: bool,
) -> float:
    """
    Greedy one-to-one matching by descending credit.
    This keeps total matched credit <= min(len(pred), len(gold)).
    """
    scored_pairs: List[Tuple[float, int, int]] = []
    for p_idx, pred in enumerate(pred_spans):
        for g_idx, gold in enumerate(gold_spans):
            if require_same_label and pred[2] != gold[2]:
                continue
            score = overlap_credit((pred[0], pred[1]), (gold[0], gold[1]))
            if score > 0.0:
                scored_pairs.append((score, p_idx, g_idx))

    scored_pairs.sort(key=lambda row: row[0], reverse=True)
    used_pred = set()
    used_gold = set()
    total = 0.0
    for score, p_idx, g_idx in scored_pairs:
        if p_idx in used_pred or g_idx in used_gold:
            continue
        used_pred.add(p_idx)
        used_gold.add(g_idx)
        total += score
    return total
