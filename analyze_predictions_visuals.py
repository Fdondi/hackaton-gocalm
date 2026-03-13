import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


@dataclass
class Example:
    doc_index: int
    text: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze prediction JSONL errors (FP/FN/partial overlaps/mismatches) "
            "and produce numbers, graphs, and examples."
        )
    )
    parser.add_argument(
        "--input",
        default="outputs_multihead/predictions.jsonl",
        help="Path to predictions JSONL with type_comparison blocks.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs_multihead/analysis",
        help="Directory for generated report files.",
    )
    parser.add_argument(
        "--examples-per-section",
        type=int,
        default=5,
        help="How many examples to include per section in markdown report.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Support both:
      1) strict JSONL (one object per line), and
      2) concatenated pretty-printed JSON objects.
    """
    content = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    length = len(content)
    row_no = 0
    while idx < length:
        while idx < length and content[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            row, end_idx = decoder.raw_decode(content, idx)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON near char {idx} ({exc})") from exc
        row_no += 1
        if not isinstance(row, dict):
            raise ValueError(f"{path}: parsed object #{row_no} is not a JSON object")
        yield row
        idx = end_idx


def safe_label(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "UNKNOWN"
    return value.strip().upper()


def safe_value(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value


def normalize_text(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def compact_text(value: str, max_len: int = 220) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def add_example(bucket: Dict[str, List[Example]], key: str, example: Example, limit: int = 20) -> None:
    items = bucket[key]
    if len(items) < limit:
        items.append(example)


def _extract_labeled_values(block: Any) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not isinstance(block, list):
        return out
    for item in block:
        if not isinstance(item, dict):
            continue
        out.append((safe_label(item.get("label")), safe_value(item.get("value"))))
    return out


def _extract_overlap_items(block: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not isinstance(block, list):
        return out
    for item in block:
        if not isinstance(item, dict):
            continue
        pred = item.get("predicted", {})
        gold = item.get("gold", {})
        if not isinstance(pred, dict) or not isinstance(gold, dict):
            continue
        out.append(
            {
                "pred_label": safe_label(pred.get("label")),
                "pred_value": safe_value(pred.get("value")),
                "gold_label": safe_label(gold.get("label")),
                "gold_value": safe_value(gold.get("value")),
            }
        )
    return out


def _find_merged_superset_matches(
    supersets: List[Dict[str, str]],
    false_negatives: List[Tuple[str, str]],
) -> Tuple[Set[int], Set[int], Counter[str]]:
    """
    If one predicted superset span with label L covers all gold split-parts with label L
    in that local group, reclassify it as a true positive.
    Returns:
      - indices of supersets to remove from partial counts
      - indices of false negatives to remove
      - extra true positives by label
    """
    used_fn_indices: Set[int] = set()
    consumed_superset_indices: Set[int] = set()
    extra_tp_by_label: Counter[str] = Counter()

    normalized_fn = [(lbl, val, normalize_text(val)) for lbl, val in false_negatives]

    for sup_idx, sup in enumerate(supersets):
        pred_label = sup["pred_label"]
        gold_label = sup["gold_label"]
        if pred_label != gold_label:
            continue
        pred_norm = normalize_text(sup["pred_value"])
        gold_norm = normalize_text(sup["gold_value"])
        if not pred_norm or not gold_norm:
            continue

        local_parts = {gold_norm}
        matched_fn_indices: List[int] = []
        for fn_idx, (fn_label, _fn_value, fn_norm) in enumerate(normalized_fn):
            if fn_idx in used_fn_indices:
                continue
            if fn_label != pred_label or not fn_norm:
                continue
            if fn_norm in pred_norm:
                local_parts.add(fn_norm)
                matched_fn_indices.append(fn_idx)

        # Require split evidence: at least two distinct gold parts covered by one prediction.
        if len(local_parts) < 2:
            continue
        if not all(part in pred_norm for part in local_parts):
            continue

        consumed_superset_indices.add(sup_idx)
        used_fn_indices.update(matched_fn_indices)
        extra_tp_by_label[pred_label] += 1

    return consumed_superset_indices, used_fn_indices, extra_tp_by_label


def gather_stats(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    fp_by_label: Counter[str] = Counter()
    fn_by_label: Counter[str] = Counter()
    tp_by_label: Counter[str] = Counter()
    subset_by_gold: Counter[str] = Counter()
    subset_by_pred: Counter[str] = Counter()
    superset_by_gold: Counter[str] = Counter()
    superset_by_pred: Counter[str] = Counter()
    mismatch_pairs: Counter[Tuple[str, str]] = Counter()
    merged_to_tp_by_label: Counter[str] = Counter()

    fp_examples: Dict[str, List[Example]] = defaultdict(list)
    fn_examples: Dict[str, List[Example]] = defaultdict(list)
    subset_examples: Dict[str, List[Example]] = defaultdict(list)
    superset_examples: Dict[str, List[Example]] = defaultdict(list)
    mismatch_examples: Dict[str, List[Example]] = defaultdict(list)

    doc_count = 0
    for doc_index, row in enumerate(rows):
        doc_count += 1
        text = row.get("text")
        if not isinstance(text, str):
            text = row.get("full_text", "")
        if not isinstance(text, str):
            text = ""

        comparison = row.get("type_comparison", {})
        if not isinstance(comparison, dict):
            continue

        tp_items = _extract_labeled_values(comparison.get("true_positives", []))
        fp_items = _extract_labeled_values(comparison.get("false_positives", []))
        fn_items = _extract_labeled_values(comparison.get("false_negatives", []))
        subset_items = _extract_overlap_items(comparison.get("value_subsets", []))
        superset_items = _extract_overlap_items(comparison.get("value_supersets", []))

        consumed_superset_indices, consumed_fn_indices, merged_tp_by_label = _find_merged_superset_matches(
            superset_items, fn_items
        )

        for label, _value in tp_items:
            tp_by_label[label] += 1
        for label, count in merged_tp_by_label.items():
            tp_by_label[label] += count
            merged_to_tp_by_label[label] += count

        for label, value in fp_items:
            fp_by_label[label] += 1
            add_example(
                fp_examples,
                label,
                Example(
                    doc_index=doc_index,
                    text=compact_text(text),
                    detail=f"predicted={label} value={value!r}",
                ),
            )

        for fn_idx, (label, value) in enumerate(fn_items):
            if fn_idx in consumed_fn_indices:
                continue
            fn_by_label[label] += 1
            add_example(
                fn_examples,
                label,
                Example(
                    doc_index=doc_index,
                    text=compact_text(text),
                    detail=f"gold={label} value={value!r}",
                ),
            )

        for item in subset_items:
            pred_label = item["pred_label"]
            gold_label = item["gold_label"]
            pred_value = item["pred_value"]
            gold_value = item["gold_value"]
            subset_by_pred[pred_label] += 1
            subset_by_gold[gold_label] += 1
            key = f"{pred_label}->{gold_label}"
            add_example(
                subset_examples,
                key,
                Example(
                    doc_index=doc_index,
                    text=compact_text(text),
                    detail=f"pred={pred_label}:{pred_value!r} | gold={gold_label}:{gold_value!r}",
                ),
            )

        for sup_idx, item in enumerate(superset_items):
            if sup_idx in consumed_superset_indices:
                continue
            pred_label = item["pred_label"]
            gold_label = item["gold_label"]
            pred_value = item["pred_value"]
            gold_value = item["gold_value"]
            superset_by_pred[pred_label] += 1
            superset_by_gold[gold_label] += 1
            key = f"{pred_label}->{gold_label}"
            add_example(
                superset_examples,
                key,
                Example(
                    doc_index=doc_index,
                    text=compact_text(text),
                    detail=f"pred={pred_label}:{pred_value!r} | gold={gold_label}:{gold_value!r}",
                ),
            )

        for item in comparison.get("type_mismatches", []) or []:
            if not isinstance(item, dict):
                continue
            pred = item.get("predicted", {})
            gold = item.get("gold", {})
            if not isinstance(pred, dict) or not isinstance(gold, dict):
                continue
            pred_label = safe_label(pred.get("label"))
            gold_label = safe_label(gold.get("label"))
            pred_value = safe_value(pred.get("value"))
            gold_value = safe_value(gold.get("value"))
            mismatch_pairs[(pred_label, gold_label)] += 1
            key = f"{pred_label}->{gold_label}"
            add_example(
                mismatch_examples,
                key,
                Example(
                    doc_index=doc_index,
                    text=compact_text(text),
                    detail=f"pred={pred_label}:{pred_value!r} | gold={gold_label}:{gold_value!r}",
                ),
            )

    labels = sorted(
        {
            *fp_by_label.keys(),
            *fn_by_label.keys(),
            *tp_by_label.keys(),
            *subset_by_pred.keys(),
            *subset_by_gold.keys(),
            *superset_by_pred.keys(),
            *superset_by_gold.keys(),
            *(p for p, _ in mismatch_pairs.keys()),
            *(g for _, g in mismatch_pairs.keys()),
        }
    )

    return {
        "num_docs": doc_count,
        "labels": labels,
        "fp_by_label": dict(fp_by_label),
        "fn_by_label": dict(fn_by_label),
        "tp_by_label": dict(tp_by_label),
        "merged_to_tp_by_label": dict(merged_to_tp_by_label),
        "subset_by_gold": dict(subset_by_gold),
        "subset_by_pred": dict(subset_by_pred),
        "superset_by_gold": dict(superset_by_gold),
        "superset_by_pred": dict(superset_by_pred),
        "mismatch_pairs": [
            {"predicted": p, "gold": g, "count": c}
            for (p, g), c in sorted(mismatch_pairs.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
        ],
        "examples": {
            "false_positives": {k: [e.__dict__ for e in v] for k, v in fp_examples.items()},
            "false_negatives": {k: [e.__dict__ for e in v] for k, v in fn_examples.items()},
            "value_subsets": {k: [e.__dict__ for e in v] for k, v in subset_examples.items()},
            "value_supersets": {k: [e.__dict__ for e in v] for k, v in superset_examples.items()},
            "type_mismatches": {k: [e.__dict__ for e in v] for k, v in mismatch_examples.items()},
        },
        "totals": {
            "false_positives": int(sum(fp_by_label.values())),
            "false_negatives": int(sum(fn_by_label.values())),
            "true_positives": int(sum(tp_by_label.values())),
            "merged_to_true_positive": int(sum(merged_to_tp_by_label.values())),
            "value_subsets": int(sum(subset_by_gold.values())),
            "value_supersets": int(sum(superset_by_gold.values())),
            "type_mismatches": int(sum(mismatch_pairs.values())),
        },
    }


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def top_counter_rows(counter_dict: Dict[str, int], limit: int = 25) -> List[List[Any]]:
    items = sorted(counter_dict.items(), key=lambda x: (-x[1], x[0]))[:limit]
    return [[k, v] for k, v in items]


def top_mismatch_rows(mismatches: List[Dict[str, Any]], limit: int = 25) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for item in mismatches[:limit]:
        rows.append([item["predicted"], item["gold"], item["count"]])
    return rows


def render_examples(block: Dict[str, List[Dict[str, Any]]], limit: int, title: str) -> str:
    keys_sorted = sorted(block.keys(), key=lambda k: (-len(block[k]), k))
    lines = [f"## {title}", ""]
    if not keys_sorted:
        lines.append("_None_")
        lines.append("")
        return "\n".join(lines)

    for key in keys_sorted:
        lines.append(f"### {key}")
        for ex in block[key][:limit]:
            lines.append(f"- doc `{ex['doc_index']}`: `{ex['detail']}`")
            lines.append(f"  - text: `{ex['text']}`")
        lines.append("")
    return "\n".join(lines)


def build_markdown(summary: Dict[str, Any], examples_per_section: int) -> str:
    totals = summary["totals"]
    lines: List[str] = []
    lines.append("# Predictions Error Analysis")
    lines.append("")
    lines.append(f"- docs analyzed: **{summary['num_docs']}**")
    lines.append(f"- true positives: **{totals['true_positives']}**")
    lines.append(f"- merged-span true positives: **{totals['merged_to_true_positive']}**")
    lines.append(f"- false positives: **{totals['false_positives']}**")
    lines.append(f"- false negatives: **{totals['false_negatives']}**")
    lines.append(f"- partial overlaps (subset): **{totals['value_subsets']}**")
    lines.append(f"- partial overlaps (superset): **{totals['value_supersets']}**")
    lines.append(f"- type mismatches: **{totals['type_mismatches']}**")
    lines.append("")

    lines.append("## False Positives Per Category")
    lines.append("")
    lines.append(markdown_table(["category", "count"], top_counter_rows(summary["fp_by_label"])))
    lines.append("")

    lines.append("## False Negatives Per Category")
    lines.append("")
    lines.append(markdown_table(["category", "count"], top_counter_rows(summary["fn_by_label"])))
    lines.append("")

    lines.append("## Partial Range Overlaps Per Category")
    lines.append("")
    lines.append("### Subset Errors (predicted is narrower than gold)")
    lines.append("")
    lines.append("By gold category:")
    lines.append("")
    lines.append(markdown_table(["gold_category", "count"], top_counter_rows(summary["subset_by_gold"])))
    lines.append("")
    lines.append("By predicted category:")
    lines.append("")
    lines.append(markdown_table(["predicted_category", "count"], top_counter_rows(summary["subset_by_pred"])))
    lines.append("")
    lines.append("### Superset Errors (predicted is wider than gold)")
    lines.append("")
    lines.append("By gold category:")
    lines.append("")
    lines.append(markdown_table(["gold_category", "count"], top_counter_rows(summary["superset_by_gold"])))
    lines.append("")
    lines.append("By predicted category:")
    lines.append("")
    lines.append(markdown_table(["predicted_category", "count"], top_counter_rows(summary["superset_by_pred"])))
    lines.append("")

    lines.append("## Category Confusions (Predicted -> Gold)")
    lines.append("")
    lines.append(markdown_table(["predicted", "gold", "count"], top_mismatch_rows(summary["mismatch_pairs"])))
    lines.append("")

    lines.append(render_examples(summary["examples"]["false_positives"], examples_per_section, "FP Examples"))
    lines.append(render_examples(summary["examples"]["false_negatives"], examples_per_section, "FN Examples"))
    lines.append(render_examples(summary["examples"]["value_subsets"], examples_per_section, "Subset Overlap Examples"))
    lines.append(render_examples(summary["examples"]["value_supersets"], examples_per_section, "Superset Overlap Examples"))
    lines.append(render_examples(summary["examples"]["type_mismatches"], examples_per_section, "Type Mismatch Examples"))
    return "\n".join(lines).strip() + "\n"


def _js(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def build_html(summary: Dict[str, Any]) -> str:
    labels = summary["labels"]
    fp = summary["fp_by_label"]
    fn = summary["fn_by_label"]
    subset_gold = summary["subset_by_gold"]
    superset_gold = summary["superset_by_gold"]
    mismatches = summary["mismatch_pairs"]
    totals = summary["totals"]

    mismatch_rows = []
    for pred in labels:
        row = []
        for gold in labels:
            count = 0
            for item in mismatches:
                if item["predicted"] == pred and item["gold"] == gold:
                    count = int(item["count"])
                    break
            row.append(count)
        mismatch_rows.append(row)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Predictions Error Analysis</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #131a31;
      --ink: #eef2ff;
      --muted: #a8b0cc;
      --accent1: #59c3c3;
      --accent2: #ff7f50;
      --accent3: #f4d35e;
      --accent4: #9ad1d4;
      --grid: #2a3356;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }}
    h1, h2 {{
      margin: 0 0 10px 0;
    }}
    .muted {{
      color: var(--muted);
    }}
    .grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--grid);
      border-radius: 12px;
      padding: 16px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 18px;
    }}
    .stat {{
      background: #0f1530;
      border: 1px solid var(--grid);
      border-radius: 10px;
      padding: 12px;
    }}
    .stat .k {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .stat .v {{
      font-size: 22px;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      border: 1px solid var(--grid);
      padding: 6px;
      text-align: center;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #182147;
      z-index: 1;
    }}
    .table-wrap {{
      max-height: 500px;
      overflow: auto;
      border: 1px solid var(--grid);
      border-radius: 8px;
    }}
  </style>
</head>
<body>
  <h1>Predictions Error Analysis</h1>
  <p class="muted">Counts are computed directly from <code>type_comparison</code> in each JSONL row.</p>
  <div class="stats">
    <div class="stat"><div class="k">docs</div><div class="v">{int(summary["num_docs"])}</div></div>
    <div class="stat"><div class="k">TP</div><div class="v">{int(totals["true_positives"])}</div></div>
    <div class="stat"><div class="k">merged TP</div><div class="v">{int(totals["merged_to_true_positive"])}</div></div>
    <div class="stat"><div class="k">FP</div><div class="v">{int(totals["false_positives"])}</div></div>
    <div class="stat"><div class="k">FN</div><div class="v">{int(totals["false_negatives"])}</div></div>
    <div class="stat"><div class="k">subset overlaps</div><div class="v">{int(totals["value_subsets"])}</div></div>
    <div class="stat"><div class="k">superset overlaps</div><div class="v">{int(totals["value_supersets"])}</div></div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>False Positives vs False Negatives</h2>
      <canvas id="fpFn"></canvas>
    </div>
    <div class="card">
      <h2>Partial Overlaps by Gold Category</h2>
      <canvas id="overlapGold"></canvas>
    </div>
    <div class="card" style="grid-column: 1 / span 2;">
      <h2>Type Confusion Matrix (Predicted -> Gold)</h2>
      <div class="table-wrap">
        <table id="confusion"></table>
      </div>
    </div>
  </div>

  <script>
    const labels = {_js(labels)};
    const fp = {_js(fp)};
    const fn = {_js(fn)};
    const subsetGold = {_js(subset_gold)};
    const supersetGold = {_js(superset_gold)};
    const matrix = {_js(mismatch_rows)};

    const barCommon = {{
      responsive: true,
      plugins: {{
        legend: {{ labels: {{ color: '#eef2ff' }} }}
      }},
      scales: {{
        x: {{ ticks: {{ color: '#eef2ff' }}, grid: {{ color: '#2a3356' }} }},
        y: {{ beginAtZero: true, ticks: {{ color: '#eef2ff' }}, grid: {{ color: '#2a3356' }} }},
      }}
    }};

    new Chart(document.getElementById('fpFn'), {{
      type: 'bar',
      data: {{
        labels,
        datasets: [
          {{ label: 'False Positives', data: labels.map(k => fp[k] || 0), backgroundColor: '#59c3c3' }},
          {{ label: 'False Negatives', data: labels.map(k => fn[k] || 0), backgroundColor: '#ff7f50' }},
        ]
      }},
      options: barCommon
    }});

    new Chart(document.getElementById('overlapGold'), {{
      type: 'bar',
      data: {{
        labels,
        datasets: [
          {{ label: 'Subset overlaps (gold category)', data: labels.map(k => subsetGold[k] || 0), backgroundColor: '#f4d35e' }},
          {{ label: 'Superset overlaps (gold category)', data: labels.map(k => supersetGold[k] || 0), backgroundColor: '#9ad1d4' }},
        ]
      }},
      options: barCommon
    }});

    const table = document.getElementById('confusion');
    const maxValue = Math.max(1, ...matrix.flat());
    const header = document.createElement('tr');
    header.appendChild(document.createElement('th'));
    for (const gold of labels) {{
      const th = document.createElement('th');
      th.textContent = gold;
      header.appendChild(th);
    }}
    table.appendChild(header);

    for (let i = 0; i < labels.length; i++) {{
      const tr = document.createElement('tr');
      const rowHead = document.createElement('th');
      rowHead.textContent = labels[i];
      tr.appendChild(rowHead);
      for (let j = 0; j < labels.length; j++) {{
        const td = document.createElement('td');
        const val = matrix[i][j];
        td.textContent = String(val);
        const alpha = 0.08 + (val / maxValue) * 0.75;
        td.style.backgroundColor = `rgba(255, 127, 80, ${{alpha.toFixed(3)}})`;
        tr.appendChild(td);
      }}
      table.appendChild(tr);
    }}
  </script>
</body>
</html>
"""
    return html


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = gather_stats(iter_jsonl(input_path))

    summary_path = out_dir / "error_summary.json"
    report_path = out_dir / "error_report.md"
    html_path = out_dir / "error_dashboard.html"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report_path.write_text(build_markdown(summary, args.examples_per_section), encoding="utf-8")
    html_path.write_text(build_html(summary), encoding="utf-8")

    print(f"Analyzed: {input_path}")
    print(f"Docs: {summary['num_docs']}")
    print(f"Wrote JSON summary: {summary_path}")
    print(f"Wrote markdown report: {report_path}")
    print(f"Wrote html dashboard: {html_path}")


if __name__ == "__main__":
    main()
