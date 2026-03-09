import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pii_labels import PII_LABELS, PII_LABELS_CSV


LABELS = PII_LABELS
LABELS_CSV = PII_LABELS_CSV
MIN_ITEMS = 10
MAX_ITEMS = 40
MIN_NON_PII = 3
MIN_PII_LOOKALIKE = 1
MIN_REAL_PII = 2
ERROR_LOG_FILENAME = "generation_errors.jsonl"
DEFAULT_CALL_TIMEOUT_SECONDS = 600
FUTURE_POLL_SECONDS = 5
LOCAL_MAX_NEW_TOKENS = 2200
MODEL_PRICING_PER_M = {
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-5.3-chat-latest": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.2-chat-latest": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.3-codex": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.2-codex": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1-codex-max": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.1-codex": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-codex": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.2-pro": {"input": 21.00, "cached_input": None, "output": 168.00},
    "gpt-5-pro": {"input": 15.00, "cached_input": None, "output": 120.00},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 5.00, "cached_input": None, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-realtime": {"input": 4.00, "cached_input": 0.40, "output": 16.00},
    "gpt-realtime-1.5": {"input": 4.00, "cached_input": 0.40, "output": 16.00},
    "gpt-realtime-mini": {"input": 0.60, "cached_input": 0.06, "output": 2.40},
    "gpt-4o-realtime-preview": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
    "gpt-4o-mini-realtime-preview": {"input": 0.60, "cached_input": 0.30, "output": 2.40},
}
CONTEXT_TIER_PRICING_PER_M = {
    "gpt-5.4": {
        "lt_eq_272k": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
        "gt_272k": {"input": 5.00, "cached_input": 0.50, "output": 22.50},
    },
    "gpt-5.4-pro": {
        "lt_eq_272k": {"input": 30.00, "cached_input": None, "output": 180.00},
        "gt_272k": {"input": 60.00, "cached_input": None, "output": 270.00},
    },
}
CONTEXT_TIER_THRESHOLD_TOKENS = 272_000
LOCAL_GENERATOR_LOCK = threading.Lock()
LOCAL_GENERATORS: Dict[str, object] = {}

SYSTEM_PROMPT = """You generate training data for character-level span extraction.
Output only valid JSON.
"""

USER_PROMPT_TEMPLATE = """Create unique examples for NER span extraction.

Domain:
- PII detection in messy real-world text.
- Text should look like OCR/PDF grabs: line breaks, broken punctuation, weird spacing, occasional garbled fragments, partial words, duplicated symbols, merged columns.

Rules:
* Annotate spans in-place using this exact syntax: [[ROOT_CATEGORY:SUBCATEGORY|value]]. Note to NOT repeat the value outside this. ALWAYS follow this ternary pattern in the right order, or the parser will reject it.
   - ROOT_CATEGORY must be one of: NON_PII, PII_LOOKALIKE, REAL_PII
   - For REAL_PII and PII_LOOKALIKE, SUBCATEGORY must be one of: {labels_csv}
   - PII_LOOKALIKE examples: only include values that have the same technical structure as REAL_PII labels, using the same label categories, but where context clearly indicates the value is public/non-sensitive (e.g., institutional hotline phone, public office email, well-known public address).
   - For NON_PII, SUBCATEGORY is a short descriptor.

* Introduce difficulty:
   - some examples with multiple entities near each other; other split by long uninteresting text.
   - some with formatting artifacts that split tokens or introduce typos
   - some with the keywords that explain the context (such as Tel, address) either missing or on a different line, as could happen with OCR.
   - Keep it realistic but intentionally challenging. Add plenty of filler text. It should be a real text with real context, not only a list of PIIs.
   - Make sure the filler text does NOT have any PIIs or PII lookalikes. Anything looking like PII should be annotated.
* Composition guidance per example:
   - include between 15 to 40 total annotated spans in "annotated_text"
   - target approximately 50% NON_PII, 20% PII_LOOKALIKE, 30% REAL_PII
   - hard minimums: at least 5 NON_PII, at least 2 PII_LOOKALIKE, at least 3 REAL_PII
* classification guidelines:   
   - note that for ORG and associated data, they are REAL_PII if it's a small one the person is one of the few owners, clients, or employees of. Being owners, clients, or employees of a large, public organization is not sensitive information.
   - The contact details of a large organization are also likely to be public. However infomration such as passwords, or non-public emails, phone numbers, locations, IP addresses, etc, will be sensitive even if they belong to a large organization.
   - Any other "confusing" patterns that do not map to REAL_PII/PII_LOOKALIKE label structure must be annotated as NON_PII.
   - NON_PII examples: monetary amounts, form IDs, quantities with units, law refs, measurements, version numbers, paragraph numbers, tables of numbers, etc.
* Before returning JSON, self-check each example:
   - if spans are too few or proportions are off, add more annotations until constraints are satisfied
* Keep each text under 1000 characters.
* Do not include explanations, markdown, or extra keys.
* Do NOT often redact passwords and other PII or add indications that they are fake, the point is to learn to protect customers from sharing unredacted real ones.

Output format (JSON object):
{{
  "examples": [
    {{
      "annotated_text": "Please find us at [[REAL_PII:EMAIL|example@prov.ch]], or [[REAL_PII:PHONE|+393331232345]], thank you."
    }}
  ]
}}
"""
USER_PROMPT = USER_PROMPT_TEMPLATE.format(labels_csv=LABELS_CSV)


class ResponseFormatError(RuntimeError):
    def __init__(self, message: str, raw_content: str):
        super().__init__(message)
        self.raw_content = raw_content


def _clean_json_text(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _append_error_log(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl_row(path: Path, row: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_insufficient_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "insufficient_quota" in msg or "exceeded your current quota" in msg


def _to_int(value) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _pricing_for_model(model: str, input_tokens: Optional[int] = None) -> Dict[str, Optional[float]]:
    model_key = model.strip().lower()
    if model_key in CONTEXT_TIER_PRICING_PER_M:
        token_count = input_tokens or 0
        tier_key = "gt_272k" if token_count > CONTEXT_TIER_THRESHOLD_TOKENS else "lt_eq_272k"
        return CONTEXT_TIER_PRICING_PER_M[model_key][tier_key]
    pricing = MODEL_PRICING_PER_M.get(model_key)
    if pricing is None:
        # Fallback keeps cost estimation enabled for compatible models.
        pricing = MODEL_PRICING_PER_M["gpt-5-mini"]
    return pricing


def _is_known_priced_model(model: str) -> bool:
    model_key = model.strip().lower()
    return model_key in MODEL_PRICING_PER_M or model_key in CONTEXT_TIER_PRICING_PER_M


def _extract_usage_info(completion, model: str) -> Dict:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {
            "usage_available": False,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "uncached_input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    input_tokens = _to_int(getattr(usage, "prompt_tokens", None))
    if input_tokens == 0:
        input_tokens = _to_int(getattr(usage, "input_tokens", None))

    output_tokens = _to_int(getattr(usage, "completion_tokens", None))
    if output_tokens == 0:
        output_tokens = _to_int(getattr(usage, "output_tokens", None))

    input_details = getattr(usage, "prompt_tokens_details", None)
    if input_details is None:
        input_details = getattr(usage, "input_tokens_details", None)

    cached_input_tokens = _to_int(getattr(input_details, "cached_tokens", None))
    if cached_input_tokens == 0 and isinstance(input_details, dict):
        cached_input_tokens = _to_int(input_details.get("cached_tokens"))
    if cached_input_tokens > input_tokens:
        cached_input_tokens = input_tokens

    uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
    pricing_per_m = _pricing_for_model(model, input_tokens=input_tokens)
    cached_input_price = pricing_per_m["cached_input"]
    if cached_input_price is None:
        cached_input_price = pricing_per_m["input"]

    input_per_token = pricing_per_m["input"] / 1_000_000
    cached_input_per_token = cached_input_price / 1_000_000
    output_per_token = pricing_per_m["output"] / 1_000_000
    estimated_cost_usd = (
        uncached_input_tokens * input_per_token
        + cached_input_tokens * cached_input_per_token
        + output_tokens * output_per_token
    )
    return {
        "usage_available": True,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "pricing_per_m": {
            "input": pricing_per_m["input"],
            "cached_input": cached_input_price,
            "output": pricing_per_m["output"],
        },
    }


def _extract_first_json_object(raw: str) -> str:
    start = raw.find("{")
    if start < 0:
        return raw
    depth = 0
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return raw


def _is_gguf_model_path(model: str) -> bool:
    return Path(model).suffix.lower() == ".gguf"


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("local_base_url cannot be empty")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _detect_local_api_base_url_for_model(
    model: str,
    api_key: Optional[str],
) -> Optional[str]:
    default_base_url = "http://127.0.0.1:1234/v1"
    try:
        client = OpenAI(base_url=default_base_url, api_key=api_key or "lm-studio")
        model_list = client.models.list()
    except Exception:
        return None

    for entry in getattr(model_list, "data", []):
        if getattr(entry, "id", None) == model:
            return default_base_url
    return None


def _get_local_generator(model: str):
    with LOCAL_GENERATOR_LOCK:
        cached = LOCAL_GENERATORS.get(model)
        if cached is not None:
            return cached

        if _is_gguf_model_path(model):
            model_path = Path(model).expanduser()
            if not model_path.exists():
                raise RuntimeError(
                    f"GGUF model path does not exist: {model_path}. "
                    "Pass a valid local file path to a .gguf model."
                )
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError(
                    "GGUF local fallback requested, but 'llama-cpp-python' is not installed. "
                    "Install it in this Python environment (for example: pip install llama-cpp-python)."
                ) from exc
            try:
                generator = Llama(model_path=str(model_path), verbose=False)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load GGUF model from '{model_path}': {exc}"
                ) from exc
            runtime = {"backend": "gguf", "generator": generator}
            LOCAL_GENERATORS[model] = runtime
            return runtime

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Local fallback requested, but 'transformers' is not installed in this runtime. "
                "Install it in this Python environment and retry."
            ) from exc

        try:
            generator = pipeline("text-generation", model=model, tokenizer=model)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download/load local model '{model}'. "
                "Provide a valid local/HuggingFace model identifier available to this runtime."
            ) from exc
        runtime = {"backend": "transformers", "generator": generator}
        LOCAL_GENERATORS[model] = runtime
        return runtime


def _request_examples_via_chat_api(
    client: OpenAI,
    model: str,
    count: int,
    call_timeout_seconds: int,
    labels: set,
    extra_instruction: Optional[str] = None,
    prefer_json_object_response: bool = True,
) -> Tuple[List[Dict], Dict]:
    user_messages = [
        {"role": "user", "content": USER_PROMPT},
        {"role": "user", "content": f"Generate exactly {count} examples. Labels allowed: {LABELS_CSV}."},
    ]
    if extra_instruction:
        user_messages.append({"role": "user", "content": extra_instruction})

    response_format = (
        {"type": "json_object"} if prefer_json_object_response else {"type": "text"}
    )
    completion = client.chat.completions.create(
        model=model,
        response_format=response_format,
        timeout=call_timeout_seconds,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *user_messages,
        ],
    )
    content = completion.choices[0].message.content or ""
    cleaned = _clean_json_text(content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ResponseFormatError(f"response is not valid JSON: {exc}", raw_content=content) from exc
    examples = parsed.get("examples")
    if not isinstance(examples, list):
        raise ResponseFormatError("model output missing examples list", raw_content=content)
    normalized_examples: List[Dict] = []
    for idx, example in enumerate(examples):
        try:
            normalized_examples.append(_normalize_model_example(example, labels=labels))
        except Exception as exc:
            raise ResponseFormatError(
                f"example[{idx}] has invalid annotation format: {exc}", raw_content=content
            ) from exc
    return normalized_examples, _extract_usage_info(completion, model=model)


def _model_slug(model: str) -> str:
    slug = model.strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "model"


class ValidationStatus(str, Enum):
    OK = "OK"
    FAIL = "FAIL"
    RETRY = "RETRY"


def _build_retry_instruction(retry_reason_counts: Dict[str, int]) -> Optional[str]:
    if not retry_reason_counts:
        return None
    top_reasons = sorted(retry_reason_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    reason_text = "; ".join(f"{reason} (x{count})" for reason, count in top_reasons)
    return (
        "Retry because previous validation rejected examples for these reasons: "
        f"{reason_text}. "
        "Fix these issues. If spans are too few or proportions are off, add/fix annotations so each "
        "example satisfies the per-category minimums and target mix."
    )


def _extract_marked_content(text: str, start_idx: int) -> Tuple[str, int]:
    idx = start_idx + 2
    while idx < len(text):
        if text[idx : idx + 2] == "]]":
            return text[start_idx + 2 : idx], idx + 2
        idx += 1
    raise ValueError("unterminated annotation marker; expected closing ']]'")


def _parse_annotated_text_to_example(annotated_text: str, labels: set) -> Dict:
    if not isinstance(annotated_text, str) or not annotated_text.strip():
        raise ValueError("missing/empty annotated_text")

    out_chunks: List[str] = []
    items: List[Dict] = []
    plain_length = 0
    i = 0
    while i < len(annotated_text):
        if annotated_text[i : i + 2] != "[[":
            out_chunks.append(annotated_text[i])
            plain_length += 1
            i += 1
            continue

        marker_content, next_idx = _extract_marked_content(annotated_text, i)
        if "|" not in marker_content or ":" not in marker_content:
            raise ValueError(
                "invalid marker format; expected [[ROOT_CATEGORY:SUBCATEGORY|value]]"
            )
        left, value = marker_content.split("|", 1)
        root_category, subcategory = left.split(":", 1)
        root_category = root_category.strip().upper()
        subcategory = subcategory.strip()

        if not value:
            raise ValueError("empty marker value is not allowed")

        start = plain_length
        out_chunks.append(value)
        plain_length += len(value)
        end = plain_length

        if root_category == "REAL_PII":
            label = subcategory.upper()
            if label not in labels:
                raise ValueError(
                    f"invalid REAL_PII label in marker: {label!r}; allowed={sorted(labels)}"
                )
            item = {
                "start": start,
                "end": end,
                "category": "REAL_PII",
                "label": label,
                "value": value,
            }
        elif root_category == "PII_LOOKALIKE":
            label = subcategory.upper()
            if label not in labels:
                raise ValueError(
                    f"invalid PII_LOOKALIKE label in marker: {label!r}; allowed={sorted(labels)}"
                )
            item = {
                "start": start,
                "end": end,
                "category": root_category,
                "label": label,
                "value": value,
                "note": "public-context lookalike",
            }
        elif root_category == "NON_PII":
            item = {
                "start": start,
                "end": end,
                "category": root_category,
                "value": value,
                "note": subcategory or "annotated",
            }
        else:
            raise ValueError(
                "invalid ROOT_CATEGORY in marker: "
                f"{root_category!r}; expected NON_PII, PII_LOOKALIKE, or REAL_PII"
            )
        items.append(item)
        i = next_idx

    return {"text": "".join(out_chunks), "items": items, "original_text": annotated_text}


def _normalize_model_example(example: Dict, labels: set) -> Dict:
    if not isinstance(example, dict):
        raise ValueError("example is not an object")

    # Primary path: new in-place annotation format.
    annotated_text = example.get("annotated_text")
    if annotated_text is not None:
        return _parse_annotated_text_to_example(annotated_text, labels)

    # Backward-compatible fallback: old offset-based payload.
    text = example.get("text")
    items = example.get("items")
    if isinstance(text, str) and isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            start = item.get("start")
            end = item.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if not isinstance(item.get("value"), str):
                safe_start = max(0, min(start, len(text)))
                safe_end = max(safe_start, min(end, len(text)))
                item["value"] = text[safe_start:safe_end]
        return {"text": text, "items": items, "original_text": text}

    raise ValueError("example missing both annotated_text and legacy text/items")


def _validate_example(example: Dict, labels: set) -> Tuple[ValidationStatus, Optional[str]]:
    fail = ValidationStatus.FAIL
    retry = ValidationStatus.RETRY

    if not isinstance(example, dict):
        return fail, "example is not an object"
    text = example.get("text")
    items = example.get("items")
    if not isinstance(text, str) or not text.strip():
        return fail, "missing/empty text"
    if not isinstance(items, list) or not items:
        return fail, "missing/empty items"

    if len(items) < MIN_ITEMS or len(items) > MAX_ITEMS:
        return (
            retry,
            f"items must contain between {MIN_ITEMS} and {MAX_ITEMS} entries; got={len(items)}",
        )

    category_counts = {"NON_PII": 0, "PII_LOOKALIKE": 0, "REAL_PII": 0}
    real_items = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            return fail, f"item[{idx}] is not an object (type={type(item).__name__})"
        missing = {"start", "end", "category"} - set(item.keys())
        if missing:
            return fail, f"item[{idx}] missing keys: {sorted(missing)}"
        if not isinstance(item["start"], int) or not isinstance(item["end"], int):
            return (
                fail,
                f"item[{idx}] start/end must be int; got start={item.get('start')!r} "
                f"({type(item.get('start')).__name__}), end={item.get('end')!r} "
                f"({type(item.get('end')).__name__})",
            )
        if item["start"] < 0 or item["end"] <= item["start"]:
            return (
                fail,
                f"item[{idx}] invalid item bounds ordering: start={item['start']} end={item['end']}",
            )
        if item["end"] > len(text):
            return (
                fail,
                f"item[{idx}] out of range: start={item['start']} end={item['end']} "
                f"valid_range=[0,{len(text)}]",
            )
        if not text[item["start"] : item["end"]].strip():
            return (
                fail,
                f"item[{idx}] points to only whitespace: start={item['start']} end={item['end']}",
            )

        category = item["category"]
        # Accept label-style categories (e.g. PERSON/OTHER) as REAL_PII shorthand.
        if category in labels:
            item["category"] = "REAL_PII"
            if item.get("label") is None:
                item["label"] = category
            category = "REAL_PII"
        if category not in category_counts:
            return (
                fail,
                f"invalid item category at item[{idx}]: {category}; "
                "allowed categories are NON_PII, PII_LOOKALIKE, REAL_PII "
                "or any valid REAL_PII label used as shorthand category",
            )
        category_counts[category] += 1

        if category == "REAL_PII":
            label = item.get("label")
            if label not in labels:
                return (
                    fail,
                    f"REAL_PII item[{idx}] requires valid label; got={label!r}; "
                    f"allowed={sorted(labels)}",
                )
            key = (item["start"], item["end"])
            if key in real_items and real_items[key] != label:
                return (
                    fail,
                    "conflicting REAL_PII labels for same span "
                    f"start={item['start']} end={item['end']}: "
                    f"existing={real_items[key]} new={label}",
                )
            real_items[key] = label
        if category == "PII_LOOKALIKE":
            label = item.get("label")
            if isinstance(label, str):
                label = label.upper()
                item["label"] = label
            if label not in labels:
                return (
                    fail,
                    f"PII_LOOKALIKE item[{idx}] requires valid label; got={label!r}; "
                    f"allowed={sorted(labels)}",
                )

    if category_counts["NON_PII"] < MIN_NON_PII:
        return (
            retry,
            f"must have at least {MIN_NON_PII} NON_PII items; "
            f"got={category_counts['NON_PII']}",
        )
    if category_counts["PII_LOOKALIKE"] < MIN_PII_LOOKALIKE:
        return (
            retry,
            f"must have at least {MIN_PII_LOOKALIKE} PII_LOOKALIKE items; "
            f"got={category_counts['PII_LOOKALIKE']}",
        )
    if category_counts["REAL_PII"] < MIN_REAL_PII:
        return (
            retry,
            f"must have at least {MIN_REAL_PII} REAL_PII items; got={category_counts['REAL_PII']}",
        )

    total_items = len(items)
    non_pii_ratio = category_counts["NON_PII"] / total_items
    lookalike_ratio = category_counts["PII_LOOKALIKE"] / total_items
    real_pii_ratio = category_counts["REAL_PII"] / total_items

    # Loose guardrails: encourage the intended 50/20/30 mix without being rigid.
    if not (0.20 <= non_pii_ratio <= 0.70):
        return (
            retry,
            "NON_PII ratio is too far from target: "
            f"ratio={non_pii_ratio:.3f} expected=[0.200,0.700] "
            f"counts={category_counts['NON_PII']}/{total_items}",
        )
    if not (0.10 <= lookalike_ratio <= 0.40):
        return (
            retry,
            "PII_LOOKALIKE ratio is too far from target: "
            f"ratio={lookalike_ratio:.3f} expected=[0.100,0.400] "
            f"counts={category_counts['PII_LOOKALIKE']}/{total_items}",
        )
    if not (0.10 <= real_pii_ratio <= 0.50):
        return (
            retry,
            "REAL_PII ratio is too far from target: "
            f"ratio={real_pii_ratio:.3f} expected=[0.100,0.500] "
            f"counts={category_counts['REAL_PII']}/{total_items}",
        )

    return ValidationStatus.OK, None


def _spans_from_items(text: str, items: List[Dict]) -> List[Dict]:
    dedup: Dict[Tuple[int, int], Dict] = {}
    for item in items:
        if item.get("category") != "REAL_PII":
            continue
        key = (item["start"], item["end"])
        value = item.get("value")
        if not isinstance(value, str):
            safe_start = max(0, min(item["start"], len(text)))
            safe_end = max(safe_start, min(item["end"], len(text)))
            value = text[safe_start:safe_end]
        dedup[key] = {"label": item["label"], "value": value}
    spans = [
        {"start": s, "end": e, "label": data["label"], "value": data["value"]}
        for (s, e), data in dedup.items()
    ]
    spans.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return spans


def _pii_lookalike_from_items(text: str, items: List[Dict]) -> List[Dict]:
    dedup: Dict[Tuple[int, int], Dict] = {}
    for item in items:
        if item.get("category") != "PII_LOOKALIKE":
            continue
        key = (item["start"], item["end"])
        value = item.get("value")
        if not isinstance(value, str):
            safe_start = max(0, min(item["start"], len(text)))
            safe_end = max(safe_start, min(item["end"], len(text)))
            value = text[safe_start:safe_end]
        dedup[key] = {"label": item.get("label"), "value": value}
    lookalikes = [
        {"start": s, "end": e, "label": data["label"], "value": data["value"]}
        for (s, e), data in dedup.items()
    ]
    lookalikes.sort(key=lambda x: (x["start"], x["end"], str(x.get("label"))))
    return lookalikes


def _request_examples(
    client: OpenAI,
    model: str,
    count: int,
    call_timeout_seconds: int,
    labels: set,
    extra_instruction: Optional[str] = None,
    local_openai_client: Optional[OpenAI] = None,
) -> Tuple[List[Dict], Dict]:
    if not _is_known_priced_model(model):
        if local_openai_client is not None:
            if _is_gguf_model_path(model):
                raise RuntimeError(
                    "When using --local-base-url, --model must be the local API model id "
                    "(not a .gguf file path)."
                )
            return _request_examples_via_chat_api(
                client=local_openai_client,
                model=model,
                count=count,
                call_timeout_seconds=call_timeout_seconds,
                labels=labels,
                extra_instruction=extra_instruction,
                prefer_json_object_response=False,
            )

        local_runtime = _get_local_generator(model)
        backend = local_runtime.get("backend")
        generator = local_runtime.get("generator")
        instruction_parts = [
            SYSTEM_PROMPT.strip(),
            USER_PROMPT.strip(),
            f"Generate exactly {count} examples.",
        ]
        if extra_instruction:
            instruction_parts.append(extra_instruction.strip())
        instruction_parts.append("Return ONLY valid JSON object with key 'examples'.")
        prompt = "\n\n".join(part for part in instruction_parts if part)

        try:
            if backend == "transformers":
                outputs = generator(
                    prompt,
                    max_new_tokens=LOCAL_MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.2,
                    return_full_text=False,
                )
                if not outputs or not isinstance(outputs, list) or "generated_text" not in outputs[0]:
                    raise ResponseFormatError(
                        "local model output is empty or malformed", raw_content=str(outputs)
                    )
                content = outputs[0]["generated_text"] or ""
            elif backend == "gguf":
                completion = generator.create_completion(
                    prompt=prompt,
                    max_tokens=LOCAL_MAX_NEW_TOKENS,
                    temperature=0.2,
                    top_p=0.95,
                    echo=False,
                )
                choices = completion.get("choices") if isinstance(completion, dict) else None
                if not choices or not isinstance(choices, list):
                    raise ResponseFormatError(
                        "local GGUF output is empty or malformed", raw_content=str(completion)
                    )
                first_choice = choices[0] if isinstance(choices[0], dict) else {}
                content = first_choice.get("text", "") or ""
                if not content and isinstance(first_choice.get("message"), dict):
                    content = first_choice["message"].get("content", "") or ""
            else:
                raise RuntimeError(
                    f"Unsupported local backend '{backend}' for model '{model}'."
                )
        except Exception as exc:
            raise RuntimeError(f"Local generation failed for model '{model}': {exc}") from exc

        cleaned = _clean_json_text(content)
        if not cleaned.startswith("{"):
            cleaned = _extract_first_json_object(cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            cleaned_json_candidate = _extract_first_json_object(content)
            try:
                parsed = json.loads(_clean_json_text(cleaned_json_candidate))
            except json.JSONDecodeError:
                raise ResponseFormatError(
                    f"local model response is not valid JSON: {exc}", raw_content=content
                ) from exc

        examples = parsed.get("examples")
        if not isinstance(examples, list):
            raise ResponseFormatError("local model output missing examples list", raw_content=content)
        normalized_examples: List[Dict] = []
        for idx, example in enumerate(examples):
            try:
                normalized_examples.append(_normalize_model_example(example, labels=labels))
            except Exception as exc:
                raise ResponseFormatError(
                    f"example[{idx}] has invalid annotation format: {exc}", raw_content=content
                ) from exc
        return (
            normalized_examples,
            {
                "usage_available": False,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "uncached_input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
        )

    return _request_examples_via_chat_api(
        client=client,
        model=model,
        count=count,
        call_timeout_seconds=call_timeout_seconds,
        labels=labels,
        extra_instruction=extra_instruction,
        prefer_json_object_response=True,
    )


def _load_existing_jsonl(path: Path) -> Tuple[int, set]:
    if not path.exists():
        return 0, set()

    count = 0
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {i}: {exc}") from exc
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Missing/invalid 'text' in {path} at line {i}")
            seen.add(text)
            count += 1
    return count, seen


def generate_dataset(
    model: str,
    train_size: int,
    valid_size: int,
    train_path: Path,
    valid_path: Path,
    batch_size: int,
    parallel: int,
    max_calls: Optional[int],
    max_dollars: float,
    call_timeout_seconds: int,
    seed: int,
    overwrite: bool,
    local: bool,
    local_base_url: Optional[str],
    local_api_key: Optional[str],
) -> None:
    load_dotenv()
    local_model_mode = local or bool(local_base_url) or not _is_known_priced_model(model)
    cost_tracking_enabled = not local_model_mode
    if not local_model_mode and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in a .env file or export it in the environment."
        )

    random.seed(seed)
    if local_model_mode:
        if local and _is_known_priced_model(model):
            print(
                f"--local enabled: forcing local mode for model '{model}' "
                "even though it matches a known priced model name."
            )
        else:
            print(
                f"Model '{model}' is not in known pricing tables. "
                "Will attempt local inference."
            )
    else:
        pricing_per_m = _pricing_for_model(model)
        cached_price_label = (
            f"${pricing_per_m['cached_input']}/1M"
            if pricing_per_m["cached_input"] is not None
            else "same as input (no separate cached tier)"
        )
        print(
            "Cost model "
            f"{model}: input=${pricing_per_m['input']}/1M "
            f"cached_input={cached_price_label} "
            f"output=${pricing_per_m['output']}/1M"
        )
        if model.strip().lower() in CONTEXT_TIER_PRICING_PER_M:
            print(
                f"{model} uses context-tier pricing; threshold={CONTEXT_TIER_THRESHOLD_TOKENS} input tokens "
                "(<= threshold vs > threshold) is applied per call."
            )
    labels_set = set(LABELS)
    target_total = train_size + valid_size
    if target_total <= 0:
        raise ValueError("train_size + valid_size must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if parallel <= 0:
        raise ValueError("parallel must be > 0")
    if local_model_mode and parallel > 1:
        print(
            f"Local model mode active; forcing parallel from {parallel} to 1 "
            "to avoid concurrent local generation conflicts."
        )
        parallel = 1
    if max_calls is not None and max_calls <= 0:
        raise ValueError("max_calls must be > 0 when provided")
    if max_dollars < 0:
        raise ValueError("max_dollars must be >= 0")
    if call_timeout_seconds <= 0:
        raise ValueError("call_timeout_seconds must be > 0")
    if cost_tracking_enabled:
        if max_dollars == 0:
            print("WARNING: --max-dollars=0 selected (unlimited spend).")
        else:
            print(f"Estimated cost cap enabled: ${max_dollars:.2f} (soft cap).")

    seen_texts = set()
    duplicate_count = 0
    invalid_count = 0
    retry_invalid_count = 0
    fail_invalid_count = 0
    retry_reason_counts: Dict[str, int] = {}
    invalid_reason_counts: Dict[str, int] = {}
    quota_exhausted = False
    budget_exhausted = False
    running_input_tokens = 0
    running_cached_input_tokens = 0
    running_output_tokens = 0
    running_cost_usd = 0.0

    client = OpenAI()
    local_openai_client: Optional[OpenAI] = None
    selected_local_base_url = local_base_url
    if local_model_mode and not selected_local_base_url:
        # Convenience path: if LM Studio is running locally and serves this model id,
        # auto-route local requests through its OpenAI-compatible endpoint.
        selected_local_base_url = _detect_local_api_base_url_for_model(
            model=model,
            api_key=local_api_key,
        )
        if selected_local_base_url:
            print(
                "Detected local API model on LM Studio; "
                f"using base_url={selected_local_base_url}."
            )

    if local_model_mode and selected_local_base_url:
        normalized_base_url = _normalize_openai_base_url(selected_local_base_url)
        local_openai_client = OpenAI(
            base_url=normalized_base_url,
            api_key=local_api_key or "lm-studio",
        )
        print(
            "Local API mode enabled "
            f"(base_url={normalized_base_url}, model={model})."
        )
    elif local_model_mode:
        # Fail fast once instead of repeatedly attempting the same invalid local loader setup.
        _get_local_generator(model)
    attempts = 0

    train_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    error_log_path = train_path.parent / ERROR_LOG_FILENAME

    if overwrite:
        train_written = 0
        valid_written = 0
        train_path.write_text("", encoding="utf-8")
        valid_path.write_text("", encoding="utf-8")
    else:
        train_written, train_seen = _load_existing_jsonl(train_path)
        valid_written, valid_seen = _load_existing_jsonl(valid_path)
        seen_texts.update(train_seen)
        seen_texts.update(valid_seen)

    if train_written > train_size or valid_written > valid_size:
        raise ValueError(
            "Existing files already exceed requested target sizes. "
            f"existing train={train_written}/{train_size}, valid={valid_written}/{valid_size}. "
            "Increase --train-size/--valid-size or run with --overwrite."
        )

    if not overwrite:
        print(
            f"Append mode: existing rows train={train_written}/{train_size}, "
            f"valid={valid_written}/{valid_size}"
        )

    remaining_total = target_total - (train_written + valid_written)
    if remaining_total <= 0:
        print("Targets already satisfied from existing data; nothing to generate.")
        return

    theoretical_min_attempts = math.ceil(remaining_total / batch_size)
    auto_max_attempts = theoretical_min_attempts * 3
    effective_max_attempts = max_calls if max_calls is not None else auto_max_attempts
    if effective_max_attempts < theoretical_min_attempts:
        raise ValueError(
            "Inconsistent parameters: max_calls is lower than theoretical minimum calls needed. "
            f"max_calls={effective_max_attempts}, required_min_calls={theoretical_min_attempts}, "
            f"remaining_total={remaining_total}, batch_size={batch_size}."
        )
    if max_calls is None:
        print(
            f"Auto max_calls enabled: using 3x theoretical minimum "
            f"({theoretical_min_attempts} -> {effective_max_attempts})."
        )

    # Rolling scheduler: keep up to `parallel` calls in-flight and launch new ones as calls complete.
    calls_submitted = 0
    pending_futures: Dict[concurrent.futures.Future, Dict] = {}
    wait_started = time.monotonic()
    last_return_monotonic = wait_started

    row_id = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        while (train_written + valid_written) < target_total and (
            calls_submitted < effective_max_attempts or pending_futures
        ):
            needed = target_total - (train_written + valid_written)
            inflight_requested = sum(m["requested_count"] for m in pending_futures.values())
            fillable = max(0, needed - inflight_requested)

            # Refill pipeline immediately whenever slots are available.
            while (
                fillable > 0
                and len(pending_futures) < parallel
                and calls_submitted < effective_max_attempts
                and not quota_exhausted
                and not budget_exhausted
            ):
                if cost_tracking_enabled and max_dollars > 0 and running_cost_usd >= max_dollars:
                    budget_exhausted = True
                    print(
                        "Estimated cost cap reached; stopping new call submissions and waiting for in-flight calls."
                    )
                    _append_error_log(
                        error_log_path,
                        {
                            "type": "budget_cap_reached",
                            "model": model,
                            "timestamp": _now_iso(),
                            "max_dollars": max_dollars,
                            "running_cost_usd": running_cost_usd,
                            "calls_submitted": calls_submitted,
                            "calls_completed": attempts,
                        },
                    )
                    break
                request_n = min(batch_size, fillable)
                call_id = calls_submitted + 1
                retry_instruction = _build_retry_instruction(retry_reason_counts)
                is_retry = retry_instruction is not None
                print(f"[call {call_id}] starting requested_count={request_n}")
                if is_retry:
                    print(
                        f"[call {call_id}] retrying with validation guidance "
                        f"(retry_rejects={retry_invalid_count}, total_rejects={invalid_count})"
                    )
                _append_error_log(
                    error_log_path,
                    {
                        "type": "call_started",
                        "call_id": call_id,
                        "model": model,
                        "requested_count": request_n,
                        "is_retry": is_retry,
                        "retry_reason": (
                            "retry guidance from validation RETRY reasons" if is_retry else None
                        ),
                        "retryable_rejects": retry_invalid_count,
                        "total_rejects": invalid_count,
                        "retry_reason_counts_snapshot": (
                            dict(sorted(retry_reason_counts.items(), key=lambda x: x[1], reverse=True)[:5])
                            if is_retry
                            else {}
                        ),
                        "timestamp": _now_iso(),
                    },
                )
                started_monotonic = time.monotonic()
                future = executor.submit(
                    _request_examples,
                    client=client,
                    model=model,
                    count=request_n,
                    call_timeout_seconds=call_timeout_seconds,
                    labels=labels_set,
                    extra_instruction=retry_instruction,
                    local_openai_client=local_openai_client,
                )
                pending_futures[future] = {
                    "call_id": call_id,
                    "requested_count": request_n,
                    "is_retry": is_retry,
                    "started_at": _now_iso(),
                    "started_monotonic": started_monotonic,
                }
                calls_submitted += 1
                fillable -= request_n

            if not pending_futures:
                break

            done, _ = concurrent.futures.wait(
                set(pending_futures.keys()),
                timeout=FUTURE_POLL_SECONDS,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                waiting_for_s = time.monotonic() - last_return_monotonic
                oldest_pending_s = max(
                    time.monotonic() - meta["started_monotonic"] for meta in pending_futures.values()
                )
                print(
                    f"Waiting on {len(pending_futures)} API calls... "
                    f"since_last_return={waiting_for_s:.1f}s oldest_pending={oldest_pending_s:.1f}s"
                )
                continue

            for future in done:
                last_return_monotonic = time.monotonic()
                attempts += 1
                meta = pending_futures.pop(future)
                elapsed_s = time.monotonic() - meta["started_monotonic"]
                try:
                    result, usage_info = future.result()
                    running_input_tokens += usage_info["input_tokens"]
                    running_cached_input_tokens += usage_info["cached_input_tokens"]
                    running_output_tokens += usage_info["output_tokens"]
                    if cost_tracking_enabled:
                        running_cost_usd += usage_info["estimated_cost_usd"]
                    if (
                        cost_tracking_enabled
                        and max_dollars > 0
                        and running_cost_usd >= max_dollars
                        and not budget_exhausted
                    ):
                        budget_exhausted = True
                        print(
                            "Estimated cost cap reached after this call; no further calls will be submitted."
                        )
                        _append_error_log(
                            error_log_path,
                            {
                                "type": "budget_cap_reached",
                                "model": model,
                                "timestamp": _now_iso(),
                                "max_dollars": max_dollars,
                                "running_cost_usd": running_cost_usd,
                                "calls_submitted": calls_submitted,
                                "calls_completed": attempts,
                                "trigger_call_id": meta["call_id"],
                            },
                        )
                    if cost_tracking_enabled:
                        print(
                            f"[call {meta['call_id']}] finished ok in {elapsed_s:.1f}s "
                            f"returned={len(result)} "
                            f"tokens(in={usage_info['input_tokens']}, cached={usage_info['cached_input_tokens']}, "
                            f"out={usage_info['output_tokens']}) "
                            f"est_cost=${usage_info['estimated_cost_usd']:.6f} total_est=${running_cost_usd:.6f}"
                        )
                    else:
                        print(
                            f"[call {meta['call_id']}] finished ok in {elapsed_s:.1f}s "
                            f"returned={len(result)} "
                            f"tokens(in={usage_info['input_tokens']}, cached={usage_info['cached_input_tokens']}, "
                            f"out={usage_info['output_tokens']})"
                        )
                    _append_error_log(
                        error_log_path,
                        {
                            "type": "call_finished",
                            "call_id": meta["call_id"],
                            "model": model,
                            "requested_count": meta["requested_count"],
                            "is_retry": meta.get("is_retry", False),
                            "started_at": meta["started_at"],
                            "timestamp": _now_iso(),
                            "status": "ok",
                            "returned_examples": len(result),
                            "usage": usage_info,
                            "running_usage": {
                                "input_tokens": running_input_tokens,
                                "cached_input_tokens": running_cached_input_tokens,
                                "output_tokens": running_output_tokens,
                                "estimated_cost_usd": running_cost_usd,
                            },
                        },
                    )

                    for ex in result:
                        status, reason = _validate_example(ex, labels_set)
                        if status != ValidationStatus.OK:
                            reason_key = reason or "unknown"
                            invalid_count += 1
                            if status == ValidationStatus.RETRY:
                                retry_invalid_count += 1
                                retry_reason_counts[reason_key] = retry_reason_counts.get(reason_key, 0) + 1
                            else:
                                fail_invalid_count += 1
                            invalid_reason_counts[reason_key] = invalid_reason_counts.get(reason_key, 0) + 1
                            candidate_items = ex.get("items")
                            candidate_items_count = (
                                len(candidate_items) if isinstance(candidate_items, list) else "n/a"
                            )
                            print(
                                f"[validation_reject][call_id={meta['call_id']}] "
                                f"status={status.value} "
                                f"total_rejected={invalid_count} "
                                f"reason_count={invalid_reason_counts[reason_key]} "
                                f"candidate_items={candidate_items_count} "
                                f"reason={reason_key}"
                            )
                            _append_error_log(
                                error_log_path,
                                {
                                    "type": "validation_error",
                                    "call": attempts,
                                    "status": status.value,
                                    "reason": reason_key,
                                    "candidate": ex,
                                },
                            )
                            continue
                        text = ex["text"]
                        if text in seen_texts:
                            duplicate_count += 1
                            continue
                        seen_texts.add(text)

                        true_spans = _spans_from_items(text, ex["items"])
                        lookalike_spans = _pii_lookalike_from_items(text, ex["items"])
                        row_id += 1
                        row = {
                            "row_id": row_id,
                            "text": text,
                            "spans": true_spans,
                            "original_text": ex.get("original_text", text),
                            "pii_lookalike": lookalike_spans,
                        }
                        remaining_train = train_size - train_written
                        remaining_valid = valid_size - valid_written

                        if remaining_train > 0 and remaining_valid > 0:
                            p_train = remaining_train / (remaining_train + remaining_valid)
                            to_train = random.random() < p_train
                        else:
                            to_train = remaining_train > 0

                        if to_train:
                            _append_jsonl_row(train_path, row)
                            train_written += 1
                        else:
                            _append_jsonl_row(valid_path, row)
                            valid_written += 1

                        if (train_written + valid_written) >= target_total:
                            break
                except Exception as exc:
                    print(
                        f"[call {meta['call_id']}] failed in {elapsed_s:.1f}s "
                        f"API/parsing error: {exc}"
                    )
                    if _is_insufficient_quota_error(exc):
                        quota_exhausted = True
                    _append_error_log(
                        error_log_path,
                        {
                            "type": "call_finished",
                            "call_id": meta["call_id"],
                            "model": model,
                            "requested_count": meta["requested_count"],
                            "is_retry": meta.get("is_retry", False),
                            "started_at": meta["started_at"],
                            "timestamp": _now_iso(),
                            "status": "error",
                            "error": str(exc),
                        },
                    )
                    error_payload = {
                        "type": "api_or_parsing_error",
                        "call": attempts,
                        "call_id": meta["call_id"],
                        "requested_count": meta["requested_count"],
                        "model": model,
                        "error": str(exc),
                    }
                    if isinstance(exc, ResponseFormatError):
                        error_payload["raw_response"] = exc.raw_content
                    _append_error_log(error_log_path, error_payload)

                if cost_tracking_enabled:
                    print(
                        f"[calls={attempts}/{theoretical_min_attempts} est (max {effective_max_attempts})] "
                        f"in_flight={len(pending_futures)} "
                        f"total={train_written + valid_written}/{target_total} "
                        f"(train={train_written}/{train_size}, valid={valid_written}/{valid_size}) "
                        f"cost_est=${running_cost_usd:.6f} "
                        f"rejects(retry={retry_invalid_count}, fail={fail_invalid_count})"
                    )
                else:
                    print(
                        f"[calls={attempts}/{theoretical_min_attempts} est (max {effective_max_attempts})] "
                        f"in_flight={len(pending_futures)} "
                        f"total={train_written + valid_written}/{target_total} "
                        f"(train={train_written}/{train_size}, valid={valid_written}/{valid_size}) "
                        f"rejects(retry={retry_invalid_count}, fail={fail_invalid_count})"
                    )

            if quota_exhausted and not pending_futures:
                break

    if cost_tracking_enabled and budget_exhausted and (train_written + valid_written) < target_total:
        raise RuntimeError(
            "Estimated cost cap reached (--max-dollars). Stopped submitting new calls. "
            f"Cap=${max_dollars:.2f}, estimated_spend=${running_cost_usd:.6f}. "
            f"Partial data saved to {train_path} and {valid_path} "
            f"(train={train_written}, valid={valid_written}). "
            f"See {error_log_path} for details."
        )

    if quota_exhausted and (train_written + valid_written) < target_total:
        raise RuntimeError(
            "OpenAI quota exhausted (insufficient_quota). Stopped submitting new calls. "
            f"Partial data saved to {train_path} and {valid_path} "
            f"(train={train_written}, valid={valid_written}). "
            f"See {error_log_path} for details."
        )

    if (train_written + valid_written) < target_total:
        cap = effective_max_attempts * batch_size
        reason_parts = ", ".join(
            f"{k}={v}" for k, v in sorted(invalid_reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        raise RuntimeError(
            "Could not produce enough valid examples from API output. "
            f"Needed {target_total}, got {train_written + valid_written} after {attempts} attempts. "
            f"Theoretical raw cap this run was {cap}. "
            f"duplicates={duplicate_count}, invalid={invalid_count}, "
            f"retry_invalid={retry_invalid_count}, fail_invalid={fail_invalid_count}, "
            f"top_invalid_reasons=[{reason_parts}]. "
            f"Partial data was saved to {train_path} and {valid_path} "
            f"(train={train_written}, valid={valid_written}). "
            f"Error details logged to {error_log_path}."
        )

    print(f"Wrote {train_written} rows to {train_path}")
    print(f"Wrote {valid_written} rows to {valid_path}")
    if cost_tracking_enabled:
        print(
            f"Estimated usage totals: input={running_input_tokens}, cached_input={running_cached_input_tokens}, "
            f"output={running_output_tokens}, estimated_cost=${running_cost_usd:.6f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate intentionally challenging JSONL span data using OpenAI or local models."
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help=(
            "OpenAI model name, local HF model id/path, or a local .gguf file path "
            "(requires llama-cpp-python)."
        ),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Force local mode and skip OpenAI pricing/cost logic. "
            "Use with --model for local model id/path."
        ),
    )
    parser.add_argument(
        "--local-base-url",
        "--local-openai-base-url",
        dest="local_base_url",
        default=os.getenv("LOCAL_BASE_URL") or os.getenv("LOCAL_OPENAI_BASE_URL"),
        help=(
            "Optional local API endpoint (for example LM Studio: "
            "http://127.0.0.1:1234/v1). Providing this flag enables local mode automatically."
        ),
    )
    parser.add_argument(
        "--local-api-key",
        "--local-openai-api-key",
        dest="local_api_key",
        default=os.getenv("LOCAL_API_KEY") or os.getenv("LOCAL_OPENAI_API_KEY", "lm-studio"),
        help="API key for --local-base-url (LM Studio accepts any non-empty string).",
    )
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--valid-size", type=int, default=100)
    parser.add_argument(
        "--train-out",
        default=None,
        help="Output train JSONL path. Default: train.<model>.json",
    )
    parser.add_argument(
        "--valid-out",
        default=None,
        help="Output valid JSONL path. Default: valid.<model>.json",
    )
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--parallel",
        type=int,
        default=25,
        help="Number of concurrent API calls per round.",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Hard cap on total API calls for this run. Default: auto = 3x theoretical minimum.",
    )
    parser.add_argument(
        "--max_dollars",
        type=float,
        default=0.0,
        help="Estimated spend cap in USD. 0 means unlimited (default).",
    )
    parser.add_argument(
        "--call-timeout-seconds",
        type=int,
        default=DEFAULT_CALL_TIMEOUT_SECONDS,
        help="Per-call API timeout in seconds (default: 600).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files instead of appending to existing data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_slug = _model_slug(args.model)
    train_out = args.train_out or f"train.{model_slug}.json"
    valid_out = args.valid_out or f"valid.{model_slug}.json"
    generate_dataset(
        model=args.model,
        train_size=args.train_size,
        valid_size=args.valid_size,
        train_path=Path(train_out),
        valid_path=Path(valid_out),
        batch_size=args.batch_size,
        parallel=args.parallel,
        max_calls=args.max_calls,
        max_dollars=args.max_dollars,
        call_timeout_seconds=args.call_timeout_seconds,
        seed=args.seed,
        overwrite=args.overwrite,
        local=args.local,
        local_base_url=args.local_base_url,
        local_api_key=args.local_api_key,
    )
