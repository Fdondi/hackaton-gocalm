from typing import Dict, Optional

from pii_labels import PII_LABELS


NONE_LABEL = "NONE"
TYPE_LABELS = [NONE_LABEL, *PII_LABELS]
TYPE_LABEL_TO_ID: Dict[str, int] = {label: i for i, label in enumerate(TYPE_LABELS)}
TYPE_ID_TO_LABEL: Dict[int, str] = {i: label for label, i in TYPE_LABEL_TO_ID.items()}

BIO_LABELS = ["O", "B-ENTITY", "I-ENTITY"]
BIO_LABEL_TO_ID: Dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}
BIO_ID_TO_LABEL: Dict[int, str] = {i: label for label, i in BIO_LABEL_TO_ID.items()}

SENSITIVITY_LABELS = ["REDACT", "KEEP"]
SENSITIVITY_LABEL_TO_ID: Dict[str, int] = {
    label: i for i, label in enumerate(SENSITIVITY_LABELS)
}
SENSITIVITY_ID_TO_LABEL: Dict[int, str] = {
    i: label for label, i in SENSITIVITY_LABEL_TO_ID.items()
}

ITEM_CATEGORY_TO_SENSITIVITY = {
    "REAL_PII": "REDACT",
    "PII_LOOKALIKE": "KEEP",
}


def sensitivity_from_item_category(category: Optional[str]) -> Optional[str]:
    if not isinstance(category, str):
        return None
    return ITEM_CATEGORY_TO_SENSITIVITY.get(category.upper())
