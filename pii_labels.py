PII_LABELS = [
    "PERSON",
    "ORG",
    "ADDRESS",
    "EMAIL",
    "PHONE",
    "USERNAME",
    "PASSWORD",
    "IP_ADDRESS",
    "IBAN",
    "CREDIT_CARD",
    "ID_NUMBER",
    "ACCOUNT_NUMBER",
    "OTHER",
]

# Preformatted once; reuse everywhere prompts need labels.
PII_LABELS_CSV = ", ".join(PII_LABELS)
PII_LABELS_BULLETS = "\n".join(f"- {label}" for label in PII_LABELS)

# Training taxonomy includes the background class.
TRAINING_LABELS = ["NONE", *PII_LABELS]
