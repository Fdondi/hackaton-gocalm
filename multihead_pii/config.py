import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class MultiHeadConfig:
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    max_span_len: int = 12
    train_batch_size: int = 2
    eval_batch_size: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 100
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-3
    seed: int = 42
    dropout: float = 0.1
    span_width_vocab_size: int = 64
    proposal_loss_weight: float = 1.0
    type_loss_weight: float = 1.0
    sensitivity_loss_weight: float = 1.0
    lookalike_redact_target: float = 0.8
    no_info_keep_target: float = 1.0
    negative_sample_rate: float = 0.2
    include_regex_candidates: bool = True
    regex_only_inference: bool = False
    redact_score_threshold: float = 0.5
    nms_iou_threshold: float = 0.5
    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str) -> MultiHeadConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must be an object.")
    return MultiHeadConfig(**payload)


def save_config(config: MultiHeadConfig, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
