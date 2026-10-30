"""GroundedGuard API settings, read from environment variables.

GG_MODEL               modernbert | distilbert         (configs/inference.yaml)    default modernbert
GG_PRECISION           fp32 | bf16                     (bf16 needs a GPU)          default fp32
GG_DEVICE              auto | cuda | cuda:N | cpu                                  default auto
GG_MAX_RESPONSE_CHARS  max characters in a response                                default 10000
GG_MAX_CONTEXT_CHARS   max characters of serialized context (question + passages,
                       document, or serialized data)                              default 60000
GG_MAX_BATCH           max requests in one /detect/batch call                      default 16
GG_LOG_CONTENT         true | false: log (truncated) request text; off by default   default false
"""

import os
from dataclasses import dataclass

import yaml

from groundedguard.inference.predictor import INFERENCE_CONFIG


def _bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"not a boolean: {value!r}")


@dataclass(frozen=True)
class Settings:
    model: str = "modernbert"
    precision: str = "fp32"
    device: str = "auto"
    max_response_chars: int = 10_000
    max_context_chars: int = 60_000
    max_batch: int = 16
    log_content: bool = False

    def __post_init__(self):
        models = yaml.safe_load(INFERENCE_CONFIG.read_text())["models"]
        if self.model not in models:
            raise ValueError(f"GG_MODEL must be one of {sorted(models)}, got {self.model!r}")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError(f"GG_PRECISION must be fp32 or bf16, got {self.precision!r}")
        if self.precision == "bf16" and self.device == "cpu":
            raise ValueError("GG_PRECISION=bf16 needs a GPU device")
        for name in ("max_response_chars", "max_context_chars", "max_batch"):
            if getattr(self, name) < 1:
                raise ValueError(f"GG_{name.upper()} must be >= 1")

    @property
    def checkpoint(self) -> str:
        return yaml.safe_load(INFERENCE_CONFIG.read_text())["models"][self.model]["checkpoint"]

    @classmethod
    def from_env(cls, env=os.environ) -> "Settings":
        return cls(
            model=env.get("GG_MODEL", cls.model),
            precision=env.get("GG_PRECISION", cls.precision),
            device=env.get("GG_DEVICE", cls.device),
            max_response_chars=int(env.get("GG_MAX_RESPONSE_CHARS", cls.max_response_chars)),
            max_context_chars=int(env.get("GG_MAX_CONTEXT_CHARS", cls.max_context_chars)),
            max_batch=int(env.get("GG_MAX_BATCH", cls.max_batch)),
            log_content=_bool(env.get("GG_LOG_CONTENT", "false")),
        )
