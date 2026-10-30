"""Module 6: GroundedGuard inference. Flags spans of a response that the context does not support.

Everything that touches the data reuses the training code: context serialization
(data/serialize.py), tokenization, offset trimming and window building (data/windowing.py),
batching, window aggregation and scoring (models/dataset.py, models/aggregate.py,
models/train.py). If an input is longer than the model's max_len, the context is split into
sliding windows and scored with the same min-over-windows rule as in training.

Python:
    p = GroundedGuardPredictor("checkpoints/modernbert_lr5e-5_s42")   # threshold: frozen value
    p.predict("The library opened in 2003.", context="The library opened in 1998.")

CLI (from the project root):
    python -m groundedguard.inference.predictor --response "..." [--context_file f.txt]
        [--question "..."] [--structured_json f.json] [--model distilbert|modernbert]
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForTokenClassification, AutoTokenizer

from groundedguard.config import DOC_STRIDE, OUTPUTS_DIR, PROJECT_ROOT
from groundedguard.data.serialize import context_parts
from groundedguard.data.windowing import build_windows, encode_responses
from groundedguard.models.dataset import Collator
from groundedguard.models.train import model_kwargs, predict_probs, spans_at

INFERENCE_CONFIG = PROJECT_ROOT / "configs" / "inference.yaml"


def frozen_threshold(checkpoint_dir: Path) -> float:
    """The pre-registered threshold of this checkpoint, from outputs/module5*/frozen_choice.json."""
    rel = str(checkpoint_dir.resolve().relative_to(PROJECT_ROOT))
    found = [(f, r["threshold"]) for f in sorted(OUTPUTS_DIR.glob("module5*/frozen_choice.json"))
             for r in json.loads(f.read_text())["runs"] if r["checkpoint"] == rel]
    if len(found) != 1:
        raise ValueError(f"expected exactly one frozen threshold for {rel}, found {found}; pass threshold=")
    return found[0][1]


class _Encoded(torch.utils.data.Dataset):
    """The interface predict_probs expects (ids, n_windows, window_lens, items)."""

    def __init__(self, items: list[dict]):
        self.items = items
        self.ids = [it["id"] for it in items]
        self.n_windows = [len(it["input_ids"]) for it in items]
        self.window_lens = [[len(w) for w in it["input_ids"]] for it in items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


class GroundedGuardPredictor:
    def __init__(self, checkpoint_dir, threshold: float | None = None, device: str = "auto",
                 max_len: int | None = None, stride: int = DOC_STRIDE, precision: str = "auto"):
        """precision: "auto" = bf16 autocast on GPU (as in training/evaluation), fp32 on CPU;
        "fp32" on GPU gives batch-composition-independent outputs (bf16 rounding depends on
        the batch shape, so batch-1 and batched bf16 results can differ near the threshold)."""
        self.checkpoint_dir = (PROJECT_ROOT / checkpoint_dir).resolve()
        self.cfg = yaml.safe_load((self.checkpoint_dir / "config.yaml").read_text())
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if precision not in ("auto", "fp32", "bf16"):
            raise ValueError(f"precision must be auto, fp32 or bf16, got {precision!r}")
        if precision == "bf16" and self.device.type != "cuda":
            raise ValueError("bf16 is only used on GPU")
        self.bf16 = self.device.type == "cuda" if precision == "auto" else precision == "bf16"

        self.tok = AutoTokenizer.from_pretrained(self.checkpoint_dir)
        self.tok.model_max_length = 10**6  # windows are built explicitly; silences the length warning
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.checkpoint_dir, **model_kwargs(self.cfg)).to(self.device)
        self.model.eval()
        self.collate = Collator(self.tok.pad_token_id)

        self.max_len = max_len or self.model.config.max_position_embeddings
        self.stride = stride
        self.threshold = frozen_threshold(self.checkpoint_dir) if threshold is None else float(threshold)
        self.name = f"{self.cfg['model_name']} ({self.checkpoint_dir.name})"

    # ------------------------------------------------------------------------- inputs
    @staticmethod
    def _context(context=None, question=None, structured=None) -> tuple[str, str]:
        """(prefix, body) exactly as in training: question -> QA, structured -> Data2txt,
        plain context -> Summary-style body."""
        if structured is not None:
            if context is not None or question is not None:
                raise ValueError("structured cannot be combined with context or question")
            if not isinstance(structured, dict):
                raise ValueError("structured must be a dict")
            return context_parts("Data2txt", structured)
        if question is not None:
            if not isinstance(question, str):
                raise ValueError("question must be a string")
            return context_parts("QA", {"question": question, "passages": context or ""})
        if context is not None:
            if not isinstance(context, str):
                raise ValueError("context must be a string")
            return context_parts("Summary", context)
        return "", ""

    def _encode(self, inputs: list[dict]) -> tuple[list[dict], list[list[tuple[int, int]]]]:
        for i, x in enumerate(inputs):
            r = x.get("response")
            if not isinstance(r, str) or r == "":
                raise ValueError(f"input {i}: response must be a non-empty string")
        parts = [self._context(x.get("context"), x.get("question"), x.get("structured")) for x in inputs]
        pre = self.tok([p for p, _ in parts], add_special_tokens=False)["input_ids"]
        body = self.tok([b for _, b in parts], add_special_tokens=False)["input_ids"]
        encoded = encode_responses(self.tok, [x["response"] for x in inputs])

        items, offsets = [], []
        for i, ((r_ids, r_off), p_ids, c_ids) in enumerate(zip(encoded, pre, body)):
            wins = build_windows(p_ids, c_ids, r_ids, self.tok.cls_token_id, self.tok.sep_token_id,
                                 self.max_len, self.stride)
            if wins is None:
                raise ValueError(f"input {i}: response ({len(r_ids)} tokens) is too long for max_len {self.max_len}")
            items.append({"id": str(i), "input_ids": [w["input_ids"] for w in wins],
                          "response_start": [w["response_start"] for w in wins], "labels": [0] * len(r_ids)})
            offsets.append(r_off)
        return items, offsets

    # ------------------------------------------------------------------------- prediction
    def predict_batch(self, inputs: list[dict], return_token_probs: bool = False) -> list[dict]:
        """inputs: [{"response", "context"?, "question"?, "structured"?}]. latency_ms is the
        batch wall time divided by the number of inputs. return_token_probs adds the per-token
        probabilities and their char offsets (for debugging and the parity tests)."""
        t0 = time.perf_counter()
        items, offsets = self._encode(inputs)
        ds = _Encoded(items)
        with torch.inference_mode():
            probs = predict_probs(self.model, ds, self.collate, self.device, self.cfg["eval_max_windows_per_batch"],
                                  self.bf16, 0, self.cfg.get("aggregation", "min"), self.cfg.get("tau", 1.0),
                                  self.cfg.get("eval_max_tokens_per_batch"))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency = 1000 * (time.perf_counter() - t0) / len(inputs)

        out = []
        for x, it, off in zip(inputs, items, offsets):
            p = probs[it["id"]]
            spans = []
            for s, e in spans_at({"response_offsets": off}, p, self.threshold):
                inside = [float(q) for (a, b), q in zip(off, p) if a < b and a < e and s < b]
                spans.append({"start": s, "end": e, "text": x["response"][s:e], "score": max(inside)})
            n = len(it["input_ids"])
            out.append({"has_hallucination": bool(spans), "response_score": float(p.max()) if len(p) else 0.0,
                        "spans": spans, "threshold": self.threshold, "model": self.name,
                        "windowed": n > 1, "n_windows": n, "latency_ms": latency})
            if return_token_probs:
                out[-1].update({"token_offsets": [list(o) for o in off], "token_probs": [float(q) for q in p]})
        return out

    def predict(self, response: str, context: str | None = None, question: str | None = None,
                structured: dict | None = None) -> dict:
        return self.predict_batch([{"response": response, "context": context, "question": question,
                                    "structured": structured}])[0]


def ragtruth_input(row: dict, source: dict) -> dict:
    """predict() arguments for a RAGTruth response row and its RAW source_info record."""
    info = source["source_info"]
    if row["task_type"] == "QA":
        return {"response": row["response"], "question": info["question"], "context": info["passages"]}
    if row["task_type"] == "Summary":
        return {"response": row["response"], "context": info}
    return {"response": row["response"], "structured": info}


def mark_spans(response: str, spans: list[dict]) -> str:
    out, last = [], 0
    for sp in spans:
        out += [response[last:sp["start"]], "[[", response[sp["start"]:sp["end"]], "]]"]
        last = sp["end"]
    return "".join(out + [response[last:]])


def main() -> None:
    cfg = yaml.safe_load(INFERENCE_CONFIG.read_text())
    p = argparse.ArgumentParser(description="Flag unsupported spans in a response.")
    p.add_argument("--response", required=True)
    p.add_argument("--context_file", help="plain-text context (Summary-style), or the passages with --question")
    p.add_argument("--question", help="QA: the question; the context file holds the passages")
    p.add_argument("--structured_json", help="Data2txt: a JSON object")
    p.add_argument("--model", choices=sorted(cfg["models"]), default=cfg["default_model"])
    p.add_argument("--checkpoint", help="overrides --model")
    p.add_argument("--threshold", type=float, help="overrides the frozen threshold")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    context = Path(args.context_file).read_text(encoding="utf-8") if args.context_file else None
    structured = json.loads(Path(args.structured_json).read_text(encoding="utf-8")) if args.structured_json else None
    predictor = GroundedGuardPredictor(args.checkpoint or cfg["models"][args.model]["checkpoint"],
                                       threshold=args.threshold, device=args.device)
    result = predictor.predict(args.response, context=context, question=args.question, structured=structured)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n" + mark_spans(args.response, result["spans"]))


if __name__ == "__main__":
    main()
