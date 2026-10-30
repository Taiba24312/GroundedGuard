"""Module 3b: non-learned baselines, scored with the official metric.

  B0  predict nothing
  B1  the whole response is one span
  B2  lexical overlap: a response word (lowercased, alphanumeric, not a stopword) that does
      not appear anywhere in the context is flagged; runs of consecutive flagged words become
      spans (a stopword or a supported word ends a run). Only knob: minimum span length in
      words, tuned on DEV by All span-F1, then applied once to test.

Run from the project root:
    python -m groundedguard.baselines.heuristic

Writes:
    outputs/module3/baselines.json
"""

import json
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from groundedguard.config import OUTPUTS_DIR, PROCESSED_DIR
from groundedguard.data.io import read_jsonl
from groundedguard.data.serialize import join_context
from groundedguard.eval.metrics import evaluate, format_table

OUT_DIR = OUTPUTS_DIR / "module3"
WORD = re.compile(r"[^\W_]+")  # unicode letters/digits
MIN_WORDS_CANDIDATES = (1, 2, 3)


def predict_nothing(row) -> list[list[int]]:
    return []


def predict_everything(row) -> list[list[int]]:
    return [[0, len(row["response"])]] if row["response"] else []


def lexical_overlap(row, context_vocab: set[str], min_words: int) -> list[list[int]]:
    spans, run = [], []

    def flush():
        if len(run) >= min_words:
            spans.append([run[0].start(), run[-1].end()])
        run.clear()

    for m in WORD.finditer(row["response"]):
        w = m.group().lower()
        if w in ENGLISH_STOP_WORDS or w in context_vocab:
            flush()
        else:
            run.append(m)
    flush()
    return spans


def score(rows, predict) -> dict:
    return evaluate([{"id": r["id"], "task_type": r["task_type"], "response": r["response"],
                      "gold_spans": r["hallucination_spans"], "pred_spans": predict(r)} for r in rows])


def main() -> None:
    splits = {s: read_jsonl(PROCESSED_DIR / f"{s}.jsonl") for s in ("dev", "test")}
    vocab = {}  # source_id -> context words, shared by the 6 responses of a source
    for rows in splits.values():
        for r in rows:
            if r["source_id"] not in vocab:
                ctx = join_context(r["context_prefix"], r["context_body"])
                vocab[r["source_id"]] = {w.lower() for w in WORD.findall(ctx)}

    def b2(k):
        return lambda r: lexical_overlap(r, vocab[r["source_id"]], k)

    # Tune B2 on dev only.
    tuning = {k: score(splits["dev"], b2(k)) for k in MIN_WORDS_CANDIDATES}
    best_k = max(MIN_WORDS_CANDIDATES, key=lambda k: tuning[k]["All"]["span"]["f1"])

    results = {"dev": {"B0 nothing": score(splits["dev"], predict_nothing),
                       "B1 whole response": score(splits["dev"], predict_everything),
                       **{f"B2 lexical k={k}": tuning[k] for k in MIN_WORDS_CANDIDATES}},
               "test": {"B0 nothing": score(splits["test"], predict_nothing),
                        "B1 whole response": score(splits["test"], predict_everything),
                        f"B2 lexical k={best_k}": score(splits["test"], b2(best_k))}}

    report = {
        "config": {"stopwords": f"sklearn ENGLISH_STOP_WORDS ({len(ENGLISH_STOP_WORDS)})",
                   "word_regex": WORD.pattern, "b2_min_words_candidates": list(MIN_WORDS_CANDIDATES),
                   "b2_tuning": "dev, All span F1", "b2_chosen_min_words": best_k},
        "b2_dev_tuning_all_span_f1": {k: tuning[k]["All"]["span"]["f1"] for k in MIN_WORDS_CANDIDATES},
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "baselines.json").write_text(json.dumps(report, indent=2))

    table = {f"{split:4s} {name}": res for split, by_name in results.items() for name, res in by_name.items()}
    print("=" * 98 + "\nModule 3: baselines (official RAGTruth metric)\n" + "=" * 98)
    print("B2 dev tuning (All span F1): " + ", ".join(
        f"k={k}: {f1:.4f}" for k, f1 in report["b2_dev_tuning_all_span_f1"].items()) + f"  -> chosen k={best_k}\n")
    print(format_table(table))
    print(f"\nsaved -> {OUT_DIR}/baselines.json")


if __name__ == "__main__":
    main()
