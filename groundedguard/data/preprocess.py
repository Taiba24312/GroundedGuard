"""Module 2a: clean, join, split and serialize RAGTruth.

What it does
  1. keep responses with quality == "good"  (same as the official baseline)
  2. join each response with its source and build the context string, stored as
     `context_prefix` (QA question, repeated in every window) + `context_body`
  3. merge overlapping label spans into a union  -> `hallucination_spans`
  4. dev split = 50 train sources per task, random.seed(2024)  (official recipe)

Run from the project root:
    python -m groundedguard.data.preprocess

Writes:
    data/processed/{train,dev,test}.jsonl
    outputs/module2/preprocess_report.json
"""

import json
import random
from collections import Counter

from groundedguard.config import (
    DEV_SEED,
    DEV_SOURCES_PER_TASK,
    DEV_TASK_ORDER,
    KEEP_QUALITY,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
)
from groundedguard.data.io import read_jsonl, write_jsonl
from groundedguard.data.serialize import context_parts, join_context

OUT_DIR = OUTPUTS_DIR / "module2"


def merge_spans(labels: list[dict]) -> list[list[int]]:
    """Union of [start, end) char intervals; overlapping or touching spans are merged."""
    intervals = sorted((int(l["start"]), int(l["end"])) for l in labels)
    merged: list[list[int]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def official_dev_source_ids(train_rows: list[dict]) -> set[str]:
    """Reproduce the official sampling: one global seed, tasks in fixed order,
    source ids in order of first appearance among (good-quality) train responses."""
    rng = random.Random(DEV_SEED)
    dev = set()
    for task in DEV_TASK_ORDER:
        ids = list(dict.fromkeys(r["source_id"] for r in train_rows if r["task_type"] == task))
        if len(ids) < DEV_SOURCES_PER_TASK:
            raise ValueError(f"only {len(ids)} {task} train sources, need {DEV_SOURCES_PER_TASK}")
        dev.update(rng.sample(ids, DEV_SOURCES_PER_TASK))
    return dev


def build_row(r: dict, src: dict) -> dict:
    labels = r.get("labels") or []
    prefix, body = context_parts(src["task_type"], src["source_info"])
    return {
        "id": str(r["id"]),
        "source_id": str(r["source_id"]),
        "split": r["split"],
        "task_type": src["task_type"],
        "source": src.get("source"),
        "model": r.get("model"),
        "quality": r.get("quality"),
        "context_prefix": prefix,
        "context_body": body,
        "response": r["response"],
        "hallucination_spans": merge_spans(labels),   # training / metric target
        "labels": labels,                             # raw annotations kept for error analysis
    }


def main() -> None:
    responses = read_jsonl(RAW_DIR / "response.jsonl")
    sources = {str(s["source_id"]): s for s in read_jsonl(RAW_DIR / "source_info.jsonl")}

    dropped = Counter((r["split"], r.get("quality")) for r in responses if r.get("quality") not in KEEP_QUALITY)
    dropped_with_labels = sum(1 for r in responses if r.get("quality") not in KEEP_QUALITY and r.get("labels"))
    kept = [r for r in responses if r.get("quality") in KEEP_QUALITY]

    rows = [build_row(r, sources[str(r["source_id"])]) for r in kept]

    # Sanity: merged spans must stay inside the response and cover every original span char.
    for row in rows:
        n = len(row["response"])
        gold = {i for l in row["labels"] for i in range(l["start"], l["end"])}
        merged = {i for s, e in row["hallucination_spans"] for i in range(s, e)}
        assert gold == merged, f"span merge changed coverage for id={row['id']}"
        assert all(0 <= s < e <= n for s, e in row["hallucination_spans"]), f"bad span id={row['id']}"

    train_all = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    dev_ids = official_dev_source_ids(train_all)
    train = [r for r in train_all if r["source_id"] not in dev_ids]
    dev = [r for r in train_all if r["source_id"] in dev_ids]
    for r in dev:
        r["split"] = "dev"

    # Leakage checks between our three splits.
    s_train, s_dev, s_test = ({r["source_id"] for r in x} for x in (train, dev, test))
    assert not (s_train & s_dev) and not (s_train & s_test) and not (s_dev & s_test), "source leakage!"

    for name, data in (("train", train), ("dev", dev), ("test", test)):
        write_jsonl(data, PROCESSED_DIR / f"{name}.jsonl")

    def summary(data):
        by_task = Counter(r["task_type"] for r in data)
        hal = Counter(r["task_type"] for r in data if r["hallucination_spans"])
        return {
            "responses": len(data),
            "sources": len({r["source_id"] for r in data}),
            "hallucinated_%": round(100 * sum(hal.values()) / max(len(data), 1), 2),
            "by_task": {t: {"responses": by_task[t], "hallucinated_%": round(100 * hal[t] / by_task[t], 2)}
                        for t in sorted(by_task)},
        }

    n_raw_spans = sum(len(r["labels"]) for r in rows)
    n_merged = sum(len(r["hallucination_spans"]) for r in rows)
    report = {
        "dropped_low_quality": {f"{s}|{q}": c for (s, q), c in sorted(dropped.items())},
        "dropped_low_quality_with_labels": dropped_with_labels,
        "spans_raw": n_raw_spans,
        "spans_after_merge": n_merged,
        "splits": {"train": summary(train), "dev": summary(dev), "test": summary(test)},
        "dev_source_ids_sample": sorted(dev_ids)[:10],
        "context_chars_by_task": {
            t: round(sum(len(join_context(r["context_prefix"], r["context_body"])) for r in rows if r["task_type"] == t)
                     / max(1, sum(1 for r in rows if r["task_type"] == t)), 1)
            for t in DEV_TASK_ORDER
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "preprocess_report.json").write_text(json.dumps(report, indent=2))

    # One serialized context per task, so you can eyeball the format.
    examples = {}
    for r in rows:
        if r["task_type"] not in examples:
            examples[r["task_type"]] = join_context(r["context_prefix"], r["context_body"])[:1200]
    (OUT_DIR / "context_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False))

    print("=" * 70 + "\nModule 2a: preprocessing\n" + "=" * 70)
    print(f"dropped (low quality): {report['dropped_low_quality']}  (with labels: {dropped_with_labels})")
    print(f"spans: raw={n_raw_spans}  after merge={n_merged}")
    for name in ("train", "dev", "test"):
        s = report["splits"][name]
        print(f"{name:5s}: {s['responses']:>6,} responses | {s['sources']:>5,} sources | "
              f"hallucinated {s['hallucinated_%']}% | {s['by_task']}")
    print(f"avg context chars by task: {report['context_chars_by_task']}")
    print(f"saved -> {PROCESSED_DIR}/{{train,dev,test}}.jsonl, {OUT_DIR}/preprocess_report.json, context_examples.json")


if __name__ == "__main__":
    main()