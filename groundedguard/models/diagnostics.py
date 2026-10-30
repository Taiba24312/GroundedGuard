"""Module 4b: where does a trained checkpoint fail? Dev only, no training.

  - span P/R/F1 by number of context windows per response (overall and Summary)
  - char recall for Evident vs Subtle gold spans (and the 4 fine label types), using
    the raw `labels` to decide which gold chars belong to which type. A char covered by
    labels of two types counts for both.

The checkpoint is scored exactly as in training: its own windows file, aggregation and
best dev threshold.

Run from the project root:
    python -m groundedguard.models.diagnostics --run distilbert_minpool_v1

Writes:
    outputs/module4/diagnostics.json   (or --out)
"""

import argparse
import json
from collections import defaultdict

import torch
import yaml
from transformers import AutoModelForTokenClassification, AutoTokenizer

from groundedguard.config import CHECKPOINTS_DIR, OUTPUTS_DIR, PROCESSED_DIR
from groundedguard.data.io import read_jsonl
from groundedguard.data.windowing import tokens_to_spans
from groundedguard.eval.metrics import evaluate, prf
from groundedguard.models.dataset import Collator, GroupedWindowDataset
from groundedguard.models.train import predict_probs

BUCKETS = (("1", 1, 1), ("2-3", 2, 3), ("4-6", 4, 6), ("7-10", 7, 10), (">10", 11, 10**9))


def span_row(res: dict, key: str) -> dict:
    if key not in res:
        return {"n": 0}
    s = res[key]["span"]
    return {"n": res[key]["n"], "precision": s["precision"], "recall": s["recall"], "f1": s["f1"],
            "gold_chars": s["gold"], "pred_chars": s["pred"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev diagnostics for a trained checkpoint.")
    parser.add_argument("--run", required=True, help="checkpoints/<run>")
    parser.add_argument("--out", default=str(OUTPUTS_DIR / "module4" / "diagnostics.json"))
    args = parser.parse_args()

    run_dir = CHECKPOINTS_DIR / args.run
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    threshold = json.loads((run_dir / "best_threshold.json").read_text())["threshold"]
    aggregation, tau = cfg.get("aggregation", "min"), cfg.get("tau", 1.0)
    sfx = cfg.get("windows_suffix") or ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = bool(cfg["bf16"]) and device.type == "cuda"
    ds = GroupedWindowDataset(PROCESSED_DIR / f"dev_windows{sfx}.pkl")
    tok = AutoTokenizer.from_pretrained(run_dir)
    model = AutoModelForTokenClassification.from_pretrained(run_dir).to(device)
    probs = predict_probs(model, ds, Collator(tok.pad_token_id), device, cfg["eval_max_windows_per_batch"],
                          bf16, 2, aggregation, tau)

    examples = []
    for rid, m in ds.responses.items():
        pred = tokens_to_spans(m["response_offsets"], (probs[rid] >= threshold).astype(int).tolist())
        examples.append({"id": rid, "task_type": m["task_type"], "response": m["response"],
                         "gold_spans": m["hallucination_spans"], "pred_spans": pred, "n_windows": m["n_windows"]})

    # 1. by number of windows
    buckets = {}
    for name, lo, hi in BUCKETS:
        res = evaluate([e for e in examples if lo <= e["n_windows"] <= hi])
        buckets[name] = {"overall": span_row(res, "All"), "summary": span_row(res, "Summary")}

    # 2. recall by label type (chars), from the raw annotations
    raw = {r["id"]: r["labels"] for r in read_jsonl(PROCESSED_DIR / "dev.jsonl")}
    hit, tot = defaultdict(int), defaultdict(int)
    for e in examples:
        pred = {i for s, t in e["pred_spans"] for i in range(s, t)}
        chars = defaultdict(set)
        for lab in raw[e["id"]]:
            lt = lab["label_type"]
            span = range(lab["start"], lab["end"])
            chars[lt].update(span)
            chars[lt.split()[0]].update(span)  # "Evident" / "Subtle"
        for key, cs in chars.items():
            for k in (key, f"{e['task_type']}|{key}"):
                tot[k] += len(cs)
                hit[k] += len(cs & pred)
    label_recall = {k: {"gold_chars": tot[k], "recalled_chars": hit[k],
                        "recall": prf(hit[k], 0, tot[k])["recall"]} for k in sorted(tot)}

    report = {"run": args.run, "threshold": threshold, "aggregation": aggregation, "tau": tau,
              "windows_file": f"dev_windows{sfx}.pkl", "overall": evaluate(examples)["All"]["span"],
              "by_n_windows": buckets, "recall_by_label_type": label_recall}
    out = args.out
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"run {args.run} | dev | threshold {threshold} | aggregation {aggregation}")
    print(f"\n{'n_windows':>9s} | {'overall: n':>10s} {'P':>7s} {'R':>7s} {'F1':>7s} | "
          f"{'Summary: n':>10s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    for name, b in buckets.items():
        cells = []
        for part in ("overall", "summary"):
            r = b[part]
            cells.append(f"{r['n']:>10d} " + (f"{r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f}"
                                               if r["n"] else f"{'-':>7s} {'-':>7s} {'-':>7s}"))
        print(f"{name:>9s} | {cells[0]} | {cells[1]}")
    print(f"\n{'label type':28s} {'gold chars':>10s} {'recall':>7s}")
    for k, v in label_recall.items():
        if "|" not in k:
            print(f"{k:28s} {v['gold_chars']:>10d} {v['recall']:7.4f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
