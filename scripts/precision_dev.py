"""Module 7, step 0: fp32 vs bf16 for the served ModernBERT checkpoint. DEV ONLY.

For each precision: dev All span F1 (batched predict_batch, frozen threshold), the number of
dev responses whose spans differ between the two, and GPU latency of predict() at batch 1
(median/p95 over the same 200 dev responses as Module 6, after 10 warm-up calls).
Also span-granularity statistics of the predictions vs gold (dev), for the model card.

Decision rule: fp32 is the API default unless its median GPU latency is above 100 ms.

Run from the project root, GPU otherwise idle:
    CUDA_VISIBLE_DEVICES=0 python scripts/precision_dev.py

Writes:
    outputs/module7/precision_dev.json
"""

import json
import random
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groundedguard.config import OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR, SEED  # noqa: E402
from groundedguard.data.io import read_jsonl  # noqa: E402
from groundedguard.eval.metrics import evaluate  # noqa: E402
from groundedguard.inference.predictor import GroundedGuardPredictor, ragtruth_input  # noqa: E402

CHECKPOINT = "checkpoints/modernbert_lr5e-5_s42"
N_LATENCY, WARMUP, MAX_MEDIAN_MS = 200, 10, 100.0
SENT_END = re.compile(r"[.!?][\"')\]]*$")


def granularity(rows: list[dict], spans_by_id: dict) -> dict:
    """Length of spans and how often they start/end on a sentence boundary."""
    lens, ends, starts = [], [], []
    for r in rows:
        text = r["response"]
        for s, e in spans_by_id[r["id"]]:
            lens.append(e - s)
            ends.append(bool(SENT_END.search(text[s:e].rstrip())))
            before = text[:s].rstrip()
            starts.append(before == "" or before[-1] in ".!?\n:" or text[s - 1:s] == "\n")
    return {"n_spans": len(lens), "mean_chars": statistics.mean(lens), "median_chars": statistics.median(lens),
            "ends_at_sentence_end_%": 100 * sum(ends) / len(ends),
            "starts_at_sentence_start_%": 100 * sum(starts) / len(starts)}


def main() -> None:
    rows = read_jsonl(PROCESSED_DIR / "dev.jsonl")
    sources = {str(s["source_id"]): s for s in read_jsonl(RAW_DIR / "source_info.jsonl")}
    xs = [ragtruth_input(r, sources[r["source_id"]]) for r in rows]
    timed = random.Random(SEED).sample(rows, N_LATENCY)  # same sample as groundedguard.inference.latency
    timed_xs = [ragtruth_input(r, sources[r["source_id"]]) for r in timed]

    result, spans = {}, {}
    for precision in ("bf16", "fp32"):
        pred = GroundedGuardPredictor(CHECKPOINT, device="cuda", precision=precision)
        outs = pred.predict_batch(xs)
        spans[precision] = {r["id"]: [[s["start"], s["end"]] for s in o["spans"]] for r, o in zip(rows, outs)}
        m = evaluate([{"id": r["id"], "task_type": r["task_type"], "response": r["response"],
                       "gold_spans": r["hallucination_spans"], "pred_spans": spans[precision][r["id"]]} for r in rows])
        for x in timed_xs[:WARMUP]:
            pred.predict(**x)
        lat = [pred.predict(**x)["latency_ms"] for x in timed_xs]
        result[precision] = {
            "threshold": pred.threshold,
            "dev_all_span": {k: m["All"]["span"][k] for k in ("precision", "recall", "f1")},
            "dev_all_case_f1": m["All"]["case"]["f1"],
            "dev_span_f1_by_task": {t: m[t]["span"]["f1"] for t in m if t != "All"},
            "gpu_latency_ms_batch1": {"median": statistics.median(lat), "p95": float(np.percentile(lat, 95))},
        }
        print(f"{precision}: dev All span F1 {m['All']['span']['f1']:.4f} | case F1 {m['All']['case']['f1']:.4f} | "
              f"GPU batch-1 median {statistics.median(lat):.1f} ms p95 {np.percentile(lat, 95):.1f} ms", flush=True)
        del pred

    differ = [r["id"] for r in rows if spans["fp32"][r["id"]] != spans["bf16"][r["id"]]]
    gold = {r["id"]: r["hallucination_spans"] for r in rows}
    fp32_median = result["fp32"]["gpu_latency_ms_batch1"]["median"]
    report = {
        "checkpoint": CHECKPOINT, "split": "dev", "n_responses": len(rows),
        "note": "threshold frozen on dev in bf16; fp32 uses the same threshold",
        "bf16_batched": result["bf16"], "fp32_batched": result["fp32"],
        "delta_fp32_minus_bf16_all_span_f1": result["fp32"]["dev_all_span"]["f1"] - result["bf16"]["dev_all_span"]["f1"],
        "responses_with_different_spans": len(differ), "differing_ids": differ,
        "latency_sample": f"{N_LATENCY} dev responses, random.Random({SEED}), {WARMUP} warm-up calls",
        "decision": {"rule": f"fp32 default unless fp32 median GPU latency > {MAX_MEDIAN_MS} ms",
                     "fp32_median_ms": fp32_median, "api_default": "fp32" if fp32_median <= MAX_MEDIAN_MS else "STOP"},
        "span_granularity_dev": {"gold": granularity(rows, gold), "fp32_predicted": granularity(rows, spans["fp32"])},
    }
    out = OUTPUTS_DIR / "module7" / "precision_dev.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"responses with different spans (fp32 vs bf16): {len(differ)}/{len(rows)}")
    print(f"decision: {report['decision']['api_default']}  -> {out}")


if __name__ == "__main__":
    main()
