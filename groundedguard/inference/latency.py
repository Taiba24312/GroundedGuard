"""Module 6: end-to-end latency of GroundedGuardPredictor.predict (batch 1).

200 dev responses (seeded sample), both models, on GPU (bf16) and CPU (fp32, 4 threads).
Latency = predict()'s latency_ms: tokenization + windows + forward + aggregation + spans.
10 warm-up calls are not counted.

Run from the project root, with the GPU otherwise idle:
    CUDA_VISIBLE_DEVICES=0 python -m groundedguard.inference.latency

Writes:
    outputs/module6/latency.json
"""

import json
import random
import statistics

import numpy as np
import torch
import yaml

from groundedguard.config import OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR, SEED
from groundedguard.data.io import read_jsonl
from groundedguard.inference.predictor import INFERENCE_CONFIG, GroundedGuardPredictor, ragtruth_input

N, WARMUP, CPU_THREADS = 200, 10, 4


def main() -> None:
    torch.set_num_threads(CPU_THREADS)
    models = yaml.safe_load(INFERENCE_CONFIG.read_text())["models"]
    sources = {str(s["source_id"]): s for s in read_jsonl(RAW_DIR / "source_info.jsonl")}
    rows = random.Random(SEED).sample(read_jsonl(PROCESSED_DIR / "dev.jsonl"), N)
    xs = [ragtruth_input(r, sources[r["source_id"]]) for r in rows]

    results = []
    for name, m in models.items():
        for device in ("cuda", "cpu"):
            pred = GroundedGuardPredictor(m["checkpoint"], device=device)
            for x in xs[:WARMUP]:
                pred.predict(**x)
            outs = [pred.predict(**x) for x in xs]
            lat = [o["latency_ms"] for o in outs]
            results.append({"model": name, "checkpoint": m["checkpoint"], "device": device,
                            "precision": "bf16" if pred.bf16 else "fp32",
                            "threads": torch.get_num_threads() if device == "cpu" else None,
                            "n": len(lat), "median_ms": statistics.median(lat),
                            "p95_ms": float(np.percentile(lat, 95)), "mean_n_windows": statistics.mean(o["n_windows"] for o in outs)})
            r = results[-1]
            print(f"{name:10s} {device:4s} {r['precision']}: median {r['median_ms']:.1f} ms | p95 {r['p95_ms']:.1f} ms "
                  f"| mean windows {r['mean_n_windows']:.2f}", flush=True)
            del pred
            torch.cuda.empty_cache()

    out = OUTPUTS_DIR / "module6" / "latency.json"
    out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "cpu_threads": CPU_THREADS, "batch": 1,
                               "n_responses": N, "warmup": WARMUP, "sample": f"dev, random.Random({SEED})",
                               "results": results}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
