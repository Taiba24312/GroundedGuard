"""Module 7: light load test of a running GroundedGuard API.

Phase 1: 200 sequential /detect requests. Phase 2: 4 concurrent clients x 50 requests.
Bodies are dev responses (seeded sample, all three tasks), built from the raw source_info.
Latency is measured by the client (request sent -> response read); the server's own
latency_ms (model time) is reported alongside.

Start the server first, then, from the project root:
    python scripts/load_test.py [--url http://127.0.0.1:8000]

Writes:
    outputs/module7/load_test.json
"""

import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    import httpx2 as httpx  # the environment ships httpx2; same client API
except ImportError:
    import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groundedguard.config import OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR, SEED  # noqa: E402
from groundedguard.data.io import read_jsonl  # noqa: E402

N_SEQUENTIAL, N_CLIENTS, N_PER_CLIENT = 200, 4, 50


def body(row: dict, src: dict) -> dict:
    info = src["source_info"]
    if row["task_type"] == "QA":
        return {"task": "qa", "question": info["question"], "passages": info["passages"], "response": row["response"]}
    if row["task_type"] == "Summary":
        return {"task": "summary", "document": info, "response": row["response"]}
    return {"task": "data2txt", "data": info, "response": row["response"]}


def send(client, url: str, b: dict) -> dict:
    t0 = time.perf_counter()
    try:
        r = client.post(f"{url}/detect", json=b, timeout=60)
        ms = 1000 * (time.perf_counter() - t0)
        ok = r.status_code == 200
        return {"ok": ok, "status": r.status_code, "client_ms": ms, "server_ms": r.json()["latency_ms"] if ok else None}
    except Exception as exc:
        return {"ok": False, "status": type(exc).__name__, "client_ms": 1000 * (time.perf_counter() - t0), "server_ms": None}


def summarize(results: list[dict], wall_s: float) -> dict:
    lat = [r["client_ms"] for r in results if r["ok"]]
    server = [r["server_ms"] for r in results if r["ok"]]
    errors = [r["status"] for r in results if not r["ok"]]
    return {"requests": len(results), "errors": len(errors), "error_statuses": sorted(set(map(str, errors))),
            "wall_s": wall_s, "throughput_rps": len(results) / wall_s,
            "client_latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                                  "p99": float(np.percentile(lat, 99)), "max": max(lat)},
            "server_model_ms_median": statistics.median(server)}


def main() -> None:
    p = argparse.ArgumentParser(description="Light load test of the GroundedGuard API.")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    url = p.parse_args().url

    sources = {str(s["source_id"]): s for s in read_jsonl(RAW_DIR / "source_info.jsonl")}
    rows = random.Random(SEED).sample(read_jsonl(PROCESSED_DIR / "dev.jsonl"), N_SEQUENTIAL)
    bodies = [body(r, sources[r["source_id"]]) for r in rows]

    with httpx.Client() as c:
        health = c.get(f"{url}/health").json()
        assert health["status"] == "ok", health
        t0 = time.perf_counter()
        seq = [send(c, url, b) for b in bodies]
        seq_summary = summarize(seq, time.perf_counter() - t0)

    def client_run(k: int) -> list[dict]:
        with httpx.Client() as c:
            return [send(c, url, b) for b in bodies[k * N_PER_CLIENT:(k + 1) * N_PER_CLIENT]]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(N_CLIENTS) as pool:
        conc = [r for rs in pool.map(client_run, range(N_CLIENTS)) for r in rs]
    conc_summary = summarize(conc, time.perf_counter() - t0)

    report = {"url": url, "server": health, "bodies": f"{N_SEQUENTIAL} dev responses, random.Random({SEED})",
              "tasks": {t: sum(b["task"] == t for b in bodies) for t in ("qa", "summary", "data2txt")},
              "sequential": seq_summary, f"concurrent_{N_CLIENTS}x{N_PER_CLIENT}": conc_summary}
    out = OUTPUTS_DIR / "module7" / "load_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for name, s in (("sequential 200", seq_summary), (f"{N_CLIENTS} clients x {N_PER_CLIENT}", conc_summary)):
        l = s["client_latency_ms"]
        print(f"{name:16s}: p50 {l['p50']:.1f} ms | p95 {l['p95']:.1f} | p99 {l['p99']:.1f} | "
              f"throughput {s['throughput_rps']:.1f} req/s | errors {s['errors']} | server model ms (median) "
              f"{s['server_model_ms_median']:.1f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
