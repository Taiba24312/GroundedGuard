"""Module 7 API tests (FastAPI TestClient).

Tests marked `slow` load the real ModernBERT checkpoint on the GPU in fp32 (the API default);
the others run without a model (TestClient without its context manager skips the lifespan).

    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_api.py -v
    python -m pytest tests/test_api.py -m "not slow"
"""

import json
import logging
import random

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings
from groundedguard.config import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR
from groundedguard.data.io import read_jsonl

DEMOS = json.loads((PROJECT_ROOT / "examples" / "demos.json").read_text())
PASSAGES = (PROJECT_ROOT / DEMOS["passages_file"]).read_text(encoding="utf-8")
MARKER = "ZQXJ-private-text-7781"


def api_body(row: dict, src: dict) -> dict:
    """/detect request for a RAGTruth row, from the RAW source_info."""
    info = src["source_info"]
    if row["task_type"] == "QA":
        return {"task": "qa", "question": info["question"], "passages": info["passages"], "response": row["response"]}
    if row["task_type"] == "Summary":
        return {"task": "summary", "document": info, "response": row["response"]}
    return {"task": "data2txt", "data": info, "response": row["response"]}


def predictor_args(body: dict) -> dict:
    if body["task"] == "qa":
        return {"response": body["response"], "question": body["question"], "context": body["passages"]}
    if body["task"] == "summary":
        return {"response": body["response"], "context": body["document"]}
    return {"response": body["response"], "structured": body["data"]}


def expected(predictor, body: dict) -> dict:
    o = predictor.predict(**predictor_args(body))
    return {"has_hallucination": o["has_hallucination"], "score": o["response_score"], "threshold": o["threshold"],
            "spans": o["spans"], "windowed": o["windowed"], "n_windows": o["n_windows"], "model": o["model"]}


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("latency_ms", "request_id")}


@pytest.fixture(scope="module")
def dev_bodies():
    sources = {str(s["source_id"]): s for s in read_jsonl(RAW_DIR / "source_info.jsonl")}
    rows = random.Random(7).sample(read_jsonl(PROCESSED_DIR / "dev.jsonl"), 50)
    return [api_body(r, sources[r["source_id"]]) for r in rows]


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(Settings(model="modernbert", precision="fp32", device="cuda"))) as c:
        assert c.app.state.predictor is not None, "model failed to load"
        yield c


def no_model_client(**limits) -> TestClient:
    return TestClient(create_app(Settings(device="cpu", **limits)))


# ----------------------------------------------------------------------------- no model needed
def test_health_and_detect_503_without_model():
    c = no_model_client()
    r = c.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "unavailable"
    r = c.post("/detect", json={"task": "summary", "document": "d", "response": "r"})
    assert r.status_code == 503 and r.json()["detail"] == "model not loaded"


@pytest.mark.parametrize("body, limits, needle", [
    ({"task": "summary", "response": "r"}, {}, "Field required"),                                   # missing field
    ({"task": "qa", "question": "q", "response": "r"}, {}, "Field required"),                       # missing passages
    ({"task": "chat", "response": "r"}, {}, "does not match any of the expected tags"),             # wrong task
    ({"document": "d", "response": "r"}, {}, "Unable to extract tag"),                              # no task
    ({"task": "summary", "document": "d", "response": ""}, {}, "at least 1 character"),            # empty response
    ({"task": "data2txt", "data": "not an object", "response": "r"}, {}, "valid dictionary"),       # data not an object
    ({"task": "summary", "document": "d", "response": "x" * 51}, {"max_response_chars": 50}, "the limit is 50"),
    ({"task": "summary", "document": "d" * 51, "response": "r"}, {"max_context_chars": 50}, "the limit is 50"),
])
def test_validation_422(body, limits, needle):
    r = no_model_client(**limits).post("/detect", json=body)
    assert r.status_code == 422, r.text
    assert needle in r.text and "Traceback" not in r.text and r.json()["request_id"]


@pytest.mark.parametrize("n", [0, 3])
def test_batch_size_422(n):
    body = [{"task": "summary", "document": "d", "response": "r"}] * n
    r = no_model_client(max_batch=2).post("/detect/batch", json=body)
    assert r.status_code == 422 and "GG_MAX_BATCH" in r.text


def test_internal_error_has_no_stack_trace():
    class Broken:
        def predict_batch(self, xs):
            raise RuntimeError(f"secret detail {MARKER}")

    c = no_model_client()
    c.app.state.predictor = Broken()
    r = c.post("/detect", json={"task": "summary", "document": "d", "response": "r"})
    assert r.status_code == 500 and r.json()["detail"] == "internal server error"
    assert "Traceback" not in r.text and MARKER not in r.text


# ----------------------------------------------------------------------------- real model (GPU)
@pytest.mark.slow
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    h = r.json()
    assert h["status"] == "ok" and h["precision"] == "fp32" and h["device"].startswith("cuda")
    assert h["threshold"] == 0.85 and "modernbert_lr5e-5_s42" in h["model"] and h["version"]


def test_docs_examples():
    """Every task has a /docs example, including a supported one; demo text is not duplicated wrongly."""
    from api.schemas import DEMO_PASSAGES, DEMO_QUESTION, EXAMPLES
    assert DEMO_PASSAGES == PASSAGES and DEMO_QUESTION == DEMOS["question"]
    assert EXAMPLES["qa_supported"]["value"]["response"] in [d["response"] for d in DEMOS["demos"]]
    spec = no_model_client().get("/openapi.json").json()
    examples = spec["paths"]["/detect"]["post"]["requestBody"]["content"]["application/json"]["examples"]
    assert {"qa", "qa_supported", "summary", "data2txt"} <= set(examples)


@pytest.mark.slow
def test_demos_equal_predictor(client):
    pred = client.app.state.predictor
    for d in DEMOS["demos"]:
        body = {"task": "qa", "question": DEMOS["question"], "passages": PASSAGES, "response": d["response"]}
        r = client.post("/detect", json=body)
        assert r.status_code == 200
        assert strip(r.json()) == expected(pred, body), d["name"]


@pytest.mark.slow
def test_dev_responses_equal_predictor(client, dev_bodies):
    pred = client.app.state.predictor
    same = 0
    for body in dev_bodies:
        r = client.post("/detect", json=body)
        assert r.status_code == 200
        same += strip(r.json()) == expected(pred, body)
    print(f"\n/detect == predictor.predict (fp32): {same}/{len(dev_bodies)} dev responses "
          f"({sorted({b['task'] for b in dev_bodies})})")
    assert same == len(dev_bodies)


@pytest.mark.slow
def test_batch_equals_individual_calls(client, dev_bodies):
    """fp32: batching changes probabilities by < 1e-4, so spans and flags must be identical."""
    bodies = dev_bodies[:16]
    r = client.post("/detect/batch", json=bodies)
    assert r.status_code == 200
    batch = r.json()["results"]
    single = [client.post("/detect", json=b).json() for b in bodies]
    max_diff = 0.0
    for b, s in zip(batch, single):
        assert [(x["start"], x["end"], x["text"]) for x in b["spans"]] == [(x["start"], x["end"], x["text"]) for x in s["spans"]]
        assert (b["has_hallucination"], b["n_windows"], b["windowed"]) == (s["has_hallucination"], s["n_windows"], s["windowed"])
        max_diff = max([max_diff, abs(b["score"] - s["score"])] + [abs(x["score"] - y["score"]) for x, y in zip(b["spans"], s["spans"])])
    print(f"\n/detect/batch vs /detect: spans identical for {len(bodies)}/{len(bodies)}, max score diff {max_diff:.2e}")
    assert max_diff < 1e-4


@pytest.mark.slow
def test_logs_never_contain_request_text(client, caplog):
    caplog.set_level(logging.DEBUG)
    ok = {"task": "qa", "question": f"q {MARKER}", "passages": f"p {MARKER}", "response": f"A {MARKER} answer."}
    assert client.post("/detect", json=ok).status_code == 200
    assert client.post("/detect/batch", json=[ok, {**ok, "task": "summary", "document": MARKER}]).status_code == 422
    assert client.post("/detect/batch", json=[ok, {"task": "summary", "document": MARKER, "response": MARKER}]).status_code == 200
    too_long = {"task": "summary", "document": MARKER, "response": MARKER * 1000}
    assert client.post("/detect", json=too_long).status_code == 422
    assert client.post("/detect", json={"task": "summary", "document": MARKER}).status_code == 422
    assert "request_id" in caplog.text and '"status": 200' in caplog.text and '"status": 422' in caplog.text
    assert MARKER not in caplog.text
