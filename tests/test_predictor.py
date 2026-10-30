"""Module 6 tests: predictor parity with the training/evaluation code, plus unit tests.

Parity is a REPRODUCTION check, not an evaluation: the predictor's spans must equal
  - test: the saved predictions (outputs/module5b/preds_seed42.jsonl, outputs/module5/preds_seed42.jsonl)
  - dev:  the training code path (models.train.predict_probs on the stored dev windows + spans_at
          at the frozen threshold), regenerated here because training did not save dev predictions.
No metric is computed. Results (exact-match %, every mismatch with its reason) go to
outputs/module6/parity.json.

Run from the project root (GPU):
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_predictor.py -v -s
"""

import json

import numpy as np
import pytest
import torch

from groundedguard.config import OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR
from groundedguard.data.io import read_jsonl
from groundedguard.eval.freeze_choice import run_probs
from groundedguard.inference.predictor import GroundedGuardPredictor, mark_spans, ragtruth_input
from groundedguard.models.dataset import GroupedWindowDataset
from groundedguard.models.train import spans_at

MODELS = {
    "modernbert": {"checkpoint": "checkpoints/modernbert_lr5e-5_s42", "test_preds": "module5b/preds_seed42.jsonl",
                   "windows_suffix": "_modernbert"},
    "distilbert": {"checkpoint": "checkpoints/B_min_full_e8", "test_preds": "module5/preds_seed42.jsonl",
                   "windows_suffix": ""},
}
NEAR_THRESHOLD = 0.01  # batch mode: a flip this close to the threshold would be attributed to numerics
PARITY: list[dict] = []


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session", autouse=True)
def write_parity_report():
    yield
    if PARITY:
        out = OUTPUTS_DIR / "module6" / "parity.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(PARITY, indent=2, ensure_ascii=False))
        print(f"\nparity report -> {out}")
        for r in PARITY:
            print(f"  {r['model']:10s} {r['split']:4s} {r['mode']:6s} exact {r['exact']}/{r['n']} "
                  f"({r['exact_match_%']:.2f}%) | input_ids identical to stored windows: {r['input_ids_identical']} "
                  f"| max |Δprob| {r['max_abs_prob_diff']:.2e} | mismatches {len(r['mismatches'])}")
            for m in r["mismatches"]:
                print(f"     id {m['id']}: {m['reason']}")


@pytest.fixture(scope="session")
def predictors():
    assert torch.cuda.is_available(), "parity is checked on GPU (bf16), as the saved predictions were"
    return {name: GroundedGuardPredictor(m["checkpoint"], device="cuda") for name, m in MODELS.items()}


@pytest.fixture(scope="session")
def sources():
    return {str(s["source_id"]): s for s in read_jsonl(RAW_DIR / "source_info.jsonl")}


def reference(name: str, split: str, predictor: GroundedGuardPredictor) -> dict:
    """{id: (spans, token_probs)} from the saved test predictions or the training code path (dev)."""
    if split == "test":
        return {p["id"]: (p["pred_spans"], np.array(p["token_probs"]))
                for p in read_jsonl(OUTPUTS_DIR / MODELS[name]["test_preds"])}
    ds = GroupedWindowDataset(PROCESSED_DIR / f"dev_windows{MODELS[name]['windows_suffix']}.pkl")
    probs = run_probs(predictor.cfg, predictor.tok, predictor.model, ds, predictor.device)
    return {rid: (spans_at(m, probs[rid], predictor.threshold), probs[rid]) for rid, m in ds.responses.items()}


# ----------------------------------------------------------------------------- parity
@pytest.mark.parametrize("mode", ["batch", "single"])
@pytest.mark.parametrize("split", ["dev", "test"])
@pytest.mark.parametrize("name", list(MODELS))
def test_parity(name, split, mode, predictors, sources):
    pred = predictors[name]
    rows = read_jsonl(PROCESSED_DIR / f"{split}.jsonl")
    inputs = [ragtruth_input(r, sources[r["source_id"]]) for r in rows]
    ref = reference(name, split, pred)

    # The predictor must build exactly the stored windows.
    stored = GroupedWindowDataset(PROCESSED_DIR / f"{split}_windows{MODELS[name]['windows_suffix']}.pkl")
    items, _ = pred._encode(inputs)
    by_id = dict(zip(stored.ids, stored.windows))
    ids_identical = all([w["input_ids"] for w in by_id[r["id"]]] == it["input_ids"] for r, it in zip(rows, items))
    assert pred.max_len == stored.config["max_len"] and pred.stride == stored.config["stride"]

    if mode == "batch":
        outs = pred.predict_batch(inputs, return_token_probs=True)
    else:
        outs = [pred.predict_batch([x], return_token_probs=True)[0] for x in inputs]

    exact, mismatches, max_diff = 0, [], 0.0
    for r, o in zip(rows, outs):
        ref_spans, ref_p = ref[r["id"]]
        p = np.array(o["token_probs"])
        max_diff = max(max_diff, float(np.abs(p - ref_p).max()) if len(p) else 0.0)
        got = [[s["start"], s["end"]] for s in o["spans"]]
        if got == [list(s) for s in ref_spans]:
            exact += 1
            continue
        flipped = np.nonzero((p >= o["threshold"]) != (ref_p >= o["threshold"]))[0]
        gap = float(np.abs(ref_p[flipped] - o["threshold"]).max()) if len(flipped) else float("nan")
        # A span difference must come from token decisions that flipped. In batch mode the batches
        # are the saved run's, so no flip is expected; batch-1 bf16 differs in rounding (see
        # test_fp32_padding_invariance for the proof that padding itself changes nothing).
        explained = len(flipped) > 0 and (mode == "single" or gap <= NEAR_THRESHOLD)
        reason = (f"bf16 rounding, batch-1 vs batched: {len(flipped)} token decision(s) flipped, reference "
                  f"probs within {gap:.4f} of threshold {o['threshold']}" if explained
                  else f"UNEXPLAINED: {len(flipped)} flipped tokens, max gap {gap}")
        mismatches.append({"id": r["id"], "task_type": r["task_type"], "predictor_spans": got,
                           "reference_spans": ref_spans, "flipped_tokens": len(flipped), "reason": reason,
                           "explained": explained})

    PARITY.append({"model": name, "split": split, "mode": mode, "n": len(rows), "exact": exact,
                   "exact_match_%": 100 * exact / len(rows), "input_ids_identical": ids_identical,
                   "max_abs_prob_diff": max_diff, "mismatches": mismatches})
    assert ids_identical, "predictor windows differ from the stored training windows"
    assert all(m["explained"] for m in mismatches), "unexplained mismatches"
    if mode == "batch":  # same batches as the saved run -> must be identical
        assert exact == len(rows)


@pytest.mark.parametrize("name", list(MODELS))
def test_fp32_padding_invariance(name, sources):
    """In fp32, a response scored alone and inside padded batches gets the same probabilities:
    the single-mode differences above are bf16 rounding, not a masking/padding error.
    Uses the dev responses that differed in single mode (up to 40), else the first 40."""
    rows = {r["id"]: r for r in read_jsonl(PROCESSED_DIR / "dev.jsonl")}
    bad = [m["id"] for r in PARITY if (r["model"], r["split"], r["mode"]) == (name, "dev", "single")
           for m in r["mismatches"]][:40]
    ids = bad or list(rows)[:40]
    xs = [ragtruth_input(rows[i], sources[rows[i]["source_id"]]) for i in ids]
    pred = GroundedGuardPredictor(MODELS[name]["checkpoint"], device="cuda", precision="fp32")
    batched = pred.predict_batch(xs, return_token_probs=True)
    alone = [pred.predict_batch([x], return_token_probs=True)[0] for x in xs]
    diff = max(float(np.abs(np.array(a["token_probs"]) - np.array(b["token_probs"])).max()) for a, b in zip(batched, alone))
    offsets = lambda o: [(s["start"], s["end"]) for s in o["spans"]]
    same = sum(offsets(a) == offsets(b) for a, b in zip(batched, alone))
    print(f"\nfp32 {name}: {len(ids)} dev responses ({'single-mode bf16 mismatches' if bad else 'first 40'}): "
          f"spans identical alone vs batched {same}/{len(ids)}, max |Δprob| {diff:.2e}")
    assert same == len(ids) and diff < 1e-4


# ----------------------------------------------------------------------------- unit tests
ROUND_TRIP = [
    "  Leading and trailing spaces stay.  ",
    "Line one.\n\nLine two with a tab\tinside.",
    "Accents: café, naïve, Zürich; dashes – and — quotes “like this”.",
    "Emoji 🙂 and CJK 東京 and numbers 1,234.56%.",
    "x",
]


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("response", ROUND_TRIP)
def test_offsets_round_trip(name, response, predictors):
    pred = predictors[name]
    saved = pred.threshold
    pred.threshold = 0.0  # every token flagged -> one span from the first to the last non-space char
    try:
        out = pred.predict(response, context="Some context.")
    finally:
        pred.threshold = saved
    for sp in out["spans"]:
        assert 0 <= sp["start"] < sp["end"] <= len(response)
        assert response[sp["start"]:sp["end"]] == sp["text"]
    # One span; it starts at the first non-space char and covers every non-space char. Its end
    # may include trailing whitespace when the tokenizer has a whitespace-only token there
    # (byte-level BPE); the evaluated span logic (windowing.tokens_to_spans) behaves the same.
    assert len(out["spans"]) == 1
    (sp,) = out["spans"]
    assert sp["start"] == len(response) - len(response.lstrip())
    assert sp["end"] >= len(response.rstrip())
    assert mark_spans(response, out["spans"]).replace("[[", "").replace("]]", "") == response


@pytest.mark.parametrize("name", list(MODELS))
def test_empty_context(name, predictors):
    out = predictors[name].predict("Paris is the capital of France.")
    assert out["n_windows"] == 1 and out["windowed"] is False
    assert isinstance(out["has_hallucination"], bool) and 0.0 <= out["response_score"] <= 1.0
    for sp in out["spans"]:
        assert "Paris is the capital of France."[sp["start"]:sp["end"]] == sp["text"]


def test_windowed_fallback(predictors, sources):
    """ModernBERT forced to max_len=512 on the dev response with the longest context."""
    pred = predictors["modernbert"]
    rows = read_jsonl(PROCESSED_DIR / "dev.jsonl")
    row = max(rows, key=lambda r: len(r["context_prefix"]) + len(r["context_body"]))
    x = ragtruth_input(row, sources[row["source_id"]])
    full = pred.predict(**x)
    saved = pred.max_len
    pred.max_len = 512
    try:
        out = pred.predict(**x)
    finally:
        pred.max_len = saved
    assert full["windowed"] is False and full["n_windows"] == 1
    assert out["windowed"] is True and out["n_windows"] > 1
    for sp in out["spans"]:
        assert row["response"][sp["start"]:sp["end"]] == sp["text"]
    print(f"\nwindowed fallback: id {row['id']} ({row['task_type']}) -> {out['n_windows']} windows at max_len 512")


@pytest.mark.parametrize("name", list(MODELS))
def test_determinism(name, predictors, sources):
    rows = read_jsonl(PROCESSED_DIR / "dev.jsonl")[:20]
    xs = [ragtruth_input(r, sources[r["source_id"]]) for r in rows]
    drop = lambda outs: [{k: v for k, v in o.items() if k != "latency_ms"} for o in outs]
    assert drop(predictors[name].predict_batch(xs)) == drop(predictors[name].predict_batch(xs))
    assert drop([predictors[name].predict(**xs[0])]) == drop([predictors[name].predict(**xs[0])])


@pytest.mark.parametrize("bad", [
    {"response": ""}, {"response": None}, {"response": 42},
    {"response": "ok", "structured": {"a": 1}, "context": "c"},
    {"response": "ok", "structured": "not a dict"},
])
def test_input_validation(bad, predictors):
    with pytest.raises(ValueError):
        predictors["distilbert"].predict_batch([bad])
