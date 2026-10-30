"""Module 1b: explore and sanity-check the raw RAGTruth files.

Nothing here assumes the README is correct: it reads the real files, reports the
actual schema, and flags anything unexpected. No data is modified.

Run from the project root:
    python -m groundedguard.data.explore
    python -m groundedguard.data.explore --tokenizer distilbert-base-uncased   # + token-length stats

Writes:
    outputs/module1/dataset_report.json   (all statistics)
    outputs/module1/samples.json          (one example per task, long text shortened)
    outputs/module1/span_mismatches.json  (spans whose offsets don't match their text, if any)
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from groundedguard.config import (
    EXPECTED_LABEL_FIELDS,
    EXPECTED_RESPONSE_FIELDS,
    EXPECTED_SOURCE_FIELDS,
    OUTPUTS_DIR,
    RAW_DIR,
)

OUT_DIR = OUTPUTS_DIR / "module1"
MAX_LEN = 512  # DistilBERT position limit


# ----------------------------------------------------------------------------- helpers
def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m groundedguard.data.download")
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} line {i}: invalid JSON ({exc})") from exc
    return rows


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


def describe(values: list[float]) -> dict:
    """min / percentiles / max / mean without numpy."""
    if not values:
        return {"n": 0}
    v = sorted(values)

    def q(p: float) -> float:
        k = (len(v) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (k - lo)

    return {
        "n": len(v),
        "min": v[0],
        "p50": round(q(0.50), 1),
        "p90": round(q(0.90), 1),
        "p95": round(q(0.95), 1),
        "p99": round(q(0.99), 1),
        "max": v[-1],
        "mean": round(sum(v) / len(v), 1),
    }


def schema_of(rows: list[dict]) -> dict:
    """For every key: how many rows have it and which Python types appear."""
    present, types = Counter(), defaultdict(Counter)
    for r in rows:
        for k, val in r.items():
            present[k] += 1
            types[k][type(val).__name__] += 1
    return {k: {"present_in": present[k], "types": dict(types[k])} for k in sorted(present)}


def shorten(x, n: int = 300):
    if isinstance(x, str):
        return x if len(x) <= n else x[:n] + f"... [+{len(x) - n} chars]"
    if isinstance(x, dict):
        return {k: shorten(v, n) for k, v in x.items()}
    if isinstance(x, list):
        return [shorten(v, n) for v in x[:5]] + ([f"... [+{len(x) - 5} items]"] if len(x) > 5 else [])
    return x


def context_text(source_info) -> str:
    """Rough context string for length estimates only. Module 2 decides the real format."""
    return source_info if isinstance(source_info, str) else json.dumps(source_info, ensure_ascii=False)


# ----------------------------------------------------------------------------- checks
def check_schema(responses, sources, report):
    resp_schema, src_schema = schema_of(responses), schema_of(sources)
    label_rows = [lab for r in responses for lab in (r.get("labels") or [])]
    label_schema = schema_of(label_rows)
    report["schema"] = {
        "response": resp_schema,
        "source": src_schema,
        "label": label_schema,
        "missing_vs_readme": {
            "response": sorted(EXPECTED_RESPONSE_FIELDS - resp_schema.keys()),
            "source": sorted(EXPECTED_SOURCE_FIELDS - src_schema.keys()),
            "label": sorted(EXPECTED_LABEL_FIELDS - label_schema.keys()),
        },
        "extra_vs_readme": {
            "response": sorted(resp_schema.keys() - EXPECTED_RESPONSE_FIELDS),
            "source": sorted(src_schema.keys() - EXPECTED_SOURCE_FIELDS),
            "label": sorted(label_schema.keys() - EXPECTED_LABEL_FIELDS),
        },
    }


def check_ids_and_join(responses, sources, report):
    resp_ids = Counter(str(r["id"]) for r in responses)
    src_ids = Counter(str(s["source_id"]) for s in sources)
    resp_src = {str(r["source_id"]) for r in responses}
    per_source = Counter(str(r["source_id"]) for r in responses)
    report["ids"] = {
        "n_responses": len(responses),
        "n_sources": len(sources),
        "duplicate_response_ids": sum(1 for c in resp_ids.values() if c > 1),
        "duplicate_source_ids": sum(1 for c in src_ids.values() if c > 1),
        "responses_with_unknown_source": sum(1 for r in responses if str(r["source_id"]) not in src_ids),
        "sources_without_responses": len(set(src_ids) - resp_src),
        "responses_per_source": dict(sorted(Counter(per_source.values()).items())),
    }


def check_splits(responses, src_by_id, report):
    task_of = lambda r: src_by_id.get(str(r["source_id"]), {}).get("task_type", "UNKNOWN")
    split_counts = Counter(r.get("split") for r in responses)
    split_task = Counter((r.get("split"), task_of(r)) for r in responses)

    # Leakage check: does any source appear in more than one split?
    splits_per_source = defaultdict(set)
    for r in responses:
        splits_per_source[str(r["source_id"])].add(r.get("split"))
    leaking = [sid for sid, s in splits_per_source.items() if len(s) > 1]

    report["splits"] = {
        "responses_per_split": dict(split_counts),
        "responses_per_split_and_task": {f"{s}|{t}": c for (s, t), c in sorted(split_task.items(), key=str)},
        "sources_in_multiple_splits": len(leaking),
        "example_leaking_source_ids": leaking[:10],
    }


def check_hallucination_rates(responses, src_by_id, report):
    task_of = lambda r: src_by_id.get(str(r["source_id"]), {}).get("task_type", "UNKNOWN")
    groups = {
        "overall": lambda r: "all",
        "by_split": lambda r: r.get("split"),
        "by_task": task_of,
        "by_model": lambda r: r.get("model"),
    }
    out = {}
    for name, key in groups.items():
        tot, hal, spans = Counter(), Counter(), Counter()
        for r in responses:
            k = key(r)
            n = len(r.get("labels") or [])
            tot[k] += 1
            spans[k] += n
            hal[k] += n > 0
        out[name] = {
            str(k): {"responses": tot[k], "hallucinated": hal[k], "rate_%": pct(hal[k], tot[k]), "spans": spans[k]}
            for k in sorted(tot, key=str)
        }
    report["hallucination"] = out


def check_labels(responses, src_by_id, report):
    """Label types, meta flags, and whether char offsets really point at the span text."""
    label_types, by_task = Counter(), defaultdict(Counter)
    flag_values = defaultdict(Counter)  # due_to_null / implicit_true, whatever values exist
    span_chars, mismatches, bad_bounds = [], [], 0
    exact = stripped_only = 0
    overlapping_pairs = responses_with_overlap = 0
    quality = Counter(r.get("quality") for r in responses)

    for r in responses:
        text = r.get("response", "")
        task = src_by_id.get(str(r["source_id"]), {}).get("task_type", "UNKNOWN")
        labels = r.get("labels") or []
        intervals = []
        for lab in labels:
            lt = lab.get("label_type")
            label_types[lt] += 1
            by_task[task][lt] += 1
            for flag in ("due_to_null", "implicit_true"):
                if flag in lab:
                    flag_values[flag][repr(lab[flag])] += 1

            s, e = lab.get("start"), lab.get("end")
            if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text)):
                bad_bounds += 1
                mismatches.append({"id": r["id"], "reason": "bad_bounds", "start": s, "end": e,
                                   "response_len": len(text), "label_text": lab.get("text")})
                continue
            intervals.append((s, e))
            span_chars.append(e - s)
            piece, target = text[s:e], lab.get("text", "")
            if piece == target:
                exact += 1
            elif piece.strip() == str(target).strip():
                stripped_only += 1
            else:
                mismatches.append({"id": r["id"], "reason": "text_mismatch", "start": s, "end": e,
                                   "label_text": target, "text_at_offsets": piece})

        intervals.sort()
        n_ov = sum(1 for (a0, a1), (b0, b1) in zip(intervals, intervals[1:]) if b0 < a1)
        overlapping_pairs += n_ov
        responses_with_overlap += n_ov > 0

    n_spans = sum(label_types.values())
    report["labels"] = {
        "n_spans": n_spans,
        "label_type_counts": dict(label_types.most_common()),
        "label_type_by_task": {t: dict(c.most_common()) for t, c in sorted(by_task.items())},
        "flag_values": {k: dict(v) for k, v in flag_values.items()},
        "quality_counts": dict(quality.most_common()),
        "offset_check": {
            "exact_match": exact,
            "match_after_strip": stripped_only,
            "bad_bounds": bad_bounds,
            "text_mismatch": sum(1 for m in mismatches if m["reason"] == "text_mismatch"),
            "exact_match_%": pct(exact, n_spans),
        },
        "overlap": {"overlapping_span_pairs": overlapping_pairs, "responses_with_overlap": responses_with_overlap},
        "span_length_chars": describe(span_chars),
        "spans_per_response": describe([len(r.get("labels") or []) for r in responses]),
    }
    return mismatches


def check_sources(sources, report):
    info_types = defaultdict(Counter)
    dict_keys = defaultdict(Counter)
    origin = Counter()
    for s in sources:
        t = s.get("task_type")
        info = s.get("source_info")
        info_types[t][type(info).__name__] += 1
        origin[(t, s.get("source"))] += 1
        if isinstance(info, dict):
            dict_keys[t].update(info.keys())
    report["sources"] = {
        "task_counts": dict(Counter(s.get("task_type") for s in sources)),
        "task_and_origin": {f"{t}|{o}": c for (t, o), c in sorted(origin.items(), key=str)},
        "source_info_type_by_task": {t: dict(c) for t, c in info_types.items()},
        "source_info_dict_keys_by_task": {t: dict(c.most_common()) for t, c in dict_keys.items()},
    }


def check_lengths(responses, src_by_id, report, tokenizer_name: str | None):
    report["char_lengths"] = {
        "response": describe([len(r.get("response", "")) for r in responses]),
        "context_estimate": describe([len(context_text(s.get("source_info"))) for s in src_by_id.values()]),
    }
    if not tokenizer_name:
        return
    try:
        from transformers import AutoTokenizer
    except ImportError:
        report["token_lengths"] = "skipped: transformers not installed"
        print("[warn] transformers not installed -> token stats skipped")
        return

    print(f"[info] tokenizing with {tokenizer_name} ...")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    enc = lambda texts: [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]

    sids = list(src_by_id)
    ctx_len = dict(zip(sids, enc([context_text(src_by_id[s].get("source_info")) for s in sids])))
    resp_len = enc([r.get("response", "") for r in responses])

    # [CLS] context [SEP] response [SEP] -> 3 special tokens
    totals = [ctx_len.get(str(r["source_id"]), 0) + rl + 3 for r, rl in zip(responses, resp_len)]
    over_by_task = Counter()
    n_by_task = Counter()
    for r, t in zip(responses, totals):
        task = src_by_id.get(str(r["source_id"]), {}).get("task_type", "UNKNOWN")
        n_by_task[task] += 1
        over_by_task[task] += t > MAX_LEN

    report["token_lengths"] = {
        "tokenizer": tokenizer_name,
        "note": "context = raw source_info (dicts JSON-serialized); an estimate, not the final input format",
        "response_tokens": describe(resp_len),
        "context_tokens": describe(list(ctx_len.values())),
        "pair_tokens": describe(totals),
        "responses_alone_over_512": sum(1 for x in resp_len if x + 2 > MAX_LEN),
        "pairs_over_512_%_by_task": {t: pct(over_by_task[t], n_by_task[t]) for t in sorted(n_by_task)},
        "pairs_over_512_%_overall": pct(sum(over_by_task.values()), len(totals)),
    }


# ----------------------------------------------------------------------------- output
def print_summary(report):
    ids, sch, lab = report["ids"], report["schema"], report["labels"]
    print("\n" + "=" * 70 + "\nRAGTruth exploration summary\n" + "=" * 70)
    print(f"responses: {ids['n_responses']:,} | sources: {ids['n_sources']:,}")
    print(f"missing vs README: {sch['missing_vs_readme']}")
    print(f"extra vs README  : {sch['extra_vs_readme']}")
    print(f"join problems    : unknown_source={ids['responses_with_unknown_source']}, "
          f"dup_resp_ids={ids['duplicate_response_ids']}, dup_src_ids={ids['duplicate_source_ids']}")
    print(f"responses/source : {ids['responses_per_source']}")
    print(f"splits           : {report['splits']['responses_per_split']}")
    print(f"sources in >1 split (leakage): {report['splits']['sources_in_multiple_splits']}")
    print("\nhallucinated-response rate:")
    for grp in ("by_split", "by_task", "by_model"):
        for k, v in report["hallucination"][grp].items():
            print(f"  {grp:9s} {k:22s} {v['hallucinated']:>5}/{v['responses']:<5} = {v['rate_%']:>5}%  spans={v['spans']}")
    print(f"\nlabel types      : {lab['label_type_counts']}")
    print(f"flags            : {lab['flag_values']}")
    print(f"quality          : {lab['quality_counts']}")
    print(f"offset check     : {lab['offset_check']}")
    print(f"overlaps         : {lab['overlap']}")
    print(f"span len (chars) : {lab['span_length_chars']}")
    print(f"source_info types: {report['sources']['source_info_type_by_task']}")
    tl = report.get("token_lengths")
    if isinstance(tl, dict):
        print(f"\nresponse tokens  : {tl['response_tokens']}")
        print(f"context tokens   : {tl['context_tokens']}")
        print(f"pairs > 512 (%)  : {tl['pairs_over_512_%_by_task']}  overall={tl['pairs_over_512_%_overall']}")
        print(f"responses alone > 512: {tl['responses_alone_over_512']}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore RAGTruth.")
    parser.add_argument("--tokenizer", default=None,
                        help="HF tokenizer for token-length stats, e.g. distilbert-base-uncased")
    args = parser.parse_args()

    responses = load_jsonl(RAW_DIR / "response.jsonl")
    sources = load_jsonl(RAW_DIR / "source_info.jsonl")
    src_by_id = {str(s["source_id"]): s for s in sources}

    report: dict = {}
    check_schema(responses, sources, report)
    check_ids_and_join(responses, sources, report)
    check_splits(responses, src_by_id, report)
    check_hallucination_rates(responses, src_by_id, report)
    mismatches = check_labels(responses, src_by_id, report)
    check_sources(sources, report)
    check_lengths(responses, src_by_id, report, args.tokenizer)

    # One example per task, preferring a response that has labels.
    samples = {}
    for r in sorted(responses, key=lambda r: -len(r.get("labels") or [])):
        task = src_by_id.get(str(r["source_id"]), {}).get("task_type")
        if task not in samples:
            samples[task] = {"response_row": shorten(r), "source_row": shorten(src_by_id[str(r["source_id"])])}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dataset_report.json").write_text(json.dumps(report, indent=2, default=str))
    (OUT_DIR / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    (OUT_DIR / "span_mismatches.json").write_text(json.dumps(mismatches[:200], indent=2, ensure_ascii=False))

    print_summary(report)
    print(f"\nsaved -> {OUT_DIR}/dataset_report.json, samples.json, span_mismatches.json")


if __name__ == "__main__":
    main()