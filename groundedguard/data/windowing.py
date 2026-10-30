"""Module 2b: tokenize, build sliding context windows, align token labels.

Input format for every window (no truncation anywhere):
    [CLS] context_prefix context_body_window [SEP] full_response [SEP]

- The response is ALWAYS complete; only the context body is split into overlapping
  windows. The prefix (QA question) is repeated in every window.
- Label per response token: 1 if the token overlaps any hallucination char, else 0.
  Labels are stored ONCE per response (`token_labels`), not per window: they are
  judged against the full context, which a single window may not show.
- A token's score = MIN over its windows ("supported if any window supports it"),
  in training (loss on the aggregated score, groundedguard/models/aggregate.py) and at inference.

Run from the project root:
    python -m groundedguard.data.windowing
    python -m groundedguard.data.windowing --stride 128 --tokenizer distilbert-base-uncased
    python -m groundedguard.data.windowing --stride 64 --suffix _s64 --splits train,dev
    python -m groundedguard.data.windowing --tokenizer answerdotai/ModernBERT-base --max_len 8192 --suffix _modernbert

Writes:
    data/processed/{split}_windows{suffix}.pkl
    outputs/module2/windowing_report{suffix}.json     statistics only
    outputs/module2/windowing_examples{suffix}.json   decoded gold spans (RAGTruth text; git-ignored)
"""

import argparse
import json
from collections import Counter, defaultdict

from groundedguard.config import (
    DOC_STRIDE,
    MAX_LEN,
    MIN_CONTEXT_TOKENS,
    MODEL_NAME,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
)
from groundedguard.data.explore import describe
from groundedguard.data.io import read_jsonl, save_pickle
from groundedguard.data.serialize import serialize_context
from groundedguard.eval.metrics import char_overlap, prf

OUT_DIR = OUTPUTS_DIR / "module2"
N_SPECIAL = 3  # [CLS], [SEP], [SEP]


# ----------------------------------------------------------------------------- core logic
def window_starts(n_ctx: int, budget: int, stride: int) -> list[tuple[int, int]]:
    """Overlapping [start, end) windows over n_ctx context tokens."""
    if n_ctx <= budget:
        return [(0, n_ctx)]
    step = budget - stride
    out, start = [], 0
    while True:
        end = min(start + budget, n_ctx)
        out.append((start, end))
        if end == n_ctx:
            return out
        start += step


def trim_offsets(text: str, offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop leading/trailing whitespace from token char ranges. Byte-level BPE tokens such
    as 'Ġworld' include the preceding space; WordPiece offsets never do (no-op there).
    A whitespace-only token becomes empty and so never overlaps a span."""
    out = []
    for a, b in offsets:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        out.append((a, b))
    return out


def token_labels(offsets: list[tuple[int, int]], spans: list[list[int]]) -> list[int]:
    """1 if the token's [a, b) char range overlaps any span, else 0."""
    labels = []
    for a, b in offsets:
        labels.append(int(any(a < e and s < b for s, e in spans)))
    return labels


def tokens_to_spans(offsets: list[tuple[int, int]], labels: list[int]) -> list[list[int]]:
    """Contiguous runs of label-1 tokens -> char spans [first.start, last.end).
    The same function will turn model predictions into spans in Module 6."""
    spans, cur = [], None
    for (a, b), y in zip(offsets, labels):
        if y == 1:
            cur = [a, b] if cur is None else [cur[0], b]
        elif cur is not None:
            spans.append(cur)
            cur = None
    if cur is not None:
        spans.append(cur)
    return spans


def encode_responses(tok, responses: list[str]) -> list[tuple[list[int], list[tuple[int, int]]]]:
    """Per response: token ids (no special tokens) and whitespace-trimmed char offsets into
    the ORIGINAL string. Shared by build_split and the inference predictor."""
    enc = tok(responses, add_special_tokens=False, return_offsets_mapping=True)
    return [(ids, trim_offsets(text, [tuple(o) for o in off]))
            for text, ids, off in zip(responses, enc["input_ids"], enc["offset_mapping"])]


def build_windows(p_ids: list[int], c_ids: list[int], r_ids: list[int], cls_id: int, sep_id: int,
                  max_len: int, stride: int) -> list[dict] | None:
    """Windows of ONE response: [CLS] prefix body[a:b] [SEP] response [SEP], at most max_len
    tokens each. None if the response leaves fewer than MIN_CONTEXT_TOKENS of body per
    window. Shared by build_split and the inference predictor."""
    budget = max_len - N_SPECIAL - len(r_ids) - len(p_ids)  # body tokens per window
    if budget < MIN_CONTEXT_TOKENS:
        return None
    out = []
    for w, (a, b) in enumerate(window_starts(len(c_ids), budget, min(stride, budget // 2))):
        input_ids = [cls_id] + p_ids + c_ids[a:b] + [sep_id] + r_ids + [sep_id]
        assert len(input_ids) <= max_len
        out.append({
            "window": w,
            "ctx_token_range": (a, b),  # slice of the context BODY tokens
            "response_start": len(p_ids) + b - a + 2,  # index of first response token in input_ids
            "input_ids": input_ids,
        })
    return out


def build_split(rows, tok, stride: int, max_len: int = MAX_LEN):
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id

    # Tokenize each distinct context once. Whitespace separates prefix and body, so
    # tokenizing them apart gives the same tokens as tokenizing the joined context.
    parts_by_source = {r["source_id"]: (r["context_prefix"], r["context_body"]) for r in rows}
    sids = list(parts_by_source)
    pre_ids = dict(zip(sids, tok([parts_by_source[s][0] for s in sids], add_special_tokens=False)["input_ids"]))
    body_ids = dict(zip(sids, tok([parts_by_source[s][1] for s in sids], add_special_tokens=False)["input_ids"]))

    encoded = encode_responses(tok, [r["response"] for r in rows])

    windows, responses, skipped = [], {}, []
    for r, (r_ids, r_off) in zip(rows, encoded):
        p_ids, c_ids = pre_ids[r["source_id"]], body_ids[r["source_id"]]
        wins = build_windows(p_ids, c_ids, r_ids, cls_id, sep_id, max_len, stride)
        if wins is None:
            skipped.append({"id": r["id"], "task_type": r["task_type"], "response_tokens": len(r_ids),
                            "prefix_tokens": len(p_ids)})
            continue

        y = token_labels(r_off, r["hallucination_spans"])
        for w in wins:
            windows.append({"id": r["id"], "n_windows": len(wins), **w})

        responses[r["id"]] = {
            "task_type": r["task_type"],
            "model": r["model"],
            "response": r["response"],
            "hallucination_spans": r["hallucination_spans"],
            "response_offsets": r_off,
            "token_labels": y,
            "n_windows": len(wins),
            "context_tokens": len(p_ids) + len(c_ids),
            "response_tokens": len(r_ids),
        }
    return windows, responses, skipped


# ----------------------------------------------------------------------------- checks
def alignment_report(responses: dict) -> dict:
    """Project token labels back to chars and score them with the official metric.
    This is the best span-F1 any token classifier could reach with this tokenization."""
    by_task = defaultdict(lambda: [0, 0, 0])
    lost_spans, n_spans = 0, 0
    for meta in responses.values():
        gold = meta["hallucination_spans"]
        proj = tokens_to_spans(meta["response_offsets"], meta["token_labels"])
        for s, e in gold:
            n_spans += 1
            if not any(a < e and s < b and y for (a, b), y in zip(meta["response_offsets"], meta["token_labels"])):
                lost_spans += 1
        for key in (meta["task_type"], "All"):
            tp, p, g = char_overlap(gold, proj)
            by_task[key][0] += tp
            by_task[key][1] += p
            by_task[key][2] += g

    out = {k: {m: round(v, 4) for m, v in prf(tp, p, g).items()} for k, (tp, p, g) in sorted(by_task.items())}
    return {"oracle_char_metric": out, "gold_spans": n_spans, "spans_with_no_labeled_token": lost_spans}


def decode_examples(responses: dict, tok, k: int = 3) -> list[dict]:
    """Gold span text vs. the tokens that got label 1 (decoded) - eyeball check."""
    ex = []
    for rid, meta in responses.items():
        if not meta["hallucination_spans"]:
            continue
        s, e = meta["hallucination_spans"][0]
        ids = tok(meta["response"], add_special_tokens=False)["input_ids"]
        pos = [t for t, ((a, b), y) in enumerate(zip(meta["response_offsets"], meta["token_labels"]))
               if y and a < e and s < b]
        ex.append({"id": rid, "gold_text": meta["response"][s:e], "decoded_label1_tokens": tok.decode([ids[t] for t in pos])})
        if len(ex) == k:
            break
    return ex


def split_stats(windows, responses, skipped) -> dict:
    by_task = defaultdict(list)
    for m in responses.values():
        by_task[m["task_type"]].append(m)
    pos = sum(sum(m["token_labels"]) for m in responses.values())
    tot = sum(len(m["token_labels"]) for m in responses.values())
    return {
        "responses": len(responses),
        "windows": len(windows),
        "skipped_response_too_long": skipped,
        "positive_token_%": round(100 * pos / max(tot, 1), 2),
        "windows_per_response": describe([m["n_windows"] for m in responses.values()]),
        "windows_per_response_by_task": {t: describe([m["n_windows"] for m in ms]) for t, ms in sorted(by_task.items())},
        "single_window_%_by_task": {t: round(100 * sum(m["n_windows"] == 1 for m in ms) / len(ms), 2)
                                    for t, ms in sorted(by_task.items())},
        "window_lengths": describe([len(w["input_ids"]) for w in windows]),
    }


def data2txt_format_comparison(tok) -> dict:
    """Tokens for the official str(dict) vs our compact serializer (Data2txt only)."""
    srcs = [s for s in read_jsonl(RAW_DIR / "source_info.jsonl") if s["task_type"] == "Data2txt"]
    repr_len = [len(x) for x in tok([f"{s['source_info']}" for s in srcs], add_special_tokens=False)["input_ids"]]
    ours_len = [len(x) for x in tok([serialize_context("Data2txt", s["source_info"]) for s in srcs],
                                    add_special_tokens=False)["input_ids"]]
    return {"official_str_dict": describe(repr_len), "compact_ours": describe(ours_len),
            "token_reduction_%": round(100 * (1 - sum(ours_len) / sum(repr_len)), 2)}


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Sliding-window tokenization for RAGTruth.")
    parser.add_argument("--tokenizer", default=MODEL_NAME, help="HF name or local tokenizer dir")
    parser.add_argument("--stride", type=int, default=DOC_STRIDE, help="context-token overlap between windows")
    parser.add_argument("--max_len", type=int, default=MAX_LEN, help="tokens per window incl. [CLS] and 2x [SEP]")
    parser.add_argument("--suffix", default="", help="output name suffix, e.g. _s64 -> train_windows_s64.pkl")
    parser.add_argument("--splits", default="train,dev,test", help="comma-separated splits to build")
    args = parser.parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if not tok.is_fast:
        raise RuntimeError("a fast tokenizer is required for offset_mapping")
    tok.model_max_length = 10**6  # we never feed > max_len; this only silences the length warning

    report = {"config": {"tokenizer": args.tokenizer, "max_len": args.max_len, "stride": args.stride,
                         "min_context_tokens": MIN_CONTEXT_TOKENS},
              "data2txt_serialization": data2txt_format_comparison(tok)}

    examples = {}
    for split in splits:
        rows = read_jsonl(PROCESSED_DIR / f"{split}.jsonl")
        windows, responses, skipped = build_split(rows, tok, args.stride, args.max_len)
        report[split] = split_stats(windows, responses, skipped)
        report[split]["alignment"] = alignment_report(responses)
        examples[split] = decode_examples(responses, tok)
        save_pickle({"config": report["config"], "windows": windows, "responses": responses, "skipped": skipped},
                    PROCESSED_DIR / f"{split}_windows{args.suffix}.pkl")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"windowing_report{args.suffix}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (OUT_DIR / f"windowing_examples{args.suffix}.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False))

    print("=" * 70 + "\nModule 2b: sliding-window tokenization\n" + "=" * 70)
    d2t = report["data2txt_serialization"]
    print(f"Data2txt tokens: official str(dict) p50={d2t['official_str_dict']['p50']} | "
          f"ours p50={d2t['compact_ours']['p50']} | reduction {d2t['token_reduction_%']}%")
    for split in splits:
        s = report[split]
        al = s["alignment"]
        print(f"\n[{split}] responses={s['responses']:,} windows={s['windows']:,} "
              f"skipped={len(s['skipped_response_too_long'])} positive_tokens={s['positive_token_%']}%")
        print(f"  windows/response by task: " + ", ".join(
            f"{t}: p50={d['p50']} p95={d['p95']} max={d['max']}" for t, d in s["windows_per_response_by_task"].items()))
        print(f"  single-window %: {s['single_window_%_by_task']}")
        print(f"  window length: {s['window_lengths']}")
        oracle = ", ".join(f"{k}={v['f1']}" for k, v in al["oracle_char_metric"].items())
        print(f"  alignment: lost_spans={al['spans_with_no_labeled_token']}/{al['gold_spans']} | "
              f"oracle char-F1 {oracle}")
        for ex in examples[split][:2]:
            print(f"    gold: {ex['gold_text'][:70]!r}\n    toks: {ex['decoded_label1_tokens'][:70]!r}")
        if s["skipped_response_too_long"]:
            print(f"  skipped: {s['skipped_response_too_long']}")
    print(f"\nsaved -> {PROCESSED_DIR}/{{{','.join(splits)}}}_windows{args.suffix}.pkl, "
          f"{OUT_DIR}/windowing_report{args.suffix}.json")


if __name__ == "__main__":
    main()