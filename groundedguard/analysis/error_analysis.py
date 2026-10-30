"""Module 5a, step 3: error analysis of seed 42's TEST predictions (no model, no re-scoring).

  - char recall by label_type (Evident/Subtle x Baseless/Conflict), overall and per task
  - span P/R/F1 by generator model and by n_windows bucket
  - char recall on gold spans flagged implicit_true / due_to_null vs unflagged
  - 15 false-positive and 15 false-negative examples (5 per task, seeded sample):
    facts only (label_type, annotator meta, token probabilities), plus a 300-char context
    snippet around the best lexical match of the span.

A false positive is a predicted span that overlaps no gold char; a false negative is a raw
gold label span that overlaps no predicted char. Recall is over chars; a char covered by
labels of two categories counts for both. "detected" = the gold span overlaps >= 1
predicted char.

Run from the project root:
    python -m groundedguard.analysis.error_analysis
    python -m groundedguard.analysis.error_analysis --preds outputs/module5b/preds_seed42.jsonl \
        --out_dir outputs/module5b --bucket_from outputs/module5/preds_seed42.jsonl

--bucket_from takes n_windows from another model's predictions (e.g. the DistilBERT
windows), so n_windows rows line up across models.

Writes (default out_dir outputs/module5):
    <out_dir>/error_analysis_tables.json          numbers only (no dataset text): safe to publish
    <out_dir>/error_analysis_examples.{md,json}   the FP/FN examples, which quote RAGTruth text
                                                  (git-ignored: the dataset is not redistributed)
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

from groundedguard.baselines.heuristic import WORD
from groundedguard.config import PROJECT_ROOT, OUTPUTS_DIR, PROCESSED_DIR, SEED
from groundedguard.data.io import read_jsonl
from groundedguard.data.serialize import join_context
from groundedguard.eval.metrics import TASKS, evaluate
from groundedguard.models.diagnostics import BUCKETS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

OUT_DIR = OUTPUTS_DIR / "module5"
LABEL_TYPES = ("Evident Baseless Info", "Evident Conflict", "Subtle Baseless Info", "Subtle Conflict")
MODELS = ("gpt-4-0613", "gpt-3.5-turbo-0613", "llama-2-7b-chat", "llama-2-13b-chat", "llama-2-70b-chat",
          "mistral-7B-instruct")
N_EXAMPLES_PER_TASK = 5
SNIPPET = 300
RESPONSE_CHARS = 400  # shown on each side of the marked span when the response is long


# ----------------------------------------------------------------------------- helpers
def chars(spans) -> set[int]:
    return {i for s, e in spans for i in range(s, e)}


def recall_table(items: list[tuple[str, tuple[int, int], set[int]]]) -> dict:
    """items: (category, (start, end), predicted chars of that response) -> per-category recall."""
    acc = defaultdict(lambda: {"spans": 0, "detected_spans": 0, "gold_chars": 0, "recalled_chars": 0})
    for cat, (s, e), pred in items:
        span = set(range(s, e))
        a = acc[cat]
        a["spans"] += 1
        a["detected_spans"] += bool(span & pred)
        a["gold_chars"] += len(span)
        a["recalled_chars"] += len(span & pred)
    for a in acc.values():
        a["char_recall"] = a["recalled_chars"] / a["gold_chars"] if a["gold_chars"] else 0.0
        a["detected_%"] = 100 * a["detected_spans"] / a["spans"] if a["spans"] else 0.0
    return dict(acc)


def span_prf_rows(subsets: dict[str, list[dict]]) -> dict:
    out = {}
    for name, ex in subsets.items():
        if not ex:
            out[name] = {"n": 0}
            continue
        a = evaluate(ex)["All"]
        out[name] = {"n": a["n"], "hallucinated_%": 100 * sum(bool(e["gold_spans"]) for e in ex) / len(ex),
                     "case_f1": a["case"]["f1"], "span_precision": a["span"]["precision"],
                     "span_recall": a["span"]["recall"], "span_f1": a["span"]["f1"]}
    return out


def span_probs(pred: dict, s: int, e: int) -> dict:
    ps = [p for (a, b), p in zip(pred["token_offsets"], pred["token_probs"]) if a < e and s < b]
    return {"max_token_prob": round(max(ps), 4), "mean_token_prob": round(sum(ps) / len(ps), 4)} if ps else {}


def mark(text: str, s: int, e: int) -> str:
    left, right = max(0, s - RESPONSE_CHARS), min(len(text), e + RESPONSE_CHARS)
    return (("…" if left else "") + text[left:s] + "[[" + text[s:e] + "]]" + text[e:right]
            + ("…" if right < len(text) else ""))


def best_lexical_match(context: str, span_text: str) -> dict:
    """Context window of the span's word length sharing the most content words with it."""
    span_words = [w.lower() for w in WORD.findall(span_text)]
    content = {w for w in span_words if w not in ENGLISH_STOP_WORDS}
    ctx = list(WORD.finditer(context))
    k = max(1, len(span_words))
    best, best_i = 0, None
    for i in range(len(ctx)):
        score = len(content & {m.group().lower() for m in ctx[i:i + k]})
        if score > best:
            best, best_i = score, i
    if best_i is None:
        return {"shared_content_words": 0, "snippet": None}
    a, b = ctx[best_i].start(), ctx[min(best_i + k, len(ctx)) - 1].end()
    mid = (a + b) // 2
    lo, hi = max(0, mid - SNIPPET // 2), min(len(context), mid + SNIPPET // 2)
    return {"shared_content_words": best, "of_content_words": len(content),
            "snippet": ("…" if lo else "") + context[lo:hi] + ("…" if hi < len(context) else "")}


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Error analysis of one seed's test predictions.")
    parser.add_argument("--preds", default=str(OUT_DIR / "preds_seed42.jsonl"))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--bucket_from", default=None, help="preds file whose n_windows define the buckets")
    args = parser.parse_args()
    out_dir = PROJECT_ROOT / Path(args.out_dir)
    preds = {p["id"]: p for p in read_jsonl(args.preds)}
    bucket_n = ({p["id"]: p["n_windows"] for p in read_jsonl(args.bucket_from)} if args.bucket_from
                else {i: p["n_windows"] for i, p in preds.items()})
    rows = {r["id"]: r for r in read_jsonl(PROCESSED_DIR / "test.jsonl")}
    assert preds.keys() == rows.keys(), "predictions and test rows differ"
    threshold = next(iter(preds.values()))["threshold"]

    examples = [{"id": i, "task_type": r["task_type"], "response": r["response"], "gold_spans": r["hallucination_spans"],
                 "pred_spans": preds[i]["pred_spans"]} for i, r in rows.items()]

    # 1. recall by label type (2x2), overall and per task
    lt_items = defaultdict(list)
    flag_items = []
    for i, r in rows.items():
        pred = chars(preds[i]["pred_spans"])
        for lab in r["labels"]:
            se = (lab["start"], lab["end"])
            lt_items["All"].append((lab["label_type"], se, pred))
            lt_items[r["task_type"]].append((lab["label_type"], se, pred))
            flags = [f for f in ("implicit_true", "due_to_null") if lab.get(f) is True]
            for f in flags or ["unflagged"]:
                flag_items.append((f, se, pred))
    label_recall = {scope: recall_table(items) for scope, items in lt_items.items()}

    # 2-3. by generator model, by n_windows
    by_model = span_prf_rows({m: [e for e in examples if rows[e["id"]]["model"] == m] for m in MODELS})
    by_windows = span_prf_rows({name: [e for e in examples if lo <= bucket_n[e["id"]] <= hi]
                                for name, lo, hi in BUCKETS})
    # 4. flags
    flag_recall = recall_table(flag_items)

    # 5. examples
    rng = random.Random(SEED)
    fps, fns = defaultdict(list), defaultdict(list)
    for i, r in sorted(rows.items(), key=lambda kv: int(kv[0])):
        gold, p = chars(r["hallucination_spans"]), preds[i]
        for s, e in p["pred_spans"]:
            if not set(range(s, e)) & gold:
                fps[r["task_type"]].append((i, s, e, None))
        pred = chars(p["pred_spans"])
        for lab in r["labels"]:
            if not set(range(lab["start"], lab["end"])) & pred:
                fns[r["task_type"]].append((i, lab["start"], lab["end"], lab))

    def build(kind, i, s, e, lab):
        r, p = rows[i], preds[i]
        ex = {"kind": kind, "id": i, "task_type": r["task_type"], "generator": r["model"],
              "n_windows": p["n_windows"], "span": [s, e], "span_text": r["response"][s:e], **span_probs(p, s, e),
              "response_gold_spans": len(r["hallucination_spans"]), "response_pred_spans": len(p["pred_spans"]),
              "response_marked": mark(r["response"], s, e),
              "context_match": best_lexical_match(join_context(r["context_prefix"], r["context_body"]),
                                                  r["response"][s:e])}
        if lab is not None:
            ex.update({"label_type": lab["label_type"], "implicit_true": lab.get("implicit_true"),
                       "due_to_null": lab.get("due_to_null"), "meta": lab.get("meta")})
        return ex

    picked = {"false_positives": [], "false_negatives": []}
    candidates = {"false_positives": fps, "false_negatives": fns}
    for kind, by_task in candidates.items():
        for task in TASKS:
            pool = by_task[task]
            for c in rng.sample(pool, min(N_EXAMPLES_PER_TASK, len(pool))):
                picked[kind].append(build(kind, *c))

    rel = lambda path: str(Path(path).resolve().relative_to(PROJECT_ROOT))
    tables = {"predictions": rel(args.preds), "bucket_from": rel(args.bucket_from or args.preds),
              "threshold": threshold, "n_responses": len(rows),
              "overall": evaluate(examples)["All"],
              "recall_by_label_type": label_recall, "by_generator_model": by_model, "by_n_windows": by_windows,
              "recall_by_flag": flag_recall,
              "candidate_counts": {k: {t: len(v[t]) for t in TASKS} for k, v in candidates.items()}}
    ex_report = {"predictions": tables["predictions"], "threshold": threshold,
                 "candidate_counts": tables["candidate_counts"], "examples": picked}
    (out_dir / "error_analysis_tables.json").write_text(json.dumps(tables, indent=2, ensure_ascii=False))
    (out_dir / "error_analysis_examples.json").write_text(json.dumps(ex_report, indent=2, ensure_ascii=False))
    (out_dir / "error_analysis_examples.md").write_text(render_examples(ex_report))
    print(render_tables(tables))
    print(f"saved -> {out_dir}/error_analysis_tables.json, error_analysis_examples.md, error_analysis_examples.json")


# ----------------------------------------------------------------------------- markdown
def render_tables(rep: dict) -> str:
    f4 = lambda x: f"{x:.4f}"
    L = [f"# Error analysis on test",
         f"", f"Predictions `{rep['predictions']}`, threshold {rep['threshold']}, {rep['n_responses']} responses. "
         f"Overall span P/R/F1 = {f4(rep['overall']['span']['precision'])} / {f4(rep['overall']['span']['recall'])} / "
         f"{f4(rep['overall']['span']['f1'])}.", "",
         "Recall is over gold characters (raw annotations); *detected* = the gold span overlaps at least one "
         "predicted character.", ""]

    L += ["## 1. Recall by label type", "",
          "| scope | label type | spans | detected % | gold chars | char recall |", "|---|---|---|---|---|---|"]
    for scope in ["All"] + list(TASKS):
        for lt in LABEL_TYPES:
            a = rep["recall_by_label_type"].get(scope, {}).get(lt)
            if a:
                L.append(f"| {scope} | {lt} | {a['spans']} | {a['detected_%']:.1f} | {a['gold_chars']} | "
                         f"{f4(a['char_recall'])} |")
            else:
                L.append(f"| {scope} | {lt} | 0 | – | 0 | – |")

    for title, key, label in (("2. By generator model", "by_generator_model", "generator"),
                              (f"3. By number of context windows (buckets from `{rep['bucket_from']}`)", "by_n_windows", "n_windows")):
        L += ["", f"## {title}", "",
              f"| {label} | n | hallucinated % | case F1 | span P | span R | span F1 |", "|---|---|---|---|---|---|---|"]
        for name, r in rep[key].items():
            if r["n"]:
                L.append(f"| {name} | {r['n']} | {r['hallucinated_%']:.1f} | {f4(r['case_f1'])} | "
                         f"{f4(r['span_precision'])} | {f4(r['span_recall'])} | {f4(r['span_f1'])} |")

    L += ["", "## 4. Recall on flagged gold spans", "",
          "| flag | spans | detected % | gold chars | char recall |", "|---|---|---|---|---|"]
    for name in ("implicit_true", "due_to_null", "unflagged"):
        a = rep["recall_by_flag"].get(name)
        if a:
            L.append(f"| {name} | {a['spans']} | {a['detected_%']:.1f} | {a['gold_chars']} | {f4(a['char_recall'])} |")
    return "\n".join(L) + "\n"


def render_examples(rep: dict) -> str:
    """Markdown of the FP/FN examples (quotes RAGTruth text; not committed)."""
    L = ["# Error analysis on test: examples", "",
         f"Predictions `{rep['predictions']}`, threshold {rep['threshold']}.", "",
         "## Examples", "",
          f"5 per task, sampled with seed {SEED} from all candidates "
          f"(false positives: {rep['candidate_counts']['false_positives']}; "
          f"false negatives: {rep['candidate_counts']['false_negatives']}). "
          "Facts only: no explanation is inferred. The span is marked [[like this]]; long responses are cut "
          f"to {RESPONSE_CHARS} characters around it. The context snippet ({SNIPPET} chars) is centred on the "
          "context window that shares the most content words with the span.", ""]
    for kind, title in (("false_positives", "False positives (predicted, not in gold)"),
                        ("false_negatives", "False negatives (gold, not predicted)")):
        L += [f"### {title}", ""]
        for n, ex in enumerate(rep["examples"][kind], 1):
            facts = [f"task **{ex['task_type']}**", f"generator `{ex['generator']}`", f"id {ex['id']}",
                     f"windows {ex['n_windows']}", f"chars {ex['span'][0]}–{ex['span'][1]}"]
            if "max_token_prob" in ex:
                facts.append(f"token prob max {ex['max_token_prob']} / mean {ex['mean_token_prob']}")
            facts.append(f"response has {ex['response_gold_spans']} gold / {ex['response_pred_spans']} predicted spans")
            L += [f"**{kind[:-1].replace('_', ' ').upper()} {n}**: " + " · ".join(facts), ""]
            if "label_type" in ex:
                L.append(f"- label_type: `{ex['label_type']}` · implicit_true: `{ex['implicit_true']}` · "
                         f"due_to_null: `{ex['due_to_null']}`")
                if ex.get("meta"):
                    L.append("- annotator meta: " + ex["meta"].replace("\n", " / "))
            L += ["", "Response:", "", "```text", ex["response_marked"], "```", ""]
            cm = ex["context_match"]
            if cm["snippet"]:
                L += [f"Context snippet (shares {cm['shared_content_words']} of {cm['of_content_words']} "
                      "content words of the span):", "", "```text", cm["snippet"], "```", ""]
            else:
                L += ["Context snippet: no content word of the span occurs in the context.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
