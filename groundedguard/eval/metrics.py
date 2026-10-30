"""Module 3a: the official RAGTruth metric (CodingLL/RAGTruth_Eval, compute_metrics.py).

Span level: per response, the union of gold char positions vs the union of predicted
char positions; tp, |pred| and |gold| are summed over all responses (micro) -> P, R, F1.
Case level: a response is predicted positive if it has at least one predicted span,
and is truly positive if it has at least one gold span.
Division by zero gives 0, as in the official code.

Input: a list of {id, task_type, response, gold_spans, pred_spans}, spans = [start, end)
char offsets into `response`. The official script takes predicted span STRINGS and
locates them with str.find; we take offsets directly, so a repeated substring is never
matched to the wrong occurrence.
"""

from collections import defaultdict

TASKS = ("QA", "Summary", "Data2txt")


def char_overlap(gold: list[list[int]], pred: list[list[int]]) -> tuple[int, int, int]:
    """Official span metric pieces for one response: (|gold ∩ pred|, |pred|, |gold|) over char sets."""
    g = {i for s, e in gold for i in range(s, e)}
    p = {i for s, e in pred for i in range(s, e)}
    return len(g & p), len(p), len(g)


def prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def evaluate(examples: list[dict]) -> dict:
    """{task: {"n", "case": {P, R, F1, tp, fp, fn, tn}, "span": {P, R, F1, tp, pred, gold}}}
    for every task present, in TASKS order, plus "All"."""
    span = defaultdict(lambda: [0, 0, 0])   # tp, |pred|, |gold|
    case = defaultdict(lambda: [0, 0, 0, 0])  # tp, fp, fn, tn
    n = defaultdict(int)
    for ex in examples:
        gold, pred, length = ex["gold_spans"], ex["pred_spans"], len(ex["response"])
        for s, e in list(gold) + list(pred):
            if not 0 <= s <= e <= length:
                raise ValueError(f"id={ex['id']}: span [{s}, {e}) outside response of length {length}")
        tp, n_pred, n_gold = char_overlap(gold, pred)
        has_pred, has_gold = bool(pred), bool(gold)
        cell = (0 if has_pred and has_gold else 1 if has_pred else 2 if has_gold else 3)
        for key in (ex["task_type"], "All"):
            n[key] += 1
            span[key][0] += tp
            span[key][1] += n_pred
            span[key][2] += n_gold
            case[key][cell] += 1

    out = {}
    for key in [t for t in TASKS if t in n] + sorted(set(n) - set(TASKS) - {"All"}) + ["All"]:
        if key not in n:
            continue
        ctp, cfp, cfn, ctn = case[key]
        stp, spred, sgold = span[key]
        out[key] = {
            "n": n[key],
            "case": {**prf(ctp, ctp + cfp, ctp + cfn), "tp": ctp, "fp": cfp, "fn": cfn, "tn": ctn},
            "span": {**prf(stp, spred, sgold), "tp": stp, "pred": spred, "gold": sgold},
        }
    return out


def format_table(results: dict[str, dict], title: str = "") -> str:
    """results: {row label: evaluate(...) output} -> one aligned text table."""
    head = f"{'':24s} {'task':9s} {'n':>5s} | {'case P':>7s} {'case R':>7s} {'case F1':>7s} | " \
           f"{'span P':>7s} {'span R':>7s} {'span F1':>7s}"
    lines = ([title] if title else []) + [head, "-" * len(head)]
    for label, res in results.items():
        for task, m in res.items():
            c, s = m["case"], m["span"]
            lines.append(f"{label:24s} {task:9s} {m['n']:>5d} | {c['precision']:7.4f} {c['recall']:7.4f} "
                         f"{c['f1']:7.4f} | {s['precision']:7.4f} {s['recall']:7.4f} {s['f1']:7.4f}")
        lines.append("")
    return "\n".join(lines).rstrip()
