"""Module 5a, step 2: score the frozen checkpoints on TEST, once.

Reads checkpoints and thresholds ONLY from outputs/module5/frozen_choice.json and checks
each checkpoint's sha256 against it. Refuses to run if test results already exist.
A test response with no windows (response too long) would count as "no prediction";
the number of such responses is reported.

Run from the project root (same GPU type as training):
    CUDA_VISIBLE_DEVICES=0 python -m groundedguard.eval.evaluate_test [--out_dir outputs/module5b]

Writes (default out_dir outputs/module5):
    <out_dir>/test_results.json
    <out_dir>/preds_seed{42,43,44}.jsonl
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from groundedguard.config import OUTPUTS_DIR, PROCESSED_DIR, PROJECT_ROOT
from groundedguard.data.io import read_jsonl, write_jsonl
from groundedguard.eval.freeze_choice import OUT_DIR, load_run, run_probs, sha256_file
from groundedguard.eval.metrics import TASKS, evaluate
from groundedguard.models.dataset import GroupedWindowDataset
from groundedguard.models.train import spans_at

BASELINES = OUTPUTS_DIR / "module3" / "baselines.json"
METRICS = ("precision", "recall", "f1")
ROWS = [t for t in TASKS] + ["All"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the frozen checkpoints on test, once.")
    parser.add_argument("--out_dir", default=str(OUT_DIR), help="folder holding frozen_choice.json")
    out_dir = PROJECT_ROOT / Path(parser.parse_args().out_dir)
    FROZEN, RESULTS = out_dir / "frozen_choice.json", out_dir / "test_results.json"
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} exists: test has already been scored once.")
    if not FROZEN.exists():
        raise SystemExit(f"{FROZEN} not found: run groundedguard.eval.freeze_choice first.")
    frozen = json.loads(FROZEN.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_rows = read_jsonl(PROCESSED_DIR / "test.jsonl")

    per_seed = {}
    for entry in frozen["runs"]:
        run_dir = PROJECT_ROOT / entry["checkpoint"]
        digest = sha256_file(run_dir / entry["weights_file"])
        if digest != entry["weights_sha256"]:
            raise SystemExit(f"{run_dir}: weights sha256 {digest} != frozen {entry['weights_sha256']}")
        cfg, tok, model = load_run(run_dir, device)
        ds = GroupedWindowDataset(PROCESSED_DIR / f"test_windows{entry['windows_suffix']}.pkl")
        probs = run_probs(cfg, tok, model, ds, device)
        thr = entry["threshold"]

        examples, preds, no_windows = [], [], []
        for r in test_rows:
            meta = ds.responses.get(r["id"])
            if meta is None:
                no_windows.append(r["id"])
            pred = spans_at(meta, probs[r["id"]], thr) if meta else []
            examples.append({"id": r["id"], "task_type": r["task_type"], "response": r["response"],
                             "gold_spans": r["hallucination_spans"], "pred_spans": pred})
            preds.append({"id": r["id"], "task_type": r["task_type"], "model": r["model"],
                          "n_windows": meta["n_windows"] if meta else 0, "threshold": thr,
                          "pred_spans": pred, "gold_spans": r["hallucination_spans"],
                          "token_offsets": [list(o) for o in meta["response_offsets"]] if meta else [],
                          "token_probs": [round(float(p), 5) for p in probs[r["id"]]] if meta else []})
        write_jsonl(preds, out_dir / f"preds_seed{entry['seed']}.jsonl")
        per_seed[entry["seed"]] = {"run_name": entry["run_name"], "threshold": thr,
                                   "responses_without_windows": no_windows, "metrics": evaluate(examples)}
        a = per_seed[entry["seed"]]["metrics"]["All"]
        print(f"seed {entry['seed']} @ {thr}: test All span F1 {a['span']['f1']:.4f} | case F1 {a['case']['f1']:.4f} "
              f"| responses without windows: {len(no_windows)}", flush=True)
        del model
        torch.cuda.empty_cache()

    seeds = list(per_seed)
    summary = {task: {level: {m: {"mean": statistics.mean(per_seed[s]["metrics"][task][level][m] for s in seeds),
                                  "std": statistics.stdev(per_seed[s]["metrics"][task][level][m] for s in seeds)}
                              for m in METRICS} for level in ("case", "span")} for task in ROWS}
    base = json.loads(BASELINES.read_text())["results"]["test"]
    baselines = {name: res for name, res in base.items() if name.startswith(("B1", "B2"))}

    RESULTS.write_text(json.dumps({
        "frozen_choice": str(FROZEN.relative_to(PROJECT_ROOT)), "n_test_responses": len(test_rows),
        "std": "sample std (n-1) over seeds", "mean_std_over_seeds": summary,
        "per_seed": per_seed, "baselines_from_module3": baselines,
    }, indent=2))

    head = (f"{'method':22s} {'task':9s} | {'case P':>15s} {'case R':>15s} {'case F1':>15s} | "
            f"{'span P':>15s} {'span R':>15s} {'span F1':>15s}")
    print("\n" + head + "\n" + "-" * len(head))
    for task in ROWS:
        cells = " | ".join(" ".join(f"{summary[task][lv][m]['mean']:.4f}±{summary[task][lv][m]['std']:.4f}"
                                    for m in METRICS) for lv in ("case", "span"))
        print(f"{'model (3 seeds)':22s} {task:9s} | {cells}")
        for name, res in baselines.items():
            cells = " | ".join(" ".join(f"{res[task][lv][m]:15.4f}" for m in METRICS) for lv in ("case", "span"))
            print(f"{name:22s} {task:9s} | {cells}")
        print()
    print(f"saved -> {RESULTS}, {out_dir}/preds_seed*.jsonl")


if __name__ == "__main__":
    main()
