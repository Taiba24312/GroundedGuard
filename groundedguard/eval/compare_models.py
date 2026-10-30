"""Module 5b, step 4: windowed DistilBERT vs full-context ModernBERT, on test.

Reads the frozen choices, test results and error analyses already written by
freeze_choice / evaluate_test / error_analysis (nothing is re-scored), and adds:
  - parameters
  - train time per epoch: median over epochs of the given timing runs' history.json.
    Timing runs must have trained alone on the GPU (the DistilBERT seed runs shared it).
  - peak GPU memory per training process (train + dev eval; max over epochs and seeds;
    per-process, so unaffected by other jobs on the GPU)
  - inference ms per response on GPU: batch = one response (all its windows), bf16,
    median over the same 200 test responses (seeded sample), seed-42 checkpoint, after
    20 warm-up responses; forward + window aggregation + sigmoid.

Run from the project root, with the GPU otherwise idle:
    CUDA_VISIBLE_DEVICES=0 python -m groundedguard.eval.compare_models

Writes:
    outputs/module5b/comparison.json
    outputs/module5b/comparison.md
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from groundedguard.analysis.error_analysis import LABEL_TYPES
from groundedguard.config import CHECKPOINTS_DIR, OUTPUTS_DIR, PROCESSED_DIR, PROJECT_ROOT, SEED
from groundedguard.eval.freeze_choice import load_run
from groundedguard.eval.metrics import TASKS
from groundedguard.models.aggregate import aggregate_windows
from groundedguard.models.dataset import Collator, GroupedWindowDataset
from groundedguard.models.diagnostics import BUCKETS
from groundedguard.models.train import to_device

ROWS = list(TASKS) + ["All"]
N_TIMED, N_WARMUP = 200, 20


def history(run: str) -> list[dict]:
    return json.loads((CHECKPOINTS_DIR / run / "history.json").read_text())


@torch.no_grad()
def ms_per_response(cfg, tok, model, ds, ids, device) -> float:
    model.eval()
    collate = Collator(tok.pad_token_id)
    index = {rid: i for i, rid in enumerate(ds.ids)}
    bf16 = bool(cfg["bf16"]) and device.type == "cuda"
    times = []
    for k, rid in enumerate(ids[:N_WARMUP] + ids):
        b = to_device(collate([ds[index[rid]]]), device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.squeeze(-1)
        torch.sigmoid(aggregate_windows(logits.float(), b, cfg.get("aggregation", "min"), cfg.get("tau", 1.0)))
        torch.cuda.synchronize()
        if k >= N_WARMUP:
            times.append(1000 * (time.perf_counter() - t0))
    return statistics.median(times)


def describe_model(name: str, out_dir: Path, timing_runs: list[str], ids: list[str], device) -> dict:
    frozen = json.loads((out_dir / "frozen_choice.json").read_text())
    results = json.loads((out_dir / "test_results.json").read_text())
    runs = [e["run_name"] for e in frozen["runs"]]
    seed42 = next(e for e in frozen["runs"] if e["seed"] == 42)
    cfg, tok, model = load_run(PROJECT_ROOT / seed42["checkpoint"], device)
    ds = GroupedWindowDataset(PROCESSED_DIR / f"test_windows{seed42['windows_suffix']}.pkl")
    epoch_times = [h["train_time_s"] for r in timing_runs for h in history(r)]
    info = {
        "name": name, "out_dir": str(out_dir.relative_to(PROJECT_ROOT)), "runs": runs,
        "thresholds": {e["seed"]: e["threshold"] for e in frozen["runs"]},
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "train_s_per_epoch_median": statistics.median(epoch_times), "timing_runs": timing_runs,
        "peak_gpu_mem_gib": max(h["peak_gpu_mem_gib"] for r in runs for h in history(r)),
        "infer_ms_per_response_median": ms_per_response(cfg, tok, model, ds, ids, device),
        "test": results["mean_std_over_seeds"],
        "error_analysis": json.loads((out_dir / "error_analysis_tables.json").read_text()),
    }
    del model
    torch.cuda.empty_cache()
    return info


def main() -> None:
    p = argparse.ArgumentParser(description="DistilBERT (windows) vs ModernBERT (full context) on test.")
    p.add_argument("--a", default="outputs/module5", help="first model's out_dir")
    p.add_argument("--a_name", default="DistilBERT-minpool")
    p.add_argument("--a_timing_runs", default="distilbert_minpool_v1",
                   help="runs that trained alone on the GPU (same windows/batching as the frozen config)")
    p.add_argument("--b", default="outputs/module5b")
    p.add_argument("--b_name", default="ModernBERT-full")
    p.add_argument("--b_timing_runs", required=True)
    args = p.parse_args()
    device = torch.device("cuda")

    test_ids = sorted(GroupedWindowDataset(PROCESSED_DIR / "test_windows.pkl").ids, key=int)
    ids = random.Random(SEED).sample(test_ids, N_TIMED)
    A = describe_model(args.a_name, PROJECT_ROOT / args.a, args.a_timing_runs.split(","), ids, device)
    B = describe_model(args.b_name, PROJECT_ROOT / args.b, args.b_timing_runs.split(","), ids, device)

    out = {"gpu": torch.cuda.get_device_name(0), "timed_response_ids": ids, "models": [A, B]}
    out_dir = PROJECT_ROOT / args.b
    (out_dir / "comparison.json").write_text(json.dumps(out, indent=2))
    md = render(A, B, out["gpu"])
    (out_dir / "comparison.md").write_text(md)
    print(md)
    print(f"saved -> {out_dir}/comparison.json, comparison.md")


def ms(x: dict) -> str:
    return f"{x['mean']:.4f} ± {x['std']:.4f}"


def render(A: dict, B: dict, gpu: str) -> str:
    L = [f"# {A['name']} vs {B['name']} (test, mean ± sample std over 3 seeds)", "",
         f"| task | metric | {A['name']} | {B['name']} | Δ (B − A) |", "|---|---|---|---|---|"]
    for task in ROWS:
        for level, metric, label in (("case", "f1", "case F1"), ("span", "precision", "span P"),
                                     ("span", "recall", "span R"), ("span", "f1", "span F1")):
            a, b = A["test"][task][level][metric], B["test"][task][level][metric]
            L.append(f"| {task} | {label} | {ms(a)} | {ms(b)} | {b['mean'] - a['mean']:+.4f} |")
    L += ["", f"Efficiency ({gpu}):", "", f"| | {A['name']} | {B['name']} |", "|---|---|---|",
          f"| parameters | {A['params_M']:.1f} M | {B['params_M']:.1f} M |",
          f"| train time / epoch (median; runs: {', '.join(A['timing_runs'])} / {', '.join(B['timing_runs'])}) | "
          f"{A['train_s_per_epoch_median']:.1f} s | {B['train_s_per_epoch_median']:.1f} s |",
          f"| inference / response, batch 1, median of {N_TIMED} | {A['infer_ms_per_response_median']:.2f} ms | "
          f"{B['infer_ms_per_response_median']:.2f} ms |",
          f"| peak GPU memory per training process | {A['peak_gpu_mem_gib']:.1f} GiB | {B['peak_gpu_mem_gib']:.1f} GiB |",
          f"| frozen thresholds (seed: thr) | {A['thresholds']} | {B['thresholds']} |"]

    ea, eb = A["error_analysis"], B["error_analysis"]
    L += ["", "## Seed 42: char recall by label type", "",
          f"| scope | label type | spans | {A['name']} recall | {B['name']} recall | "
          f"{A['name']} detected % | {B['name']} detected % |", "|---|---|---|---|---|---|---|"]
    for scope in ["All"] + list(TASKS):
        for lt in LABEL_TYPES:
            ra = ea["recall_by_label_type"].get(scope, {}).get(lt)
            rb = eb["recall_by_label_type"].get(scope, {}).get(lt)
            if ra:
                L.append(f"| {scope} | {lt} | {ra['spans']} | {ra['char_recall']:.4f} | {rb['char_recall']:.4f} | "
                         f"{ra['detected_%']:.1f} | {rb['detected_%']:.1f} |")
    L += ["", f"## Seed 42: by number of DistilBERT windows (buckets from `{eb['bucket_from']}`)", "",
          f"| n_windows | n | {A['name']} span P / R / F1 | {B['name']} span P / R / F1 |", "|---|---|---|---|"]
    for name, _, _ in BUCKETS:
        a, b = ea["by_n_windows"][name], eb["by_n_windows"][name]
        assert a["n"] == b["n"], "bucket sizes differ: buckets are not aligned"
        L.append(f"| {name} | {a['n']} | {a['span_precision']:.4f} / {a['span_recall']:.4f} / {a['span_f1']:.4f} | "
                 f"{b['span_precision']:.4f} / {b['span_recall']:.4f} / {b['span_f1']:.4f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
