"""Module 5a, step 1: pre-register the test-time checkpoints and thresholds. DEV ONLY.

For each seed checkpoint of the frozen config (B_min_full_e8), re-sweep the dev threshold
over 0.05..0.90 (step 0.05) and 0.91..0.99 (step 0.01), pick the best by dev All span F1,
and write everything needed to score test, plus the sha256 of each checkpoint's weights.
Refuses to run if the file already exists: after it is written, nothing changes.

Run from the project root (same GPU type as training, for identical bf16 numerics):
    CUDA_VISIBLE_DEVICES=0 python -m groundedguard.eval.freeze_choice
    CUDA_VISIBLE_DEVICES=0 python -m groundedguard.eval.freeze_choice --out_dir outputs/module5b \
        --runs modernbert_lr5e-5_s42,modernbert_lr5e-5_s43,modernbert_lr5e-5_s44 --description "..."

Writes:
    <out_dir>/frozen_choice.json   (default outputs/module5)
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForTokenClassification, AutoTokenizer

from groundedguard.config import CHECKPOINTS_DIR, OUTPUTS_DIR, PROCESSED_DIR, PROJECT_ROOT
from groundedguard.eval.metrics import TASKS
from groundedguard.models.dataset import Collator, GroupedWindowDataset
from groundedguard.models.train import model_kwargs, predict_probs, sweep_thresholds

OUT_DIR = OUTPUTS_DIR / "module5"
DEFAULT_RUNS = "B_min_full_e8,B_min_full_e8_seed43,B_min_full_e8_seed44"
DEFAULT_DESCRIPTION = "B_min_full_e8 (min aggregation, pos_weight full, 8 epochs, stride 128)"
THRESHOLD_GRID = [round(0.05 * i, 2) for i in range(1, 19)] + [round(0.90 + 0.01 * i, 2) for i in range(1, 10)]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run(run_dir, device):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    tok = AutoTokenizer.from_pretrained(run_dir)
    model = AutoModelForTokenClassification.from_pretrained(run_dir, **model_kwargs(cfg)).to(device)
    return cfg, tok, model


def run_probs(cfg, tok, model, ds, device) -> dict:
    bf16 = bool(cfg["bf16"]) and device.type == "cuda"
    return predict_probs(model, ds, Collator(tok.pad_token_id), device, cfg["eval_max_windows_per_batch"], bf16,
                         cfg["num_workers"], cfg.get("aggregation", "min"), cfg.get("tau", 1.0),
                         cfg.get("eval_max_tokens_per_batch"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-register test checkpoints and thresholds (dev only).")
    parser.add_argument("--runs", default=DEFAULT_RUNS, help="comma-separated checkpoints/<run> names")
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    args = parser.parse_args()
    out_dir = PROJECT_ROOT / Path(args.out_dir)
    FROZEN = out_dir / "frozen_choice.json"
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} already exists; the choice is frozen and must not change.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    entries = []
    for run in args.runs.split(","):
        run_dir = CHECKPOINTS_DIR / run
        cfg, tok, model = load_run(run_dir, device)
        seed = cfg["seed"]
        sfx = cfg.get("windows_suffix") or ""
        ds = GroupedWindowDataset(PROCESSED_DIR / f"dev_windows{sfx}.pkl")
        best, sweep, m = sweep_thresholds(ds, run_probs(cfg, tok, model, ds, device), THRESHOLD_GRID)
        trained = json.loads((run_dir / "best_threshold.json").read_text())
        entries.append({
            "seed": seed, "run_name": run, "model_name": cfg["model_name"], "config_diff": cfg.get("config_diff"), "checkpoint": str(run_dir.relative_to(PROJECT_ROOT)),
            "weights_file": "model.safetensors", "weights_sha256": sha256_file(run_dir / "model.safetensors"),
            "threshold": best, "training_sweep_threshold": trained["threshold"], "best_epoch": trained["epoch"],
            "aggregation": cfg.get("aggregation", "min"), "tau": cfg.get("tau", 1.0), "windows_suffix": sfx,
            "dev": {"all_case": {k: m["All"]["case"][k] for k in ("precision", "recall", "f1")},
                    "all_span": {k: m["All"]["span"][k] for k in ("precision", "recall", "f1")},
                    "span_f1_by_task": {t: m[t]["span"]["f1"] for t in TASKS if t in m}},
            "dev_all_span_f1_by_threshold": sweep,
        })
        print(f"seed {seed} ({run}): dev threshold {best} (training sweep: {trained['threshold']}) | "
              f"dev All span F1 {m['All']['span']['f1']:.4f} | case F1 {m['All']['case']['f1']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps({
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "config": args.description,
        "selection_rule": "per seed: threshold maximizing dev All span F1 (first maximum on ties)",
        "threshold_grid": THRESHOLD_GRID,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "runs": entries,
    }, indent=2))
    print(f"frozen -> {FROZEN}")


if __name__ == "__main__":
    main()
