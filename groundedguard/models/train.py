"""Module 4: train DistilBERT on MIN-over-windows token scores.

Run from the project root:
    python -m groundedguard.models.train --config configs/train_distilbert.yaml --run_name NAME
    python -m groundedguard.models.train --config configs/train_distilbert.yaml --run_name smoke \
        --max_train_responses 64 --epochs 40 --eval_on_train          # overfit smoke test

Only the train and dev splits are read. The test split is never loaded here.

After every epoch the dev set is scored with the official metric at thresholds
0.05..0.95; the threshold with the best dev All span F1 is kept for that epoch, and
the epoch with the best dev All span F1 is saved.

Writes checkpoints/<run_name>/:
    model + tokenizer (best epoch), best_threshold.json, config.yaml (resolved), history.json
and appends one row to the config's runs_csv (relative to the project root).
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from groundedguard.config import CHECKPOINTS_DIR, PROCESSED_DIR, PROJECT_ROOT
from groundedguard.data.windowing import tokens_to_spans
from groundedguard.eval.metrics import TASKS, evaluate, format_table
from groundedguard.models.aggregate import AGGREGATIONS, aggregate_windows, grouped_loss
from groundedguard.models.dataset import Collator, GroupedWindowDataset, WindowBudgetSampler

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
CLI_KEYS = {
    "model_name": str, "lr": float, "weight_decay": float, "warmup_ratio": float, "epochs": int,
    "grad_clip": float, "max_windows_per_batch": int, "eval_max_windows_per_batch": int, "seed": int,
    "pos_weight": str, "num_workers": int, "log_every": int, "max_train_responses": int,
    "aggregation": str, "tau": float, "windows_suffix": str,
    "max_tokens_per_batch": int, "eval_max_tokens_per_batch": int, "attn_implementation": str,
    "runs_csv": str,
}


# ----------------------------------------------------------------------------- setup
def load_config(argv=None) -> tuple[dict, bool]:
    p = argparse.ArgumentParser(description="Train DistilBERT with MIN-over-windows token scores.")
    p.add_argument("--config", required=True, help="YAML file; any key below overrides it")
    p.add_argument("--run_name", default=None, help="checkpoints/<run_name>/ (default: timestamp)")
    p.add_argument("--overwrite", action="store_true", help="reuse an existing non-empty run dir")
    for key, typ in CLI_KEYS.items():
        p.add_argument(f"--{key}", type=typ, default=None)
    p.add_argument("--eval_on_train", action="store_true", default=None)
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    base = dict(cfg)
    missing = (set(CLI_KEYS) | {"bf16", "eval_on_train"}) - cfg.keys()
    if missing:
        raise ValueError(f"{args.config} is missing keys: {sorted(missing)}")
    for key in list(CLI_KEYS) + ["eval_on_train"]:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.no_bf16:
        cfg["bf16"] = False
    if cfg["aggregation"] not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {cfg['aggregation']!r}")
    cfg["config_diff"] = "; ".join(f"{k}={cfg[k]}" for k in base if cfg[k] != base[k]) or "none"
    cfg["run_name"] = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    cfg["config_file"] = str(args.config)
    return cfg, args.overwrite


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def token_counts(ds: GroupedWindowDataset) -> tuple[int, int]:
    pos = sum(sum(m["token_labels"]) for m in ds.responses.values())
    tot = sum(len(m["token_labels"]) for m in ds.responses.values())
    return pos, tot - pos


def resolve_pos_weight(spec, pos: int, neg: int) -> float | None:
    spec = "none" if spec is None else str(spec).strip().lower()
    if spec in ("none", "null"):
        return None
    if spec == "sqrt":
        return math.sqrt(neg / pos)
    if spec == "full":
        return neg / pos
    return float(spec)


def exact_total_steps(sampler: WindowBudgetSampler, epochs: int) -> int:
    """Batches per epoch depend on that epoch's shuffle, so count them for every epoch."""
    total = 0
    for e in range(epochs):
        sampler.set_epoch(e)
        total += sum(1 for _ in sampler)
    return total


def model_kwargs(cfg: dict) -> dict:
    """Extra from_pretrained kwargs; attn_implementation only when the config sets one."""
    return {"attn_implementation": cfg["attn_implementation"]} if cfg.get("attn_implementation") else {}


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict_probs(model, ds, collate, device, max_windows: int, bf16: bool, num_workers: int,
                  aggregation: str = "min", tau: float = 1.0, max_tokens: int | None = None) -> dict:
    """{response id: np.array of per-token hallucination probabilities}, windows aggregated
    exactly as in training."""
    model.eval()
    sampler = WindowBudgetSampler(ds.n_windows, max_windows, window_lens=ds.window_lens, max_tokens=max_tokens)
    loader = DataLoader(ds, batch_sampler=sampler,
                        collate_fn=collate, num_workers=num_workers)
    out = {}
    for b in loader:
        t = to_device(b, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            logits = model(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).logits.squeeze(-1)
        probs = torch.sigmoid(aggregate_windows(logits.float(), t, aggregation, tau)).cpu().numpy()
        for i, rid in enumerate(b["ids"]):
            out[rid] = probs[i, : int(b["response_len"][i])]
    return out


def spans_at(meta: dict, token_probs, threshold: float) -> list[list[int]]:
    """Char spans of the runs of response tokens with prob >= threshold."""
    return tokens_to_spans(meta["response_offsets"], (token_probs >= threshold).astype(int).tolist())


def sweep_thresholds(ds, probs: dict, thresholds=THRESHOLDS) -> tuple[float, dict, dict]:
    """Official metric at every threshold -> (best threshold, {t: All span F1}, metrics at best)."""
    results = {}
    for t in thresholds:
        results[t] = evaluate([
            {"id": rid, "task_type": m["task_type"], "response": m["response"],
             "gold_spans": m["hallucination_spans"], "pred_spans": spans_at(m, probs[rid], t)}
            for rid, m in ds.responses.items()])
    sweep = {t: results[t]["All"]["span"]["f1"] for t in thresholds}
    best = max(thresholds, key=lambda t: sweep[t])
    return best, sweep, results[best]


def epoch_table(history: list[dict], best_epoch: int) -> str:
    head = (f"{'ep':>3s} {'loss':>7s} {'thr':>5s} | {'caseF1':>6s} | {'spanP':>6s} {'spanR':>6s} {'spanF1':>6s} | "
            + " ".join(f"{t[:7] + ' F1':>10s}" for t in TASKS) + f" | {'train s':>7s} {'eval s':>6s} {'GiB':>5s}")
    rows = [head, "-" * len(head)]
    for h in history:
        m = h["dev"]["metrics"]
        a = m["All"]
        row = (f"{h['epoch']:>3d} {h['train_loss']:7.4f} {h['dev']['best_threshold']:5.2f} | "
               f"{a['case']['f1']:6.4f} | {a['span']['precision']:6.4f} {a['span']['recall']:6.4f} "
               f"{a['span']['f1']:6.4f} | " + " ".join(f"{m[t]['span']['f1']:10.4f}" for t in TASKS if t in m)
               + f" | {h['train_time_s']:7.1f} {h['eval_time_s']:6.1f} {h['peak_gpu_mem_gib']:5.1f}")
        if "train_eval" in h:
            te = h["train_eval"]
            row += f" | train@{te['best_threshold']:.2f} span F1 {te['metrics']['All']['span']['f1']:.4f}"
        rows.append(row + ("  *best" if h["epoch"] == best_epoch else ""))
    return "\n".join(rows)


def append_run_row(cfg: dict, best: dict) -> None:
    m = best["dev"]["metrics"]
    row = {"run_name": cfg["run_name"], "config_diff": cfg["config_diff"], "best_epoch": best["epoch"],
           "best_threshold": best["dev"]["best_threshold"], "dev_case_f1": round(m["All"]["case"]["f1"], 4),
           **{f"dev_span_{name}": round(m["All"]["span"][k], 4)
              for name, k in (("P", "precision"), ("R", "recall"), ("F1", "f1"))},
           **{f"dev_span_f1_{t}": round(m[t]["span"]["f1"], 4) if t in m else "" for t in TASKS}}
    runs_csv = PROJECT_ROOT / cfg["runs_csv"]
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not runs_csv.exists()
    with runs_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------------- main
def main() -> None:
    cfg, overwrite = load_config()
    run_dir = CHECKPOINTS_DIR / cfg["run_name"]
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise SystemExit(f"{run_dir} exists and is not empty; pick another --run_name or pass --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = bool(cfg["bf16"]) and device.type == "cuda"

    sfx = cfg["windows_suffix"] or ""
    train_ds = GroupedWindowDataset(PROCESSED_DIR / f"train_windows{sfx}.pkl")
    dev_ds = GroupedWindowDataset(PROCESSED_DIR / f"dev_windows{sfx}.pkl")
    for name, ds in (("train", train_ds), ("dev", dev_ds)):
        if ds.config["tokenizer"] != cfg["model_name"]:
            raise ValueError(f"{name}_windows{sfx}.pkl was built with {ds.config['tokenizer']!r}, "
                             f"model is {cfg['model_name']!r}; re-run windowing")
    if cfg["max_train_responses"]:
        train_ds.subset(cfg["max_train_responses"], cfg["seed"])

    pos, neg = token_counts(train_ds)
    pos_weight = resolve_pos_weight(cfg["pos_weight"], pos, neg)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForTokenClassification.from_pretrained(cfg["model_name"], **model_kwargs(cfg), num_labels=1).to(device)
    collate = Collator(tok.pad_token_id)

    sampler = WindowBudgetSampler(train_ds.n_windows, cfg["max_windows_per_batch"], shuffle=True, seed=cfg["seed"],
                                  window_lens=train_ds.window_lens, max_tokens=cfg["max_tokens_per_batch"])
    total_steps = exact_total_steps(sampler, cfg["epochs"])
    warmup_steps = round(cfg["warmup_ratio"] * total_steps)
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=cfg["num_workers"])

    # No weight decay on 1-D parameters (biases, LayerNorm).
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for p in params if p.ndim >= 2], "weight_decay": cfg["weight_decay"]},
                             {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}], lr=cfg["lr"])
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    resolved = {**cfg, "resolved": {
        "pos_weight_value": pos_weight, "train_pos_tokens": pos, "train_neg_tokens": neg,
        "n_train_responses": len(train_ds), "n_train_windows": sum(train_ds.n_windows),
        "windows_stride": train_ds.config["stride"], "n_dev_responses": len(dev_ds), "total_steps": total_steps, "warmup_steps": warmup_steps,
        "thresholds": THRESHOLDS, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "bf16_active": bf16, "torch": str(torch.__version__), "transformers": transformers.__version__,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    print(yaml.safe_dump(resolved, sort_keys=False), flush=True)

    history, best_epoch, best_f1, step = [], 0, -1.0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        sampler.set_epoch(epoch - 1)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        t0, losses = time.time(), []
        for b in loader:
            t = to_device(b, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                logits = model(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).logits.squeeze(-1)
            loss = grouped_loss(logits.float(), t, pos_weight, cfg["aggregation"], cfg["tau"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            step += 1
            if step % cfg["log_every"] == 0:
                print(f"  step {step}/{total_steps}  loss(last {cfg['log_every']}) "
                      f"{np.mean(losses[-cfg['log_every']:]):.4f}  lr {sched.get_last_lr()[0]:.2e}", flush=True)
        train_time = time.time() - t0

        t0 = time.time()
        args = (collate, device, cfg["eval_max_windows_per_batch"], bf16, cfg["num_workers"],
                cfg["aggregation"], cfg["tau"], cfg["eval_max_tokens_per_batch"])
        thr, sweep, dev_metrics = sweep_thresholds(dev_ds, predict_probs(model, dev_ds, *args))
        entry = {"epoch": epoch, "step": step, "train_loss": float(np.mean(losses)), "lr_end": sched.get_last_lr()[0],
                 "dev": {"best_threshold": thr, "all_span_f1_by_threshold": sweep, "metrics": dev_metrics}}
        if cfg["eval_on_train"]:
            tthr, _, train_metrics = sweep_thresholds(train_ds, predict_probs(model, train_ds, *args))
            entry["train_eval"] = {"best_threshold": tthr, "metrics": train_metrics}
        entry["train_time_s"] = round(train_time, 1)
        entry["eval_time_s"] = round(time.time() - t0, 1)
        entry["peak_gpu_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0.0
        history.append(entry)

        dev_f1 = dev_metrics["All"]["span"]["f1"]
        if dev_f1 > best_f1:
            best_epoch, best_f1 = epoch, dev_f1
            model.save_pretrained(run_dir)
            tok.save_pretrained(run_dir)
            (run_dir / "best_threshold.json").write_text(json.dumps(
                {"threshold": thr, "epoch": epoch, "dev_all_span_f1": dev_f1,
                 "dev_all_case_f1": dev_metrics["All"]["case"]["f1"]}, indent=2))
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        print(f"\n[epoch {epoch}/{cfg['epochs']}] train loss {entry['train_loss']:.4f} | "
              f"dev best threshold {thr} | dev All span F1 {dev_f1:.4f}"
              f"{'  (new best, saved)' if best_epoch == epoch else ''} | train {entry['train_time_s']}s "
              f"eval {entry['eval_time_s']}s | peak GPU {entry['peak_gpu_mem_gib']} GiB", flush=True)
        print(format_table({f"dev ep{epoch} @{thr}": dev_metrics}), flush=True)
        if "train_eval" in entry:
            print(format_table({f"train ep{epoch} @{entry['train_eval']['best_threshold']}": entry["train_eval"]["metrics"]}),
                  flush=True)

    assert step == total_steps, f"ran {step} steps, scheduler planned {total_steps}"
    print("\n" + "=" * 100 + f"\nper-epoch dev summary ({cfg['run_name']}), best epoch {best_epoch}\n" + "=" * 100)
    print(epoch_table(history, best_epoch))
    append_run_row(cfg, history[best_epoch - 1])
    print(f"\nsaved -> {run_dir}  (best epoch {best_epoch}, dev All span F1 {best_f1:.4f}); row added to {cfg['runs_csv']}",
          flush=True)


if __name__ == "__main__":
    main()
