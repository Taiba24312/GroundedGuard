#!/usr/bin/env bash
# GroundedGuard v1.0: end-to-end reproduction, from the raw RAGTruth download to the final
# model comparison. Runs the FROZEN configurations only (the ablations that led to them are
# documented in outputs/module4/runs.csv and outputs/module5b/runs.csv).
#
#   bash scripts/reproduce.sh --dry-run   # checks every module, flag, config and script; runs nothing
#   bash scripts/reproduce.sh             # runs everything (several hours of GPU time)
#
# A real run needs a fresh checkout: freeze_choice / evaluate_test refuse to overwrite existing
# frozen files and train.py refuses existing checkpoint folders. Runtimes are approximate, for one
# RTX PRO 6000 (96 GB) running one job at a time; GPU training is not bit-exact across runs or GPUs.
# Python: the `groundedguard` conda env (Python 3.11, torch 2.11.0+cu128, transformers 5.21.1).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FAILS=0

# run <description> -- <command...>
# Dry run: for `python -m <module>`, the module must import and print --help, and every --flag used
# must appear in that help; every referenced file under configs/, scripts/, examples/ or tests/ must
# exist. Other paths are produced by earlier steps and are only reported.
run() {
  local desc="$1"; shift; [[ "$1" == "--" ]] && shift
  if [[ $DRY -eq 0 ]]; then
    echo ">>> $desc"; "$@"; return
  fi
  local status="ok" notes=() args=("$@")
  for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "-m" && "${args[$((i + 1))]}" == groundedguard.* ]]; then
      local mod="${args[$((i + 1))]}" help
      if ! help="$("$PY" -m "$mod" --help 2>/dev/null)"; then
        status="FAIL"; notes+=("module $mod does not run --help")
      else
        for a in "${args[@]}"; do
          if [[ "$a" == --* && "$a" != "--help" ]] && ! grep -q -- "${a%%=*}" <<<"$help"; then
            status="FAIL"; notes+=("$mod has no flag $a")
          fi
        done
      fi
    fi
  done
  for a in "$@"; do
    for f in ${a//,/ }; do
      if [[ "$f" == configs/* || "$f" == scripts/* || "$f" == examples/* || "$f" == tests/* ]]; then
        [[ -e "$f" ]] || { status="FAIL"; notes+=("missing $f"); }
      elif [[ "$f" == outputs/* || "$f" == data/* ]]; then
        notes+=("$f produced earlier: $([[ -e "$f" ]] && echo present || echo absent)")
      fi
    done
  done
  [[ "$status" == "FAIL" ]] && FAILS=$((FAILS + 1))
  printf '[%s] %s\n' "$status" "$desc"
  for n in "${notes[@]+"${notes[@]}"}"; do printf '       %s\n' "$n"; done
}

# ---------------------------------------------------------------- Module 1: data (CPU, network)
# ~1 min: two JSONL files from github.com/ParticleMedia/RAGTruth + manifest with sha256
run "download RAGTruth" -- "$PY" -m groundedguard.data.download
# ~1 min (CPU; downloads the DistilBERT tokenizer): schema, label and length report
run "explore the raw data" -- "$PY" -m groundedguard.data.explore --tokenizer distilbert-base-uncased

# ---------------------------------------------------------------- Module 2: preprocessing (CPU)
# < 1 min: quality filter, span merge, official dev split, context serialization
run "preprocess (train/dev/test jsonl)" -- "$PY" -m groundedguard.data.preprocess
# < 1 min each: tokenization and context windows
run "windows, DistilBERT (512 tokens, stride 128)" -- "$PY" -m groundedguard.data.windowing
run "windows, ModernBERT (8192 tokens, one window per response)" -- \
  "$PY" -m groundedguard.data.windowing --tokenizer answerdotai/ModernBERT-base --max_len 8192 --suffix _modernbert

# ---------------------------------------------------------------- Module 3: baselines (CPU)
# < 1 min: B0 nothing, B1 whole response, B2 lexical overlap (min words tuned on dev)
run "baselines" -- "$PY" -m groundedguard.baselines.heuristic

# ---------------------------------------------------------------- Module 4: DistilBERT, frozen config B (GPU)
# ~10 min each on one GPU (66 s/epoch x 8 epochs + dev eval), ~6 GiB GPU memory
for s in 42 43 44; do
  name="B_min_full_e8"; [[ $s -ne 42 ]] && name="B_min_full_e8_seed$s"
  run "train DistilBERT seed $s -> checkpoints/$name" -- \
    "$PY" -m groundedguard.models.train --config configs/train_distilbert.yaml --epochs 8 --pos_weight full \
      --seed "$s" --run_name "$name"
done

# ---------------------------------------------------------------- Module 5b: ModernBERT, full context (GPU)
# ~24 min each (171 s/epoch x 8 epochs + dev eval), ~47 GiB GPU memory
for s in 42 43 44; do
  run "train ModernBERT seed $s -> checkpoints/modernbert_lr5e-5_s$s" -- \
    "$PY" -m groundedguard.models.train --config configs/train_modernbert.yaml --lr 5e-5 --seed "$s" \
      --run_name "modernbert_lr5e-5_s$s"
done

# ---------------------------------------------------------------- Module 5: freeze on dev, then test ONCE (GPU)
# ~1 min each: dev threshold sweep + sha256 of the weights, written BEFORE test is loaded
run "freeze DistilBERT choice (dev only)" -- "$PY" -m groundedguard.eval.freeze_choice
run "freeze ModernBERT choice (dev only)" -- \
  "$PY" -m groundedguard.eval.freeze_choice --out_dir outputs/module5b \
    --runs modernbert_lr5e-5_s42,modernbert_lr5e-5_s43,modernbert_lr5e-5_s44 \
    --description "ModernBERT-base full context (1 window, MAX_LEN 8192), lr 5e-5 (tuned over {2e-5, 5e-5}), min aggregation, pos_weight full, 8 epochs, sdpa"
# ~1 min each: scores test once with the frozen checkpoints and thresholds
run "score test once, DistilBERT" -- "$PY" -m groundedguard.eval.evaluate_test
run "score test once, ModernBERT" -- "$PY" -m groundedguard.eval.evaluate_test --out_dir outputs/module5b

# ---------------------------------------------------------------- analysis (CPU) and comparison (GPU)
# < 1 min each
run "error analysis, DistilBERT seed 42" -- "$PY" -m groundedguard.analysis.error_analysis
run "error analysis, ModernBERT seed 42 (DistilBERT window buckets)" -- \
  "$PY" -m groundedguard.analysis.error_analysis --preds outputs/module5b/preds_seed42.jsonl \
    --out_dir outputs/module5b --bucket_from outputs/module5/preds_seed42.jsonl
# ~2 min, GPU otherwise idle (inference timing). In this script every run trained alone, so the
# frozen DistilBERT runs serve as timing runs (v1.0 used distilbert_minpool_v1, same windows/batching).
run "compare DistilBERT vs ModernBERT" -- \
  "$PY" -m groundedguard.eval.compare_models --a_timing_runs B_min_full_e8,B_min_full_e8_seed43,B_min_full_e8_seed44 \
    --b_timing_runs modernbert_lr5e-5_s42,modernbert_lr5e-5_s43,modernbert_lr5e-5_s44

# ---------------------------------------------------------------- Modules 6-7: inference checks (GPU)
# ~3 min: predictor parity with the saved predictions + unit tests; ~10 s: API tests
run "predictor tests" -- "$PY" -m pytest tests/test_predictor.py -q
run "API tests" -- "$PY" -m pytest tests/test_api.py -q
# ~10 min (CPU part dominates): batch-1 latency, both models, GPU and CPU
run "latency" -- "$PY" -m groundedguard.inference.latency
# ~2 min: fp32 vs bf16 on dev
run "precision check (dev)" -- "$PY" scripts/precision_dev.py
# The API load test needs a running server: see README ("Measured performance").

if [[ $DRY -eq 1 ]]; then
  echo
  [[ $FAILS -eq 0 ]] && echo "dry run: all referenced modules, flags, configs and scripts OK" \
                     || { echo "dry run: $FAILS step(s) FAILED"; exit 1; }
fi
