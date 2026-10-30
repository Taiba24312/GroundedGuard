#!/usr/bin/env bash
# Module 6 demos: three hand-written responses to one hand-written context (examples/demos.json).
# Demos, not evaluation. Run from the project root:  bash examples/run_demos.sh [modernbert|distilbert]
set -euo pipefail
MODEL="${1:-modernbert}"
python - "$MODEL" <<'PY'
import json, subprocess, sys
spec = json.load(open("examples/demos.json"))
for d in spec["demos"]:
    print(f"=== {d['name']}", flush=True)
    subprocess.run([sys.executable, "-m", "groundedguard.inference.predictor", "--model", sys.argv[1],
                    "--question", spec["question"], "--context_file", spec["passages_file"],
                    "--response", d["response"]], check=True)
    print(flush=True)
PY
