"""Central paths and constants for GroundedGuard.

All paths are derived from this file's location, so the code works no matter
which directory you launch it from (as long as you use `python -m ...` from the
project root).
"""

from pathlib import Path

# <project_root>/groundedguard/config.py -> parents[1] == <project_root>
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "ragtruth"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Official RAGTruth repo (ParticleMedia/RAGTruth, ACL 2024).
RAGTRUTH_REPO = "ParticleMedia/RAGTruth"
RAGTRUTH_BRANCH = "main"
RAGTRUTH_RAW_BASE = f"https://raw.githubusercontent.com/{RAGTRUTH_REPO}/{RAGTRUTH_BRANCH}/dataset"
RAGTRUTH_FILES = ["response.jsonl", "source_info.jsonl"]

# Field names as documented in the official README. Module 1 checks the real
# files against these and reports anything missing or extra.
EXPECTED_RESPONSE_FIELDS = {
    "id", "source_id", "model", "temperature", "labels", "split", "quality", "response",
}
EXPECTED_SOURCE_FIELDS = {"source_id", "task_type", "source", "source_info", "prompt"}
EXPECTED_LABEL_FIELDS = {"start", "end", "text", "label_type"}  # README also mentions meta, due_to_null, implicit_true

SEED = 42

# ----------------------------------------------------------------------------- Module 2
# Same recipe as the official baseline (ParticleMedia/RAGTruth/baseline/prepare_dataset.py):
# keep quality == "good", dev = 50 train sources per task, random.seed(2024).
KEEP_QUALITY = {"good"}
DEV_SOURCES_PER_TASK = 50
DEV_SEED = 2024
DEV_TASK_ORDER = ["QA", "Summary", "Data2txt"]  # order matters for reproducing the sample

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 512       # DistilBERT position limit, incl. [CLS] and 2x [SEP]
DOC_STRIDE = 128    # tokens of overlap between consecutive context windows
MIN_CONTEXT_TOKENS = 64  # a window must show at least this much context